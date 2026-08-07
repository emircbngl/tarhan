"""The forge display — TARHAN's live progress, driven by work rather than a clock.

Not wired to any command yet, and that is deliberate. The criterion for showing a
progress display is that there is something to watch: today a 1D solve takes
milliseconds and the whole suite about fifteen seconds, so a monitor would be
decoration with a maintenance cost. This module exists so that the day a run
does take thirty seconds — a 3D device, a large 2D sweep — wiring it up is
mechanical rather than a design exercise.

Three properties are the whole point, and each is a mistake this file refuses to
make.

**It never clears the screen.** Redrawing with ``ESC[2J`` wipes whatever the user
had above: the command they ran, the warning that explains the run, the previous
solve they were comparing against. Only this display's own lines are redrawn, by
walking the cursor back over them, so scrollback survives and the session still
reads as a log afterwards.

**Progress is never a timer.** A bar interpolated across an expected duration
shows a number that looks measured and is not. Here it is
``completed_stages / total_stages``, refined only by a fraction the caller
actually knows — Newton iteration k of its cap, bias point i of n. The hammer
advances on :meth:`Forge.tick`, so if the work stalls the hammer stops. A hammer
swinging over stalled work is a lie told ten times a second.

**It disappears when nobody is watching.** No terminal means no animation: one
plain line per completed stage, so a CI log gets a record instead of a screenful
of block characters.

Styling is monochrome by construction — weight and reverse video, never hue. A
display that picks no colour cannot clash with the user's terminal theme, which
is the usual reason someone else's TUI looks wrong on your machine.
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from tarhan import cliout

ESC = "\x1b["
HIDE_CURSOR = f"{ESC}?25l"
SHOW_CURSOR = f"{ESC}?25h"
ERASE_LINE = f"{ESC}2K"

FULL_WIDTH = 76          # anvil, wordmark and the stage list all fit
COMPACT_WIDTH = 46       # wordmark dropped, anvil and stage list kept
# below COMPACT_WIDTH: a single status line and nothing else

# The working indicator: the forge, shrunk to four cells.
#
# A generic spinner would have been easier and would have looked like every
# other CLI. These four cells keep the anvil on screen the whole time and let
# the hammer strike it — raised, falling, contact, recovering. The anvil never
# moves, so the eye reads the motion as a blow rather than a rotation, and the
# spark appears on exactly one frame, so a spark still means metal was hit.
FORGE_CELL = ("╱▂▄▂", "╲▂▄▂", "✦▂▄▂", "╲▂▄▂")
FORGE_CELL_ASCII = ("/-=-", "\\-=-", "*-=-", "\\-=-")
STRIKE_CELL = 2

# Orbitron-inspired geometry drawn as terminal cells, because a CLI cannot pick
# the user's font.
WORDMARK = (
    "█████   ███   ████   █   █   ███   █   █",
    "  █    █   █  █   █  █   █  █   █  ██  █",
    "  █    █████  ████   █████  █████  █ █ █",
    "  █    █   █  █ █    █   █  █   █  █  ██",
    "  █    █   █  █  ██  █   █  █   █  █   █",
)

WORDMARK_ASCII = (
    "TTTTT   AAA   RRRR   H   H   AAA   N   N",
    "  T    A   A  R   R  H   H  A   A  NN  N",
    "  T    AAAAA  RRRR   HHHHH  AAAAA  N N N",
    "  T    A   A  R  R   H   H  A   A  N  NN",
    "  T    A   A  R   R  H   H  A   A  N   N",
)

# Horn on the left, tapering body, narrow waist, wide foot.
ANVIL = (
    "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄  ",
    "  ▀▀█████████████████████ █████",
    "      ▀████████████████████████",
    "          ▀████████████████████",
    "             ▀█████████████▀▀▀ ",
    "               ████████        ",
    "               ████████        ",
    "         ▄▄▄▄▄██████████▄▄▄▄▄  ",
    "       ▄██████████████████████▄",
)

ANVIL_ASCII = (
    "=======================  ===== ",
    "  ########################### ",
    "     ######################### ",
    "        ###################### ",
    "            ###############    ",
    "               ########        ",
    "               ########        ",
    "         ##################    ",
    "       ######################  ",
)

HAMMER = (
    ("             ▄████████▄       ",
     "              ▀███████        ",
     "                  ██          ",
     "                   ██         ",
     "                    ██        "),
    ("                              ",
     "          ▄████████▄          ",
     "           ▀███████           ",
     "               ██             ",
     "                 ██           "),
    ("                     ██       ",
     "                  ▄██         ",
     "                ▄██           ",
     "              ▄██             ",
     "       ✦  ▄████████▄  ✦       "),
    ("                              ",
     "          ▄████████▄          ",
     "           ▀███████           ",
     "               ██             ",
     "                 ██           "),
)

HAMMER_ASCII = (
    ("             /########\\       ",
     "              \\######/        ",
     "                  ||          ",
     "                   ||         ",
     "                    ||        "),
    ("                              ",
     "          /########\\          ",
     "           \\######/           ",
     "               ||             ",
     "                 ||           "),
    ("                     ||       ",
     "                  /||         ",
     "                /||           ",
     "              /||             ",
     "       *  /########\\  *       "),
    ("                              ",
     "          /########\\          ",
     "           \\######/           ",
     "               ||             ",
     "                 ||           "),
)

STRIKE_FRAME = 2


@dataclass
class Stage:
    """One real step of the work. ``detail`` is filled in as it is learned."""

    name: str
    detail: str = ""
    done: bool = False
    failed: bool = False
    elapsed: float = 0.0


class Forge:
    """The live display. Real work calls it; it never advances on its own.

    ::

        out = cliout.Output()
        with Forge(["MESH", "ASSEMBLY", "SOLVE"], out) as forge:
            forge.begin("MESH", "reading 2,847 nodes")
            forge.tick(within=k / n)          # one swing per unit of work
            forge.finish("2,847 nodes · Delaunay ok")
            ...
            forge.converged("12 Newton iterations · residual 8.3e-11")

    Everything it writes goes to the :class:`~tarhan.cliout.Output`'s stderr, so
    a ``--format json`` consumer never sees a frame of it.
    """

    def __init__(self, stage_names: Sequence[str],
                 out: Optional[cliout.Output] = None,
                 unicode: Optional[bool] = None,
                 animate: Optional[bool] = None) -> None:
        self.out = out if out is not None else cliout.Output()
        self.stream = self.out.stderr
        self.stages: List[Stage] = [Stage(name) for name in stage_names]
        self._index = -1
        self._frame = 0
        self._within: Optional[float] = None
        self._drawn = 0
        self._started = time.monotonic()
        self._stage_started = self._started
        self._final: Optional[str] = None
        self._failed = False
        self._restored = False
        self._last_logged: Optional[Stage] = None

        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.animate = (tty and not self.out.quiet) if animate is None else animate
        if unicode is None:
            unicode = cliout._encodable(self.stream, "█▄▀✦━─✓›✕╱╲▂")
        self.unicode = unicode

        self._anvil = ANVIL if unicode else ANVIL_ASCII
        self._hammer = HAMMER if unicode else HAMMER_ASCII
        self._wordmark = WORDMARK if unicode else WORDMARK_ASCII
        self._cell = FORGE_CELL if unicode else FORGE_CELL_ASCII
        self._bar_full = "━" if unicode else "="
        self._bar_empty = "─" if unicode else "-"
        self._tick_mark = "✓" if unicode else "+"
        self._cross = "✕" if unicode else "x"
        self._arrow = "›" if unicode else ">"
        # U+00B7 is not ASCII. It reached the indicator through a hard-coded
        # separator and the ascii-console test caught it — the same class of
        # bug that once made `demo` exit non-zero on Windows for a purely
        # cosmetic reason. Every glyph goes through this switch, no exceptions.
        self._sep = " · " if unicode else " | "

    # --- lifecycle --------------------------------------------------------

    def __enter__(self) -> "Forge":
        if self.animate:
            self.stream.write(HIDE_CURSOR)
            self.stream.flush()
            # A SIGTERM would otherwise leave the cursor hidden in the user's
            # shell long after this process is gone: `finally` does not run for
            # it, a signal handler does.
            self._install_handlers()
            atexit.register(self._restore)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and self._final is None:
            self.failed(f"{exc_type.__name__}: {exc}")
        elif self._final is None:
            self._render(force=True)
        self._restore()
        return False

    def _install_handlers(self) -> None:
        def handler(signum, _frame):
            self._restore()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass          # not the main thread, or the platform refuses

    def _restore(self) -> None:
        if self._restored:
            return
        self._restored = True
        if self.animate:
            try:
                self.stream.write(SHOW_CURSOR)
                self.stream.flush()
            except (ValueError, OSError):
                pass

    # --- what real work calls --------------------------------------------

    def begin(self, name: str, detail: str = "") -> None:
        self._index = next(i for i, s in enumerate(self.stages)
                           if s.name == name)
        self.stages[self._index].detail = detail
        self._within = None
        self._stage_started = time.monotonic()
        self._render()

    def tick(self, detail: Optional[str] = None,
             within: Optional[float] = None) -> None:
        """One unit of real work: swing the hammer, optionally refine the bar.

        ``within`` is a fraction the CALLER knows to be true. It is never
        invented here, and no code path advances it with elapsed time.
        """
        self._frame += 1
        if detail is not None and self._index >= 0:
            self.stages[self._index].detail = detail
        if within is not None:
            self._within = min(1.0, max(0.0, within))
        self._render()

    def finish(self, detail: Optional[str] = None) -> None:
        if self._index < 0:
            return
        stage = self.stages[self._index]
        stage.done = True
        stage.elapsed = time.monotonic() - self._stage_started
        if detail is not None:
            stage.detail = detail
        self._within = None
        self._render()

    def converged(self, summary: str) -> None:
        self._final = summary
        self._render(force=True)

    def failed(self, summary: str) -> None:
        self._final = summary
        self._failed = True
        if 0 <= self._index < len(self.stages):
            self.stages[self._index].failed = True
        self._render(force=True)

    def log(self, text: str) -> None:
        """Print a line above the indicator, keeping the indicator at the bottom.

        A terminal cannot truly pin a line to the bottom without the alternate
        buffer — which destroys the scrollback this design exists to protect —
        or a scroll region, which is inconsistent across terminals. Erasing,
        printing and redrawing gets the same feel at none of that cost.
        """
        if not self.animate:
            self.stream.write(text + "\n")
            self.stream.flush()
            return
        out = []
        if self._drawn:
            out.append(f"{ESC}{self._drawn}A")
        out.append(ERASE_LINE + text + "\n")
        self._drawn = 0
        self.stream.write("".join(out))
        self.stream.flush()
        self._render()

    # --- drawing ----------------------------------------------------------

    @property
    def progress(self) -> float:
        """Completed stages, plus only the part of the current one we know."""
        total = len(self.stages) or 1
        done = sum(1 for s in self.stages if s.done)
        if self._within is not None and done < total:
            done += self._within
        return min(1.0, done / total)

    @property
    def _current(self) -> Optional[Stage]:
        if 0 <= self._index < len(self.stages):
            return self.stages[self._index]
        return None

    def _paint(self, text: str, style: str) -> str:
        return self.out.paint(text, style)

    def _bar(self, width: int) -> str:
        width = max(4, width)
        filled = round(self.progress * width)
        return (self._paint(self._bar_full * filled, "bold")
                + self._paint(self._bar_empty * (width - filled), "dim"))

    def _indicator(self, width: int) -> str:
        """The working line: the forge cell, the stage, and what is known."""
        index = self._frame % len(self._cell)
        cell = self._paint(self._cell[index],
                           "reverse" if index == STRIKE_CELL else "")
        stage = self._current
        name = stage.name if stage else "READY"
        elapsed = time.monotonic() - self._started
        facts = [f"{elapsed:.1f}s"]
        if self._within is not None:
            facts.append(f"{round(self._within * 100)}% of {name.lower()}")
        # Count the stage being NAMED on this line, not the number completed:
        # deriving it from the done-count made the line read "MESH … stage 2/4"
        # in the instant between finishing one stage and beginning the next.
        shown = self._index + 1 if self._index >= 0 else 1
        facts.append(f"stage {shown}/{len(self.stages)}")
        joined = self._sep.join(facts)
        tail = self._paint("(" + joined + ")", "dim")
        room = max(0, width - len(name) - len(joined) - 12)
        detail = (stage.detail if stage else "")[:room]
        return f"{cell} {self._paint(name, 'bold')}  {detail}  {tail}"

    def _logo(self, width: int, frame: Optional[int] = None) -> List[str]:
        """The mark. With ``frame``, the hammer is mid-swing; without it, at rest.

        At rest the hammer is gone entirely rather than parked above the anvil:
        a hammer frozen in the air reads as something stalled, which is the one
        impression this display must never give when nothing is wrong.
        """
        if frame is None:
            rows, word_offset, hot_row = list(self._anvil), 0, None
        else:
            rows = list(self._hammer[frame]) + list(self._anvil)
            word_offset, hot_row = 5, (4 if frame == STRIKE_FRAME else None)
        lines = []
        for i, row in enumerate(rows):
            left = self._paint(row.ljust(31),
                               "reverse" if i == hot_row else "")
            word_index = i - word_offset
            if width >= FULL_WIDTH and 0 <= word_index < len(self._wordmark):
                word = self._paint(self._wordmark[word_index], "bold")
                lines.append(f"{left}  {word}")
            else:
                lines.append(left)
        return lines

    def intro(self, strikes: int = 2, pace: float = 0.09) -> None:
        """Play the forge once, then leave the anvil and the wordmark standing.

        This is the one place a timer is the honest driver. A title sequence
        represents no work, so pacing it by the clock claims nothing; a progress
        bar paced by the clock claims something false. Same ``sleep``, opposite
        meaning, and the difference is whether there is work being described.

        The resting mark is *printed*, not held in the redraw region, so the
        indicator that follows appears beneath it and the mark scrolls away
        naturally as the run produces output. Nothing is ever cleared.
        """
        if not self.animate or strikes <= 0:
            return
        width = shutil.get_terminal_size((80, 24)).columns
        if width < COMPACT_WIDTH:
            return                      # no room for a title sequence
        for beat in range(strikes * len(self._hammer)):
            self._blit(self._logo(width, frame=beat % len(self._hammer)))
            time.sleep(max(0.0, pace))
        self._blit(self._logo(width))   # at rest: anvil and wordmark only
        self._drawn = 0                 # printed for good; the run goes below

    def _blit(self, lines: List[str]) -> None:
        """Redraw a block in place by stepping back over what we drew last."""
        out = []
        if self._drawn:
            out.append(f"{ESC}{self._drawn}A")
        for line in lines:
            out.append(ERASE_LINE + line + "\n")
        leftovers = max(0, self._drawn - len(lines))
        for _ in range(leftovers):
            out.append(ERASE_LINE + "\n")
        if leftovers:
            out.append(f"{ESC}{leftovers}A")
        self._drawn = len(lines)
        self.stream.write("".join(out))
        self.stream.flush()

    def _stage_lines(self, width: int) -> List[str]:
        lines = []
        for stage in self.stages:
            if stage.failed:
                mark, style = self._paint(self._cross, "reverse"), "reverse"
            elif stage.done:
                mark, style = self._paint(self._tick_mark, "bold"), "dim"
            elif stage is self._current:
                mark, style = self._paint(self._arrow, "bold"), "bold"
            else:
                dot = "·" if self.unicode else "."
                mark, style = self._paint(dot, "dim"), "dim"
            timing = f"{stage.elapsed:6.2f}s" if stage.done else " " * 7
            detail = stage.detail[: max(0, width - 24)]
            lines.append(f"  {mark} {self._paint(stage.name.ljust(10), style)} "
                         f"{self._paint(timing, 'dim')} {detail}")
        return lines

    def _summary(self, width: int) -> List[str]:
        """The closing block: what happened, not the mark again.

        The anvil belongs at the top of a run, once, as the title sequence.
        Reprinting it here would make a short run mostly logo — and the second
        copy would carry no information the first did not.
        """
        if width < COMPACT_WIDTH:
            return [self._status_line()]
        lines = [self._paint(self._bar_empty * min(width - 1, FULL_WIDTH),
                             "dim")]
        lines.extend(self._stage_lines(width))
        lines.append("")
        lines.append(f"       {self._bar(min(48, width - 14))}  "
                     f"{round(self.progress * 100):3d}%")
        lines.append("")
        lines.append("  " + self._status_line())
        return lines

    def _status_line(self) -> str:
        elapsed = time.monotonic() - self._started
        clock = self._paint(f"{elapsed:.2f}s", "dim")
        if self._final and self._failed:
            return (self._paint(f"{self._cross} FAILED", "reverse")
                    + f"  {self._final}  {clock}")
        if self._final:
            return (self._paint(f"{self._tick_mark} CONVERGED", "bold")
                    + f"  {self._final}  {clock}")
        stage = self._current
        return (self._paint(stage.name if stage else "READY", "bold")
                + f"  {clock}")

    def _render(self, force: bool = False) -> None:
        if not self.animate:
            stage = self._current
            if force or (stage is not None and stage.done
                         and stage is not self._last_logged):
                self._last_logged = stage
                self.stream.write(self._plain_line() + "\n")
                self.stream.flush()
            return

        width = shutil.get_terminal_size((80, 24)).columns
        self._blit(self._summary(width) if self._final
                   else [self._indicator(width)])
        if self._final:
            # The summary is the record of the run. Nothing may overwrite it,
            # and whatever is printed next belongs below it.
            self._drawn = 0

    def _plain_line(self) -> str:
        if self._final:
            verdict = "FAILED" if self._failed else "converged"
            return f"[tarhan] {verdict}: {self._final}"
        stage = self._current
        if stage is None:
            return "[tarhan] ready"
        timing = f" ({stage.elapsed:.2f}s)" if stage.done else ""
        return f"[tarhan] {stage.name.lower()}{timing}: {stage.detail}"

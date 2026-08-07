"""Layer-0 for the forge display — the three claims, asserted not described.

A progress display is the easiest place in a codebase to tell a small lie:
nobody diffs a spinner. So the properties that make this one honest are pinned
here, and most are checked against real escape-sequence output rather than
against the code's intentions.

1. It never clears the screen (scrollback is somebody's evidence).
2. Progress comes from stages, never from elapsed time.
3. It is silent when nothing is watching.
"""
import io
import os
import re
import select
import shutil
import sys
import time

import pytest

from tarhan import cliout
from tarhan.forge import COMPACT_WIDTH, FORGE_CELL, FORGE_CELL_ASCII, Forge


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal, so the animated path is taken.

    ``encoding`` matters and is easy to forget: a bare StringIO reports None,
    the glyph check falls back to "ascii", and every test here would quietly
    exercise the ASCII art while believing it tested the real thing. A real
    terminal reports utf-8, so this one does too.
    """

    encoding = "utf-8"

    def isatty(self):
        return True


def _tty_forge(stages=("MESH", "SOLVE"), **kw):
    out = cliout.Output(color="always", stderr=FakeTTY())
    return Forge(list(stages), out, **kw), out


# --- 1. the screen is never cleared ---------------------------------------

def test_the_display_never_clears_the_screen():
    """ESC[2J would wipe whatever the user had above; it must never be emitted.

    This is the whole reason the display redraws by walking the cursor back
    instead of repainting the terminal.
    """
    forge, out = _tty_forge()
    with forge:
        forge.begin("MESH", "reading nodes")
        for k in range(6):
            forge.tick(within=(k + 1) / 6)
        forge.finish("done")
        forge.begin("SOLVE", "newton")
        forge.tick()
        forge.converged("12 iterations")
    written = out.stderr.getvalue()
    assert "\x1b[2J" not in written
    assert "\x1b[3J" not in written        # the scrollback-erasing variant
    assert re.search(r"\x1b\[\d+A", written), "nothing was redrawn in place"


def test_only_one_line_is_redrawn_while_work_runs():
    """The indicator is one line. Redrawing twenty per Newton iteration would
    bury whatever the user was reading and cost more than the solve does."""
    forge, out = _tty_forge()
    with forge:
        forge.begin("MESH", "reading nodes")
        for _ in range(5):
            forge.tick()
        forge.converged("done")
    ups = re.findall(r"\x1b\[(\d+)A", out.stderr.getvalue())
    assert ups.count("1") >= 5, ups


def test_a_logged_line_survives_and_is_never_overwritten():
    forge, out = _tty_forge()
    with forge:
        forge.begin("MESH", "reading nodes")
        forge.tick()
        forge.log("  note: damping engaged")
        forge.tick()
        forge.converged("done")
    written = out.stderr.getvalue()
    assert written.count("note: damping engaged") == 1


def test_the_cursor_is_restored_even_when_the_body_raises():
    forge, out = _tty_forge()
    with pytest.raises(RuntimeError):
        with forge:
            forge.begin("MESH", "reading nodes")
            raise RuntimeError("solver exploded")
    written = out.stderr.getvalue()
    assert written.count("\x1b[?25l") == 1      # hidden once
    assert written.count("\x1b[?25h") == 1      # and shown again
    assert "FAILED" in written and "solver exploded" in written


# --- the title sequence ----------------------------------------------------

def test_the_intro_plays_the_hammer_then_leaves_the_anvil_standing():
    """Big animation first, then only the anvil and the wordmark.

    The hammer is removed at rest rather than parked above the anvil: a hammer
    frozen in mid-air reads as something stalled, which is exactly the wrong
    impression while a run is starting normally.
    """
    forge, out = _tty_forge()
    forge.intro(strikes=2, pace=0.0)
    written = out.stderr.getvalue()
    assert "\x1b[2J" not in written
    assert "✦" in written, "the strike frame never played"
    assert "█" in written, "the anvil never appeared"

    resting = "\n".join(forge._logo(100))
    assert "▄████████▄" not in resting, "the hammer is still in the resting mark"
    assert "█████   ███   ████" in resting, "the wordmark is missing at rest"


def test_the_intro_is_silent_when_nothing_is_watching():
    """A title sequence in a CI log is noise nobody asked for."""
    out = cliout.Output(stderr=io.StringIO())
    forge = Forge(["MESH"], out)
    forge.intro(strikes=3, pace=0.0)
    assert out.stderr.getvalue() == ""


def test_the_intro_is_skippable_and_skipped_when_it_would_not_fit(monkeypatch):
    forge, out = _tty_forge()
    forge.intro(strikes=0)
    assert out.stderr.getvalue() == ""

    monkeypatch.setattr(shutil, "get_terminal_size",
                        lambda *_: os.terminal_size((COMPACT_WIDTH - 6, 24)))
    forge2, out2 = _tty_forge()
    forge2.intro(strikes=2, pace=0.0)
    assert out2.stderr.getvalue() == "", "no room for a title sequence"


def test_the_closing_summary_does_not_reprint_the_mark():
    """The anvil belongs at the top of a run, once.

    A second copy carries no information the first did not, and on a short run
    it would make the output mostly logo.
    """
    forge, out = _tty_forge()
    with forge:
        forge.begin("MESH", "reading nodes")
        forge.converged("done")
    summary = out.stderr.getvalue()
    assert "█████   ███   ████" not in summary
    assert "CONVERGED" in summary


# --- 2. progress is stages, never a clock ----------------------------------

def test_progress_does_not_move_with_time():
    """The property that makes the bar worth showing at all.

    If elapsed time could advance it, the number would look measured while
    meaning nothing. Here, waiting changes nothing.
    """
    forge, _ = _tty_forge(stages=("A", "B", "C", "D"), animate=False)
    forge.begin("A")
    before = forge.progress
    time.sleep(0.05)
    forge.tick()
    assert forge.progress == before


def test_progress_is_completed_stages_over_total():
    forge, _ = _tty_forge(stages=("A", "B", "C", "D"), animate=False)
    assert forge.progress == 0.0
    forge.begin("A")
    forge.finish()
    assert forge.progress == pytest.approx(0.25)
    forge.begin("B")
    forge.finish()
    assert forge.progress == pytest.approx(0.5)


def test_a_within_fraction_refines_but_never_exceeds_the_stage():
    """`within` is knowledge the caller has — iteration k of a cap. It may
    refine the current stage and must not spill into the next one."""
    forge, _ = _tty_forge(stages=("A", "B"), animate=False)
    forge.begin("A")
    forge.tick(within=0.5)
    assert forge.progress == pytest.approx(0.25)
    forge.tick(within=5.0)                     # absurd input, clamped
    assert forge.progress == pytest.approx(0.5)
    forge.tick(within=-3.0)
    assert forge.progress == pytest.approx(0.0)


def test_the_indicator_names_the_stage_it_is_counting():
    """It read "MESH … stage 2/4" in the instant between finishing one stage
    and beginning the next, because the counter came from the done-count."""
    forge, _ = _tty_forge(stages=("MESH", "SOLVE"), animate=False)
    forge.begin("MESH")
    forge.finish()
    line = forge._indicator(100)
    assert "MESH" in line and "stage 1/2" in line


# --- 3. silent when nobody is watching -------------------------------------

def test_a_pipe_gets_plain_lines_and_no_escape_sequences():
    out = cliout.Output(stderr=io.StringIO())      # a StringIO is not a tty
    forge = Forge(["MESH", "SOLVE"], out)
    assert forge.animate is False
    with forge:
        forge.begin("MESH", "reading nodes")
        forge.tick()
        forge.finish("2,847 nodes")
        forge.begin("SOLVE", "newton")
        forge.converged("12 iterations")
    written = out.stderr.getvalue()
    assert "\x1b[" not in written
    assert "[tarhan] mesh" in written
    assert "[tarhan] converged: 12 iterations" in written
    for glyph in ("█", "▄", "✦"):
        assert glyph not in written


def test_quiet_turns_the_display_off_entirely():
    out = cliout.Output(quiet=True, stderr=FakeTTY())
    assert Forge(["A"], out).animate is False


def test_a_console_without_unicode_gets_ascii_art():
    """The Windows lesson: a decorative glyph must never crash a run."""
    class AsciiTTY(FakeTTY):
        encoding = "ascii"

    out = cliout.Output(color="never", stderr=AsciiTTY())
    forge = Forge(["MESH"], out)
    assert forge.unicode is False
    with forge:
        forge.begin("MESH", "reading nodes")
        forge.tick()
        forge.converged("done")
    written = out.stderr.getvalue()
    written.encode("ascii")                     # raises if a glyph slipped in
    assert any(cell in written for cell in FORGE_CELL_ASCII)


def test_the_forge_cell_is_four_frames_with_one_strike():
    """The anvil is on screen the whole time and the spark marks contact.

    A generic spinner would have been easier; this is the identity, so it is
    asserted rather than left for a future refactor to quietly drop.
    """
    assert len(FORGE_CELL) == 4
    assert sum("✦" in cell for cell in FORGE_CELL) == 1
    assert {cell[1:] for cell in FORGE_CELL} == {"▂▄▂"}, \
        "the anvil must not move between frames"


def test_a_narrow_terminal_drops_the_logo(monkeypatch):
    monkeypatch.setattr(shutil, "get_terminal_size",
                        lambda *_: os.terminal_size((COMPACT_WIDTH - 6, 24)))
    forge, out = _tty_forge()
    with forge:
        forge.begin("MESH", "reading nodes")
        forge.converged("done")
    assert "█" not in out.stderr.getvalue(), "the logo does not fit and must go"


# --- the real thing, through a real terminal -------------------------------

@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="pty is Unix-only; importing it on Windows raises")
def test_through_a_real_pty_scrollback_survives():
    """The checks above use a fake tty; this one uses a real one.

    A pty is the only way to be sure the animated path is what a user actually
    gets, rather than what a StringIO subclass persuaded the code to take.

    Two hazards, both handled rather than hoped away. ``pty`` does not exist on
    Windows, so the whole test is skipped there — and the import is local, since
    a module-level one would fail collection for the other thirteen tests in
    this file. And ``forkpty`` from a multi-threaded process can deadlock, so
    the read has a deadline: a hung child fails this test instead of hanging CI
    until the job is killed with no useful output.
    """
    import pty

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script = (
        "import sys; sys.path.insert(0, %r);"
        "from tarhan import cliout;"
        "from tarhan.forge import Forge;"
        "print('BEFORE-THE-RUN', flush=True);"
        "f = Forge(['MESH', 'SOLVE'], cliout.Output());"
        "f.__enter__();"
        "f.begin('MESH', 'reading');"
        "[f.tick(within=(k+1)/4) for k in range(4)];"
        "f.finish('ok');"
        "f.begin('SOLVE', 'newton');"
        "f.tick();"
        "f.converged('12 iterations');"
        "f.__exit__(None, None, None)"
    ) % os.path.join(repo, "src")

    pid, fd = pty.fork()
    if pid == 0:                                # pragma: no cover — the child
        os.execv(sys.executable, [sys.executable, "-c", script])
    chunks = []
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not select.select([fd], [], [], 1.0)[0]:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    else:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        pytest.fail("the pty child produced no EOF within 30 s")
    os.waitpid(pid, 0)
    raw = b"".join(chunks).decode("utf-8", "replace")

    assert "BEFORE-THE-RUN" in raw, "scrollback above the display was destroyed"
    assert "\x1b[2J" not in raw
    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", raw)
    assert "CONVERGED" in plain

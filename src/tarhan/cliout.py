"""The CLI output contract: one stream for results, another for everything else.

The rule this module enforces is a single sentence — **a human and a machine
never share stdout**. Results go to stdout in the requested format; progress,
warnings, colour and the anvil feedback go to stderr in every mode, including
the human one. That is stricter than a human needs, deliberately: the moment
progress is allowed onto stdout "just for the interactive case", someone adds a
spinner and a ``--format json`` consumer starts failing on a partial parse it
cannot explain.

Exit codes are part of the same contract. An agent or a CI job should learn what
happened from the status, not by grepping prose that will be reworded.

Which codes are wired today, stated plainly rather than implied:

* ``EXIT_OK`` and ``EXIT_INPUT`` are used by ``capabilities``.
* ``EXIT_UNAVAILABLE`` is returned by ``capabilities show`` for a blocked or
  planned capability. The record still prints in full — the non-zero status is
  the machine-readable half of the same answer, so a script does not have to
  parse the word "blocked" out of a paragraph.
* ``EXIT_NO_CONVERGENCE`` is returned by ``run solve`` when the solver gives up.
  No artifact is written in that case: a partial state would claim more than the
  run earned. (This entry read "no call site yet" from the day the numbering was
  fixed until ``run solve`` existed to use it.)
* ``EXIT_INTERNAL`` is the catch-all for an unexpected exception.

``tarhan demo`` keeps its existing 0/1 contract untouched. AGENTS.md documents
it, and changing it would break the one command people already run.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
from typing import Any, Dict, Sequence

EXIT_OK = 0
EXIT_INPUT = 2                 # bad input, schema or units
EXIT_UNAVAILABLE = 3           # capability is blocked or merely planned
EXIT_NO_CONVERGENCE = 4        # solver gave up; a partial artifact may exist
EXIT_INTERNAL = 5              # a bug here, not in the caller's request

FORMATS = ("table", "json", "csv")
COLOR_MODES = ("auto", "always", "never")


_ANSI = {"dim": "\033[2m", "bold": "\033[1m", "reverse": "\033[7m",
         "red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m",
         "reset": "\033[0m"}


def _encodable(stream, text: str) -> bool:
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class Output:
    """Where results go, and in what shape.

    ``stdout`` and ``stderr`` are injectable so the tests can assert the
    separation rather than trust it.
    """

    def __init__(self, fmt: str = "table", color: str = "auto",
                 quiet: bool = False, stdout=None, stderr=None) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
        if color not in COLOR_MODES:
            raise ValueError(f"color must be one of {COLOR_MODES}, got {color!r}")
        self.fmt = fmt
        self.quiet = quiet
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self._color = self._decide_color(color)

    def _decide_color(self, mode: str) -> bool:
        if mode == "never":
            return False
        if mode == "always":
            return True
        # auto. NO_COLOR is honoured because it is the convention users already
        # have, and a machine-readable format never gets colour regardless.
        if os.environ.get("NO_COLOR"):
            return False
        if self.fmt != "table":
            return False
        return bool(getattr(self.stderr, "isatty", lambda: False)())

    @property
    def color(self) -> bool:
        return self._color

    def paint(self, text: str, style: str) -> str:
        if not self._color or style not in _ANSI:
            return text
        return f"{_ANSI[style]}{text}{_ANSI['reset']}"

    # --- the results half -------------------------------------------------

    def emit(self, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
        """Write the result set to stdout — and nothing else to stdout, ever."""
        if self.fmt == "json":
            json.dump([{c: r.get(c) for c in columns} for r in rows],
                      self.stdout, indent=2, ensure_ascii=False)
            self.stdout.write("\n")
            return
        if self.fmt == "csv":
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(columns),
                                    extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
            self.stdout.write(buf.getvalue())
            return
        self._emit_table(rows, columns)

    def _emit_table(self, rows, columns) -> None:
        if not rows:
            if not self.quiet:
                self.stdout.write("  ".join(columns) + "\n")
            return
        widths = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows])
                  for c in columns}
        header = "  ".join(c.ljust(widths[c]) for c in columns)
        self.stdout.write(self.paint(header, "bold") + "\n")
        self.stdout.write("-" * len(header) + "\n")
        for row in rows:
            self.stdout.write("  ".join(
                str(row.get(c, "")).ljust(widths[c]) for c in columns) + "\n")

    def detail(self, text: str) -> None:
        """Free-form result text — stdout, and refused in machine formats."""
        if self.fmt != "table":
            raise RuntimeError(
                "detail() would put prose on stdout in a machine-readable "
                "format; that is the one thing this contract forbids")
        self.stdout.write(text)

    # --- the everything-else half ----------------------------------------

    def note(self, text: str) -> None:
        if not self.quiet:
            self.stderr.write(text + "\n")

    def warn(self, text: str) -> None:
        if not self.quiet:
            self.stderr.write(self.paint("warning: " + text, "yellow") + "\n")

    def error(self, text: str) -> None:
        """Critical errors survive --quiet; silence here would be a lie."""
        self.stderr.write(self.paint("error: " + text, "red") + "\n")


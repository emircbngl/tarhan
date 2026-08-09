"""Every command printed in the documentation must actually be a command.

README.md advertised ``tarhan capabilities list --format json``. That is not a
thing: ``--format`` is a global, so argparse sees it after the subcommand and
exits 2. The claim had been in the README since the flag was added, the whole
suite was green, and it was caught by a reviewer typing it — which is the worst
way to find out, because it is the way a new user finds out.

Nothing here checks prose. It checks the one property a documented command has
to have: **the CLI accepts it as written**. The shell blocks are the source, so
a command that is edited in the README and not here is still covered, and a
command that is deleted stops being tested automatically.

Each line goes through the real parser, which accepts or rejects it exactly as
the command line would, without running a solve. The obvious cheaper trick —
appending ``--help`` to a subprocess — was tried first and is WRONG: argparse
fires a subparser's help before the parent reports an unrecognised global, so
``capabilities list --format json --help`` exits 0. It would have passed on the
very defect it was written for. The two fast commands are additionally run as
real subprocesses, because parsing is not the same claim as working.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tarhan.cli import build_parser

REPO = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "AGENTS.md"]

#: Placeholders the documentation uses instead of concrete values. Each is
#: replaced by something real so the line can be parsed as typed.
PLACEHOLDERS = {
    "<id>": "semiconductor.pn.drift-diffusion.1d.steady",
    "<capability-id>": "semiconductor.pn.drift-diffusion.1d.steady",
    "<run-id>": "0000000000000000",
    "<run-a>": "0000000000000000",
    "<run-b>": "1111111111111111",
}


def documented_commands():
    """Every ``tarhan …`` line inside a fenced block, comments stripped."""
    found = []
    for name in DOCS:
        text = (REPO / name).read_text(encoding="utf-8")
        for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
            for raw in block.splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line.startswith("tarhan "):
                    continue
                for token, value in PLACEHOLDERS.items():
                    line = line.replace(token, value)
                found.append((name, line))
    return found


COMMANDS = documented_commands()


def test_the_documentation_actually_contains_commands():
    """Guards the harvester itself.

    If the fence regex ever stopped matching, every parametrised case below
    would silently vanish and the suite would still be green — a test that
    tests nothing is worse than no test, because it reads as coverage.
    """
    assert len(COMMANDS) >= 8
    assert any("--format json" in cmd for _, cmd in COMMANDS)


@pytest.mark.parametrize("source,command",
                         COMMANDS, ids=[f"{s}:{c}" for s, c in COMMANDS])
def test_every_documented_command_is_accepted_as_written(source, command):
    argv = command.split()[1:]          # drop the "tarhan" console-script name
    parser, _ = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as exit_:                    # argparse's way of erroring
        pytest.fail(f"{source} documents `{command}`, which the CLI rejects "
                    f"with exit {exit_.code}")


@pytest.mark.parametrize("command", [
    "tarhan capabilities list",
    "tarhan --format json capabilities list",
])
def test_the_documented_capability_commands_really_run(command):
    """Parsing is not working. These two are fast enough to prove outright."""
    assert command in [c for _, c in COMMANDS], \
        f"`{command}` is no longer documented; update or drop this test"
    proc = subprocess.run([sys.executable, "-m", "tarhan.cli",
                           *command.split()[1:]],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert proc.stdout.strip(), "the command produced no result on stdout"

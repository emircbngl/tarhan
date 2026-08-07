"""Layer-0 for the CLI contract: the two streams, and the exit codes.

The claim under test is not "the CLI prints something useful". It is the one
promise a machine consumer relies on: **stdout carries the result and nothing
else**. That is asserted the only way it can honestly be asserted — by running
the real command in a real subprocess with real pipes, and parsing its stdout
strictly. An in-process check with a StringIO would still pass if the command
wrote a progress bar straight to the terminal's file descriptor.

The exit codes get the same treatment. A caller that has to grep prose to learn
what happened breaks the next time a sentence is reworded.
"""
import csv
import io
import json
import subprocess
import sys

import pytest

from tarhan import cliout

CMD = [sys.executable, "-m", "tarhan.cli"]


def run(*args):
    """Run the CLI with pipes on both streams — never a terminal."""
    return subprocess.run(CMD + list(args), capture_output=True, text=True,
                          timeout=120)


# --- the promise a machine depends on --------------------------------------

def test_json_stdout_is_byte_for_byte_valid_json():
    """The whole reason the contract exists.

    ``json.loads`` over the entire stream, not a search for a JSON-looking
    substring: one stray progress line and this fails, which is the point.
    """
    proc = run("--format", "json", "capabilities", "list")
    assert proc.returncode == cliout.EXIT_OK
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list) and payload
    assert {"id", "status", "dimension", "time"} <= set(payload[0])


def test_the_note_goes_to_stderr_not_stdout():
    """The human-facing summary must not contaminate the machine stream."""
    proc = run("--format", "json", "capabilities", "list")
    assert "not runnable today" in proc.stderr
    assert "not runnable today" not in proc.stdout


def test_csv_stdout_parses_as_csv_and_only_csv():
    proc = run("--format", "csv", "capabilities", "list")
    assert proc.returncode == cliout.EXIT_OK
    rows = list(csv.DictReader(io.StringIO(proc.stdout)))
    assert rows and rows[0]["id"]
    assert all(r["status"] in ("validated", "experimental", "blocked", "planned")
               for r in rows)


def test_show_in_json_is_a_single_valid_record():
    proc = run("--format", "json", "capabilities", "show",
               "semiconductor.pn.drift-diffusion.2d.steady")
    assert proc.returncode == cliout.EXIT_OK
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    assert payload[0]["evidence"], "a validated record must carry its evidence"


@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_no_ansi_escape_reaches_a_machine_format_even_with_color_always(fmt):
    """An explicit --color always must still lose to a machine format.

    This assertion started out stronger — no colour on stdout at all — and the
    test failed against a bolded table header. The stronger claim was the wrong
    one: `table` IS the human format, and styling its header is what every
    other CLI does. The invariant worth defending is narrower and sharper: a
    stream something will parse never carries escapes, no matter what the user
    asked for.
    """
    proc = run("--color", "always", "--format", fmt, "capabilities", "list")
    assert "\033[" not in proc.stdout


def test_a_piped_table_has_no_colour_by_default():
    """auto means auto: no terminal, no escapes, even in the human format."""
    proc = run("capabilities", "list")
    assert "\033[" not in proc.stdout


def test_doctor_json_survives_a_dependency_that_prints_from_c():
    """The regression that made this test exist.

    DEVSIM prints a BLAS/UMFPACK banner at import time from C, straight onto
    file descriptor 1. ``contextlib.redirect_stdout`` cannot see it, so it
    landed in the middle of the JSON and ``json.loads`` failed on a stream that
    looked fine to anyone reading it by eye. The fix redirects the descriptor;
    this asserts the outcome rather than the fix, so any future chatty import
    is caught the same way.
    """
    proc = run("--format", "json", "capabilities", "doctor")
    payload = json.loads(proc.stdout)
    names = [row["check"] for row in payload]
    assert {"numpy", "scipy", "registry", "evidence"} <= set(names)
    assert all(row["status"] in ("ok", "FAILED", "absent") for row in payload)


def test_doctor_reports_a_healthy_install_as_zero():
    proc = run("capabilities", "doctor")
    assert proc.returncode == cliout.EXIT_OK
    assert proc.stdout == "", "table mode puts the whole report on stderr"


def test_doctor_counts_checks_and_not_seconds():
    """The bar is a count. Nothing in the doctor path may advance it with time."""
    from tarhan import cli

    assert len(cli.DOCTOR_CHECKS) >= 4
    for name, detail, check in cli.DOCTOR_CHECKS:
        assert name and detail and callable(check)
    ok, said = cli._check_registry()
    assert ok is True and "capabilities" in said


# --- exit codes -------------------------------------------------------------

@pytest.mark.parametrize("capability_id,expected", [
    ("semiconductor.pn.drift-diffusion.1d.steady", cliout.EXIT_OK),
    ("semiconductor.pn.drift-diffusion.2d.steady", cliout.EXIT_OK),
    ("semiconductor.mos.capacitance.2d.ac", cliout.EXIT_UNAVAILABLE),
    ("semiconductor.mosfet.drift-diffusion.2d.steady", cliout.EXIT_UNAVAILABLE),
    ("semiconductor.device.drift-diffusion.3d.transient", cliout.EXIT_UNAVAILABLE),
])
def test_show_exit_status_matches_runnability(capability_id, expected):
    assert run("capabilities", "show", capability_id).returncode == expected


def test_a_blocked_capability_still_prints_in_full():
    """Exit 3 is a verdict, not a refusal to answer."""
    proc = run("capabilities", "show",
               "semiconductor.mosfet.drift-diffusion.2d.steady")
    assert proc.returncode == cliout.EXIT_UNAVAILABLE
    assert "Delaunay" in proc.stdout
    assert "does not mean" in proc.stdout


def test_unknown_capability_is_an_input_error_not_an_internal_one():
    proc = run("capabilities", "show", "semiconductor.nope.9d.steady")
    assert proc.returncode == cliout.EXIT_INPUT
    assert "no such capability" in proc.stderr
    assert proc.stdout == ""


def test_quiet_silences_notes_but_never_errors():
    quiet = run("--quiet", "capabilities", "show", "does.not.exist.1d.steady")
    assert quiet.returncode == cliout.EXIT_INPUT
    assert "no such capability" in quiet.stderr        # the error survives
    assert "capabilities list" not in quiet.stderr     # the hint does not


def test_an_unexpected_exception_becomes_exit_5_not_a_traceback_and_zero():
    """The catch-all, proven rather than assumed.

    Forced in-process because a subprocess cannot be made to fail on demand
    without inventing a failure mode that does not exist. What matters is that
    the boundary turns a bug into a distinguishable status instead of letting
    it escape as an unhandled traceback with whatever code Python picks.
    """
    from tarhan import cli

    def boom():
        raise RuntimeError("simulated internal failure")

    original = cli.all_capabilities
    cli.all_capabilities = boom
    try:
        assert cli.main(["capabilities", "list"]) == cliout.EXIT_INTERNAL
    finally:
        cli.all_capabilities = original


def test_exit_codes_are_the_documented_numbers():
    """Pinned because they are a contract with scripts, not an implementation
    detail. Renumbering them silently would break callers who never see this
    repository."""
    assert (cliout.EXIT_OK, cliout.EXIT_INPUT, cliout.EXIT_UNAVAILABLE,
            cliout.EXIT_NO_CONVERGENCE, cliout.EXIT_INTERNAL) == (0, 2, 3, 4, 5)


# --- the anvil never reaches a pipe ---------------------------------------

def test_no_forge_glyph_when_stderr_is_a_pipe():
    """Every run in this file is piped, so the display must be invisible.

    The Forge display itself is tested in test_forge.py; this asserts the one
    thing the CLI contract owns — that nothing decorative escapes into a stream
    something might read.
    """
    proc = run("capabilities", "list")
    for glyph in ("\u2571\u2582\u2584\u2582", "\u2726", "\u2588"):
        assert glyph not in proc.stdout and glyph not in proc.stderr


# --- the guard that keeps prose off the machine stream ---------------------

@pytest.mark.parametrize("fmt", ["json", "csv"])
def test_detail_refuses_to_write_prose_in_a_machine_format(fmt):
    out = cliout.Output(fmt=fmt, stdout=io.StringIO(), stderr=io.StringIO())
    with pytest.raises(RuntimeError, match="forbids"):
        out.detail("a human sentence")


def test_colour_is_never_enabled_for_a_machine_format():
    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    assert cliout.Output(fmt="json", color="auto", stderr=FakeTTY()).color is False
    assert cliout.Output(fmt="table", color="auto", stderr=FakeTTY()).color is True


def test_no_color_env_is_honoured(monkeypatch):
    class FakeTTY(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    assert cliout.Output(fmt="table", color="auto", stderr=FakeTTY()).color is False


@pytest.mark.parametrize("kwargs", [{"fmt": "yaml"}, {"color": "sometimes"}])
def test_output_refuses_an_unknown_mode(kwargs):
    with pytest.raises(ValueError):
        cliout.Output(**kwargs)

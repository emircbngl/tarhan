"""`run solve`, `run show`, `compare runs` — the statuses a script branches on.

Every assertion is made through a real subprocess, because the thing under test
is a contract with a caller: the exit status. Checking a return value in-process
would pass even if argparse swallowed the code on its way out.

The status table this pins:

===  ==============================================================
  0  it worked
  2  bad input — an unknown id, or two runs that cannot be compared
  3  the capability is blocked, planned, or not wired to `run solve`
  4  the solver gave up, and NO artifact was written
===  ==============================================================

Code 4 is the reason this file exists. It was defined with no call site the day
the numbering was fixed, and both cliout.py and AGENTS.md said so in as many
words. `run solve` is what it was reserved for.
"""
import json
import subprocess
import sys

from tarhan import cliout

CMD = [sys.executable, "-m", "tarhan.cli"]
PN1D = "semiconductor.pn.drift-diffusion.1d.steady"


def run(*args):
    return subprocess.run(CMD + list(args), capture_output=True, text=True,
                          timeout=300)


def solve(out_dir, *extra):
    proc = run("--format", "json", "run", "solve", PN1D,
               "--output", str(out_dir), *extra)
    assert proc.returncode == cliout.EXIT_OK, proc.stderr
    return json.loads(proc.stdout)[0]["run_id"]


# --- solve ----------------------------------------------------------------

def test_a_solve_writes_a_run_that_reads_back(tmp_path):
    run_id = solve(tmp_path)
    directory = tmp_path / run_id
    for name in ("manifest.json", "input.lock.toml", "provenance.json",
                 "metrics.json", "stdout.log", "report.md", "fields.npz"):
        assert (directory / name).exists(), name

    shown = run("--format", "json", "run", "show", run_id,
                "--output", str(tmp_path))
    assert shown.returncode == cliout.EXIT_OK
    record = json.loads(shown.stdout)[0]
    assert record["capability"] == PN1D
    assert record["capability_status"] == "validated"
    assert record["status"] == "converged"


def test_the_same_problem_reuses_the_same_directory(tmp_path):
    """Re-running must overwrite, not accumulate. Two directories differing
    only by a clock are two answers to one question."""
    first = solve(tmp_path)
    second = solve(tmp_path)
    assert first == second
    assert len(list(tmp_path.iterdir())) == 1


def test_a_different_tolerance_is_a_different_problem(tmp_path):
    assert solve(tmp_path) != solve(tmp_path, "--tol", "1e-7")
    assert len(list(tmp_path.iterdir())) == 2


# --- the display advances DURING the solve, not after it ------------------

def test_the_gummel_loop_reports_every_outer_iteration():
    """Without this the progress display is decoration.

    The whole solve is one blocking call, so a caller has no other moment to
    draw in. Measured before the callback existed: zero bytes were written
    between begin() and finish(), which meant a long solve showed nothing while
    anyone was waiting and the indicator only appeared once the work was over.
    Reported by Codex against the published commit.
    """
    from tarhan.models.pn1d import PNDiode1D, solve_bias

    seen = []
    state = solve_bias(PNDiode1D(), 0.30,
                       on_iteration=lambda i, n: seen.append((i, n)))
    assert seen, "the solver never reported progress"
    assert len(seen) == state["gummel_iters"]
    assert [i for i, _ in seen] == list(range(len(seen)))
    assert all(total == 60 for _, total in seen)


def test_the_runner_forwards_the_callback():
    from tarhan import cli

    seen = []
    cli._solve_pn1d_steady({"bias_v": 0.3, "tol": 1e-9, "max_iter": 60},
                           on_iteration=lambda i, n: seen.append(i))
    assert seen


def test_the_indicator_actually_advances_while_the_solver_runs():
    """The regression this fix exists for, asserted in bytes.

    A display that only draws after the work is finished is worse than none:
    it looks like feedback and never arrives while anyone is waiting.
    """
    import io

    from tarhan import cli, cliout
    from tarhan.forge import Forge

    class FakeTTY(io.StringIO):
        encoding = "utf-8"

        def isatty(self):
            return True

    out = cliout.Output(color="always", stdout=io.StringIO(), stderr=FakeTTY())
    forge = Forge(["SOLVE"], out, pin="auto", pin_after=0.0)
    with forge:
        forge.begin("SOLVE", "newton")
        before = len(out.stderr.getvalue())
        cli._solve_pn1d_steady(
            {"bias_v": 0.3, "tol": 1e-9, "max_iter": 60},
            on_iteration=lambda i, n: forge.tick(within=(i + 1) / n))
        during = len(out.stderr.getvalue()) - before
        forge.finish("done")
        forge.converged("done")
    assert during > 0, "nothing was drawn while the solver was running"
    assert out.stdout.getvalue() == "", "the display must never touch stdout"


# --- the refusals ---------------------------------------------------------

def test_a_blocked_capability_is_refused_before_any_work(tmp_path):
    proc = run("run", "solve", "semiconductor.mosfet.drift-diffusion.2d.steady",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_UNAVAILABLE
    assert "refusing to solve" in proc.stderr
    assert "Delaunay" in proc.stderr, "the reason must travel with the refusal"
    assert not list(tmp_path.iterdir()), "a refused run must leave nothing"


def test_a_validated_capability_with_no_runner_says_which_half_is_missing(
        tmp_path):
    """Being proven and being wired up are different facts.

    2D steady is validated against DEVSIM and still not runnable from the CLI.
    Reporting that as "blocked" would blame the physics for a missing command.
    """
    proc = run("run", "solve", "semiconductor.pn.drift-diffusion.2d.steady",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_UNAVAILABLE
    assert "not wired" in proc.stderr


def test_an_unknown_capability_is_an_input_error(tmp_path):
    proc = run("run", "solve", "nope.9d.steady", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT


def test_a_solver_that_gives_up_exits_4_and_writes_nothing(tmp_path):
    """The call site EXIT_NO_CONVERGENCE was reserved for.

    One Gummel iteration cannot converge this device. The status has to be
    distinguishable from a crash (5) and from bad input (2), because a caller
    does something different about each — and no artifact may be left behind,
    since a partial state would claim more than the run earned.
    """
    proc = run("run", "solve", PN1D, "--max-iter", "1",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_NO_CONVERGENCE
    assert "did not converge" in proc.stderr
    assert not list(tmp_path.iterdir())


# --- compare --------------------------------------------------------------

def test_two_identical_runs_compare_with_zero_deltas(tmp_path):
    run_id = solve(tmp_path)
    proc = run("--format", "json", "compare", "runs", run_id, run_id,
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    rows = json.loads(proc.stdout)
    assert {r["metric"] for r in rows} >= {"current_a_cm2", "grid_points"}
    assert all(r["delta"] == 0 for r in rows)


def test_a_changed_solver_contract_is_refused_rather_than_ranked(tmp_path):
    """The feature is the refusal.

    Two solves at different tolerances give two numbers that can be subtracted,
    and the difference means nothing. Printing it anyway is how a spurious
    ranking reaches a report.
    """
    left = solve(tmp_path)
    right = solve(tmp_path, "--tol", "1e-7")
    proc = run("compare", "runs", left, right, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "not comparable" in proc.stderr
    assert "different solver" in proc.stderr


def test_a_changed_input_is_refused_too(tmp_path):
    left = solve(tmp_path)
    right = solve(tmp_path, "--bias", "0.25")
    proc = run("compare", "runs", left, right, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "different inputs" in proc.stderr


def test_the_refusal_is_machine_readable_too(tmp_path):
    """A script must not have to parse the sentence to learn it was refused."""
    left = solve(tmp_path)
    right = solve(tmp_path, "--tol", "1e-7")
    proc = run("--format", "json", "compare", "runs", left, right,
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    payload = json.loads(proc.stdout)
    assert payload[0]["comparable"] is False
    assert "solver" in payload[0]["differing"]


def test_comparing_a_missing_run_is_an_input_error(tmp_path):
    run_id = solve(tmp_path)
    proc = run("compare", "runs", run_id, "deadbeef1234",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT


# --- the contract stays on stderr ----------------------------------------

def test_json_output_stays_parseable_through_a_solve(tmp_path):
    """The whole-session invariant, re-checked on the newest command."""
    proc = run("--format", "json", "run", "solve", PN1D,
               "--output", str(tmp_path))
    json.loads(proc.stdout)
    assert "artifact written" in proc.stderr
    assert "artifact written" not in proc.stdout

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

    1D transient is validated — the steady state is an exact fixed point of its
    RHS, and a perturbation relaxes back — and still not runnable from the CLI,
    because `run solve` has no bias waveform to give it. Reporting that as
    "blocked" would blame the physics for a missing command.

    This test used to make the same point with 2D steady. That stopped being
    true the moment 2D was wired, and the example was replaced rather than the
    assertion weakened: the distinction is what matters, not which capability
    happens to illustrate it today.
    """
    proc = run("run", "solve", "semiconductor.pn.drift-diffusion.1d.transient",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_UNAVAILABLE
    assert "not wired" in proc.stderr


def test_every_wired_capability_is_one_the_registry_calls_runnable():
    """The map and the registry must not drift apart.

    A runner for a blocked capability would be reachable only through a check
    that exists two branches earlier — the kind of pairing that stays correct
    until someone reorders the function.
    """
    from tarhan.capability_registry import get
    from tarhan.cli import RUNNERS

    for capability_id in RUNNERS:
        assert get(capability_id).runnable, \
            f"{capability_id} has a runner but the registry refuses it"


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


# --- the build is a term of the comparability contract ---------------------

def _restamp_build(source_dir, build):
    """Copy a run and relabel the build that produced it.

    The honest way to make two builds would be to check out two commits and
    solve twice, which costs a checkout per assertion. What `compare` reads is
    the recorded build id, so that is what is varied — and the manifest is not
    covered by its own checksums, so nothing else has to be forged for the
    directory to stay valid.
    """
    import json as _json
    import shutil

    target = source_dir.parent / f"{source_dir.name.split('-')[0]}-{build}"
    shutil.copytree(source_dir, target, dirs_exist_ok=True)
    manifest_path = target / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest["code_id"] = build
    manifest["run_id"] = target.name
    manifest_path.write_text(_json.dumps(manifest, indent=2) + "\n")
    return target.name


def test_two_builds_are_refused_by_default(tmp_path):
    """Reported in review: the contract covered capability, inputs and solver,
    so two results from DIFFERENT CODE compared silently. With the inputs held
    fixed, a delta from a code change is indistinguishable from a physical
    effect — which is the single most misreadable output this command has."""
    left = solve(tmp_path)
    right = _restamp_build(tmp_path / left, "deadbeef")

    proc = run("compare", "runs", left, right, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "different build" in proc.stderr
    assert "--allow-build-diff" in proc.stderr


def test_the_waiver_compares_and_flags_every_delta(tmp_path):
    """Comparing across builds IS a real question — "did my change move this
    number?" — so the flag exists. What it must never do is produce output that
    reads like an ordinary comparison."""
    left = solve(tmp_path)
    right = _restamp_build(tmp_path / left, "deadbeef")

    proc = run("compare", "runs", left, right, "--allow-build-diff",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert "BUILDS DIFFER" in proc.stderr
    assert "may be the code changing rather than the physics" in proc.stderr


def test_the_waiver_does_not_excuse_any_other_term(tmp_path):
    """One waiver, one term. No flag makes two solver tolerances comparable."""
    left = solve(tmp_path)
    right = solve(tmp_path, "--bias", "0.35")
    proc = run("compare", "runs", left, right, "--allow-build-diff",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "different inputs" in proc.stderr


def test_a_legacy_run_says_it_was_never_verified(tmp_path):
    """`run show` on a directory written before checksums existed."""
    import json as _json

    run_id = solve(tmp_path)
    manifest_path = tmp_path / run_id / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest.pop("files")
    manifest.pop("schema_version")
    manifest_path.write_text(_json.dumps(manifest, indent=2) + "\n")

    proc = run("run", "show", run_id, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert "nothing has verified" in proc.stderr


# --- 2D steady, now that a device exists -----------------------------------

PN2D = "semiconductor.pn.drift-diffusion.2d.steady"


def test_2d_steady_is_wired_and_writes_a_full_artifact(tmp_path):
    """It was validated long before it was runnable, and the CLI said so.

    `run solve` exited 3 with "the physics is proven; the command surface for
    it is not built" — true, and the missing half was a DEVICE: PNDiode2D needs
    a mesh, and nothing in the package could produce one.
    """
    proc = run("--format", "json", "run", "solve", PN2D,
               "--bias", "0.3", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK, proc.stderr
    record = json.loads(proc.stdout)[0]
    assert record["grid_points"] == 625
    directory = tmp_path / record["run_id"]
    for name in ("manifest.json", "input.lock.toml", "metrics.json",
                 "fields.npz", "report.md"):
        assert (directory / name).exists(), name


def test_the_2d_lock_file_records_the_mesh_as_scalars(tmp_path):
    """A lock holding 625 coordinates would be a record nobody reads and a
    hash nobody can reproduce by hand. The generator is narrow enough that
    these scalars determine the mesh exactly."""
    import tomllib

    run_id = None
    proc = run("--format", "json", "run", "solve", PN2D,
               "--bias", "0.3", "--output", str(tmp_path))
    run_id = json.loads(proc.stdout)[0]["run_id"]
    lock = tomllib.loads(
        (tmp_path / run_id / "input.lock.toml").read_text(encoding="utf-8"))
    assert {"len_p", "len_n", "height", "h0", "gamma", "ny", "Na", "Nd"} <= \
        set(lock), "the mesh is an input and must be reproducible from the lock"
    assert {"ni", "ut", "eps_s", "q", "mu_n", "mu_p"} <= set(lock)
    assert "points" not in lock and "triangles" not in lock


def test_the_2d_current_agrees_with_the_1d_capability(tmp_path):
    """The check that makes wiring it worth anything.

    Both capabilities describe the same ideal diode, so the current densities
    must agree — and they are computed by two different discretisations on two
    different meshes, so agreement is evidence rather than tautology. The 2D
    terminal current is per unit depth; the reported `current_a_cm2` is what
    divides out the device height.
    """
    one = run("--format", "json", "run", "solve", PN1D,
              "--bias", "0.3", "--output", str(tmp_path))
    two = run("--format", "json", "run", "solve", PN2D,
              "--bias", "0.3", "--output", str(tmp_path))
    j_1d = json.loads(one.stdout)[0]["current_a_cm2"]
    j_2d = json.loads(two.stdout)[0]["current_a_cm2"]
    assert abs(j_2d / j_1d - 1.0) < 1e-3


def test_two_capabilities_are_not_comparable_to_each_other(tmp_path):
    """Same physics, same bias, different capability — and `compare` must
    still refuse. The contract is not "do the numbers look close"."""
    one = json.loads(run("--format", "json", "run", "solve", PN1D,
                         "--bias", "0.3", "--output",
                         str(tmp_path)).stdout)[0]["run_id"]
    two = json.loads(run("--format", "json", "run", "solve", PN2D,
                         "--bias", "0.3", "--output",
                         str(tmp_path)).stdout)[0]["run_id"]
    proc = run("compare", "runs", one, two, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "different capability" in proc.stderr


def test_each_capability_carries_its_own_iteration_budget(tmp_path):
    """60 Gummel iterations is generous in 1D and tight in 2D. A default on
    the flag would silently impose the 1D budget on everything added later."""
    import tomllib

    for capability, expected in ((PN1D, 60), (PN2D, 200)):
        record = json.loads(run("--format", "json", "run", "solve", capability,
                                "--bias", "0.3", "--output",
                                str(tmp_path)).stdout)[0]
        manifest = json.loads(
            (tmp_path / record["run_id"] / "manifest.json").read_text())
        assert manifest["solver"]["max_iter"] == expected

    # ...and an explicit flag still wins over the capability's own default.
    record = json.loads(run("--format", "json", "run", "solve", PN2D,
                            "--bias", "0.3", "--max-iter", "17",
                            "--output", str(tmp_path)).stdout)[0]
    manifest = json.loads(
        (tmp_path / record["run_id"] / "manifest.json").read_text())
    assert manifest["solver"]["max_iter"] == 17

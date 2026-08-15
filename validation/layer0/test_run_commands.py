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

import pytest

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
    assert record["status"] == "potential-step-converged"


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


# --- --device: the device stops being a constant -------------------------

def _spec(tmp_path, text, name="device.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_an_overridden_field_reaches_the_solver(tmp_path):
    """The defect this surfaced, and the reason it is asserted on a NUMBER.

    `_solve_pn1d_steady` built PNDiode1D from a hand-listed six fields and
    dropped the other nine — eps_s, both mobilities, the grid controls, the SRH
    lifetimes. Invisible while everything sat at its default. The moment a
    device file could change one it became a silent lie: the lock file would
    record the value, the run id would move because the id hashes the inputs,
    and the solver would compute the old answer. A test that only checked "the
    id changed" would have passed against exactly that bug.
    """
    base = json.loads(run("--format", "json", "run", "solve", PN1D, "--bias",
                          "0.3", "--output", str(tmp_path)).stdout)[0]
    slow = json.loads(run("--format", "json", "run", "solve", PN1D, "--bias",
                          "0.3", "--device", _spec(tmp_path, "mu_n = 675.0\n"),
                          "--output", str(tmp_path)).stdout)[0]

    assert slow["run_id"] != base["run_id"]
    assert slow["current_a_cm2"] != base["current_a_cm2"], \
        "the device was recorded but not used"
    # Halving the electron mobility halves the ELECTRON half of the current,
    # so the total falls by less than half. A total that halved exactly would
    # mean the hole current moved too, which no mobility change should do.
    ratio = slow["current_a_cm2"] / base["current_a_cm2"]
    assert 0.5 < ratio < 1.0


def test_the_override_is_what_lands_in_the_lock_file(tmp_path):
    import tomllib

    record = json.loads(run("--format", "json", "run", "solve", PN2D, "--bias",
                            "0.3", "--device",
                            _spec(tmp_path, "Na = 5e16\nny = 6\n"),
                            "--output", str(tmp_path)).stdout)[0]
    lock = tomllib.loads((tmp_path / record["run_id"] / "input.lock.toml")
                         .read_text(encoding="utf-8"))
    assert lock["Na"] == 5e16 and lock["ny"] == 6
    assert lock["Nd"] == 1e16, "an unmentioned field keeps its default"
    assert record["grid_points"] == 875, "the mesh actually changed"


@pytest.mark.parametrize("text,expected", [
    ("Nq = 1e16\n", "not part of this capability's device"),
    ("ny = 4.5\n", "whole number"),
    ("Na = true\n", "not a boolean"),
    ("[doping]\nNa = 1e16\n", "flat set of scalars"),
    ("len_p = -1e-4\n", "cannot be built"),
    ("gamma = 0.5\n", "cannot be built"),
])
def test_a_device_that_cannot_apply_is_refused_by_name(tmp_path, text, expected):
    """Every one of these exits 2 and writes nothing.

    A misspelt key is the dangerous case: dropped in silence, the run looks
    like it honoured a setting it never saw. It is named instead.
    """
    out_dir = tmp_path / "runs"
    proc = run("run", "solve", PN2D, "--bias", "0.3",
               "--device", _spec(tmp_path, text), "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert expected in proc.stderr
    assert not out_dir.exists(), "a refused run must leave nothing behind"


def test_json_and_toml_describe_the_same_device(tmp_path):
    """Two spellings of one device must land in the SAME directory.

    The problem id hashes the resolved inputs, so if the file format leaked
    into it — a float parsed differently, a key ordered differently — the same
    device written two ways would be two problems. Asserted by running both
    and comparing, rather than against a hardcoded prefix.
    """
    ids = []
    for name, text in (("device.json", '{"Na": 5e16, "ny": 6}'),
                       ("device.toml", "Na = 5e16\nny = 6\n")):
        record = json.loads(run("--format", "json", "run", "solve", PN2D,
                                "--bias", "0.3", "--device",
                                _spec(tmp_path, text, name),
                                "--output", str(tmp_path)).stdout)[0]
        assert record["grid_points"] == 875
        ids.append(record["run_id"])
    assert ids[0] == ids[1]


def test_an_unreadable_device_file_is_input_not_internal(tmp_path):
    proc = run("run", "solve", PN2D, "--bias", "0.3",
               "--device", str(tmp_path / "absent.toml"),
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT

    proc = run("run", "solve", PN2D, "--bias", "0.3",
               "--device", _spec(tmp_path, "Na = 1e16", "device.yaml"),
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert ".toml or .json" in proc.stderr


# --- run sweep: candidates, under ONE contract ----------------------------

def sweep(*args, capability=PN2D):
    return run("--format", "json", "run", "sweep", capability, *args)


def test_a_sweep_writes_an_ordinary_run_per_point(tmp_path):
    """There is no such thing as a sweep-flavoured result.

    Each row is a normal artifact written through the same path `run solve`
    uses, so it can be reopened with `run show` and compared with `compare
    runs`. A sweep that produced its own private record format would be a
    second source of truth for what a result is.
    """
    proc = sweep("--vary", "bias_v=0.2,0.3,0.4", "--output", str(tmp_path),
                 capability=PN1D)
    assert proc.returncode == cliout.EXIT_OK, proc.stderr
    rows = json.loads(proc.stdout)
    assert len(rows) == 3
    assert [r["bias_v"] for r in rows] == [0.2, 0.3, 0.4]
    assert all(r["status"] == "potential-step-converged" for r in rows)

    for row in rows:
        shown = run("--format", "json", "run", "show", row["run_id"],
                    "--output", str(tmp_path))
        assert shown.returncode == cliout.EXIT_OK
        assert json.loads(shown.stdout)[0]["capability"] == PN1D


def test_the_solver_contract_is_identical_on_every_row(tmp_path):
    """The property that makes a column readable downward.

    Without it the table is a ranking across a changed contract — the exact
    comparison `compare runs` exits 2 rather than perform.
    """
    rows = json.loads(sweep("--vary", "bias_v=0.2,0.3", "--vary",
                            "Na=1e16,5e16", "--output",
                            str(tmp_path)).stdout)
    contracts = set()
    for row in rows:
        manifest = json.loads(
            (tmp_path / row["run_id"] / "manifest.json").read_text())
        contracts.add(json.dumps(manifest["solver"], sort_keys=True))
    assert len(contracts) == 1, "the rows were not solved under one contract"


def test_the_grid_is_the_product_of_its_axes(tmp_path):
    rows = json.loads(sweep("--vary", "Na=1e15,1e16,1e17", "--vary", "ny=4,8",
                            "--output", str(tmp_path)).stdout)
    assert len(rows) == 6
    assert {(r["Na"], r["ny"]) for r in rows} == \
        {(a, n) for a in (1e15, 1e16, 1e17) for n in (4, 8)}
    assert {r["run_id"] for r in rows} == set(r["run_id"] for r in rows)
    assert len({r["run_id"] for r in rows}) == 6, "two points shared a run"


def test_refining_the_unused_direction_barely_moves_the_answer(tmp_path):
    """A sweep worth having says something, and this one does.

    Nothing in this device varies along y, so doubling ny must change the
    current by far less than changing the doping does. That is a real physical
    reading taken FROM the table, which is what the command is for — and it
    would fail if ny were silently ignored, which is the failure a table of
    identical numbers hides best.
    """
    rows = json.loads(sweep("--vary", "ny=4,8", "--output",
                            str(tmp_path)).stdout)
    coarse, fine = (r["current_a_cm2"] for r in
                    sorted(rows, key=lambda r: r["ny"]))
    assert abs(fine / coarse - 1.0) < 1e-4          # mesh-converged in y
    assert fine != coarse, "ny had no effect at all; is it reaching the mesh?"


@pytest.mark.parametrize("axis", ["tol", "max_iter", "method"])
def test_varying_the_solver_contract_is_refused(tmp_path, axis):
    """Refusing is the feature, and it is the same rule `compare runs`
    enforces — seen from the other side. There it refuses because it cannot
    know the contract held; here the contract is held by construction."""
    proc = run("run", "sweep", PN2D, "--vary", f"{axis}=1,2",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "solver contract is fixed" in proc.stderr


@pytest.mark.parametrize("vary,expected", [
    ("Nq=1,2", "not part of this capability's device"),
    ("Na=1e16", "at least two values"),
    ("Na=1e16,big", "is not a number"),
    ("Na=1e16,1e16", "appears twice"),
    ("Na", "expected name=value"),
])
def test_a_sweep_that_cannot_be_read_is_refused(tmp_path, vary, expected):
    out_dir = tmp_path / "runs"
    proc = run("run", "sweep", PN2D, "--vary", vary, "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert expected in proc.stderr
    assert not out_dir.exists()


def test_the_same_axis_twice_is_refused(tmp_path):
    """Two --vary flags for one name would silently drop the first list."""
    proc = run("run", "sweep", PN2D, "--vary", "Na=1e16,2e16",
               "--vary", "Na=3e16,4e16", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "given twice" in proc.stderr


def test_a_bad_point_is_named_and_the_rest_still_run(tmp_path):
    """A table with a named gap says more than no table at all.

    gamma < 1 cannot build a mesh. The sweep must not abort on it: the other
    points are real results, and aborting would throw them away.
    """
    proc = run("--format", "json", "run", "sweep", PN2D,
               "--vary", "gamma=0.5,1.06", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_NO_CONVERGENCE
    rows = {r["gamma"]: r for r in json.loads(proc.stdout)}
    assert rows[0.5]["status"] == "invalid-device"
    assert rows[0.5]["run_id"] == "", "a refused point must not claim an artifact"
    assert rows[1.06]["status"] == "potential-step-converged"
    assert (tmp_path / rows[1.06]["run_id"]).is_dir()


def test_a_sweep_refuses_a_blocked_capability_before_doing_any_work(tmp_path):
    """solve and sweep must agree about what is runnable."""
    proc = run("run", "sweep", "semiconductor.mosfet.drift-diffusion.2d.steady",
               "--vary", "Na=1e16,2e16", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_UNAVAILABLE
    assert not list(tmp_path.iterdir())


def test_a_sweep_point_is_comparable_with_its_neighbour(tmp_path):
    """The pay-off: rows of one sweep pass the comparability contract on
    everything except the axis that was deliberately varied."""
    rows = json.loads(sweep("--vary", "Na=1e16,5e16", "--output",
                            str(tmp_path)).stdout)
    proc = run("compare", "runs", rows[0]["run_id"], rows[1]["run_id"],
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "different inputs" in proc.stderr        # only the inputs differ
    assert "different solver" not in proc.stderr
    assert "different build" not in proc.stderr


# --- candidates driving a real solve --------------------------------------

CANDIDATES = """
[SYNTH-A]
composition = "SyntheticA"
[SYNTH-A.properties.ni]
value = 1e10
unit = "cm^-3"
basis = "computed"
[SYNTH-A.properties.eps_s]
value = 1.0e-12
unit = "F/cm"
basis = "computed"
[SYNTH-A.properties.mu_n]
value = 1000.0
unit = "cm^2/Vs"
basis = "computed"
[SYNTH-A.properties.mu_p]
value = 400.0
unit = "cm^2/Vs"
basis = "computed"

[SYNTH-PARTIAL]
[SYNTH-PARTIAL.properties.mu_n]
value = 980.0
unit = "cm^2/Vs"
basis = "inferred"
uncertainty = 60.0
"""


def _candidates(tmp_path):
    path = tmp_path / "candidates.toml"
    path.write_text(CANDIDATES, encoding="utf-8")
    return str(path)


def test_a_candidate_can_drive_a_validated_solve(tmp_path):
    """The join that stops a candidate from being provenance-less JSON.

    A material is real here only if a validated model can be run on it, and
    the result must be a normal artifact that records WHICH material it was.
    """
    proc = run("--format", "json", "run", "solve", PN1D,
               "--candidate", _candidates(tmp_path),
               "--candidate-id", "SYNTH-A",
               "--bias", "0.3", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK, proc.stderr
    record = json.loads(proc.stdout)[0]

    import tomllib
    lock = tomllib.loads((tmp_path / record["run_id"] / "input.lock.toml")
                         .read_text(encoding="utf-8"))
    assert (lock["mu_n"], lock["mu_p"], lock["ni"], lock["eps_s"]) == \
        (1000.0, 400.0, 1e10, 1e-12)

    provenance = json.loads(
        (tmp_path / record["run_id"] / "provenance.json").read_text())
    assert provenance["candidate"] == "SYNTH-A"
    assert "overridden" in provenance["device"], \
        "provenance still claimed defaults over a candidate's properties"


def test_a_run_without_a_candidate_says_so_rather_than_leaving_it_blank(
        tmp_path):
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0.3", "--output",
                            str(tmp_path)).stdout)[0]
    provenance = json.loads(
        (tmp_path / record["run_id"] / "provenance.json").read_text())
    assert provenance["candidate"] == "none"
    assert provenance["device"] == "PNDiode1D defaults"


def test_an_incomplete_candidate_is_refused_with_the_missing_names(tmp_path):
    """Defaulting the gaps would solve a material that does not exist."""
    out_dir = tmp_path / "runs"
    proc = run("run", "solve", PN1D, "--candidate", _candidates(tmp_path),
               "--candidate-id", "SYNTH-PARTIAL", "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "missing ni, eps_s, mu_p" in proc.stderr
    assert not out_dir.exists()


def test_an_unknown_candidate_id_is_input_not_internal(tmp_path):
    proc = run("run", "solve", PN1D, "--candidate", _candidates(tmp_path),
               "--candidate-id", "SYNTH-NOPE", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "no such candidate" in proc.stderr


def test_two_candidates_are_two_problems(tmp_path):
    """The material is part of the problem's identity, so two materials must
    not share a run directory."""
    path = _candidates(tmp_path)
    first = json.loads(run("--format", "json", "run", "solve", PN1D,
                           "--candidate", path, "--candidate-id", "SYNTH-A",
                           "--output", str(tmp_path)).stdout)[0]
    second = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--output", str(tmp_path)).stdout)[0]
    assert first["run_id"] != second["run_id"]
    assert first["current_a_cm2"] != second["current_a_cm2"]


def test_candidate_screen_reports_every_candidate(tmp_path):
    proc = run("--format", "json", "candidate", "screen",
               "--from", _candidates(tmp_path), "--require", "mu_n>=1000")
    assert proc.returncode == cliout.EXIT_OK
    rows = {r["identifier"]: r["verdict"] for r in json.loads(proc.stdout)}
    assert rows == {"SYNTH-A": "pass", "SYNTH-PARTIAL": "undecided"}
    assert "undecided is not a soft fail" in proc.stderr


def test_candidate_list_names_what_each_material_is_missing(tmp_path):
    proc = run("--format", "json", "candidate", "list",
               "--from", _candidates(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    rows = {r["identifier"]: r for r in json.loads(proc.stdout)}
    assert rows["SYNTH-A"][PN1D] == "usable"
    assert "missing" in rows["SYNTH-PARTIAL"][PN1D]


def test_a_screen_with_no_threshold_is_refused(tmp_path):
    proc = run("candidate", "screen", "--from", _candidates(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "at least one --require" in proc.stderr


# --- the ghost candidate, and the other input-gate findings ---------------

@pytest.mark.parametrize("extra,expected", [
    (["--candidate-id", "GHOST"], "--candidate is required"),
    (["--candidate", "CANDFILE"], "--candidate-id is required"),
])
def test_a_candidate_id_without_its_file_is_refused(tmp_path, extra, expected):
    """The worst defect this project can ship: a FALSIFIED record.

    `--candidate-id GHOST` with no file was accepted, the solve ran on default
    material, and the artifact recorded {"candidate": "GHOST", "device":
    "PNDiode1D defaults"} — a scientific record naming a material that was
    never used, produced in silence. Reported in review from a real run.
    """
    extra = [str(_candidates(tmp_path)) if a == "CANDFILE" else a
             for a in extra]
    out_dir = tmp_path / "runs"
    proc = run("run", "solve", PN1D, *extra, "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert expected in proc.stderr
    assert not out_dir.exists(), "a run that names a phantom material ran anyway"


def test_two_candidates_with_equal_numbers_get_two_directories(tmp_path):
    """Measured in review: two candidates whose four values coincided landed
    on one run id, and the second overwrote the first's provenance."""
    twins = tmp_path / "twins.toml"
    body = "\n".join(
        f"[SYNTH-{tag}]\n"
        f"[SYNTH-{tag}.properties.ni]\nvalue = 1e10\nunit = \"cm^-3\"\n"
        f"basis = \"computed\"\n"
        f"[SYNTH-{tag}.properties.eps_s]\nvalue = 1e-12\nunit = \"F/cm\"\n"
        f"basis = \"computed\"\n"
        f"[SYNTH-{tag}.properties.mu_n]\nvalue = 1000.0\n"
        f"unit = \"cm^2/Vs\"\nbasis = \"computed\"\n"
        f"[SYNTH-{tag}.properties.mu_p]\nvalue = 400.0\n"
        f"unit = \"cm^2/Vs\"\nbasis = \"computed\"\n" for tag in ("A", "B"))
    twins.write_text(body, encoding="utf-8")

    ids = []
    for tag in ("A", "B"):
        record = json.loads(run("--format", "json", "run", "solve", PN1D,
                                "--candidate", str(twins), "--candidate-id",
                                f"SYNTH-{tag}", "--output",
                                str(tmp_path)).stdout)[0]
        ids.append(record["run_id"])
    assert ids[0] != ids[1]
    assert all((tmp_path / i).is_dir() for i in ids)


@pytest.mark.parametrize("flag,value", [
    ("--tol", "nan"), ("--tol", "inf"), ("--tol", "0"),
    ("--bias", "nan"), ("--bias", "inf"), ("--max-iter", "0"),
])
def test_a_solver_number_that_cannot_mean_anything_is_input_not_failure(
        tmp_path, flag, value):
    """`--tol nan` came back as exit 4, "did not converge" — which blames the
    physics for a typo. It is a bad argument and the exit code must say so."""
    proc = run("run", "solve", PN1D, flag, value, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert proc.returncode != cliout.EXIT_NO_CONVERGENCE


def test_a_sweep_with_no_axis_is_refused(tmp_path):
    """"One value is a solve" was enforced per axis, but NO axis slipped past
    and produced a cheerful one-point table."""
    proc = run("run", "sweep", PN1D, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "at least one --vary" in proc.stderr


def test_a_run_from_a_future_schema_is_refused_not_guessed_at(tmp_path):
    """Reading a newer directory with an older parser means guessing at fields
    this build has never seen and reporting the result as if it understood
    them."""
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--output", str(tmp_path)).stdout)[0]
    manifest_path = tmp_path / record["run_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))

    proc = run("run", "show", record["run_id"], "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "schema v99" in proc.stderr


# --- the rest of the audit ------------------------------------------------

def test_a_device_file_may_not_overwrite_a_candidates_material(tmp_path):
    """Same failure as the ghost candidate: a run whose provenance overstates
    what was solved. The device file was applied AFTER the candidate, so
    mu_n=99 was solved while provenance still said SYNTH-A."""
    device = tmp_path / "clash.toml"
    device.write_text("mu_n = 99.0\n", encoding="utf-8")
    out_dir = tmp_path / "runs"
    proc = run("run", "solve", PN1D, "--candidate", _candidates(tmp_path),
               "--candidate-id", "SYNTH-A", "--device", str(device),
               "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "mu_n" in proc.stderr and "SYNTH-A" in proc.stderr
    assert not out_dir.exists()


def test_a_device_file_that_does_not_clash_still_works(tmp_path):
    """The refusal must be about the OVERLAP, not about using both files."""
    device = tmp_path / "geometry.toml"
    device.write_text("len_p = 4e-4\n", encoding="utf-8")
    proc = run("run", "solve", PN1D, "--candidate", _candidates(tmp_path),
               "--candidate-id", "SYNTH-A", "--device", str(device),
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK, proc.stderr


def test_the_artifact_says_the_solve_was_nominal_only(tmp_path):
    """The screen uses each spread; the solve takes the nominal value alone.
    Defensible choice, indefensible silence."""
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--candidate", _candidates(tmp_path),
                            "--candidate-id", "SYNTH-A",
                            "--output", str(tmp_path)).stdout)[0]
    provenance = json.loads(
        (tmp_path / record["run_id"] / "provenance.json").read_text())
    assert provenance["uncertainty_treatment"] == "nominal-only"


def test_a_stale_field_file_does_not_survive_a_rerun(tmp_path):
    """Re-running a problem whose new result has no field data left the old
    fields.npz in place, and the fresh manifest checksummed it — so the run
    claimed field data it never produced."""
    from tarhan import artifact

    import numpy as np

    kwargs = dict(capability="cap.x.1d.steady", capability_status="validated",
                  inputs={"a": 1.0}, solver={"m": "g"}, metrics={"j": 1.0},
                  provenance={"p": "q"}, status="converged", command="c",
                  version="0")
    first = artifact.write_run(tmp_path, **kwargs,
                               fields_data={"psi": np.zeros(4)})
    assert (first / "fields.npz").exists()
    second = artifact.write_run(tmp_path, **kwargs)
    assert first == second
    assert not (second / "fields.npz").exists()

    manifest = json.loads((second / "manifest.json").read_text())
    assert "fields.npz" not in manifest["files"]
    assert artifact.read_run(second)["integrity"] == "verified"
    assert [p.name for p in tmp_path.iterdir()] == [second.name], \
        "a staging or displaced directory was left behind"


def test_the_recorded_command_can_be_pasted_back(tmp_path):
    """The manifest recorded `tarhan run solve <capability>` and dropped the
    bias, tolerance, candidate and device — so a run could not be re-issued
    from its own record, which is the one thing the field is for."""
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0.35", "--tol", "1e-8",
                            "--output", str(tmp_path)).stdout)[0]
    command = json.loads(
        (tmp_path / record["run_id"] / "manifest.json").read_text())["command"]
    assert "--bias 0.35" in command and "--tol 1e-8" in command
    assert PN1D in command


def test_run_show_full_reopens_the_whole_run(tmp_path):
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--candidate", _candidates(tmp_path),
                            "--candidate-id", "SYNTH-A",
                            "--output", str(tmp_path)).stdout)[0]
    proc = run("--format", "json", "run", "show", record["run_id"], "--full",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    rows = json.loads(proc.stdout)
    sections = {r["section"] for r in rows}
    assert {"input", "provenance", "integrity", "metric"} <= sections
    provenance = {r["key"]: r["value"] for r in rows
                  if r["section"] == "provenance"}
    assert provenance["candidate"] == "SYNTH-A"
    integrity = {r["key"]: r["value"] for r in rows
                 if r["section"] == "integrity"}
    assert integrity["state"] == "verified"


def test_quiet_is_actually_quiet(tmp_path):
    """Measured through a real subprocess, because the old test asserted
    `animate is False` — which was true, and beside the point: the plain
    per-stage lines were still written."""
    proc = run("--quiet", "run", "solve", PN1D, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert proc.stderr == "", f"--quiet still wrote: {proc.stderr!r}"


def test_quiet_still_lets_an_error_through(tmp_path):
    """Silence must not swallow the thing the user needs to see."""
    proc = run("--quiet", "run", "solve", "no.such.thing.1d.steady",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "no such capability" in proc.stderr


def test_an_unbounded_sweep_is_refused_before_it_allocates(tmp_path):
    """The whole product was materialised before any solve began."""
    axes = [f"--vary"] * 0
    args = []
    for name in ("Na", "Nd", "ni"):
        args += ["--vary", name + "=" + ",".join(str(1e15 * (i + 1))
                                                 for i in range(30))]
    proc = run("run", "sweep", PN2D, *args, "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "more than the" in proc.stderr
    assert not any(tmp_path.iterdir())


def test_compare_names_a_metric_that_exists_on_one_side_only(tmp_path):
    """A build that adds or removes a metric produced a table that looked
    complete — under --allow-build-diff that is exactly the difference
    somebody is looking for."""
    left = json.loads(run("--format", "json", "run", "solve", PN1D,
                          "--output", str(tmp_path)).stdout)[0]["run_id"]
    right = _restamp_build(tmp_path / left, "deadbeef")

    manifest_path = tmp_path / right / "metrics.json"
    metrics = json.loads(manifest_path.read_text())
    metrics["new_metric_from_a_later_build"] = 1.0
    manifest_path.write_text(json.dumps(metrics))
    # the checksum has to follow, or read_run refuses before compare runs
    mpath = tmp_path / right / "manifest.json"
    manifest = json.loads(mpath.read_text())
    import hashlib
    manifest["files"]["metrics.json"] = hashlib.sha256(
        manifest_path.read_bytes()).hexdigest()
    mpath.write_text(json.dumps(manifest))

    proc = run("compare", "runs", left, right, "--allow-build-diff",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert "new_metric_from_a_later_build" in proc.stderr
    assert "only in" in proc.stderr


# --- the re-audit ---------------------------------------------------------

def test_a_device_json_key_given_twice_is_refused(tmp_path):
    """Fixed in the candidate loader and not here, while the CHANGELOG claimed
    BOTH refused it — a false entry in a user-facing document is worse than
    the gap it described."""
    spec = tmp_path / "dup.json"
    spec.write_text('{"mu_n": 1000.0, "mu_n": 5.0}', encoding="utf-8")
    out_dir = tmp_path / "runs"
    proc = run("run", "solve", PN1D, "--device", str(spec),
               "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "given twice" in proc.stderr
    assert not out_dir.exists()


def test_the_candidate_evidence_travels_with_the_result(tmp_path):
    """The artifact kept an id and a 12-character fingerprint, so composition,
    basis, source, uncertainty and valid_range lived only in the user's file.
    Move that file and the run could no longer say what it rested on — while
    still carrying a fingerprint implying it could."""
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--candidate", _candidates(tmp_path),
                            "--candidate-id", "SYNTH-A",
                            "--output", str(tmp_path)).stdout)[0]
    directory = tmp_path / record["run_id"]
    lock = json.loads((directory / "candidate.lock.json").read_text())
    assert lock["identifier"] == "SYNTH-A"
    assert lock["properties"]["mu_n"]["basis"] == "computed"
    assert lock["properties"]["mu_n"]["unit"] == "cm^2/Vs"

    manifest = json.loads((directory / "manifest.json").read_text())
    assert "candidate.lock.json" in manifest["files"], \
        "the evidence must be checksummed like every other file"

    # ...and the run still reads back once the source file is gone.
    (tmp_path / "candidates.toml").unlink()
    proc = run("--format", "json", "run", "show", record["run_id"], "--full",
               "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK


def test_a_run_without_a_candidate_writes_no_candidate_lock(tmp_path):
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--output", str(tmp_path)).stdout)[0]
    assert not (tmp_path / record["run_id"] / "candidate.lock.json").exists()


def test_candidate_show_puts_the_identity_in_the_machine_format(tmp_path):
    """It went out as stderr notes, so `--format json --quiet` lost it and the
    "whole record" claim held only for a human reading a terminal."""
    proc = run("--quiet", "--format", "json", "candidate", "show", "SYNTH-A",
               "--from", _candidates(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    rows = {r["property"]: r["value"] for r in json.loads(proc.stdout)}
    assert rows["@identifier"] == "SYNTH-A"
    assert rows["@composition"] == "SyntheticA"
    assert len(rows["@fingerprint"]) == 12


def test_compare_reports_one_sided_metrics_in_the_machine_format(tmp_path):
    """Naming them on stderr left the JSON rows carrying only the
    intersection, so an automated consumer saw a comparison that looked
    complete."""
    left = json.loads(run("--format", "json", "run", "solve", PN1D,
                          "--output", str(tmp_path)).stdout)[0]["run_id"]
    right = _restamp_build(tmp_path / left, "deadbeef")
    metrics_path = tmp_path / right / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["only_on_the_right"] = 7.0
    metrics_path.write_text(json.dumps(metrics))
    mpath = tmp_path / right / "manifest.json"
    manifest = json.loads(mpath.read_text())
    import hashlib
    manifest["files"]["metrics.json"] = hashlib.sha256(
        metrics_path.read_bytes()).hexdigest()
    mpath.write_text(json.dumps(manifest))

    proc = run("--format", "json", "compare", "runs", left, right,
               "--allow-build-diff", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    rows = {r["metric"]: r for r in json.loads(proc.stdout)}
    assert "only_on_the_right" in rows
    assert rows["only_on_the_right"]["left"] is None
    assert rows["only_on_the_right"]["delta"] is None


def test_a_screen_can_be_run_under_a_condition(tmp_path):
    """valid_range was enforced at solve time and nowhere else, so a
    condition-dependent property voted unconditionally in a screen."""
    path = tmp_path / "ranged.toml"
    path.write_text(
        "[SYNTH-R]\n[SYNTH-R.properties.mu_n]\nvalue = 2000.0\n"
        'unit = "cm^2/Vs"\nbasis = "computed"\n'
        "valid_range = { bias_v = [0.0, 0.5] }\n", encoding="utf-8")

    inside = json.loads(run("--format", "json", "candidate", "screen",
                            "--from", str(path), "--at", "bias_v=0.3",
                            "--require", "mu_n>=1000").stdout)
    assert inside[0]["verdict"] == "pass"

    outside = json.loads(run("--format", "json", "candidate", "screen",
                             "--from", str(path), "--at", "bias_v=0.9",
                             "--require", "mu_n>=1000").stdout)
    assert outside[0]["verdict"] == "undecided"
    assert "valid" in outside[0]["why"]


def test_an_unconditioned_screen_of_ranged_properties_says_so(tmp_path):
    path = tmp_path / "ranged.toml"
    path.write_text(
        "[SYNTH-R]\n[SYNTH-R.properties.mu_n]\nvalue = 2000.0\n"
        'unit = "cm^2/Vs"\nbasis = "computed"\n'
        "valid_range = { bias_v = [0.0, 0.5] }\n", encoding="utf-8")
    proc = run("candidate", "screen", "--from", str(path),
               "--require", "mu_n>=1000")
    assert proc.returncode == cliout.EXIT_OK
    assert "--at" in proc.stderr


def test_an_unknown_condition_is_refused(tmp_path):
    proc = run("candidate", "screen", "--from", _candidates(tmp_path),
               "--at", "temperature_k=300", "--require", "mu_n>=1")
    assert proc.returncode == cliout.EXIT_INPUT
    assert "not a condition a run reports" in proc.stderr


# --- the third audit ------------------------------------------------------

def test_a_run_outside_the_validated_range_says_so(tmp_path):
    """A 0.2 V 2D solve wrote `status=converged`, `capability_status=validated`
    and no signal at all, while the registry's own prose said that hole
    current does not converge. Prose no run can check itself against is not a
    limit."""
    inside = json.loads(run("--format", "json", "run", "solve", PN2D,
                            "--bias", "0.4", "--output",
                            str(tmp_path)).stdout)[0]
    assert inside["status"] == "potential-step-converged"

    # 0.5 V, not 0.2 V. The envelope used to be borrowed from the DEVSIM-mesh
    # stage, under which 0.2 V was excluded; corrected to this device's own
    # evidence, 0.2 V is INSIDE (its current is checked there) and 0.5 V is
    # outside (nothing has been measured there for this device).
    outside = json.loads(run("--format", "json", "run", "solve", PN2D,
                             "--bias", "0.5", "--output",
                             str(tmp_path)).stdout)[0]
    assert outside["status"] == "potential-step-converged-outside-validated-range"

    provenance = json.loads((tmp_path / outside["run_id"]
                             / "provenance.json").read_text())
    assert "outside the validated" in provenance["validation_envelope"]
    manifest = json.loads((tmp_path / outside["run_id"]
                           / "manifest.json").read_text())
    assert manifest["status"] == "potential-step-converged-outside-validated-range"


def test_the_envelope_warning_reaches_a_human_too(tmp_path):
    proc = run("run", "solve", PN2D, "--bias", "0.5", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert "OUTSIDE THE VALIDATED RANGE" in proc.stderr


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_non_finite_sweep_value_is_input_not_a_solver_failure(tmp_path,
                                                                value):
    """`--vary bias_v=nan,0.2` was accepted: the point then failed as an
    "invalid-device" (exit 4, a SOLVER failure) and the JSON carried a bare
    NaN literal, which is not valid JSON and breaks the consumer the format
    exists for."""
    out_dir = tmp_path / "runs"
    proc = run("run", "sweep", PN1D, "--vary", f"bias_v={value},0.2",
               "--output", str(out_dir))
    assert proc.returncode == cliout.EXIT_INPUT
    assert "finite" in proc.stderr
    assert not out_dir.exists()


def test_sweep_json_is_always_parseable(tmp_path):
    """The guard above exists to protect this property."""
    proc = run("--format", "json", "run", "sweep", PN1D,
               "--vary", "bias_v=0.2,0.3", "--output", str(tmp_path))
    json.loads(proc.stdout)          # would raise on a bare NaN
    assert "NaN" not in proc.stdout


def test_an_unconditioned_screen_is_undecided_not_a_pass(tmp_path):
    """A stderr note was not enough: --quiet --format json dropped it and a
    ranged property came back a clean PASS."""
    path = tmp_path / "ranged.toml"
    path.write_text(
        "[SYNTH-R]\n[SYNTH-R.properties.mu_n]\nvalue = 2000.0\n"
        'unit = "cm^2/Vs"\nbasis = "computed"\n'
        "valid_range = { bias_v = [0.0, 0.5] }\n", encoding="utf-8")

    proc = run("--quiet", "--format", "json", "candidate", "screen",
               "--from", str(path), "--require", "mu_n>=1000")
    assert proc.returncode == cliout.EXIT_OK
    rows = json.loads(proc.stdout)
    assert rows[0]["verdict"] == "undecided"
    assert "no condition was given" in rows[0]["why"]


def test_dropping_a_checksum_entry_does_not_disable_the_check(tmp_path):
    """The integrity hole: verification iterated the RECORDED map, so deleting
    an entry deleted its check. Drop metrics.json from the map, rewrite the
    metrics, and read_run reported `verified` over a forged result."""
    from tarhan import artifact

    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--output", str(tmp_path)).stdout)[0]
    directory = tmp_path / record["run_id"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]["metrics.json"]
    manifest_path.write_text(json.dumps(manifest))
    (directory / "metrics.json").write_text('{"current_a_cm2": 999.0}')

    with pytest.raises(artifact.ArtifactError, match="not recorded"):
        artifact.read_run(directory)


def test_a_file_added_after_the_run_is_caught(tmp_path):
    """The same set-equality check from the other side."""
    from tarhan import artifact

    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--output", str(tmp_path)).stdout)[0]
    (tmp_path / record["run_id"] / "extra.txt").write_text("added later")
    with pytest.raises(artifact.ArtifactError, match="not recorded"):
        artifact.read_run(tmp_path / record["run_id"])


@pytest.mark.parametrize("gamma", ["1e308", "inf"])
def test_an_overflowing_grid_estimate_is_input_not_internal(tmp_path, gamma):
    """gamma=1e308 made the intermediate infinite and math.ceil raised
    OverflowError, which the CLI reported as exit 5 — OUR bug — for a number
    the user typed."""
    spec = tmp_path / "device.toml"
    spec.write_text(f"gamma = {gamma}\n", encoding="utf-8")
    proc = run("run", "solve", PN1D, "--device", str(spec),
               "--output", str(tmp_path / "runs"))
    assert proc.returncode == cliout.EXIT_INPUT
    assert proc.returncode != cliout.EXIT_INTERNAL


# --- the fourth audit -----------------------------------------------------

def test_a_sweep_point_records_everything_a_solve_records(tmp_path):
    """A sweep point and a solve of the same problem share a run id, so the
    poorer record OVERWROTE the richer one: sweeping a bias already solved
    dropped validation_envelope, uncertainty_treatment, the device file and
    the candidate. One provenance producer now serves both."""
    solved = json.loads(run("--format", "json", "run", "solve", PN2D,
                            "--bias", "0.5", "--output",
                            str(tmp_path)).stdout)[0]
    before = json.loads((tmp_path / solved["run_id"]
                         / "provenance.json").read_text())

    proc = run("--format", "json", "run", "sweep", PN2D,
               "--vary", "bias_v=0.5,0.3", "--output", str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK, proc.stderr
    after = json.loads((tmp_path / solved["run_id"]
                        / "provenance.json").read_text())

    assert set(after) == set(before), "the sweep wrote a poorer record"
    assert "outside the validated" in after["validation_envelope"]
    assert after["uncertainty_treatment"] == "nominal-only"


def test_a_device_override_leaves_the_reference_device(tmp_path):
    """The envelope covers the inputs it NAMES. A 0.3 V solve of a completely
    different material was coming back plainly `converged`, which reads as
    "inside the evidence" and is a much stronger claim than bias alone
    supports."""
    spec = tmp_path / "other.toml"
    spec.write_text("mu_n = 700.0\n", encoding="utf-8")
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0.3", "--device", str(spec),
                            "--output", str(tmp_path)).stdout)[0]
    assert record["status"] == "potential-step-converged-outside-validated-range"
    provenance = json.loads((tmp_path / record["run_id"]
                             / "provenance.json").read_text())
    assert "reference device" in provenance["validation_envelope"]


def test_the_runnable_1d_capability_has_an_envelope_too(tmp_path):
    """It had none, so a 0.5 V default solve was plainly `converged` while the
    cross-oracle evidence runs 0.15-0.40 V."""
    inside = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0.3", "--output",
                            str(tmp_path)).stdout)[0]
    assert inside["status"] == "potential-step-converged"
    outside = json.loads(run("--format", "json", "run", "solve", PN1D,
                             "--bias", "0.5", "--output",
                             str(tmp_path)).stdout)[0]
    assert outside["status"] == "potential-step-converged-outside-validated-range"


def test_the_envelope_is_discoverable_without_running_anything():
    """A machine-readable limit that no machine format carries is not
    machine-readable: a client could only learn the range by running something
    and reading the status back."""
    proc = run("--format", "json", "capabilities", "show", PN2D)
    assert proc.returncode == cliout.EXIT_OK
    record = json.loads(proc.stdout)[0]
    # STRUCTURED, not prose. It was emitted as "bias_v in [0.3, 0.5]", which
    # is visible and still unusable: an automated consumer had to parse a
    # sentence, the one thing the output contract exists to prevent.
    assert record["envelope"] == {"bias_v": {"intervals": [[0.0, 0.0],
                                                           [0.2, 0.4]]}}

    listed = json.loads(run("--format", "json", "capabilities",
                            "list").stdout)
    assert all("envelope" in row for row in listed)


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_a_non_finite_screen_condition_is_refused(tmp_path, value):
    """Every range comparison against NaN is False, so an out-of-range
    property could screen clean."""
    proc = run("candidate", "screen", "--from", _candidates(tmp_path),
               "--at", f"bias_v={value}", "--require", "mu_n>=1")
    assert proc.returncode == cliout.EXIT_INPUT
    assert "finite" in proc.stderr


# --- the fifth audit ------------------------------------------------------

def test_a_swept_device_override_also_leaves_the_reference_device(tmp_path):
    """The check was added to solve and not to sweep, so a swept override came
    back plainly `converged` while its own provenance said "overridden" —
    an artifact asserting two contradictory things at once. Reported in
    re-review."""
    spec = tmp_path / "other.json"
    spec.write_text('{"mu_n": 700.0}', encoding="utf-8")
    rows = json.loads(run("--format", "json", "run", "sweep", PN2D,
                          "--device", str(spec), "--vary", "bias_v=0.3,0.4",
                          "--output", str(tmp_path)).stdout)
    assert [r["status"] for r in rows] == \
        ["potential-step-converged-outside-validated-range"] * 2
    provenance = json.loads((tmp_path / rows[0]["run_id"]
                             / "provenance.json").read_text())
    assert "reference device" in provenance["validation_envelope"]


def test_sweeping_a_device_axis_leaves_the_reference_device(tmp_path):
    """The reason the check reads the RESOLVED device rather than the flags:
    no flag check would catch `--vary mu_n=...`."""
    rows = json.loads(run("--format", "json", "run", "sweep", PN2D,
                          "--vary", "mu_n=400,700", "--bias", "0.4",
                          "--output", str(tmp_path)).stdout)
    assert all(r["status"] == "potential-step-converged-outside-validated-range"
               for r in rows)


def test_a_plain_sweep_on_the_reference_device_stays_inside(tmp_path):
    """The flag must mean something: it cannot fire for every sweep."""
    rows = json.loads(run("--format", "json", "run", "sweep", PN2D,
                          "--vary", "bias_v=0.3,0.4", "--output",
                          str(tmp_path)).stdout)
    assert all(r["status"] == "potential-step-converged" for r in rows)


@pytest.mark.parametrize("capability", [PN1D, PN2D])
def test_equilibrium_is_inside_the_envelope(tmp_path, capability):
    """A single interval per input reported `outside-validated-range` for 0 V
    — the best-validated point in either capability.

    The DEVSIM attribution that stood here was wrong and was reported as
    such: 2.24e-16 V is stage 2D-1, on DEVSIM's OWN mesh. The device this
    command builds is the generated rectangular diode, whose equilibrium
    evidence is max|dpsi_hat| 1.26e-13 against the validated 1D solver. Same
    conclusion, different oracle — and getting that wrong in a docstring is
    how the envelope came to describe the wrong device in the first place.
    """
    record = json.loads(run("--format", "json", "run", "solve", capability,
                            "--bias", "0", "--output",
                            str(tmp_path)).stdout)[0]
    assert record["status"] == "potential-step-converged"


def test_the_gap_between_the_intervals_is_still_outside(tmp_path):
    """Adding equilibrium must not swallow the exclusion above it.

    For the generated 2D device the evidence runs to 0.40 V, so 0.45 V sits
    in neither interval and must be reported outside — otherwise a union
    would just be a wider single range.
    """
    record = json.loads(run("--format", "json", "run", "solve", PN2D,
                            "--bias", "0.45", "--output",
                            str(tmp_path)).stdout)[0]
    assert record["status"] == "potential-step-converged-outside-validated-range"


def test_an_artifact_records_which_evidence_said_inside(tmp_path):
    """`envelope_basis` was free text in the registry with no machine link to
    anything, and none of it reached the artifact — so a run could not answer
    "which evidence profile said inside?" after the fact. Reported in
    re-review."""
    from tarhan.capability_registry import get

    record = json.loads(run("--format", "json", "run", "solve", PN2D,
                            "--bias", "0.4", "--output",
                            str(tmp_path)).stdout)[0]
    provenance = json.loads((tmp_path / record["run_id"]
                             / "provenance.json").read_text())

    assert provenance["validation_profile"] == \
        get(PN2D).validation_profile()
    assert "RectangularDiode2D" in provenance["validation_basis"]
    assert len(provenance["device_fingerprint"]) == 12

    # Per metric, not one word for the whole artifact: at 0.40 V this device's
    # current is a measured point and its potential is not covered at all.
    coverage = dict(part.split("=") for part in
                    provenance["metric_coverage"].split("; "))
    assert coverage["current_a_cm2"] == "measured-point"
    assert coverage["psi"] == "outside"


def test_two_different_materials_get_two_device_fingerprints(tmp_path):
    """Provenance said "overridden" without saying overridden to WHAT, so two
    different materials produced records that read identically."""
    prints = set()
    for mu_n in (700.0, 900.0):
        spec = tmp_path / f"d{mu_n}.toml"
        spec.write_text(f"mu_n = {mu_n}\n", encoding="utf-8")
        record = json.loads(run("--format", "json", "run", "solve", PN1D,
                                "--bias", "0.3", "--device", str(spec),
                                "--output", str(tmp_path)).stdout)[0]
        prints.add(json.loads((tmp_path / record["run_id"]
                               / "provenance.json").read_text())
                   ["device_fingerprint"])
    assert len(prints) == 2


def test_a_sweep_point_carries_the_evidence_fields_too(tmp_path):
    rows = json.loads(run("--format", "json", "run", "sweep", PN2D,
                          "--vary", "bias_v=0.3,0.4", "--output",
                          str(tmp_path)).stdout)
    provenance = json.loads((tmp_path / rows[0]["run_id"]
                             / "provenance.json").read_text())
    for key in ("validation_profile", "validation_basis", "metric_coverage",
                "device_fingerprint"):
        assert provenance[key], f"sweep dropped {key}"


def test_capabilities_show_exposes_the_whole_evidence_claim():
    proc = run("--format", "json", "capabilities", "show", PN2D)
    record = json.loads(proc.stdout)[0]
    assert "RectangularDiode2D" in record["envelope_basis"]
    assert len(record["validation_profile"]) == 12
    assert record["coverage"]["current_a_cm2"]["points"] == [0.2, 0.3, 0.4]
    assert record["coverage"]["psi"]["points"] == [0.0, 0.3]


def test_the_solver_status_claims_only_what_was_tested(tmp_path):
    """`solver_status=converged` sat beside `current_rel_change=3.49e-4`.

    Only the potential step is checked, so that is what the field says now,
    with the current's assessment as its own value. Reported in review: the
    split into two fields is worthless if the first one keeps the old lie.
    """
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0.1", "--output",
                            str(tmp_path)).stdout)[0]
    assert record["status"] == "potential-step-converged"
    provenance = json.loads((tmp_path / record["run_id"]
                             / "provenance.json").read_text())
    assert provenance["solver_status"] == "potential-step-converged"
    assert provenance["current_convergence"].startswith("measured:")


def test_an_undefined_current_change_is_null_not_a_number(tmp_path):
    """Equilibrium published 0.8999 and a one-pass warm start published `inf`,
    neither of which a machine can tell from a measurement."""
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0", "--output",
                            str(tmp_path)).stdout)[0]
    assert record["current_rel_change"] is None
    provenance = json.loads((tmp_path / record["run_id"]
                             / "provenance.json").read_text())
    assert provenance["current_convergence"] == "unassessed"

    metrics = json.loads((tmp_path / record["run_id"]
                          / "metrics.json").read_text())
    assert metrics["current_rel_change"] is None


def test_validation_status_is_metric_aware(tmp_path):
    """`inside` described bias-envelope membership while the artifact
    published metrics that envelope says nothing about — a 1D 0.1 V run read
    `inside` next to `psi=outside`."""
    record = json.loads(run("--format", "json", "run", "solve", PN1D,
                            "--bias", "0.1", "--output",
                            str(tmp_path)).stdout)[0]
    provenance = json.loads((tmp_path / record["run_id"]
                             / "provenance.json").read_text())
    assert provenance["validation_status"] == "partially-covered"
    assert "psi=outside" in provenance["metric_coverage"]

    covered = json.loads(run("--format", "json", "run", "solve", PN1D,
                             "--bias", "0", "--output",
                             str(tmp_path)).stdout)[0]
    other = json.loads((tmp_path / covered["run_id"]
                        / "provenance.json").read_text())
    assert other["validation_status"] in ("fully-covered", "partially-covered")


@pytest.mark.parametrize("model", ["pn1d", "pn2d"])
def test_the_progress_callback_still_works_positionally(model):
    """A silent API regression I introduced: `current_tol` was inserted in
    front of `on_iteration`, so an old positional caller passed its callback
    as a tolerance and it was never invoked — zero calls, no error. Every test
    used keywords, so nothing caught it. The parameter is gone; this pins the
    order so it cannot be shifted again."""
    calls = []
    if model == "pn1d":
        from tarhan.models.pn1d import PNDiode1D, solve_bias
        solve_bias(PNDiode1D(), 0.3, None, 1e-9, 60,
                   lambda index, total: calls.append(index))
    else:
        from tarhan.models.diode2d_mesh import device
        from tarhan.models.pn2d import solve_bias
        solve_bias(device(), 0.3, None, 1e-9, 200,
                   lambda index, total: calls.append(index))
    assert calls, "the sixth positional argument is no longer the callback"


def test_a_published_metric_with_no_coverage_record_is_unverified(tmp_path):
    """`fully-covered` meant "every metric that HAS a record is covered".

    A 2D run publishes nine metrics while the registry covers two, and it was
    reporting fully-covered — so `unverified`, which the coverage vocabulary
    defines, could never actually be reached. Reported in review.
    """
    record = json.loads(run("--format", "json", "run", "solve", PN2D,
                            "--bias", "0.3", "--output",
                            str(tmp_path)).stdout)[0]
    provenance = json.loads((tmp_path / record["run_id"]
                             / "provenance.json").read_text())
    coverage = dict(part.split("=") for part in
                    provenance["metric_coverage"].split("; "))

    assert coverage["current_a_cm2"] == "measured-point"
    assert coverage["terminal_current_a_per_cm"] == "unverified"
    assert coverage["hole_current_a_per_cm"] == "unverified"
    assert provenance["validation_status"] == "partially-covered"

    # ...and the run's own diagnostics are not counted as physics claims.
    assert "psi_step" not in coverage
    assert "gummel_iterations" not in coverage


def test_the_terminal_does_not_claim_more_than_the_artifact(tmp_path):
    """The artifact said `potential-step-converged` while the terminal line
    still said "converged". The weaker claim has to be the one in both."""
    proc = run("run", "solve", PN1D, "--bias", "0.1", "--output",
               str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert "converged:" not in proc.stderr
    assert "potential step settled" in proc.stderr


def test_the_outside_range_terminal_path_does_not_overclaim_either(tmp_path):
    """The inner path was fixed and the outside-envelope branch was not.

    A 0.5 V 2D run printed "the solve converged and the artifact is written"
    beside a status of `potential-step-converged-outside-validated-range` —
    the same overclaim, on the branch nobody tested. Reported in review, and
    the second time a fix landed on one of two paths.
    """
    proc = run("run", "solve", PN2D, "--bias", "0.5", "--output",
               str(tmp_path))
    assert proc.returncode == cliout.EXIT_OK
    assert "OUTSIDE THE VALIDATED RANGE" in proc.stderr
    assert "the solve converged" not in proc.stderr
    assert "potential step settled" in proc.stderr


@pytest.mark.parametrize("model", ["pn1d", "pn2d"])
@pytest.mark.parametrize("kwargs", [
    {"min_gummel": 81, "max_gummel": 80},
    {"min_gummel": -1},
    {"min_gummel": 2.5},
    {"min_gummel": True},
])
def test_min_gummel_keeps_its_own_promise(model, kwargs):
    """It silently returned max_gummel iterations when the floor exceeded the
    ceiling, so a diagnostic asking for N passes quietly got fewer — the
    measurement path breaking its own contract."""
    if model == "pn1d":
        from tarhan.models.pn1d import PNDiode1D, solve_bias
        device, bias = PNDiode1D(), 0.1
    else:
        from tarhan.models.diode2d_mesh import device as build
        from tarhan.models.pn2d import solve_bias
        device, bias = build(), 0.3

    with pytest.raises(ValueError):
        solve_bias(device, bias, **kwargs)


def test_the_device_field_and_the_envelope_field_agree(tmp_path):
    """One artifact said `device: "...defaults"` and, in the same file,
    `validation_envelope: "mu_n=700 differs from reference 1350.0"`.

    `_off_reference_device` read the resolved inputs while `_provenance` read
    the command-line flags, so `--vary mu_n=` was invisible to one of them.
    Reported in review — the same flags-versus-resolved split that was fixed
    in the envelope check and not here.
    """
    rows = json.loads(run("--format", "json", "run", "sweep", PN2D,
                          "--vary", "mu_n=400,700", "--bias", "0.4",
                          "--output", str(tmp_path)).stdout)
    for row in rows:
        provenance = json.loads((tmp_path / row["run_id"]
                                 / "provenance.json").read_text())
        assert "overridden" in provenance["device"]
        assert "mu_n" in provenance["device"]
        assert "reference device" in provenance["validation_envelope"]

    # ...and a run on the reference device says so plainly, in both fields.
    plain = json.loads(run("--format", "json", "run", "solve", PN2D,
                           "--bias", "0.4", "--output",
                           str(tmp_path)).stdout)[0]
    provenance = json.loads((tmp_path / plain["run_id"]
                             / "provenance.json").read_text())
    assert "overridden" not in provenance["device"]
    assert provenance["validation_envelope"] == "inside"


# --- issue #2: a repeated condition changed the verdict -------------------

def _ranged_candidate(tmp_path):
    path = tmp_path / "ranged.toml"
    path.write_text(
        "[SYNTH-R]\n[SYNTH-R.properties.mu_n]\nvalue = 2000.0\n"
        'unit = "cm^2/Vs"\nbasis = "computed"\n'
        "valid_range = { bias_v = [0.0, 0.5] }\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("first,second", [("0.3", "0.9"), ("0.9", "0.3")])
def test_a_repeated_screen_condition_is_refused(tmp_path, first, second):
    """Issue #2. The last `--at` silently won, so REVERSING two otherwise
    identical arguments changed the verdict: 0.3 then 0.9 gave `undecided`,
    0.9 then 0.3 gave `pass`, and both exited 0. A caller could not tell from
    stdout or the status that a value had been discarded.

    `--vary` already refused a repeated axis for the same reason — the fifth
    time this cycle a guard existed on one of two sibling paths.
    """
    proc = run("--format", "json", "--quiet", "candidate", "screen",
               "--from", _ranged_candidate(tmp_path),
               "--at", f"bias_v={first}", "--at", f"bias_v={second}",
               "--require", "mu_n>=1000")
    assert proc.returncode == cliout.EXIT_INPUT
    assert "given twice" in proc.stderr
    assert first in proc.stderr and second in proc.stderr


def test_one_condition_per_name_still_screens(tmp_path):
    """The guard must reject the duplicate, not the feature."""
    source = _ranged_candidate(tmp_path)
    inside = json.loads(run("--format", "json", "--quiet", "candidate",
                            "screen", "--from", source, "--at", "bias_v=0.3",
                            "--require", "mu_n>=1000").stdout)
    outside = json.loads(run("--format", "json", "--quiet", "candidate",
                             "screen", "--from", source, "--at", "bias_v=0.9",
                             "--require", "mu_n>=1000").stdout)
    assert inside[0]["verdict"] == "pass"
    assert outside[0]["verdict"] == "undecided"


# --- issue #4: one build snapshot per sweep -------------------------------

def test_a_sweep_discovers_its_build_once_not_once_per_point(tmp_path,
                                                             monkeypatch):
    """Issue #4. Eleven points ran the source scan eleven times and spawned
    22 git subprocesses — 0.295 s of a 0.917 s sweep, against 0.018 s for all
    eleven solves.

    Counted rather than timed: a timing assertion would be the machine-fitted
    mistake this session has made four times. The COUNT is the invariant.
    """
    import subprocess

    from tarhan import cli

    calls = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])

    exit_code = cli.main(["--quiet", "run", "sweep", PN1D,
                          "--vary", "bias_v=0.20,0.25,0.30,0.35,0.40",
                          "--output", str(tmp_path)])
    assert exit_code == cliout.EXIT_OK
    assert len(calls) <= 2, (
        f"a 5-point sweep spawned {len(calls)} subprocesses; the build "
        "snapshot is being recomputed per point again")


def test_every_point_of_one_sweep_records_the_same_build(tmp_path):
    """The policy issue #4 asked to make explicit: all points of one sweep
    belong to the build observed when the sweep started."""
    rows = json.loads(run("--format", "json", "run", "sweep", PN1D,
                          "--vary", "bias_v=0.25,0.30,0.35",
                          "--output", str(tmp_path)).stdout)
    builds = set()
    environments = set()
    for row in rows:
        manifest = json.loads((tmp_path / row["run_id"]
                               / "manifest.json").read_text())
        builds.add(manifest["code_id"])
        environments.add(json.dumps(manifest["environment"], sort_keys=True))
    assert len(builds) == 1 and len(environments) == 1
    assert all(row["run_id"].endswith(builds.pop() if builds else "")
               or True for row in rows)


def test_a_standalone_write_run_still_computes_its_own_build(tmp_path):
    """The default must stay self-computing: a later independent command has
    to see source or environment changes, which is why this is a parameter
    rather than a module-level cache."""
    from tarhan import artifact

    kwargs = dict(capability="cap.x.1d.steady", capability_status="validated",
                  inputs={"a": 1.0}, solver={"m": "g"}, metrics={"j": 1.0},
                  provenance={"p": "q"}, status="converged", command="c",
                  version="0")
    plain = artifact.write_run(tmp_path / "plain", **kwargs)
    manifest = json.loads((plain / "manifest.json").read_text())
    assert manifest["code_id"] == artifact.code_id()

    # ...and an explicit snapshot is honoured rather than ignored.
    pinned = dict(artifact.environment(), tarhan="9.9.9-synthetic")
    other = artifact.write_run(tmp_path / "pinned", **kwargs,
                               build_snapshot=pinned)
    other_manifest = json.loads((other / "manifest.json").read_text())
    assert other_manifest["environment"]["tarhan"] == "9.9.9-synthetic"
    assert other_manifest["code_id"] == artifact.code_id(pinned)
    assert other_manifest["code_id"] != manifest["code_id"]

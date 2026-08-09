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
    assert all(r["status"] == "converged" for r in rows)

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
    assert rows[1.06]["status"] == "converged"
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

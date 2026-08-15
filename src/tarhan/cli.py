"""TARHAN CLI — the zero-config proof: ``tarhan demo``.

İki vaka (``--case``):
  cottrell  — Cottrell kronoamperometri reprodüksiyonu vs analitik G = 1/sqrt(πT)
  diode     — 1D pn-diyot (Gummel/SG amiral gemisi): I-V + band diyagramı;
              ideality 1.00±0.02 öz-denetimi
Demolar KENDİNİ DOĞRULAR: tolerans aşılırsa çıkış kodu 1 (sessiz degrade yok —
kuruluş ilkesi #6).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from tarhan import __version__, cliout, physics
from tarhan.capability_registry import CapabilityNotFound, all_capabilities, get
from tarhan.forge import Forge
from tarhan.numerics.diffusion1d import cottrell_fd_samples

_CAP_COLUMNS = ("id", "status", "dimension", "time", "source", "envelope",
                "envelope_basis", "validation_profile", "coverage")


def _envelope_text(cap) -> str:
    """The validated range as a string a machine format can carry."""
    return "; ".join(
        f"{name} in " + " or ".join(f"[{low:g}, {high:g}]"
                                    for low, high in intervals)
        for name, intervals in sorted(cap.envelope.items()))


def _cap_row(cap, fmt: str = "table") -> dict:
    # `envelope` is on the record and was NOT in this dict, so a client could
    # not discover the valid range without running something and reading the
    # status back. A machine-readable limit that no machine format carries is
    # not machine-readable. Reported in re-review.
    return {"id": cap.id, "status": cap.status, "dimension": cap.dimension,
            "time": cap.time, "source": cap.source,
            # Structured for json — an automated consumer must not have to
            # parse "bias_v in [0.3, 0.5]" out of a sentence, which is the one
            # thing the output contract exists to prevent. Text elsewhere,
            # because a table cell and a CSV field cannot hold a nested map.
            "envelope": (cap.envelope_json() if fmt == "json"
                         else _envelope_text(cap)),
            "envelope_basis": cap.envelope_basis,
            "validation_profile": cap.validation_profile(),
            "coverage": ({c.metric: {"points": list(c.points),
                                     "evidence": c.evidence}
                          for c in cap.coverage} if fmt == "json"
                         else "; ".join(f"{c.metric}@"
                                        + ",".join(f"{p:g}" for p in c.points)
                                        for c in cap.coverage))}


def _capabilities_list(out: cliout.Output) -> int:
    caps = all_capabilities()
    out.emit([_cap_row(c, out.fmt) for c in caps], _CAP_COLUMNS)
    stuck = sum(not c.runnable for c in caps)
    out.note(f"{len(caps)} capabilities, {stuck} of them not runnable today. "
             "`tarhan capabilities show <id>` says why.")
    return cliout.EXIT_OK


def _check_import(module: str):
    """Import a dependency and report its version, or why it is not usable."""
    try:
        mod = __import__(module)
    except Exception as exc:                              # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"{module} {getattr(mod, '__version__', 'unknown version')}"


def _check_registry():
    caps = all_capabilities()
    runnable = sum(c.runnable for c in caps)
    return True, (f"{len(caps)} capabilities, {runnable} runnable, "
                  f"{len(caps) - runnable} blocked or planned")


def _check_evidence():
    """Every validated claim must still name a test that exists.

    An installed copy whose evidence points at deleted files is a copy whose
    claims cannot be re-checked. In a wheel the validation tree is not shipped,
    so absence there is expected and reported as such rather than as a fault.
    """
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[2]
    if not (repo / "validation").is_dir():
        return None, "validation tree not present (expected in a wheel install)"
    missing = [ev.test for cap in all_capabilities() for ev in cap.evidence
               if not (repo / ev.test).exists()]
    if missing:
        return False, f"{len(missing)} evidence files missing, e.g. {missing[0]}"
    total = sum(len(c.evidence) for c in all_capabilities())
    return True, f"{total} evidence files present"


@contextlib.contextmanager
def _stdout_to_stderr():
    """Hold the file descriptor, not just ``sys.stdout``.

    DEVSIM prints a BLAS/UMFPACK banner from C at import time. That lands on
    file descriptor 1 directly, so ``contextlib.redirect_stdout`` does not see
    it and it appears in the middle of a ``--format json`` stream, breaking the
    one promise this CLI makes to a machine consumer. Caught by running the new
    doctor command through a pipe — a diagnostic banner is diagnostics, so it
    goes where diagnostics go.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _check_devsim():
    try:
        with _stdout_to_stderr():
            import devsim                                 # noqa: F401
    except Exception:                                     # noqa: BLE001
        return None, "not installed — cross-oracle tests will skip"
    return True, "available for cross-oracle validation"


#: (name, what-it-is, callable). A callable returns (True | False | None,
#: detail); None means optional-and-absent, reported without failing the run.
DOCTOR_CHECKS = (
    ("numpy", "array kernel", lambda: _check_import("numpy")),
    ("scipy", "sparse solver and integrators", lambda: _check_import("scipy")),
    ("matplotlib", "plotting for the demos", lambda: _check_import("matplotlib")),
    ("registry", "loading the capability records", _check_registry),
    ("evidence", "checking every claim still names its test", _check_evidence),
    ("devsim", "optional cross-oracle simulator", _check_devsim),
)


def _make_forge(out: cliout.Output, stages, *, style: str, graphics: str,
                pin=False):
    """Build the display, with the raster backend when the terminal has one.

    ``PixelForge`` is a strict extension of ``Forge``: it overrides only the
    pinned-indicator draw and falls back to the text cell when no inline
    graphics protocol is detected. So it is the right thing to construct
    everywhere — a terminal without kitty or iTerm2 gets exactly what it got
    before, and no call site has to branch.
    """
    from tarhan.forge_pixels import PixelForge

    forge = PixelForge(stages, out, style=style, pin=pin, graphics=graphics)
    if style == "indicator" and not forge.graphics_available:
        # Two backends, as the review concluded: the raster sprite where a
        # terminal can show one, and the four-row ANSI forge everywhere else.
        # One text cell can carry motion but not shape, so a terminal without
        # inline graphics gets the taller drawing rather than a worse hint.
        return Forge(stages, out, style="compact", pin=pin)
    return forge


def _capabilities_doctor(out: cliout.Output, graphics: str = "auto") -> int:
    """Bring the tools up, and say plainly whether they came up.

    The bar counts completed checks. It is not a timer, and no path advances it
    with elapsed time — the same rule the solver display follows, for the same
    reason: a number that looks measured has to be measured.
    """
    results = []
    forge = _make_forge(out, [name for name, _, _ in DOCTOR_CHECKS],
                        style="boot", graphics=graphics)
    with forge:
        for name, detail, check in DOCTOR_CHECKS:
            forge.begin(name, detail)
            forge.tick()
            ok, said = check()
            results.append((name, ok, said))
            forge.finish(said)
        broken = [n for n, ok, _ in results if ok is False]
        if broken:
            forge.failed(f"{len(broken)} of {len(results)} checks failed: "
                         + ", ".join(broken))
        else:
            optional = sum(1 for _, ok, _ in results if ok is None)
            forge.converged(
                f"{len(results) - optional} checks passed"
                + (f", {optional} optional not installed" if optional else ""))

    if out.fmt != "table":
        out.emit([{"check": n,
                   "status": {True: "ok", False: "FAILED", None: "absent"}[ok],
                   "detail": said}
                  for n, ok, said in results],
                 ("check", "status", "detail"))
    return (cliout.EXIT_UNAVAILABLE
            if any(ok is False for _, ok, _ in results) else cliout.EXIT_OK)


#: How ``None`` survives a round trip through TOML, which has no null.
NONE_IN_LOCK = "none"


def _from_lock(value):
    """Undo the lock file's spelling of ``None``."""
    return None if value == NONE_IN_LOCK else value


def _load_device_spec(path):
    """Read a device override file. TOML or JSON, decided by extension.

    Returns a flat mapping. Nested tables are refused rather than flattened:
    a device is a flat set of scalars, and quietly accepting `[doping] Na=...`
    would put a value in the lock file under a key the solver never reads.
    """
    import json
    import tomllib

    path = Path(path)
    raw = path.read_bytes()
    if path.suffix.lower() == ".json":
        # The SAME hook the candidate loader uses. Fixing duplicate keys there
        # and not here left `{"mu_n": 1000, "mu_n": 5}` silently becoming 5 —
        # and worse, the CHANGELOG claimed both loaders refused it. A false
        # entry in a user-facing document is a worse defect than the gap it
        # described. Reported in re-review.
        from tarhan.candidate import no_duplicate_keys

        spec = json.loads(raw.decode("utf-8"),
                          object_pairs_hook=no_duplicate_keys)
    elif path.suffix.lower() == ".toml":
        spec = tomllib.loads(raw.decode("utf-8"))
    else:
        raise ValueError(f"{path.name}: expected a .toml or .json device file")
    if not isinstance(spec, dict):
        raise ValueError(f"{path.name}: expected a table of scalars at the top")
    nested = sorted(k for k, v in spec.items() if isinstance(v, (dict, list)))
    if nested:
        raise ValueError(
            f"{path.name}: a device is a flat set of scalars; {nested} is not")
    return spec


def _merge_device(defaults, overrides, source):
    """Apply overrides to a resolved device, refusing what cannot apply.

    Checked HERE rather than left to the dataclass because the two questions
    are different: the dataclass knows whether a value is physically valid,
    and only this knows whether the key means anything at all for the
    capability being run. A misspelt key would otherwise be dropped in silence
    and the run would look like it honoured a setting it never saw.
    """
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(
            f"{source}: {unknown} is not part of this capability's device. "
            f"It takes: {', '.join(sorted(defaults))}")

    merged = dict(defaults)
    for key, value in overrides.items():
        default = defaults[key]
        if isinstance(value, bool):
            raise ValueError(f"{source}: {key} is a number, not a boolean")
        if default == NONE_IN_LOCK:              # an optional field, e.g. tau_n
            if value != NONE_IN_LOCK and not isinstance(value, (int, float)):
                raise ValueError(f"{source}: {key} must be a number or "
                                 f"{NONE_IN_LOCK!r}")
        elif isinstance(default, int) and not isinstance(default, bool):
            # ny is a node count. Accepting 4.5 here would silently truncate
            # and give a mesh the lock file does not describe.
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{source}: {key} must be a whole number")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"{source}: {key} must be a number")
        merged[key] = value
    return merged


def _solve_pn1d_steady(inputs, on_iteration=None):
    """The one capability wired to `run solve` today.

    Returns (metrics, fields). Raises RuntimeError when Gummel fails to
    converge — the caller turns that into EXIT_NO_CONVERGENCE rather than a
    traceback, because a solver giving up is a legitimate outcome the caller
    has to be able to branch on.

    ``on_iteration`` is forwarded to the Gummel loop. Without it the display
    cannot draw anything while the solve runs, because the whole solve is one
    blocking call.
    """
    import dataclasses

    from tarhan.models.pn1d import PNDiode1D, solve_bias

    # Every INIT field, not a hand-listed six. The old list took Na, Nd, ni,
    # ut, len_p and len_n and silently dropped eps_s, the mobilities, the grid
    # controls and the SRH lifetimes. That was invisible while everything sat
    # at its default and became a lie the moment `--device` could change one:
    # the lock file would record the value and the solver would ignore it.
    accepted = {f.name for f in dataclasses.fields(PNDiode1D) if f.init}
    device = {k: _from_lock(v) for k, v in inputs.items() if k in accepted}
    dev = PNDiode1D(**device)
    state = solve_bias(dev, inputs["bias_v"],
                       gummel_tol=inputs.get("tol", 1e-9),
                       max_gummel=int(inputs.get("max_iter", 60)),
                       on_iteration=on_iteration)
    metrics = {"current_a_cm2": float(state["j"]),
               "gummel_iterations": int(state["gummel_iters"]),
               "grid_points": int(len(state["x_hat"])),
               # What "converged" was actually worth, recorded rather than
               # asserted: the potential step AND the terminal current's
               # change per outer pass.
               "psi_step": float(state["psi_step"]),
               "current_rel_change": (None if state["current_rel_change"] is None
                                      else float(state["current_rel_change"]))}
    fields = {"x_hat": state["x_hat"], "psi": state["psi"],
              "n_hat": state["n_hat"], "p_hat": state["p_hat"]}
    return metrics, fields


def _solve_pn2d_steady(inputs, on_iteration=None):
    """2D steady drift-diffusion on the generated rectangular mesh.

    The physics here was validated against DEVSIM at stages 2D-1 and 2D-2 long
    before this function existed. What was missing was a DEVICE: PNDiode2D
    needs points, triangles, doping and contacts, and the only things that
    could produce those were a fixture inside a test file and DEVSIM's own
    oracle mesh. :mod:`tarhan.models.diode2d_mesh` is what closed that gap, and
    it is why the lock file can record eight scalars instead of 625 nodes.

    The constants are taken from the 1D device deliberately, not from
    PNDiode2D's own defaults, so `compare runs` across the two dimensions is
    comparing the mesh and not the permittivity.
    """
    from tarhan.models import pn2d
    from tarhan.models.diode2d_mesh import RectangularDiode2D, device

    spec = RectangularDiode2D(**{k: _from_lock(v) for k, v in inputs.items()
                                 if k in _SPEC_FIELDS})
    dev = device(spec, **{k: _from_lock(inputs[k]) for k in _PN2D_CONSTANTS
                          if k in inputs})
    state = pn2d.solve_bias(dev, inputs["bias_v"],
                            gummel_tol=inputs.get("tol", 1e-9),
                            max_gummel=int(inputs.get("max_iter", 200)),
                            on_iteration=on_iteration)
    # The terminal current is integrated over the contact EDGE, so it is a
    # current per unit depth; dividing by the device height gives the same
    # quantity the 1D model reports. Established by measurement against the
    # validated 1D solver (ratio 1.000000 at 0.3 V and 0.4 V), not asserted
    # from a formula — physics_verify is unavailable this session.
    residual = state["coupled_residual"]
    metrics = {"psi_step": float(state["psi_step"]),
               "current_rel_change": (None if state["current_rel_change"] is None
                                      else float(state["current_rel_change"])),
               "coupled_poisson_residual": float(residual["poisson"]),
               "current_a_cm2": float(state["i"]) / spec.height,
               "terminal_current_a_per_cm": float(state["i"]),
               "electron_current_a_per_cm": float(state["i_n"]),
               "hole_current_a_per_cm": float(state["i_p"]),
               "gummel_iterations": int(state["gummel_iters"]),
               "grid_points": int(dev.mesh.n_nodes)}
    fields = {"psi": np.asarray(state["psi"]),
              "n_hat": np.asarray(state["n_hat"]),
              "p_hat": np.asarray(state["p_hat"])}
    return metrics, fields


def _dataclass_defaults(cls) -> dict:
    """Every field that changes the answer, with its default filled in.

    Read from the dataclass rather than retyped, so a new field cannot be added
    to a device and quietly stay out of the lock file — the defect this
    replaced, where a run recorded only its bias and could not be reproduced
    from its own record.
    """
    import dataclasses

    instance = cls()
    out = {}
    for f in dataclasses.fields(instance):
        if not f.init:                      # C0, delta, L_D, mesh are derived
            continue
        value = getattr(instance, f.name)
        out[f.name] = "none" if value is None else value
    return out


def _resolved_device_1d() -> dict:
    from tarhan.models.pn1d import PNDiode1D

    return _dataclass_defaults(PNDiode1D)


def _resolved_device_2d() -> dict:
    """The mesh scalars, plus the 1D device's physical constants.

    The mesh is an input, and a lock file holding 625 coordinates would be a
    record nobody reads and a hash nobody can reproduce by hand. The generator
    is narrow enough that these scalars determine it exactly, which is what
    makes recording them equivalent to recording the mesh.
    """
    from tarhan.models.diode2d_mesh import RectangularDiode2D
    from tarhan.models.pn1d import PNDiode1D

    one = PNDiode1D()
    out = _dataclass_defaults(RectangularDiode2D)
    for name in _PN2D_CONSTANTS:
        out[name] = getattr(one, name)
    return out


_SPEC_FIELDS = ("len_p", "len_n", "height", "h0", "gamma", "ny", "Na", "Nd")

#: Constants PNDiode2D takes directly, sourced from the 1D device so that a
#: comparison across the two dimensions measures the mesh and not the physics.
_PN2D_CONSTANTS = ("ni", "ut", "eps_s", "q", "mu_n", "mu_p")


@dataclass(frozen=True)
class Runner:
    """What `run solve` needs to know to run one capability end to end."""

    solve: Callable
    device: Callable[[], dict]
    method: str
    default_max_iter: int
    describe: str


#: capability id -> its runner. A capability absent from this map is not
#: runnable from the CLI even if the registry calls it validated: being proven
#: and being wired up are different facts and the error says which.
RUNNERS = {
    "semiconductor.pn.drift-diffusion.1d.steady": Runner(
        solve=_solve_pn1d_steady, device=_resolved_device_1d,
        method="gummel", default_max_iter=60,
        describe="PNDiode1D defaults"),
    "semiconductor.pn.drift-diffusion.2d.steady": Runner(
        solve=_solve_pn2d_steady, device=_resolved_device_2d,
        method="gummel-newton", default_max_iter=200,
        describe="RectangularDiode2D defaults, PNDiode1D constants"),
}


def _load_candidates_or_none(out: cliout.Output, path):
    from tarhan import candidate as cand

    try:
        return cand.load_candidates(path)
    except (cand.CandidateError, OSError, ValueError) as exc:
        out.error(str(exc))
        return None


def _candidate_list(out: cliout.Output, args) -> int:
    from tarhan import candidate as cand

    candidates = _load_candidates_or_none(out, args.source)
    if candidates is None:
        return cliout.EXIT_INPUT

    rows = []
    for item in candidates:
        row = {"identifier": item.identifier,
               "composition": item.composition,
               "properties": len(item.properties)}
        for capability_id in sorted(cand.MATERIAL_PARAMETERS):
            fit = cand.applicability(item, capability_id)
            # The MISSING names, not a yes/no: they are the measurements
            # somebody would have to make, which is the actionable part.
            row[capability_id] = ("usable" if fit.usable
                                  else "missing " + ",".join(fit.missing))
        rows.append(row)
    out.emit(rows, tuple(rows[0]))
    return cliout.EXIT_OK


def _candidate_show(out: cliout.Output, args) -> int:
    candidates = _load_candidates_or_none(out, args.source)
    if candidates is None:
        return cliout.EXIT_INPUT

    match = next((c for c in candidates if c.identifier == args.candidate_id),
                 None)
    if match is None:
        out.error(f"no such candidate: {args.candidate_id}")
        out.note("run `tarhan candidate list --from <file>` to see the ids")
        return cliout.EXIT_INPUT

    rows = []
    for name, prop in sorted(match.properties.items()):
        low, high = prop.interval
        rows.append({"property": name, "value": prop.value, "unit": prop.unit,
                     "basis": prop.basis,
                     "uncertainty": ("" if prop.uncertainty is None
                                     else prop.uncertainty),
                     "low": low, "high": high, "source": prop.source,
                     # Stored, and previously not shown — so the CLI could not
                     # display the very range that now decides whether a solve
                     # is allowed. Reported in review.
                     "valid_range": "; ".join(
                         f"{k} in [{v[0]:g}, {v[1]:g}]"
                         for k, v in sorted(prop.valid_range.items()))})
    # The identity went out as stderr NOTES, so `--format json --quiet` lost
    # it entirely and the "whole record" claim held only for humans reading a
    # terminal. Reported in re-review. It is rows now, like everything else.
    identity = [{"property": f"@{label}", "value": value, "unit": "",
                 "basis": "", "uncertainty": "", "low": "", "high": "",
                 "source": "", "valid_range": ""}
                for label, value in (
                    ("identifier", match.identifier),
                    ("composition", match.composition),
                    ("structure", match.structure),
                    ("dimensionality", match.dimensionality),
                    ("notes", match.notes),
                    ("fingerprint", _candidate_fingerprint(match)))
                if value]
    out.emit(identity + rows,
             ("property", "value", "unit", "basis", "uncertainty",
              "low", "high", "source", "valid_range"))
    return cliout.EXIT_OK


def _candidate_fingerprint(match):
    from tarhan import candidate as cand

    return cand.fingerprint(match)


def _candidate_screen(out: cliout.Output, args) -> int:
    """Hard thresholds, with the undecided cases kept as undecided."""
    from tarhan import candidate as cand

    candidates = _load_candidates_or_none(out, args.source)
    if candidates is None:
        return cliout.EXIT_INPUT
    if not args.require:
        out.error("a screen needs at least one --require NAME>=VALUE")
        return cliout.EXIT_INPUT
    try:
        thresholds = [cand.parse_threshold(text) for text in args.require]
    except cand.CandidateError as exc:
        out.error(str(exc))
        return cliout.EXIT_INPUT

    conditions = {}
    for item in args.at:
        name, _, raw = item.partition("=")
        name = name.strip()
        if name not in cand.CONDITIONS:
            out.error(f"--at {name}: not a condition a run reports. "
                      f"Known: {', '.join(cand.CONDITIONS)}")
            return cliout.EXIT_INPUT
        try:
            value = float(raw)
        except ValueError:
            out.error(f"--at {item}: {raw!r} is not a number")
            return cliout.EXIT_INPUT
        if not math.isfinite(value):
            # `--at bias_v=nan` was accepted and every range comparison with
            # it is False, so an out-of-range property could screen clean.
            # Reported in re-review.
            out.error(f"--at {item}: {raw!r} is not a finite condition")
            return cliout.EXIT_INPUT
        if name in conditions:
            # The last occurrence used to win silently, so REVERSING two
            # otherwise identical arguments changed the verdict: 0.3 then 0.9
            # gave `undecided`, 0.9 then 0.3 gave `pass`, both exit 0. A
            # caller could not tell from stdout or the status that a value had
            # been discarded. Issue #2.
            #
            # `--vary` already refuses a repeated axis for the same reason.
            # This is the fifth time this cycle that a guard existed on one of
            # two sibling paths and not the other.
            out.error(f"--at {name}: given twice ({conditions[name]:g} and "
                      f"{value:g}); the second would silently replace the "
                      "first and the verdict would depend on argument order")
            return cliout.EXIT_INPUT
        conditions[name] = value

    # A note was not enough: `--quiet --format json` dropped it and a ranged
    # property came back a clean PASS. A missing condition is missing
    # INFORMATION, so the verdict is `undecided` — the same answer this module
    # already gives for a property nobody recorded. Reported in re-review.
    report = cand.screen(candidates, thresholds, conditions or None,
                         require_conditions=True)
    if not conditions and any(p.valid_range for c in candidates
                              for p in c.properties.values()):
        out.note("some properties state a validity range and no --at was "
                 "given, so they are undecided rather than assumed valid")
    rows = []
    for result in report["results"]:
        reasons = [j.detail for j in result["judgements"]
                   if j.verdict != "pass"]
        rows.append({"identifier": result["identifier"],
                     "verdict": result["verdict"],
                     "why": "; ".join(reasons)})
    out.emit(rows, ("identifier", "verdict", "why"))

    counts = {v: sum(1 for r in rows if r["verdict"] == v)
              for v in cand.VERDICTS}
    # Every candidate is reported, never only the survivors: a screen that
    # returns a shortlist alone hides its own selectivity, and hides which
    # candidates were dropped for want of a MEASUREMENT rather than for being
    # unsuitable.
    out.note(f"{counts['pass']} pass, {counts['fail']} fail, "
             f"{counts['undecided']} undecided out of {len(rows)}")
    if counts["undecided"]:
        out.note("undecided is not a soft fail: the uncertainty straddles the "
                 "bound, so neither answer is supportable without a better "
                 "measurement")
    return cliout.EXIT_OK


#: The axis that is not part of any device: the bias is the terminal condition,
#: not a property of the thing being biased.
BIAS_AXIS = "bias_v"

#: A sweep builds its whole Cartesian product before solving anything, so an
#: unbounded grid exhausts memory before producing a single result. The number
#: is arbitrary but the bound is not: it is chosen to be far above any sweep
#: worth reading in a terminal and far below anything that hurts.
MAX_SWEEP_POINTS = 10_000

#: Terms a sweep may never vary — see :func:`_parse_vary`.
SOLVER_TERMS = ("tol", "max_iter", "method")


def _solver_numbers_or_status(out: cliout.Output, args, default_max_iter):
    """Reject a solver contract that cannot mean anything, at the input gate.

    ``--tol nan`` used to reach the solver and come back as exit 4, "did not
    converge" — which blames the physics for a typo. A tolerance that is NaN,
    infinite or non-positive is not a hard problem, it is a bad argument, and
    the exit code has to say which. Reported in review.
    """
    tol = float(args.tol)
    if not math.isfinite(tol) or tol <= 0.0:
        out.error(f"--tol {args.tol}: must be finite and greater than zero")
        return None
    bias = float(args.bias)
    if not math.isfinite(bias):
        out.error(f"--bias {args.bias}: must be a finite voltage")
        return None
    max_iter = default_max_iter if args.max_iter is None else int(args.max_iter)
    if max_iter <= 0:
        out.error(f"--max-iter {max_iter}: must be at least one iteration")
        return None
    return {"tol": tol, "bias": bias, "max_iter": max_iter}


def _off_reference_device(runner, inputs) -> Tuple[str, ...]:
    """Which device fields differ from the ones the evidence was collected on.

    Computed from the RESOLVED device rather than from which flags were
    passed. The first version asked "was --device or --candidate given?", was
    added to `run solve`, and was not added to `run sweep` — so a swept device
    override came back plainly `converged` while its own provenance said
    "overridden". Reported in re-review. Asking the resolved values also
    catches `--vary mu_n=...`, which no flag check would have.
    """
    reference = runner.device()
    drifted = []
    for name, default in sorted(reference.items()):
        if name not in inputs:
            continue
        if inputs[name] != default:
            drifted.append(f"{name}={inputs[name]!r} differs from the "
                           f"reference {default!r}")
    return tuple(drifted)


def _envelope_breaches(cap, runner, inputs) -> Tuple[str, ...]:
    """Everything that puts a run outside the evidence: bias AND device."""
    out = list(cap.outside_envelope(inputs))
    drifted = _off_reference_device(runner, inputs)
    if drifted:
        out.append("the device differs from the reference device the evidence "
                   "was collected on (" + "; ".join(drifted) + ")")
    return tuple(out)


def _device_fingerprint(inputs) -> str:
    """A hash of the resolved device, so an artifact names WHAT it solved.

    Provenance said "overridden" without saying overridden to what, so two
    different materials produced records that read identically.
    """
    import hashlib

    payload = json.dumps({k: v for k, v in sorted(inputs.items())
                          if k != "bias_v"},
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


#: Numbers a run reports ABOUT ITSELF rather than about the physics. They are
#: not claims that need validating, so they are not counted as uncovered
#: metrics — but everything else the artifact publishes is.
_DIAGNOSTIC_METRICS = frozenset({
    "gummel_iterations", "grid_points", "psi_step", "current_rel_change",
    "coupled_poisson_residual",
})


def _statuses(cap, inputs, outside, metrics):
    """The four status fields, computed ONCE for solve and sweep alike.

    `run solve` and `run sweep` computed these separately, so the rename to
    `potential-step-converged` landed in one and not the other and a sweep
    kept publishing the old word. That is the third time in this review cycle
    that these two paths have disagreed — provenance, the reference-device
    check, and now the status itself. One producer.

    `solver_status` says only what is tested: the potential STEP settled. It
    used to say "converged" beside a current_rel_change of 3.49e-4.
    `validation_status` is metric-aware, because `inside` described
    bias-envelope membership while the artifact published metrics that
    envelope says nothing about.
    """
    change = metrics.get("current_rel_change")
    report = dict(cap.coverage_report(inputs))
    # A metric the artifact PUBLISHES but no coverage record mentions is
    # `unverified`, and saying so is the only way `unverified` can ever be
    # reached. A 2D 0.3 V run publishes nine metrics while the registry
    # covers two, and it was reporting `fully-covered`. Reported in review.
    for name in metrics:
        if name in _DIAGNOSTIC_METRICS:
            continue
        report.setdefault(name, "unverified")
    kinds = set(report.values())
    if outside:
        validation_status = "outside-validated-range"
    elif not kinds or kinds == {"unverified"}:
        validation_status = "uncovered"
    elif kinds <= {"measured-point"}:
        validation_status = "fully-covered"
    else:
        validation_status = "partially-covered"

    solver_status = "potential-step-converged"
    return {"solver_status": solver_status,
            "validation_status": validation_status,
            "metric_coverage": "; ".join(f"{m}={k}"
                                         for m, k in sorted(report.items())),
            "current_convergence": ("unassessed" if change is None
                                    else f"measured:{change:.3e}"),
            "run_status": (solver_status if not outside
                           else f"{solver_status}-outside-validated-range")}


def _provenance(cap, runner, *, scenario, outside, inputs, solver_status,
                validation_status, metric_coverage,
                current_convergence="unassessed",
                candidate_id=None, device_file=None):
    """The evidence fields EVERY run records, wherever it was launched from.

    `run sweep` built its own three-key dict, so a sweep point silently
    dropped validation_envelope, uncertainty_treatment, the device file and
    the candidate — and because a sweep point and a solve of the same problem
    share a run id, sweeping over a bias you had already solved OVERWROTE the
    richer record with the poorer one. Reported in re-review. One producer, so
    the two cannot disagree about what a result has to carry.
    """
    # Derived from the RESOLVED device, like the envelope check, not from
    # which flags were passed. `--vary mu_n=700` produced an artifact saying
    # `device: "...defaults"` and `validation_envelope: "mu_n=700 differs from
    # reference 1350.0"` at the same time — one field describing the run and
    # the other describing the command line. Reported in review, and the same
    # flags-versus-resolved-inputs split that was fixed in the envelope check
    # and not here.
    drift = _off_reference_device(runner, inputs)
    return {"model": cap.source,
            "device": (f"{runner.describe}, overridden: "
                       + "; ".join(drift) if drift else runner.describe),
            "candidate": candidate_id or "none",
            "device_file": device_file or "none",
            # The screen uses each property's spread; the SOLVE takes the
            # nominal value alone. Defensible choice, indefensible silence.
            "uncertainty_treatment": "nominal-only",
            "solver_status": solver_status,
            "current_convergence": current_convergence,
            "validation_status": validation_status,
            "validation_envelope": ("inside" if not outside
                                    else "; ".join(outside)),
            # The evidence claim, as data. `envelope_basis` was free text in
            # the registry with no machine link to anything, and none of it
            # reached the artifact — so a run could not answer "which evidence
            # profile said inside?" after the fact. Reported in re-review.
            "validation_profile": cap.validation_profile(),
            "validation_basis": cap.envelope_basis,
            # Per metric, because they are not covered at the same biases:
            # this device's potential is checked at 0.0 and 0.30 V while its
            # current is checked at 0.20, 0.30 and 0.40 V.
            "metric_coverage": metric_coverage,
            "device_fingerprint": _device_fingerprint(inputs),
            "scenario": scenario}


def _recorded_command() -> str:
    """The command actually typed, not a family name.

    The manifest recorded `tarhan run solve <capability>` and dropped the
    bias, the tolerance, the candidate and the device file — so a run could
    not be re-issued from its own record, which is the one thing the field is
    for. Reported in review. Reconstructed from argv with shlex.join so the
    result is a line that can be pasted back.
    """
    import shlex

    return "tarhan " + shlex.join(sys.argv[1:])


def _runnable_or_status(out: cliout.Output, capability_id):
    """Resolve a capability to run, or say why it cannot be.

    Returns ``(capability, None)`` or ``(None, exit_code)``. Shared by `solve`
    and `sweep` so the two cannot drift into disagreeing about what is
    runnable — the failure that would show up as a sweep happily running a
    blocked capability that `solve` refuses.
    """
    try:
        cap = get(capability_id)
    except CapabilityNotFound:
        out.error(f"no such capability: {capability_id}")
        out.note("run `tarhan capabilities list` to see the ids that exist")
        return None, cliout.EXIT_INPUT

    if not cap.runnable:
        out.error(f"{cap.id} is {cap.status}; refusing to solve it")
        out.note(cap.reason)
        if cap.needs:
            out.note(f"needs: {cap.needs}")
        return None, cliout.EXIT_UNAVAILABLE
    if cap.id not in RUNNERS:
        out.error(f"{cap.id} is {cap.status} but is not wired to `run solve`")
        out.note("the physics is proven; the command surface for it is not "
                 "built. These are different things and this is the second.")
        return None, cliout.EXIT_UNAVAILABLE
    return cap, None


def _run_solve(out: cliout.Output, args) -> int:
    """Solve, then leave a directory that can answer for the result."""
    from tarhan import artifact

    cap, refusal = _runnable_or_status(out, args.capability_id)
    if cap is None:
        return refusal

    # The RESOLVED problem, not just what was typed. Recording only the bias
    # left input.lock.toml locking nothing — a run could not be reproduced from
    # its own record, and two devices differing in doping shared a directory.
    # Reported in review against the published version.
    runner = RUNNERS[cap.id]
    device_inputs = runner.device()
    candidate_terms = {}
    candidate_snapshot = None
    material = set()
    if bool(args.candidate) != bool(args.candidate_id):
        # A --candidate-id with no file was ACCEPTED and the run went ahead on
        # default material while provenance recorded the named candidate: a
        # falsified scientific record, produced silently. Reported in review
        # after a real run wrote {"candidate": "GHOST", "device": "PNDiode1D
        # defaults"}. They are one input and must arrive together.
        missing = "--candidate" if args.candidate_id else "--candidate-id"
        out.error(f"{missing} is required alongside the other; a candidate id "
                  "without a file would record a material that was never used")
        return cliout.EXIT_INPUT

    if args.candidate:
        # The join that stops a candidate from being provenance-less JSON: a
        # material is real here only if a validated model can be run on it.
        from tarhan import candidate as cand

        candidates = _load_candidates_or_none(out, args.candidate)
        if candidates is None:
            return cliout.EXIT_INPUT
        match = next((c for c in candidates
                      if c.identifier == args.candidate_id), None)
        if match is None:
            out.error(f"no such candidate: {args.candidate_id}")
            out.note("run `tarhan candidate list --from <file>` to see the ids")
            return cliout.EXIT_INPUT
        try:
            device_inputs = _merge_device(
                device_inputs, cand.device_overrides(match, cap.id),
                f"candidate {match.identifier}")
        except cand.CandidateError as exc:
            out.error(str(exc))
            return cliout.EXIT_INPUT
        # The material is part of the PROBLEM, not just of its narration. Two
        # candidates whose four numbers coincide were landing in one directory
        # and the second overwrote the first's provenance. Both the identity
        # and a hash of the whole evidence record go into the inputs, so a
        # re-measured candidate under the same id is also a different run.
        candidate_terms = {"candidate": match.identifier,
                           "candidate_fingerprint": cand.fingerprint(match)}
        candidate_snapshot = cand.snapshot(match)
        # A range this run puts the material outside of is not a warning to be
        # scrolled past: the candidate's own file says its numbers do not
        # describe the material under these conditions, so the solve would be
        # of something nobody characterised. Reported in review as stored and
        # never consulted.
        breaches = cand.out_of_range(match, {"bias_v": float(args.bias)})
        if breaches:
            out.error(f"{match.identifier} is outside its stated validity:")
            for breach in breaches:
                out.error(f"  {breach}")
            out.note("the candidate file declares this range; solving outside "
                     "it would report numbers the material was never "
                     "characterised for")
            return cliout.EXIT_INPUT
        material = set(cand.MATERIAL_PARAMETERS[cap.id])
    if args.device:
        try:
            overrides = _load_device_spec(args.device)
            # A device file applied AFTER a candidate could rewrite the very
            # material values the candidate supplied, while provenance went on
            # saying that candidate had been solved. Two inputs disagreeing
            # about one number is not something to resolve by order of
            # application. Reported in review.
            clash = sorted(material & set(overrides))
            if clash:
                out.error(
                    f"{Path(args.device).name} sets {', '.join(clash)}, which "
                    f"{args.candidate_id} also supplies. Solving would record "
                    "that candidate while using another material's numbers")
                out.note("drop the clashing keys, or run without --candidate")
                return cliout.EXIT_INPUT
            device_inputs = _merge_device(device_inputs, overrides,
                                          Path(args.device).name)
        except (ValueError, OSError) as exc:
            out.error(str(exc))
            return cliout.EXIT_INPUT
    inputs = {"bias_v": float(args.bias), **device_inputs, **candidate_terms}
    numbers = _solver_numbers_or_status(out, args, runner.default_max_iter)
    if numbers is None:
        return cliout.EXIT_INPUT
    solver = {"method": runner.method, "tol": numbers["tol"],
              "max_iter": numbers["max_iter"]}
    forge = _make_forge(out, ["SOLVE"], style="indicator",
                        graphics=args.graphics, pin="auto")
    try:
        with forge:
            forge.begin("SOLVE", f"{cap.family} at {args.bias} V")

            def report(index, total):
                # The solve is one blocking call, so this is the ONLY moment
                # the display can advance. Measured before it existed: zero
                # bytes were written between begin() and finish(), which meant
                # a long solve showed nothing at all and the indicator only
                # appeared after the work was already over.
                forge.tick(within=(index + 1) / max(total, 1),
                           detail=f"gummel iteration {index + 1}")

            metrics, fields = runner.solve({**inputs, **solver},
                                           on_iteration=report)
            forge.finish(f"{metrics['gummel_iterations']} iterations")
            # "converged" on the terminal was the same overclaim the
            # artifact field had: only the potential step is tested.
            forge.finish(f"potential step settled; current "
                         f"{metrics['current_a_cm2']:.4e} A/cm^2")
    except ValueError as exc:
        # A device that cannot exist: a negative length, a shrinking mesh step,
        # a contact with no nodes. The dataclasses validate the physics and
        # this turns their refusal into bad INPUT rather than an internal
        # error, because the user typed it and can fix it.
        out.error(f"the device cannot be built: {exc}")
        return cliout.EXIT_INPUT
    except RuntimeError as exc:
        # The call site EXIT_NO_CONVERGENCE was defined for. A solver that
        # gives up is not a crash and must not be reported as one.
        out.error(f"solver did not converge: {exc}")
        out.note("no artifact was written; a partial state would claim more "
                 "than the run earned")
        return cliout.EXIT_NO_CONVERGENCE

    # A result outside the evidence is not a failure and must not be reported
    # as one — but it must not be reported as plain `converged` either, which
    # is what a 0.2 V 2D solve did while the registry's own prose said that
    # current does not converge. Reported in re-review.
    outside = _envelope_breaches(cap, runner, inputs)
    # TWO contracts, deliberately separate. `solver_status` is whether the
    # solver settled its own coupled system; `validation_status` is whether
    # the answer is covered by evidence. Collapsing them into one word was
    # how a run whose current was still moving reported the same "converged"
    # as one that had settled to 1e-10.
    # `solver_status` is deliberately still "the potential settled", because
    # that is the only convergence claim currently backed by a defensible
    # threshold. The terminal current's per-iteration change is MEASURED and
    # recorded beside it. Turning it into a verdict needs a scale nobody has
    # pinned down yet — see test_pn1d_grid_robustness.py for what IS
    # established — and asserting one would be the failure this keeps finding.
    statuses = _statuses(cap, inputs, outside, metrics)
    solver_status = statuses["solver_status"]
    validation_status = statuses["validation_status"]
    current_convergence = statuses["current_convergence"]
    run_status = statuses["run_status"]
    if outside:
        for line in outside:
            out.note(f"OUTSIDE THE VALIDATED RANGE: {line}")
        out.note("the potential step settled and the artifact is written, but "
                 "nothing has established that this result is right")

    path = artifact.write_run(
        args.output, capability=cap.id, capability_status=cap.status,
        inputs=inputs, solver=solver, metrics=metrics,
        provenance=_provenance(
            cap, runner, scenario=f"single bias {args.bias} V",
            outside=outside, inputs=inputs, solver_status=solver_status,
            validation_status=validation_status,
            current_convergence=current_convergence,
            metric_coverage=statuses["metric_coverage"],
            candidate_id=args.candidate_id,
            device_file=Path(args.device).name if args.device else None),
        status=run_status, command=_recorded_command(),
        version=__version__, fields_data=fields,
        candidate_snapshot=candidate_snapshot,
        report=f"# {cap.id}\n\nbias {args.bias} V\n")

    rows = [{"run_id": path.name, "capability": cap.id,
             "status": run_status, **metrics}]
    out.emit(rows, tuple(rows[0]))
    out.note(f"artifact written to {path}")
    return cliout.EXIT_OK


def _parse_vary(specs, defaults, out):
    """``name=v1,v2,...`` into an ordered mapping of axis -> values.

    Returns None after reporting the error, so the caller exits 2. Every
    refusal here is about keeping the sweep INTERPRETABLE: an axis nobody can
    name, a value that is not a number, or a single-valued axis that would
    quietly make the sweep a solve.
    """
    axes = {}
    for spec in specs:
        if "=" not in spec:
            out.error(f"--vary {spec}: expected name=value,value,...")
            return None
        name, _, raw = spec.partition("=")
        name = name.strip()
        if name in axes:
            out.error(f"--vary {name}: given twice; put every value in one list")
            return None
        if name in SOLVER_TERMS:
            # The whole point of a sweep is that one thing changes. Varying the
            # tolerance would produce rows that `compare runs` itself refuses
            # to put side by side — §5.3 — so the table would be a ranking
            # across a changed contract, which is the error this prevents.
            out.error(f"--vary {name}: the solver contract is fixed across a "
                      "sweep, or the rows would not be comparable")
            out.note("run separate sweeps and compare them deliberately")
            return None
        if name != BIAS_AXIS and name not in defaults:
            out.error(f"--vary {name}: not part of this capability's device. "
                      f"It takes: {BIAS_AXIS}, {', '.join(sorted(defaults))}")
            return None

        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token) if _looks_integral(token) else float(token)
            except ValueError:
                out.error(f"--vary {name}: {token!r} is not a number")
                return None
            if not math.isfinite(value):
                # float("nan") parses happily. The point then failed as an
                # "invalid-device" — exit 4, a SOLVER failure — and the JSON
                # carried a bare NaN literal, which is not valid JSON and
                # breaks the consumer this format exists for. Reported in
                # re-review. It is bad input, and it is caught before any
                # solve.
                out.error(f"--vary {name}: {token!r} is not a finite number")
                return None
            if value in values:
                out.error(f"--vary {name}: {token} appears twice")
                return None
            values.append(value)
        if len(values) < 2:
            out.error(f"--vary {name}: a sweep needs at least two values; "
                      "one value is a solve")
            return None
        axes[name] = values
    return axes


def _looks_integral(token: str) -> bool:
    """Whether a written value is a whole number, by its SPELLING.

    ``ny=6`` must stay an int because the device refuses a fractional node
    count, while ``6.0`` written deliberately is a float. Parsing by spelling
    rather than by value keeps `--vary ny=4,6` working without letting
    `--vary Na=1e16` become an int of a size nobody meant.
    """
    return token.lstrip("+-").isdigit()


def _sweep_points(axes):
    """Every combination, in the order the axes were given."""
    import itertools

    names = list(axes)
    for combination in itertools.product(*(axes[n] for n in names)):
        yield dict(zip(names, combination))


def _run_sweep(out: cliout.Output, args) -> int:
    """Solve a family of devices under ONE solver contract, and table them.

    `run solve` answers "what does this device do". This answers the question
    a sweep is actually for — "which of these devices does it best" — and the
    difference that makes it trustworthy is the fixed contract: the tolerance
    and iteration budget are identical for every row, so a column can be read
    down. `compare runs` refuses exactly the comparison this makes safe, and
    that is the same rule seen from the other side: it refuses because it
    cannot know the contract held, and here it is held by construction.

    Each point writes a NORMAL run artifact through the same path `run solve`
    uses. There is no such thing as a sweep-flavoured result: a row is a run,
    reopenable with `run show` and comparable with `compare runs`.
    """
    from tarhan import artifact

    cap, refusal = _runnable_or_status(out, args.capability_id)
    if cap is None:
        return refusal

    runner = RUNNERS[cap.id]
    device_inputs = runner.device()
    if args.device:
        try:
            device_inputs = _merge_device(device_inputs,
                                          _load_device_spec(args.device),
                                          Path(args.device).name)
        except (ValueError, OSError) as exc:
            out.error(str(exc))
            return cliout.EXIT_INPUT

    if not args.vary:
        # "one value is a solve" is already enforced per axis; a sweep with NO
        # axis slipped past it and produced a cheerful one-point table, which
        # contradicts the command's own contract. Reported in review.
        out.error("a sweep needs at least one --vary NAME=V1,V2,...")
        out.note("with nothing varying, `run solve` is the command you want")
        return cliout.EXIT_INPUT
    axes = _parse_vary(args.vary, device_inputs, out)
    if axes is None:
        return cliout.EXIT_INPUT

    total = 1
    for values in axes.values():
        total *= len(values)
    if total > MAX_SWEEP_POINTS:
        # The product was materialised into a list before any solve began, so
        # a few generous axes could exhaust memory before the first result.
        # Refusing with the number is more useful than dying without one.
        out.error(f"{total} points is more than the {MAX_SWEEP_POINTS} this "
                  "command will run in one go")
        out.note("narrow an axis, or split the grid across several sweeps")
        return cliout.EXIT_INPUT
    points = list(_sweep_points(axes))
    numbers = _solver_numbers_or_status(out, args, runner.default_max_iter)
    if numbers is None:
        return cliout.EXIT_INPUT
    solver = {"method": runner.method, "tol": numbers["tol"],
              "max_iter": numbers["max_iter"]}

    # One snapshot for the whole command. The policy this makes explicit,
    # which issue #4 asked for: every point in ONE sweep belongs to the build
    # observed when the sweep started. A source change mid-sweep does not
    # silently split the batch across two builds — which is what happened
    # before, since each point re-ran `git status` independently.
    from tarhan.artifact import environment

    build_snapshot = environment()

    rows, failures = [], 0
    forge = _make_forge(out, ["SWEEP"], style="indicator",
                        graphics=args.graphics, pin="auto")
    with forge:
        forge.begin("SWEEP", f"{len(points)} points of {cap.family}")
        for index, point in enumerate(points):
            inputs = dict(device_inputs)
            bias = point.get(BIAS_AXIS, float(args.bias))
            inputs.update({k: v for k, v in point.items() if k != BIAS_AXIS})
            inputs["bias_v"] = float(bias)
            forge.tick(within=(index + 1) / len(points),
                       detail=f"point {index + 1}/{len(points)}")

            breaches = _envelope_breaches(cap, runner, inputs)
            row = {**point, "status": "", "run_id": ""}
            try:
                metrics, fields = runner.solve({**inputs, **solver})
            except ValueError as exc:
                # A device in the sweep that cannot exist. The sweep does not
                # stop: the other points are still real results, and a table
                # with a named gap says more than no table at all.
                row["status"] = "invalid-device"
                row["detail"] = str(exc)
                failures += 1
                rows.append(row)
                continue
            except RuntimeError:
                row["status"] = "not-converged"
                failures += 1
                rows.append(row)
                continue

            point_statuses = _statuses(cap, inputs, breaches, metrics)
            row["status"] = point_statuses["run_status"]
            path = artifact.write_run(
                args.output, capability=cap.id, capability_status=cap.status,
                inputs=inputs, solver=solver, metrics=metrics,
                provenance=_provenance(
                    cap, runner, scenario=f"sweep point {point}",
                    outside=breaches, inputs=inputs,
                    solver_status=point_statuses["solver_status"],
                    validation_status=point_statuses["validation_status"],
                    current_convergence=point_statuses["current_convergence"],
                    metric_coverage=point_statuses["metric_coverage"],
                    device_file=(Path(args.device).name if args.device
                                 else None)),
                status=point_statuses["run_status"],
                command=_recorded_command(),
                version=__version__, fields_data=fields,
                build_snapshot=build_snapshot,
                report=f"# {cap.id}\n\nsweep point {point}\n")
            row["run_id"] = path.name
            row.update(metrics)
            rows.append(row)
        forge.finish(f"{len(points)} points")
        # converged()/failed() is what marks the display TERMINAL. Without it
        # __exit__ force-renders again and the summary line prints twice.
        settled = f"{len(points) - failures}/{len(points)} converged"
        if failures:
            forge.failed(settled)
        else:
            forge.converged(settled)

    columns = list(axes) + ["status", "run_id"]
    for row in rows:                       # metrics, in first-seen order
        for key in row:
            if key not in columns:
                columns.append(key)
    out.emit(rows, tuple(columns))
    if failures:
        out.error(f"{failures} of {len(points)} points did not produce a result")
        out.note("the table above names each one; no artifact was written for "
                 "them, because a partial state would claim more than the "
                 "point earned")
        return cliout.EXIT_NO_CONVERGENCE
    out.note(f"{len(points)} runs written to {args.output}, one solver "
             f"contract: {solver['method']}, tol {solver['tol']:g}, "
             f"max_iter {solver['max_iter']}")
    return cliout.EXIT_OK


def _run_show(out: cliout.Output, args) -> int:
    from tarhan import artifact

    path = Path(args.output) / args.run_id
    try:
        run = artifact.read_run(path)
    except (artifact.ArtifactError, OSError) as exc:
        out.error(str(exc))
        return cliout.EXIT_INPUT

    manifest = run["manifest"]
    if args.full:
        # Without this the command answered "what were the numbers" but not
        # "what produced them" — so a run could not be reopened from the CLI
        # to see which candidate and which inputs were behind it, which is the
        # whole point of writing the directory. Reported in review.
        rows = [{"section": "input", "key": k, "value": v}
                for k, v in sorted(run["inputs"].items())]
        rows += [{"section": "provenance", "key": k, "value": v}
                 for k, v in sorted(run["provenance"].items())]
        rows += [{"section": "integrity", "key": "state",
                  "value": run.get("integrity", "unknown")},
                 {"section": "integrity", "key": "schema_version",
                  "value": run.get("schema_version", 1)},
                 {"section": "integrity", "key": "command",
                  "value": manifest.get("command", "")}]
        rows += [{"section": "metric", "key": k, "value": v}
                 for k, v in sorted(run["metrics"].items())]
        out.emit(rows, ("section", "key", "value"))
        return cliout.EXIT_OK
    if run.get("integrity") == "unverified-legacy":
        # A directory written before checksums existed. Reading it in silence
        # made it indistinguishable from a verified one — reported in review.
        out.note("schema v%s: written before checksums existed, so nothing has "
                 "verified this directory's contents" % run.get("schema_version"))
    if out.fmt == "table":
        lines = [f"{k + ':':<20}{v}" for k, v in manifest.items()
                 if not isinstance(v, dict)]
        lines += [f"{'solver.' + k + ':':<20}{v}"
                  for k, v in manifest["solver"].items()]
        lines += [f"{'metric.' + k + ':':<20}{v}"
                  for k, v in run["metrics"].items()]
        out.detail("\n".join(lines) + "\n")
    else:
        out.emit([{**manifest, **run["metrics"]}],
                 tuple({**manifest, **run["metrics"]}))
    return cliout.EXIT_OK


#: The comparability contract, roadmap §5.3. Each entry is (label, extractor).
#: A difference in ANY of these makes a ranking meaningless, so `compare`
#: reports the difference instead of inventing one.
_CONTRACT = (
    ("capability", lambda r: r["manifest"]["capability"]),
    ("solver", lambda r: r["manifest"]["solver"]),
    ("inputs", lambda r: r["inputs"]),
    # The code is a term of the contract too, and it was missing — reported in
    # review. A difference here is precisely the regression case ("did my
    # change move this number?"), so it is not fatal the way a changed solver
    # is; it just must not be silent, because "same inputs, different answer"
    # is exactly how a code change gets read as a physical effect.
    ("build", lambda r: r["manifest"].get("code_id", "")),
)

#: The one term `--allow-build-diff` waives. The others are not waivable: no
#: flag makes two different solver tolerances comparable.
_WAIVABLE = "build"


def _incomparable(left, right, allow_build_diff: bool = False):
    """Which contract terms differ. Empty means the two can be compared."""
    return [label for label, get_term in _CONTRACT
            if not (allow_build_diff and label == _WAIVABLE)
            and get_term(left) != get_term(right)]


def _compare_runs(out: cliout.Output, args) -> int:
    """Compare two runs, or refuse and say exactly which term broke.

    Refusing is the feature. Two solves at different tolerances produce two
    numbers that can be subtracted, and the difference means nothing; printing
    it anyway is how a spurious ranking gets into a report.
    """
    from tarhan import artifact

    runs = []
    for run_id_arg in (args.left, args.right):
        try:
            runs.append(artifact.read_run(Path(args.output) / run_id_arg))
        except (artifact.ArtifactError, OSError) as exc:
            out.error(str(exc))
            return cliout.EXIT_INPUT

    left, right = runs
    for run in runs:
        if run.get("integrity") == "unverified-legacy":
            out.note(f"{run['path'].name}: written before checksums existed, so "
                     "nothing here has verified its contents")

    differing = _incomparable(left, right, args.allow_build_diff)
    if differing:
        out.error("not comparable: " + ", ".join(f"different {d}"
                                                 for d in differing))
        if _WAIVABLE in differing:
            out.note(f"the two runs were produced by different builds "
                     f"({left['manifest'].get('code_id', '?')} and "
                     f"{right['manifest'].get('code_id', '?')}). If comparing "
                     "them IS the question, say so with --allow-build-diff")
        out.note("§5.3 of the roadmap: a ranking across a changed contract is "
                 "not a weaker result, it is a different question")
        if out.fmt != "table":
            out.emit([{"comparable": False, "differing": ",".join(differing)}],
                     ("comparable", "differing"))
        return cliout.EXIT_INPUT

    shared = set(left["metrics"]) & set(right["metrics"])
    only_left = sorted(set(left["metrics"]) - shared)
    only_right = sorted(set(right["metrics"]) - shared)
    if only_left or only_right:
        # Comparing the intersection and saying nothing meant a build that
        # ADDED or REMOVED a metric produced a table that looked complete.
        # Under --allow-build-diff that is precisely the difference somebody
        # is looking for. Reported in review.
        if only_left:
            out.note(f"only in {left['path'].name}: {', '.join(only_left)}")
        if only_right:
            out.note(f"only in {right['path'].name}: {', '.join(only_right)}")
        out.note("a metric on one side only cannot be differenced; it is "
                 "named here rather than dropped")
    keys = sorted(shared)
    one_sided = ([{"metric": k, "left": left["metrics"][k], "right": None,
                   "delta": None} for k in only_left]
                 + [{"metric": k, "left": None, "right": right["metrics"][k],
                     "delta": None} for k in only_right])
    rows = []
    for key in keys:
        a, b = left["metrics"][key], right["metrics"][key]
        # Both sides must be numbers for a difference to mean anything: a
        # metric that changed TYPE between builds would otherwise subtract or
        # silently report None with no explanation.
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in (a, b))
        delta = (b - a) if numeric else None
        rows.append({"metric": key, "left": a, "right": b, "delta": delta})
    # Naming them on stderr left the JSON rows carrying only the intersection,
    # so an automated consumer saw a comparison that looked complete.
    # Reported in re-review.
    out.emit(rows + one_sided, ("metric", "left", "right", "delta"))
    if args.allow_build_diff and left["manifest"].get("code_id") != \
            right["manifest"].get("code_id"):
        out.note(f"BUILDS DIFFER ({left['manifest'].get('code_id', '?')} vs "
                 f"{right['manifest'].get('code_id', '?')}) — waived by "
                 "--allow-build-diff. Every delta below may be the code "
                 "changing rather than the physics")
    else:
        out.note(f"comparable: same capability, solver contract, inputs and "
                 f"build ({len(keys)} shared metrics)")
    return cliout.EXIT_OK


def _capabilities_show(out: cliout.Output, capability_id: str) -> int:
    """Print one record; the exit status repeats the verdict for a machine.

    A blocked capability still prints in full and still exits 3. Printing is the
    answer to the question that was asked; the status is that same answer in a
    form a script does not have to parse out of a paragraph.
    """
    try:
        cap = get(capability_id)
    except CapabilityNotFound:
        out.error(f"no such capability: {capability_id}")
        out.note("run `tarhan capabilities list` to see the ids that exist")
        return cliout.EXIT_INPUT

    if out.fmt == "table":
        lines = [f"{'id:':<16}{cap.id}",
                 f"{'status:':<16}{cap.status}",
                 f"{'dimension:':<16}{cap.dimension}",
                 f"{'time:':<16}{cap.time}"]
        if cap.source:
            lines.append(f"{'source:':<16}src/tarhan/{cap.source}")
        if cap.envelope:
            # `list` showed it and the detailed `show` did not, while AGENTS
            # and the CHANGELOG both said "visible in list/show". Reported in
            # re-review. The basis matters more than the numbers: an envelope
            # says nothing without the device it was measured on.
            lines.append(f"{'validated:':<16}{_envelope_text(cap)}")
            lines.append(f"{'  measured on:':<16}{cap.envelope_basis}")
        for label, items in (("inputs:", cap.inputs),
                             ("produces:", cap.produces),
                             ("limits:", cap.limits)):
            for i, item in enumerate(items):
                lines.append(f"{label if i == 0 else '':<16}{item}")
        for label, text in (("reason:", cap.reason), ("needs:", cap.needs),
                            ("does not mean:", cap.does_not_mean)):
            if text:
                lines.append(f"{label:<16}{text}")
        for ev in cap.evidence:
            lines.append(f"{'evidence:':<16}{ev.claim}")
            lines.append(f"{'':<16}  measured: {ev.measured}")
            lines.append(f"{'':<16}  test:     {ev.test}")
        out.detail("\n".join(lines) + "\n")
    else:
        row = dict(_cap_row(cap, out.fmt))
        row.update({"inputs": list(cap.inputs), "produces": list(cap.produces),
                    "limits": list(cap.limits), "reason": cap.reason,
                    "needs": cap.needs, "does_not_mean": cap.does_not_mean,
                    "evidence": [{"claim": e.claim, "measured": e.measured,
                                  "test": e.test} for e in cap.evidence]})
        out.emit([row], list(row))

    if not cap.runnable:
        out.note(f"{cap.id} is {cap.status}; exit status "
                 f"{cliout.EXIT_UNAVAILABLE} says so without parsing prose")
        return cliout.EXIT_UNAVAILABLE
    return cliout.EXIT_OK


def _demo(save: str | None, show: bool) -> int:
    targets = [0.1 * k for k in range(1, 11)]
    samples, dt, dx = cottrell_fd_samples(targets)

    print(f"TARHAN {__version__} — demo: Cottrell chronoamperometry (explicit FD)")
    print(f"grid: dX={dx:.4f}, dT={dt:.2e} (lam=0.45)\n")
    print(f"{'T':>4} | {'G_sim':>9} | {'G_analytic':>10} | {'err %':>7}")
    print("-" * 40)
    max_err = 0.0
    for t in targets:
        g_an = physics.cottrell_dimensionless(t)
        err = abs(samples[t] - g_an) / g_an * 100.0
        max_err = max(max_err, err)
        print(f"{t:4.1f} | {samples[t]:9.5f} | {g_an:10.5f} | {err:7.4f}")
    ok = max_err < 0.05
    print(f"\nmax relative error: {max_err:.4f}%  (tolerance 0.05%)  "
          f"{'PASS' if ok else 'FAIL'}")

    if save or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tt = np.linspace(0.05, 1.0, 400)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tt, 1.0 / np.sqrt(np.pi * tt), "-", label="analytic  G = 1/sqrt(pi*T)")
        ax.plot(targets, [samples[t] for t in targets], "o", label="TARHAN explicit-FD")
        ax.set_xlabel("T (dimensionless time)")
        ax.set_ylabel("G (dimensionless current)")
        ax.set_title("Cottrell reproduction - max error {:.3f}%".format(max_err))
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
            print(f"plot saved: {save}")
        if show:
            plt.show()

    return 0 if ok else 1


def _demo_diode(save: str | None, show: bool) -> int:
    from tarhan.models.pn1d import PNDiode1D, band_diagram, iv_sweep, solve_bias

    dev = PNDiode1D()
    volts = [0.05 * k for k in range(1, 9)]
    js, _ = iv_sweep(dev, volts)

    print(f"TARHAN {__version__} — demo: 1D pn-diode (Gummel + Scharfetter-Gummel)")
    print(f"Na=Nd={dev.Na:.0e} cm⁻³, ni={dev.ni:.0e}, kT/q={dev.ut} V (case inputs)\n")
    print(f"{'V [V]':>6} | {'J [A/cm²]':>12} | {'n_id':>6}")
    print("-" * 32)
    ok = True
    for i, (v, j) in enumerate(zip(volts, js)):
        nid_s = ""
        if i >= 3:                                       # −1 terimi rejimi: V ≥ 0.15
            nid = (volts[i] - volts[i - 1]) / (dev.ut * math.log(js[i] / js[i - 1]))
            nid_s = f"{nid:.4f}"
            ok &= abs(nid - 1.0) < 0.02
        print(f"{v:6.2f} | {j:12.5e} | {nid_s:>6}")
    print(f"\nideality (0.15-0.40 V) within 1.00+/-0.02: {'PASS' if ok else 'FAIL'}")

    if save or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        st = solve_bias(dev, 0.30)
        bd = band_diagram(dev, st)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.semilogy(volts, js, "o-", label="TARHAN Gummel/SG")
        vv = np.linspace(0.15, 0.40, 50)
        ax1.semilogy(vv, js[5] * np.exp((vv - volts[5]) / dev.ut), "--",
                     label="ideal slope (~60 mV/decade)")
        ax1.set_xlabel("V [V]"); ax1.set_ylabel("J [A/cm²]")
        ax1.set_title("Diode I-V"); ax1.legend()
        x_um = np.asarray(bd["x_cm"]) * 1e4
        ax2.plot(x_um, bd["Ec"], label="E_c")
        ax2.plot(x_um, bd["Ev"], label="E_v")
        ax2.plot(x_um, bd["EFn"], "--", label="E_Fn")
        ax2.plot(x_um, bd["EFp"], "--", label="E_Fp")
        ax2.set_xlabel("x [µm]"); ax2.set_ylabel("E [eV]")
        ax2.set_title("Band diagram @ 0.30 V"); ax2.legend()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
            print(f"plot saved: {save}")
        if show:
            plt.show()
    return 0 if ok else 1


def _force_utf8_stdio() -> None:
    """Windows konsolunda Türkçe çıktıyı kurtar (ilk CI koşusu yakaladı).

    Windows'ta stdout varsayılan olarak yerel kod sayfasını kullanır (ör. cp1252)
    ve çıktımızdaki 'ğ/ş/ı/ü/ö/ç' karakterlerini yazamayıp UnicodeEncodeError ile
    ÇÖKER — sayısal hiçbir sorun olmadığı hâlde demo çıkış kodu 1 döner.
    Bu CI'a özgü DEĞİL: aynı çökme gerçek Windows kullanıcısında da olur, o yüzden
    çözüm ortam değişkeni değil, akışın kendisi.

    reconfigure() Python 3.7+ TextIOWrapper'da vardır; stdout bir boruya ya da
    TextIOWrapper olmayan bir nesneye yönlendirilmişse sessizce atlanır.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _should_show(flag: bool | None) -> bool:
    """Decide whether to open an interactive plot window.

    ``plt.show()`` blocks until the window is closed. When nothing can ever
    close it — a CI job, an ssh session, a container, a piped run — the
    documented first command (`tarhan demo`) hangs forever, and because stdout
    is block-buffered when piped, it hangs having printed nothing at all. So the
    window is opt-in by context: only when stdout is a real terminal, and only
    when a display backend is actually usable. An explicit --show/--no-show
    always wins.
    """
    if flag is not None:
        return flag
    if not sys.stdout.isatty():
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def build_parser():
    """The full argument grammar, built without running anything.

    Split out of ``main`` so the documented commands can be checked
    against the real parser rather than against a subprocess: appending
    ``--help`` looks like a cheap way to do that, but argparse fires a
    subparser's help before the parent reports an unrecognised global,
    so it exits 0 on exactly the malformed line that started this
    (``capabilities list --format json``).

    Returns the parser and the three group parsers ``main`` falls back
    to when a subcommand arrives without its verb.
    """
    parser = argparse.ArgumentParser(
        prog="tarhan",
        description="TARHAN — physics-first materials simulator (pre-alpha)")
    parser.add_argument("--version", action="version", version=f"tarhan {__version__}")
    parser.add_argument("--format", choices=cliout.FORMATS, default="table",
                        help="result format on stdout (default: table). In json "
                             "and csv, stdout carries ONLY the result — every "
                             "note, warning and progress line goes to stderr")
    parser.add_argument("--color", choices=cliout.COLOR_MODES, default="auto",
                        help="colour on stderr (default: auto — only when it is "
                             "a terminal, never in json/csv, never with NO_COLOR)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress notes and progress; errors still print")
    parser.add_argument("--graphics", choices=("auto", "kitty", "iterm", "text"),
                        default="auto",
                        help="inline-graphics protocol for the forge indicator "
                             "(default: auto — detected, and text where no "
                             "protocol is available)")
    sub = parser.add_subparsers(dest="command")

    p_cap = sub.add_parser(
        "capabilities",
        help="what the engine can actually do, and where it stops")
    cap_sub = p_cap.add_subparsers(dest="cap_command")
    cap_sub.add_parser("list", help="every capability and its status")
    cap_sub.add_parser(
        "doctor",
        help="bring the tools up and check them — run this after installing",
        description="Imports the dependencies, loads the capability registry "
                    "and verifies every claim still names a test that exists. "
                    "The progress bar counts completed checks, never elapsed "
                    "time. Exits 3 if a required check fails.")
    p_show = cap_sub.add_parser(
        "show",
        help="one capability in full",
        description="Exits 3 when the capability is blocked or planned. The "
                    "record still prints; the status is the same answer in a "
                    "form a script does not have to parse.")
    p_show.add_argument("capability_id", metavar="<capability-id>")

    p_cand = sub.add_parser(
        "candidate",
        help="materials as things that can be argued with",
        description="No material database ships with TARHAN: property values "
                    "written from memory would be unverifiable numbers wearing "
                    "the authority of a package. Candidates come from a file "
                    "you supply.")
    cand_sub = p_cand.add_subparsers(dest="cand_command")
    p_clist = cand_sub.add_parser(
        "list", help="every candidate, and which models it can drive")
    p_cshow = cand_sub.add_parser(
        "show", help="one candidate's properties, with basis and uncertainty")
    p_cshow.add_argument("candidate_id", metavar="<candidate-id>")
    p_cscreen = cand_sub.add_parser(
        "screen",
        help="apply hard thresholds",
        description="Reports EVERY candidate, not only the survivors: a "
                    "shortlist alone hides how selective the screen was, and "
                    "hides which candidates were dropped for want of a "
                    "measurement rather than for being unsuitable. A value "
                    "whose uncertainty straddles a bound is `undecided`, "
                    "which is not a soft fail.")
    p_cscreen.add_argument("--at", action="append", default=[],
                           metavar="NAME=VALUE",
                           help="the operating condition to screen under; a "
                                "property outside its stated valid_range is "
                                "then undecided rather than a silent vote")
    p_cscreen.add_argument("--require", action="append", default=[],
                           metavar="NAME>=VALUE",
                           help="a hard threshold; repeat for more")
    for parser_ in (p_clist, p_cshow, p_cscreen):
        parser_.add_argument("--from", dest="source", required=True,
                             metavar="PATH",
                             help="a .toml or .json file of candidates")

    p_run = sub.add_parser("run", help="solve, and leave an artifact behind")
    run_sub = p_run.add_subparsers(dest="run_command")
    p_solve = run_sub.add_parser(
        "solve",
        help="run one capability and write a run directory",
        description="Refuses a blocked or planned capability with exit 3 "
                    "before doing any work, and exits 4 if the solver gives "
                    "up. No artifact is written for a run that did not "
                    "converge.")
    p_solve.add_argument("capability_id", metavar="<capability-id>")
    p_solve.add_argument("--bias", type=float, default=0.3,
                         help="applied bias in volts (default: 0.3)")
    p_solve.add_argument("--tol", type=float, default=1e-9,
                         help="solver tolerance; part of the run id")
    # No default here: each capability carries its own, because 60 Gummel
    # iterations is generous in 1D and tight in 2D. A default on the flag would
    # silently impose the 1D budget on every capability added later.
    p_solve.add_argument("--candidate", metavar="PATH", default=None,
                         help="a candidate file; with --candidate-id, the "
                              "material's properties become device overrides")
    p_solve.add_argument("--candidate-id", dest="candidate_id", default=None,
                         metavar="<candidate-id>")
    p_solve.add_argument("--device", metavar="PATH", default=None,
                         help="a .toml or .json file of device overrides. "
                              "Keys must belong to the capability's own "
                              "device; anything else is refused by name "
                              "rather than dropped. The merged result is what "
                              "lands in input.lock.toml and what names the run")
    p_solve.add_argument("--max-iter", dest="max_iter", type=int, default=None,
                         help="solver iteration budget (default: the "
                              "capability's own)")
    p_solve.add_argument("--output", default="runs",
                         help="directory to write run artifacts into")
    p_sweep = run_sub.add_parser(
        "sweep",
        help="solve a family of devices under one solver contract",
        description="Every point is solved with the SAME tolerance and "
                    "iteration budget, which is what makes a column of the "
                    "table readable down. Varying a solver term is refused "
                    "for that reason. Each point writes an ordinary run "
                    "artifact, so a row can be reopened with `run show`.")
    p_sweep.add_argument("capability_id", metavar="<capability-id>")
    p_sweep.add_argument("--vary", action="append", default=[],
                         metavar="NAME=V1,V2,...",
                         help="an axis to sweep; repeat for a grid. NAME is "
                              "bias_v or any field of the capability's device")
    p_sweep.add_argument("--bias", type=float, default=0.3,
                         help="bias for points that do not vary it")
    p_sweep.add_argument("--tol", type=float, default=1e-9)
    p_sweep.add_argument("--max-iter", dest="max_iter", type=int, default=None)
    p_sweep.add_argument("--device", metavar="PATH", default=None,
                         help="base device the axes are applied on top of")
    p_sweep.add_argument("--output", default="runs")

    p_rshow = run_sub.add_parser("show", help="reopen a run by id")
    p_rshow.add_argument("run_id", metavar="<run-id>")
    p_rshow.add_argument("--full", action="store_true",
                         help="also show the resolved inputs, the provenance "
                              "and the integrity state — everything needed to "
                              "tell what produced the numbers")
    p_rshow.add_argument("--output", default="runs")

    p_cmp = sub.add_parser("compare", help="compare two runs, or refuse to")
    cmp_sub = p_cmp.add_subparsers(dest="compare_command")
    p_cruns = cmp_sub.add_parser(
        "runs",
        help="compare two run directories",
        description="Exits 2 when the comparability contract does not hold, "
                    "naming the term that differs, rather than subtracting "
                    "numbers that mean different things.")
    p_cruns.add_argument("left", metavar="<run-id>")
    p_cruns.add_argument("right", metavar="<run-id>")
    p_cruns.add_argument("--output", default="runs")
    p_cruns.add_argument("--allow-build-diff", action="store_true",
                         help="compare two runs produced by different builds. "
                              "Refused by default: with the same inputs, a "
                              "delta from a code change looks exactly like a "
                              "physical effect. Every metric is flagged when "
                              "this is used")

    p_demo = sub.add_parser("demo", help="zero-config reproduction demos")
    p_demo.add_argument("--case", choices=("cottrell", "diode", "forge"),
                        default="cottrell",
                        help="demo case (default: cottrell). `forge` shows the "
                             "run indicator itself, with synthetic work")
    p_demo.add_argument("--save", metavar="PATH", default=None, help="save the plot as a PNG")
    p_demo.add_argument("--show", dest="show", action="store_true", default=None,
                        help="open an interactive plot window (default: only when "
                             "stdout is a terminal)")
    p_demo.add_argument("--no-show", dest="show", action="store_false",
                        help="never open a window (headless/CI)")

    return parser, {"capabilities": p_cap, "run": p_run,
                    "compare": p_cmp, "candidate": p_cand}


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser, groups = build_parser()
    p_cap, p_run, p_cmp = (groups["capabilities"], groups["run"],
                           groups["compare"])
    p_cand = groups["candidate"]
    args = parser.parse_args(argv)
    out = cliout.Output(fmt=args.format, color=args.color, quiet=args.quiet)

    try:
        if args.command == "demo":
            # demo keeps its own 0/1 contract, documented in AGENTS.md. It
            # predates this output contract and people already depend on it.
            if args.case == "forge":
                from tarhan.forge_pixels import demo as forge_demo
                forge_demo(args.graphics)
                return cliout.EXIT_OK
            show = _should_show(args.show)
            if args.case == "diode":
                return _demo_diode(save=args.save, show=show)
            return _demo(save=args.save, show=show)

        if args.command == "capabilities":
            if args.cap_command == "list":
                return _capabilities_list(out)
            if args.cap_command == "doctor":
                return _capabilities_doctor(out, args.graphics)
            if args.cap_command == "show":
                return _capabilities_show(out, args.capability_id)
            p_cap.print_help()
            return cliout.EXIT_INPUT

        if args.command == "candidate":
            if args.cand_command == "list":
                return _candidate_list(out, args)
            if args.cand_command == "show":
                return _candidate_show(out, args)
            if args.cand_command == "screen":
                return _candidate_screen(out, args)
            p_cand.print_help()
            return cliout.EXIT_INPUT

        if args.command == "run":
            if args.run_command == "solve":
                return _run_solve(out, args)
            if args.run_command == "sweep":
                return _run_sweep(out, args)
            if args.run_command == "show":
                return _run_show(out, args)
            p_run.print_help()
            return cliout.EXIT_INPUT

        if args.command == "compare":
            if args.compare_command == "runs":
                return _compare_runs(out, args)
            p_cmp.print_help()
            return cliout.EXIT_INPUT
    except Exception as exc:                       # noqa: BLE001 — the boundary
        # The last line of defence: an unexpected exception is OUR bug, and it
        # gets its own status so a caller can tell it apart from a rejected
        # input. The traceback still goes to stderr, where diagnostics belong.
        import traceback
        traceback.print_exc()
        out.error(f"internal error: {exc}")
        return cliout.EXIT_INTERNAL

    parser.print_help()
    print("\nhint: start with `tarhan capabilities list`, then `tarhan demo`.")
    return cliout.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

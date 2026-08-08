"""Layer-0 for run artifacts: can a result still answer for itself tomorrow?

The roadmap asks five questions of every result — which capability, which inputs
and units, which solver and version, what validation scope, and whether it is
comparable to another run. A directory that cannot answer them is a number in a
folder. These tests are about the properties that make the answers trustworthy
rather than merely present.

Two carry the weight:

* **The id is the content.** The same problem must land in the same directory,
  or "it was a different run" becomes an unfalsifiable defence for a result
  nobody can reproduce. The timestamp is therefore excluded from the hash, and
  that exclusion is asserted rather than assumed.
* **The emitter must produce real TOML.** This module writes TOML because the
  standard library only reads it, so every write is round-tripped through
  ``tomllib`` — the reader that will actually be used — rather than through a
  regex that would accept something merely TOML-shaped.
"""
import json
import tomllib

import numpy as np
import pytest

from tarhan.artifact import (FIELDS, RUN_FILES, ArtifactError, dumps_toml,
                             read_run, run_id, write_run)

INPUTS = {"bias_v": 0.3, "temperature_k": 300.0, "grid_points": 512,
          "srh": False, "contacts": ["anode", "cathode"],
          "doping": {"na_cm3": 1e16, "nd_cm3": 1e16}}
SOLVER = {"method": "gummel", "tol": 1e-9, "max_iter": 60}


def _write(tmp_path, **over):
    kwargs = dict(capability="semiconductor.pn.drift-diffusion.1d.steady",
                  capability_status="validated", inputs=INPUTS, solver=SOLVER,
                  metrics={"current_a_cm2": 1.234e-5, "iterations": 12},
                  provenance={"device": "pn1d-default", "scenario": "iv"},
                  status="converged", command="tarhan run solve",
                  version="0.2.0")
    kwargs.update(over)
    return write_run(tmp_path, **kwargs)


# --- the id is the content ------------------------------------------------

def test_the_same_problem_lands_in_the_same_directory(tmp_path):
    first = _write(tmp_path)
    second = _write(tmp_path)
    assert first == second
    assert first.name == run_id(
        "semiconductor.pn.drift-diffusion.1d.steady", INPUTS, SOLVER)


def test_a_changed_input_lands_somewhere_else():
    assert run_id("cap", dict(INPUTS, bias_v=0.31), SOLVER) != \
        run_id("cap", INPUTS, SOLVER)


def test_a_changed_solver_contract_lands_somewhere_else():
    """Two runs at the same bias with different tolerances are not the same
    result, and §5.3 will refuse to compare them. They must not share an id."""
    assert run_id("cap", INPUTS, dict(SOLVER, tol=1e-6)) != \
        run_id("cap", INPUTS, SOLVER)


def test_the_timestamp_is_not_part_of_the_id(tmp_path):
    """The property the whole scheme rests on.

    If the clock were an ingredient every run would be unique, re-running would
    never overwrite, and "the same problem lands in the same place" would be
    false while still looking implemented.
    """
    first = _write(tmp_path)
    stamp_one = json.loads((first / "manifest.json").read_text())["created"]
    second = _write(tmp_path, command="tarhan run solve --again")
    assert first == second
    stamp_two = json.loads((second / "manifest.json").read_text())["created"]
    assert stamp_one <= stamp_two          # the clock is recorded...
    assert first.name == second.name       # ...and does not move the directory


def test_a_run_must_name_its_capability():
    with pytest.raises(ArtifactError, match="name the capability"):
        run_id("", INPUTS, SOLVER)


def test_a_non_finite_input_is_refused_rather_than_hashed():
    """NaN would serialise as a non-standard literal and hash happily, handing
    a stable id to a run whose inputs were already broken."""
    with pytest.raises(ValueError):
        run_id("cap", dict(INPUTS, bias_v=float("nan")), SOLVER)


# --- the emitter must produce real TOML -----------------------------------

def test_the_written_lock_file_round_trips_through_the_stdlib_reader(tmp_path):
    """The reader that will actually be used, not a lookalike."""
    path = _write(tmp_path)
    parsed = tomllib.loads((path / "input.lock.toml").read_text())
    assert parsed["bias_v"] == pytest.approx(0.3)
    assert parsed["grid_points"] == 512
    assert parsed["srh"] is False
    assert parsed["contacts"] == ["anode", "cathode"]
    assert parsed["doping"]["na_cm3"] == pytest.approx(1e16)


@pytest.mark.parametrize("payload", [
    {"a": 1, "b": 2.5, "c": "text", "d": True, "e": [1, 2, 3]},
    {"only": {"nested": 1.0, "flag": False}},
    {"scalar": 1, "table": {"x": "y"}},
    {"empty_list": []},
])
def test_every_shape_the_emitter_accepts_parses_back(payload):
    assert tomllib.loads(dumps_toml(payload)) == payload


def test_a_boolean_is_not_written_as_an_integer():
    """bool is a subclass of int in Python, so the ORDER of the isinstance
    checks is load-bearing: get it wrong and `true` silently becomes `1`."""
    assert dumps_toml({"flag": True}).strip() == "flag = true"


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"x": float("inf")},
                                 {"x": object()}, {"x": {"y": {"z": 1}}}])
def test_the_emitter_refuses_what_it_cannot_represent(bad):
    """Refusing loudly beats emitting something the reader will reject later —
    or worse, accept as a different value."""
    with pytest.raises(ArtifactError):
        dumps_toml(bad)


# --- the directory answers for itself -------------------------------------

def test_a_complete_run_reads_back(tmp_path):
    path = _write(tmp_path, stdout="solving\n", report="# Run\n")
    for name in RUN_FILES:
        assert (path / name).exists()
    loaded = read_run(path)
    assert loaded["manifest"]["capability_status"] == "validated"
    assert loaded["manifest"]["tarhan_version"] == "0.2.0"
    assert loaded["metrics"]["iterations"] == 12
    assert loaded["provenance"]["device"] == "pn1d-default"
    assert loaded["stdout"] == "solving\n"


def test_an_incomplete_directory_is_refused(tmp_path):
    path = _write(tmp_path)
    (path / "metrics.json").unlink()
    with pytest.raises(ArtifactError, match="not a complete run"):
        read_run(path)


def test_a_directory_edited_after_the_run_is_refused(tmp_path):
    """The check that makes the id worth having.

    Edit the inputs and the recorded id no longer matches what they hash to, so
    the directory is caught claiming to be a run it is not. Without this the id
    would be decoration.
    """
    path = _write(tmp_path)
    lock = path / "input.lock.toml"
    lock.write_text(lock.read_text().replace("bias_v = 0.3", "bias_v = 0.9"))
    with pytest.raises(ArtifactError, match="edited after the run"):
        read_run(path)


def test_field_data_is_optional_and_written_when_present(tmp_path):
    bare = _write(tmp_path)
    assert not (bare / FIELDS).exists()

    withfields = _write(tmp_path, solver=dict(SOLVER, tol=1e-10),
                        fields_data={"psi": np.linspace(0, 1, 8)})
    assert (withfields / FIELDS).exists()
    with np.load(withfields / FIELDS) as data:
        assert data["psi"].shape == (8,)

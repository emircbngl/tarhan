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
import hashlib
import json
import tomllib

import numpy as np
import pytest

import tarhan
from tarhan import artifact

from tarhan.artifact import (FIELDS, RUN_FILES, ArtifactError, code_id,
                             dumps_toml, read_run, run_id, write_run)

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

def test_the_same_problem_and_the_same_code_land_in_the_same_directory(tmp_path):
    first = _write(tmp_path)
    second = _write(tmp_path)
    assert first == second
    problem, build = first.name.split("-")
    assert problem == run_id(
        "semiconductor.pn.drift-diffusion.1d.steady", INPUTS, SOLVER)
    assert build == code_id()


def test_the_directory_separates_the_problem_from_the_code_that_ran_it(tmp_path):
    """Two runs of one problem under different code are two different results.

    Naming the directory by the problem alone meant the first was silently
    overwritten by the second — reported in review, and the reason the build id
    is part of the name rather than only recorded inside it.
    """
    path = _write(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["code_id"] == code_id()
    assert manifest["environment"]["tarhan"] == tarhan.__version__
    assert set(manifest["environment"]) >= {"tarhan", "python", "numpy", "scipy"}
    assert path.name.endswith(manifest["code_id"])


def test_every_file_the_manifest_points_at_is_checksummed(tmp_path):
    """Hashing only the inputs left the results editable without trace."""
    path = _write(tmp_path, stdout="solving\n")
    manifest = json.loads((path / "manifest.json").read_text())
    assert set(manifest["files"]) == {"input.lock.toml", "provenance.json",
                                      "metrics.json", "stdout.log", "report.md"}
    assert "manifest.json" not in manifest["files"], \
        "the manifest cannot contain its own hash"


@pytest.mark.parametrize("victim", ["metrics.json", "provenance.json",
                                    "report.md"])
def test_editing_a_result_after_the_run_is_caught(tmp_path, victim):
    """The gap review found: the id covered the question, not the answer."""
    path = _write(tmp_path)
    target = path / victim
    target.write_text(target.read_text() + "\n tampered\n")
    with pytest.raises(ArtifactError, match="checksum"):
        read_run(path)


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

    # The checksum sees it first, which is the cheaper and more general catch.
    with pytest.raises(ArtifactError, match="checksum"):
        read_run(path)

    # Repair the checksum — the edit a forger would actually make — and the id
    # check is what is left standing. Both are load-bearing: the checksum
    # catches a changed file, the id catches a file changed together with its
    # own bookkeeping.
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["input.lock.toml"] = hashlib.sha256(
        lock.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
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


# --- the build id must separate two builds, not two version strings --------

def test_the_build_id_moves_when_the_source_moves(tmp_path, monkeypatch):
    """The collision the first build id did not close.

    It hashed tarhan/python/numpy/scipy versions. Every commit between two
    releases carries the same ``0.3.0.dev0``, so two genuinely different builds
    produced the same build id and wrote to one directory — reported in review.
    Simulated here by changing the source hash, which is what a commit does.
    """
    before = code_id()
    monkeypatch.setattr(artifact, "source_id", lambda: "0" * 64)
    after = code_id()
    assert before != after


def test_the_source_hash_is_over_the_bytes_that_were_imported():
    """Not the version string, and not the git commit.

    A wheel has no commit and a dirty checkout has one that lies about its own
    files; the source bytes are what the interpreter actually ran.
    """
    digest = artifact.source_id()
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert artifact.source_id() == digest, "the same tree must hash the same"
    assert artifact.environment()["source"] == digest


def test_the_git_commit_is_recorded_but_is_not_the_build_id(monkeypatch):
    """Recorded for people; hashed only incidentally.

    A checkout with edits reports a commit that no longer describes its files,
    which is why `dirty` is carried alongside it rather than left implied.
    """
    commit = artifact.git_commit()
    if commit is None:                        # an installed wheel, not a checkout
        assert "git" not in artifact.environment()
        return
    assert set(commit) == {"commit", "dirty"}
    assert len(commit["commit"]) == 40
    assert isinstance(commit["dirty"], bool)


# --- a directory written before checksums existed --------------------------

def test_a_legacy_directory_is_labelled_rather_than_silently_trusted(tmp_path):
    """The silence reported in review.

    A v1 manifest has no `files` map. Iterating it checks nothing and returns,
    so the run came back looking exactly as verified as one whose checksums had
    just been confirmed. It is now labelled, and the label is what `run show`
    and `compare runs` print.
    """
    path = _write(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]
    del manifest["schema_version"]
    manifest_path.write_text(json.dumps(manifest))

    loaded = read_run(path)                   # still readable — not a refusal
    assert loaded["integrity"] == "unverified-legacy"
    assert loaded["schema_version"] == 1


def test_a_current_directory_reports_itself_verified(tmp_path):
    loaded = read_run(_write(tmp_path))
    assert loaded["integrity"] == "verified"
    assert loaded["schema_version"] == artifact.SCHEMA_VERSION


def test_an_edited_legacy_run_is_NOT_caught_and_says_so(tmp_path):
    """The honest limit, asserted so nobody has to discover it.

    There are no checksums to check. The label is the whole protection.
    """
    path = _write(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("files")
    manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(manifest))
    (path / "metrics.json").write_text('{"current_a_cm2": 999.0}\n')

    loaded = read_run(path)                   # no error — nothing to check with
    assert loaded["metrics"]["current_a_cm2"] == 999.0
    assert loaded["integrity"] == "unverified-legacy"


def test_the_checksums_do_not_survive_an_edited_manifest(tmp_path):
    """The scope of the guarantee, pinned as a test rather than a claim.

    The manifest is unsigned. Someone who changes a result AND the digest
    beside it passes — this is accidental-corruption detection, not
    tamper-proofing, and the module docstring says so. Asserted here so the
    stronger claim cannot quietly creep back into the documentation.
    """
    path = _write(tmp_path)
    metrics = path / "metrics.json"
    metrics.write_text('{"current_a_cm2": 999.0, "iterations": 12}\n')
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["metrics.json"] = hashlib.sha256(
        metrics.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    loaded = read_run(path)                   # passes, and that is the point
    assert loaded["metrics"]["current_a_cm2"] == 999.0
    assert loaded["integrity"] == "verified"

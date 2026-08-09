"""Run artifacts — a result you can reopen, and that says where it came from.

A number printed to a terminal is gone the moment the scrollback rolls. The
roadmap's §2 asks five questions of every result, and this module exists so a
run can answer all five later, from disk, without the person who produced it:

1. which capability produced it            -> manifest.json
2. which inputs and units                  -> input.lock.toml
3. which solver, tolerance, mesh, version  -> manifest.json
4. what validation scope it falls under    -> manifest.json, from the registry
5. whether it is comparable to another run -> the solver contract in the manifest

**The id is the content, not a counter.** Two runs of the same problem must not
land in two differently-named directories, because that is how a result nobody
can reproduce ends up defended on the grounds that "it was a different run". The
id hashes the capability, the resolved inputs and the solver contract, so a
repeat lands in the same place and a genuine change lands somewhere new. The
timestamp is recorded but deliberately kept OUT of the hash — include it and
every run is unique by construction, which makes the property worthless.

**What the checksums are, and what they are NOT.** ``manifest.json`` records a
sha256 of every other file in the directory, so a result that changed after the
run is caught rather than trusted. That is **accidental-corruption and
casual-edit detection**. It is NOT tamper-proofing and must not be described as
such: the manifest is unsigned, so anyone who edits ``metrics.json`` and then
updates the digest beside it is not caught, and nothing here would notice. This
was raised in review and the honest answer is the scope, not a stronger claim —
resisting a motivated forger needs a signature over the manifest, keys to sign
with and somewhere to keep them, and none of that exists yet. The `run_id`
check is a second, independent reading of the same directory, so the two must
be edited consistently to pass both, but that is a speed bump and not a lock.

**On the TOML.** ``input.lock.toml`` is the roadmap's choice and it is kept, but
Python's standard library reads TOML and does not write it. Rather than take a
dependency for one file, this module emits the small subset it actually needs —
and every write is round-tripped through the stdlib reader in the tests, so the
emitter cannot quietly produce something that only looks like TOML.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

MANIFEST = "manifest.json"
INPUT_LOCK = "input.lock.toml"
PROVENANCE = "provenance.json"
METRICS = "metrics.json"
FIELDS = "fields.npz"
STDOUT_LOG = "stdout.log"
REPORT = "report.md"

#: Everything a complete run directory must contain. ``fields.npz`` is NOT in
#: the list: a run that produced only scalars has no field data, and inventing
#: an empty array to satisfy a checklist would be worse than its absence.
RUN_FILES = (MANIFEST, INPUT_LOCK, PROVENANCE, METRICS, STDOUT_LOG, REPORT)

ID_LENGTH = 12          # 48 bits of sha256; collision is not the failure mode
CODE_ID_LENGTH = 8      # enough to separate builds, short enough to read

#: Bumped when the manifest gains a field a reader must know about.
#:
#: 1 — the original: capability, inputs, solver. No checksums.
#: 2 — adds ``code_id``, ``environment`` and per-file ``files`` checksums.
#:
#: A v1 directory has no checksums, so a v2 reader that simply iterates
#: ``files`` finds nothing to check and returns it looking exactly like a
#: verified run. That silence was reported in review. The version is what lets
#: :func:`read_run` say ``unverified-legacy`` out loud instead.
SCHEMA_VERSION = 2


class ArtifactError(ValueError):
    """A run directory that cannot be trusted to describe itself."""


def _canonical(payload: Mapping[str, Any]) -> str:
    """Stable JSON: sorted keys, no incidental whitespace, no NaN.

    ``allow_nan=False`` matters. A NaN among the inputs would otherwise
    serialise as the non-standard literal ``NaN`` and hash happily, handing a
    stable id to a run whose inputs were already broken.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False)


def run_id(capability: str, inputs: Mapping[str, Any],
           solver: Mapping[str, Any]) -> str:
    """Content id: the same problem lands in the same directory.

    The timestamp is not an ingredient. Including it would make every run
    unique and quietly destroy the only property this id has.
    """
    if not capability:
        raise ArtifactError("a run must name the capability that produced it")
    blob = _canonical({"capability": capability,
                       "inputs": dict(inputs),
                       "solver": dict(solver)})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:ID_LENGTH]


def source_id() -> str:
    """sha256 over the package's own Python source, as imported.

    The version string is not the code. Every commit on ``main`` between two
    releases carries the same ``0.3.0.dev0``, so a build id derived from
    versions alone lets two genuinely different builds write to one directory —
    the exact collision the build id was added to prevent, reported in review
    against the first version of this function.

    A git commit was the obvious fix and is recorded separately, but it is not
    the thing to hash: it is absent from an installed wheel and it is a LIE
    about a dirty working tree, which is the normal state of the machine where
    a run is actually produced. The source bytes are neither. They are what the
    interpreter executed, in both a checkout and an install.

    **Scope, chosen deliberately: the whole package is the build.** Every
    ``.py`` under the package counts, including modules no solve imports — so a
    prototype sitting in the source tree moves the build id of a run it never
    touched. That is broader than "the code that ran", and it is the intended
    reading: which modules a solve imports is not decidable without running it,
    it changes with the inputs, and a narrower hash would quietly call two
    different working trees the same build. The cost is real and worth naming —
    an untracked scratch file makes a local build id differ from CI's for the
    same commit — but it errs toward calling two builds different, which is the
    safe direction for something whose whole job is to keep results apart.
    """
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for item in sorted(package.rglob("*.py")):
        digest.update(item.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def git_commit() -> Optional[Dict[str, Any]]:
    """The commit the source sits on, when it sits on one at all.

    Recorded for people — it is what turns a build id back into a diff — and
    deliberately NOT what identifies the build: an installed wheel has no
    commit, and a checkout with edits reports a commit that no longer describes
    its own files. ``dirty`` says which case you are looking at.
    """
    import subprocess

    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        run = lambda *a: subprocess.run(a, cwd=root, capture_output=True,
                                        text=True, timeout=10)
        head = run("git", "rev-parse", "HEAD")
        if head.returncode != 0:
            return None
        dirty = run("git", "status", "--porcelain", "--", "src")
        return {"commit": head.stdout.strip(),
                "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    """What the result was produced BY, as opposed to what it was produced FROM.

    Two runs of the same problem under different code are two different results.
    Recording only the inputs makes that indistinguishable, and the directory
    for the first would be silently overwritten by the second — reported in
    review against the first version of this module, which hashed the problem
    and nothing else.
    """
    import platform

    from tarhan import __version__

    versions = {"tarhan": __version__, "python": platform.python_version(),
                "source": source_id()}
    for name in ("numpy", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:                                 # noqa: BLE001
            versions[name] = "absent"
    commit = git_commit()
    if commit is not None:
        versions["git"] = commit
    return versions


#: Recorded in the manifest but excluded from the build hash. See :func:`code_id`.
UNHASHED_ENVIRONMENT = ("git",)


def code_id(env=None) -> str:
    """A short hash of the code and libraries that produced a result.

    Derived from :func:`environment`, whose ``source`` term is a hash of the
    package's own bytes — so two different commits under one dev version get
    two different build ids, which is what stops them sharing a directory.

    ``git`` is recorded but NOT hashed, and the exclusion has to be explicit
    rather than assumed: this function used to hash the whole ``env`` mapping
    while the documentation next to it said the commit was deliberately kept
    out, which is a documentation defect, not a wording preference — the two
    disagreed and the prose was the false one. Reported in review.

    It is excluded because it is not identifying information. The same bytes
    reached from two branches, or from a checkout and an unpacked sdist, are
    one build; hashing the commit would split them. Worse, ``dirty`` is a
    boolean over the WHOLE tree, so editing a test or the README would move the
    build id of a solver whose source had not changed at all. ``source`` is the
    term that answers "what code ran"; ``git`` answers "where did it come
    from", which is for people reading the manifest.
    """
    env = environment() if env is None else env
    hashed = {k: v for k, v in env.items() if k not in UNHASHED_ENVIRONMENT}
    return hashlib.sha256(
        _canonical(hashed).encode("utf-8")).hexdigest()[:CODE_ID_LENGTH]


def checksums(path) -> Dict[str, str]:
    """sha256 of every file in a run directory except the manifest itself.

    The manifest cannot contain its own hash, so it is the one file this does
    not cover; everything it points at is covered, which is what makes a change
    to metrics.json or fields.npz detectable rather than invisible.

    Detectable, not prevented, and not proof of anything against someone who
    means it — the manifest is unsigned, so a digest edited alongside the file
    it describes passes. See the module docstring.
    """
    path = Path(path)
    out = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.name != MANIFEST:
            out[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
    return out


# --- the small TOML the roadmap asks for ----------------------------------

def _toml_value(value: Any) -> str:
    if isinstance(value, bool):                 # before int: bool IS an int
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value
                                         or value in (float("inf"),
                                                      float("-inf"))):
            raise ArtifactError(
                f"{value!r} has no TOML representation; a run whose inputs are "
                "not finite should fail before it is recorded")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)                # TOML basic strings are JSON-ish
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ArtifactError(f"no TOML representation for {type(value).__name__}")


def dumps_toml(mapping: Mapping[str, Any]) -> str:
    """Emit the subset used here: scalars, arrays, and one level of table.

    Deliberately narrow. A general TOML writer is a dependency's worth of edge
    cases, and anything this refuses is something a run should not be recording
    as an input in the first place.
    """
    scalars, tables = [], []
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            body = "\n".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
            tables.append(f"[{key}]\n{body}\n")
        else:
            scalars.append(f"{key} = {_toml_value(value)}")
    out = "\n".join(scalars)
    if scalars:
        out += "\n"
    if scalars and tables:
        out += "\n"
    return out + "\n".join(tables)


# --- the manifest ---------------------------------------------------------

@dataclass
class RunManifest:
    """What the run was, in the terms the capability registry uses."""

    run_id: str
    capability: str
    status: str                       # converged | not-converged | failed
    tarhan_version: str
    command: str
    created: str                      # ISO-8601 UTC, e.g. 2026-08-07T21:04:00Z
    solver: Dict[str, Any] = field(default_factory=dict)
    capability_status: str = ""       # validated | blocked | planned, at run time
    notes: str = ""
    code_id: str = ""                 # what produced it, not what it was made from
    environment: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)   # sha256 per file
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "capability": self.capability,
            "capability_status": self.capability_status,
            "status": self.status,
            "tarhan_version": self.tarhan_version,
            "command": self.command,
            "created": self.created,
            "solver": self.solver,
            "notes": self.notes,
            "code_id": self.code_id,
            "environment": self.environment,
            "files": self.files,
            "schema_version": self.schema_version,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_run(root, *, capability: str, capability_status: str,
              inputs: Mapping[str, Any], solver: Mapping[str, Any],
              metrics: Mapping[str, Any], provenance: Mapping[str, Any],
              status: str, command: str, version: str,
              fields_data: Optional[Mapping[str, Any]] = None,
              stdout: str = "", report: str = "", notes: str = "") -> Path:
    """Write one run directory and return its path.

    The directory is named by :func:`run_id`, so re-running the same problem
    overwrites rather than accumulates. That is intended: two directories
    differing only in a timestamp are two answers to one question with no way
    to tell which was meant.
    """
    problem = run_id(capability, inputs, solver)
    env = environment()
    build = code_id(env)
    # The directory carries BOTH: the same problem under different code is a
    # different result, and naming it only by the problem meant the first was
    # silently overwritten by the second.
    identifier = f"{problem}-{build}"
    path = Path(root) / identifier
    path.mkdir(parents=True, exist_ok=True)

    (path / INPUT_LOCK).write_text(dumps_toml(inputs), encoding="utf-8")
    (path / PROVENANCE).write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    (path / METRICS).write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True, allow_nan=False,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    (path / STDOUT_LOG).write_text(stdout, encoding="utf-8")
    (path / REPORT).write_text(report, encoding="utf-8")
    if fields_data:
        import numpy as np
        np.savez_compressed(path / FIELDS, **dict(fields_data))

    # The manifest goes last, because it records a checksum of everything else.
    manifest = RunManifest(run_id=identifier, capability=capability,
                           capability_status=capability_status, status=status,
                           tarhan_version=version, command=command,
                           created=utc_now(), solver=dict(solver), notes=notes,
                           code_id=build, environment=env,
                           files=checksums(path))
    (path / MANIFEST).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True,
                   ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_run(path) -> Dict[str, Any]:
    """Load a run directory, refusing one that cannot describe itself."""
    import tomllib

    path = Path(path)
    missing = [name for name in RUN_FILES if not (path / name).exists()]
    if missing:
        raise ArtifactError(f"{path} is not a complete run; missing {missing}")

    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))

    # Hashing only the inputs left metrics.json, provenance.json and fields.npz
    # editable without trace — reported in review. Every file the manifest
    # points at is checked against the checksum recorded at write time, and the
    # check runs BEFORE anything is parsed: a corrupted JSON file would
    # otherwise surface as a decode error, which says the file is malformed
    # when what actually happened is that its contents changed.
    schema = manifest.get("schema_version", 1)
    recorded_files = manifest.get("files") or {}
    if schema < SCHEMA_VERSION or not recorded_files:
        # A pre-checksum directory. Iterating an absent `files` map checks
        # nothing and returns silently, which reads exactly like a verified
        # run — reported in review. Say what it is instead.
        integrity = "unverified-legacy"
    else:
        integrity = "verified"
        actual = checksums(path)
        for name, digest in recorded_files.items():
            if name not in actual:
                raise ArtifactError(f"{path}: {name} is recorded but missing")
            if actual[name] != digest:
                raise ArtifactError(
                    f"{path}: {name} does not match the checksum recorded when "
                    "the run was written; the result has changed since")

    inputs = tomllib.loads((path / INPUT_LOCK).read_text(encoding="utf-8"))
    out = {
        "path": path,
        "manifest": manifest,
        "schema_version": schema,
        "integrity": integrity,
        "inputs": inputs,
        "provenance": json.loads(
            (path / PROVENANCE).read_text(encoding="utf-8")),
        "metrics": json.loads((path / METRICS).read_text(encoding="utf-8")),
        "stdout": (path / STDOUT_LOG).read_text(encoding="utf-8"),
        "report": (path / REPORT).read_text(encoding="utf-8"),
    }
    expected = run_id(manifest["capability"], inputs, manifest["solver"])
    recorded = manifest["run_id"].split("-")[0]
    if expected != recorded:
        raise ArtifactError(
            f"{path}: the manifest says run_id {manifest['run_id']} but its own "
            f"inputs and solver hash to {expected}. Something was edited after "
            "the run, and the directory no longer describes what produced it.")
    return out

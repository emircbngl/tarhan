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
    identifier = run_id(capability, inputs, solver)
    path = Path(root) / identifier
    path.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(run_id=identifier, capability=capability,
                           capability_status=capability_status, status=status,
                           tarhan_version=version, command=command,
                           created=utc_now(), solver=dict(solver), notes=notes)
    (path / MANIFEST).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True,
                   ensure_ascii=False) + "\n", encoding="utf-8")
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
    return path


def read_run(path) -> Dict[str, Any]:
    """Load a run directory, refusing one that cannot describe itself."""
    import tomllib

    path = Path(path)
    missing = [name for name in RUN_FILES if not (path / name).exists()]
    if missing:
        raise ArtifactError(f"{path} is not a complete run; missing {missing}")

    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    inputs = tomllib.loads((path / INPUT_LOCK).read_text(encoding="utf-8"))
    out = {
        "path": path,
        "manifest": manifest,
        "inputs": inputs,
        "provenance": json.loads(
            (path / PROVENANCE).read_text(encoding="utf-8")),
        "metrics": json.loads((path / METRICS).read_text(encoding="utf-8")),
        "stdout": (path / STDOUT_LOG).read_text(encoding="utf-8"),
        "report": (path / REPORT).read_text(encoding="utf-8"),
    }
    expected = run_id(manifest["capability"], inputs, manifest["solver"])
    if expected != manifest["run_id"]:
        raise ArtifactError(
            f"{path}: the manifest says run_id {manifest['run_id']} but its own "
            f"inputs and solver hash to {expected}. Something was edited after "
            "the run, and the directory no longer describes what produced it.")
    return out

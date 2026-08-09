"""Candidates — a material as something that can be argued with.

The roadmap's §5.1 is blunt about what a candidate is not: "a candidate is not
free text and not a single score". This module is the schema that makes that
true, and the parts that carry the weight are the ones a spreadsheet leaves
out.

**A property is a value, a unit, a basis and a doubt.** ``1350`` is not a
mobility. ``1350 cm^2/Vs, computed, +/- 200, valid 250-350 K`` is one, and the
difference decides whether a screening result means anything. So ``Property``
refuses a value with no unit and refuses a basis that is not one of measured,
computed or inferred — a number whose origin nobody recorded cannot be weighed
against one that was measured, and pretending otherwise is how a literature
value and a guess end up in the same column.

**Uncertainty is not decoration; it can make a threshold undecidable.** If a
candidate's value is 1.10 +/- 0.15 eV and the threshold is "at least 1.20", the
honest answer is neither pass nor fail. :func:`screen` returns ``undecided``
for that case rather than letting the nominal value cast a vote it has not
earned. This is the single most important behaviour here: a screen that
silently resolves every borderline case in one direction produces a shortlist
whose length is a property of the rounding.

**Applicability is stated as what is MISSING.** A candidate that cannot feed a
model is not simply "unsuitable" — it is unsuitable *because* two parameters
were never supplied, and those two are the experiment somebody should run
next. :func:`applicability` names them.

**No material database ships with this module, deliberately.** Real property
values written from memory would be unverifiable numbers wearing the authority
of a package — and ``physics_verify`` is unavailable in the session this was
written in, so nothing here could have checked them. Candidates come from a
file the user supplies. Every value in the tests is synthetic and obviously so.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

#: Where a number came from. The roadmap's three, and no fourth: "unknown" is
#: not a basis, it is a missing property, and it must be absent rather than
#: present-and-vague.
BASES = ("measured", "computed", "inferred")

#: What a screen can conclude about one candidate against one threshold.
VERDICTS = ("pass", "fail", "undecided")


class CandidateError(ValueError):
    """A candidate that cannot be reasoned about."""


@dataclass(frozen=True)
class Property:
    """One physical property, with everything needed to weigh it."""

    value: float
    unit: str
    basis: str
    uncertainty: Optional[float] = None      # symmetric, same unit as value
    source: str = ""
    valid_range: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.value, (int, float)) or \
                isinstance(self.value, bool) or not math.isfinite(self.value):
            raise CandidateError(f"value={self.value!r}: must be a finite number")
        if not self.unit or not str(self.unit).strip():
            raise CandidateError(
                f"a value of {self.value} with no unit is not a property")
        if self.basis not in BASES:
            raise CandidateError(
                f"basis={self.basis!r}: must be one of {', '.join(BASES)}. A "
                "number whose origin nobody recorded cannot be weighed against "
                "one that was measured")
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, (int, float)) or \
                    isinstance(self.uncertainty, bool):
                raise CandidateError("uncertainty must be a number or absent")
            if not math.isfinite(self.uncertainty) or self.uncertainty < 0:
                raise CandidateError(
                    f"uncertainty={self.uncertainty}: must be finite and >= 0")
        if self.basis == "measured" and not self.source:
            # A measured value with no source is the one combination that
            # actively misleads: it claims the strongest basis and offers
            # nothing to check it against.
            raise CandidateError(
                f"a measured value ({self.value} {self.unit}) must name its "
                "source; 'measured' without provenance is the strongest claim "
                "with the least behind it")

    @property
    def interval(self) -> Tuple[float, float]:
        """The range the value could occupy. A point when nothing is known."""
        spread = 0.0 if self.uncertainty is None else float(self.uncertainty)
        return (self.value - spread, self.value + spread)


@dataclass(frozen=True)
class Candidate:
    """A material, identified, with properties that carry their own history."""

    identifier: str
    properties: Mapping[str, Property]
    composition: str = ""
    structure: str = ""
    dimensionality: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.identifier or not str(self.identifier).strip():
            raise CandidateError("a candidate must have a canonical identifier")
        if not self.properties:
            raise CandidateError(
                f"{self.identifier}: a candidate with no properties cannot be "
                "screened, ranked or fed to a model")
        for name, prop in self.properties.items():
            if not isinstance(prop, Property):
                raise CandidateError(
                    f"{self.identifier}.{name} is not a Property")

    def get(self, name: str) -> Optional[Property]:
        return self.properties.get(name)


# --- what a candidate can actually be used for ----------------------------

#: capability id -> the device parameters that capability needs from a
#: candidate. Deliberately NOT every field of the device: len_p, ny and the
#: mesh controls describe the thing being built, not the material it is built
#: from, and asking a material to supply them would be a category error.
MATERIAL_PARAMETERS = {
    "semiconductor.pn.drift-diffusion.1d.steady":
        ("ni", "eps_s", "mu_n", "mu_p"),
    "semiconductor.pn.drift-diffusion.2d.steady":
        ("ni", "eps_s", "mu_n", "mu_p"),
}


@dataclass(frozen=True)
class Applicability:
    """Whether a candidate can drive a capability, and what is missing if not."""

    capability: str
    usable: bool
    missing: Tuple[str, ...]

    def __str__(self):
        if self.usable:
            return f"{self.capability}: usable"
        return f"{self.capability}: missing {', '.join(self.missing)}"


def applicability(candidate: Candidate, capability_id: str) -> Applicability:
    """What stands between this candidate and this model.

    The answer is the missing parameter names, not a verdict, because the names
    are the actionable part: they are the measurements somebody would have to
    make. A bare "unsuitable" throws that away.
    """
    needed = MATERIAL_PARAMETERS.get(capability_id)
    if needed is None:
        raise CandidateError(
            f"no material-parameter mapping for {capability_id}; a capability "
            "must declare what it needs from a material before a candidate can "
            "be judged against it")
    missing = tuple(name for name in needed if candidate.get(name) is None)
    return Applicability(capability_id, not missing, missing)


def device_overrides(candidate: Candidate,
                     capability_id: str) -> Dict[str, float]:
    """The candidate's properties as device overrides for `run solve`.

    This is the join that stops a candidate from being provenance-less JSON: a
    material is only real here if a validated model can be run on it. Refuses
    an incomplete candidate rather than filling a gap with a default, because a
    default silently substitutes some OTHER material's number and the run would
    then describe a material that does not exist.
    """
    fit = applicability(candidate, capability_id)
    if not fit.usable:
        raise CandidateError(
            f"{candidate.identifier} cannot drive {capability_id}: missing "
            f"{', '.join(fit.missing)}. Filling these with defaults would "
            "solve a material that does not exist")
    return {name: float(candidate.properties[name].value)
            for name in MATERIAL_PARAMETERS[capability_id]}


# --- screening ------------------------------------------------------------

@dataclass(frozen=True)
class Threshold:
    """One hard requirement: a property, a direction, a bound."""

    prop: str
    op: str                    # ">=" or "<="
    bound: float

    def __post_init__(self):
        if self.op not in (">=", "<="):
            raise CandidateError(
                f"op={self.op!r}: a hard threshold is >= or <=. Anything "
                "softer belongs in ranking, where the trade-off is visible")
        if not isinstance(self.bound, (int, float)) or \
                isinstance(self.bound, bool) or not math.isfinite(self.bound):
            raise CandidateError(f"bound={self.bound!r}: must be a finite number")


@dataclass(frozen=True)
class Judgement:
    """One candidate against one threshold, with the reason kept."""

    threshold: Threshold
    verdict: str
    detail: str


def judge(candidate: Candidate, threshold: Threshold) -> Judgement:
    """Apply one threshold, letting uncertainty refuse to decide.

    The interesting case is the third one. If the value's interval straddles
    the bound, neither "pass" nor "fail" is supportable, and choosing one
    anyway would make the length of a shortlist a property of the rounding
    rather than of the materials. It is reported as ``undecided`` and the
    caller decides what to do with a candidate that needs a better measurement.
    """
    prop = candidate.get(threshold.prop)
    if prop is None:
        return Judgement(threshold, "undecided",
                         f"{threshold.prop} is not recorded for "
                         f"{candidate.identifier}")

    low, high = prop.interval
    if threshold.op == ">=":
        if low >= threshold.bound:
            verdict = "pass"
        elif high < threshold.bound:
            verdict = "fail"
        else:
            verdict = "undecided"
    else:
        if high <= threshold.bound:
            verdict = "pass"
        elif low > threshold.bound:
            verdict = "fail"
        else:
            verdict = "undecided"

    spread = "" if prop.uncertainty is None else f" +/- {prop.uncertainty:g}"
    detail = (f"{threshold.prop} = {prop.value:g}{spread} {prop.unit} "
              f"({prop.basis}) against {threshold.op} {threshold.bound:g}")
    if verdict == "undecided":
        detail += " - the uncertainty straddles the bound"
    return Judgement(threshold, verdict, detail)


def screen(candidates, thresholds) -> Dict[str, Any]:
    """Apply every threshold to every candidate. Nothing is dropped silently.

    A screen that returns only the survivors hides its own selectivity: you
    cannot tell a threshold that removed one candidate from one that removed
    forty, and you certainly cannot tell which ones were thrown out for want of
    a measurement rather than for being unsuitable. Every candidate comes back
    with its verdict and the reasons behind it.
    """
    thresholds = tuple(thresholds)
    results = []
    for candidate in candidates:
        judgements = [judge(candidate, t) for t in thresholds]
        if any(j.verdict == "fail" for j in judgements):
            verdict = "fail"
        elif any(j.verdict == "undecided" for j in judgements):
            verdict = "undecided"
        else:
            verdict = "pass"
        results.append({"identifier": candidate.identifier,
                        "verdict": verdict,
                        "judgements": judgements})
    return {"thresholds": thresholds, "results": results}


# --- loading --------------------------------------------------------------

_PROPERTY_FIELDS = {"value", "unit", "basis", "uncertainty", "source",
                    "valid_range"}
_CANDIDATE_FIELDS = {"properties", "composition", "structure",
                     "dimensionality", "notes"}


def _property_from(name: str, raw: Any, owner: str) -> Property:
    if not isinstance(raw, dict):
        raise CandidateError(
            f"{owner}.{name}: expected a table with at least value, unit and "
            f"basis - a bare {raw!r} is a number without a history")
    unknown = sorted(set(raw) - _PROPERTY_FIELDS)
    if unknown:
        raise CandidateError(f"{owner}.{name}: unknown field(s) {unknown}")
    missing = sorted({"value", "unit", "basis"} - set(raw))
    if missing:
        raise CandidateError(f"{owner}.{name}: missing {', '.join(missing)}")
    return Property(value=raw["value"], unit=raw["unit"], basis=raw["basis"],
                    uncertainty=raw.get("uncertainty"),
                    source=raw.get("source", ""),
                    valid_range=raw.get("valid_range", {}))


def load_candidates(path) -> Tuple[Candidate, ...]:
    """Read a .toml or .json file of candidates.

    The shape is one table per candidate, keyed by identifier, each with a
    ``properties`` table. Anything unrecognised is refused by name: a misspelt
    field dropped in silence would make a candidate quietly weaker than the
    file says it is.
    """
    import tomllib

    path = Path(path)
    raw = path.read_bytes()
    if path.suffix.lower() == ".json":
        doc = json.loads(raw.decode("utf-8"))
    elif path.suffix.lower() == ".toml":
        doc = tomllib.loads(raw.decode("utf-8"))
    else:
        raise CandidateError(
            f"{path.name}: expected a .toml or .json candidate file")
    if not isinstance(doc, dict) or not doc:
        raise CandidateError(f"{path.name}: expected candidates keyed by id")

    out = []
    for identifier, body in doc.items():
        if not isinstance(body, dict):
            raise CandidateError(f"{identifier}: expected a table")
        unknown = sorted(set(body) - _CANDIDATE_FIELDS)
        if unknown:
            raise CandidateError(f"{identifier}: unknown field(s) {unknown}")
        props_raw = body.get("properties")
        if not isinstance(props_raw, dict) or not props_raw:
            raise CandidateError(f"{identifier}: needs a properties table")
        properties = {name: _property_from(name, value, identifier)
                      for name, value in props_raw.items()}
        out.append(Candidate(identifier=identifier, properties=properties,
                             composition=body.get("composition", ""),
                             structure=body.get("structure", ""),
                             dimensionality=body.get("dimensionality", ""),
                             notes=body.get("notes", "")))
    return tuple(out)


def parse_threshold(text: str) -> Threshold:
    """``mu_n>=1000`` into a Threshold."""
    for op in (">=", "<="):
        if op in text:
            name, _, bound = text.partition(op)
            name = name.strip()
            if not name:
                raise CandidateError(f"{text!r}: no property named")
            try:
                return Threshold(name, op, float(bound))
            except ValueError:
                raise CandidateError(f"{text!r}: {bound!r} is not a number")
    raise CandidateError(
        f"{text!r}: expected NAME>=VALUE or NAME<=VALUE. A hard screen is a "
        "bound; anything softer belongs in ranking")

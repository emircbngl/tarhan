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

#: Operating conditions a run can actually report, and therefore the only ones
#: a ``valid_range`` may name. The list is short on purpose: a range over a
#: condition nothing supplies cannot be enforced, and an unenforced range is
#: worse than an absent one because it reads as a guarantee.
CONDITIONS = ("bias_v",)


def out_of_range(candidate: "Candidate", conditions: Mapping[str, float]):
    """Which properties are being used outside their stated validity.

    Returns a tuple of human-readable strings, empty when everything is in
    range. Storing ``valid_range`` and never consulting it let a candidate be
    solved far outside the window its own file declared, and still be reported
    as "usable" — reported in review.
    """
    breaches = []
    for name, prop in sorted(candidate.properties.items()):
        for condition, (low, high) in prop.valid_range.items():
            if condition not in conditions:
                continue
            value = conditions[condition]
            if not (low <= value <= high):
                breaches.append(
                    f"{name} is stated valid for {condition} in "
                    f"[{low:g}, {high:g}] but the run uses {value:g}")
    return tuple(breaches)


class CandidateError(ValueError):
    """A candidate that cannot be reasoned about."""


#: Canonical unit per property, and the alternative spellings accepted for it
#: with the factor that converts INTO the canonical unit.
#:
#: Why this table has to exist: units used to be checked only for emptiness.
#: ``mu_n = 0.1 m^2/Vs`` and ``mu_n = 1000 cm^2/Vs`` are the same mobility, and
#: the raw number went to the solver either way — so the first silently solved
#: a material ten thousand times slower than the one described. Worse,
#: ``mu_n = 3 kg`` was accepted and passed through, and so was ``unit = 123``.
#: Reported in review with a real run: ni in seconds, eps_s in eV, mu_p in
#: bananas, and the candidate came back "usable".
#:
#: Every factor below is an exact decimal relation between metric prefixes, so
#: each is written with its derivation and can be checked by reading:
#:   1 m = 100 cm  =>  1 m^3 = 1e6 cm^3  =>  1 m^-3 = 1e-6 cm^-3
#:   1 m^2 = 1e4 cm^2                    =>  1 m^2/Vs = 1e4 cm^2/Vs
#:   1 m = 100 cm                        =>  1 F/m = 1e-2 F/cm
#: NOT verified by physics_verify (the physicist server is unavailable); they
#: are arithmetic on the definition of "centi", stated here so the arithmetic
#: is the thing under review rather than a remembered constant.
PROPERTY_UNITS = {
    "ni":     ("cm^-3",     {"m^-3": 1e-6}),
    "eps_s":  ("F/cm",      {"F/m": 1e-2}),
    "mu_n":   ("cm^2/Vs",   {"m^2/Vs": 1e4}),
    "mu_p":   ("cm^2/Vs",   {"m^2/Vs": 1e4}),
    "tau_n":  ("s",         {"ms": 1e-3, "us": 1e-6, "ns": 1e-9}),
    "tau_p":  ("s",         {"ms": 1e-3, "us": 1e-6, "ns": 1e-9}),
    "band_gap": ("eV",      {"meV": 1e-3}),
}


def canonical_unit(name: str):
    """The unit a property is stored and compared in, or None if unconstrained."""
    entry = PROPERTY_UNITS.get(name)
    return None if entry is None else entry[0]


def to_canonical(name: str, value: float, unit: str) -> float:
    """Convert a value into the property's canonical unit, or refuse.

    Refusing is the whole point. A property this package knows how to feed to a
    solver may only arrive in a unit this package knows how to read; anything
    else is a number whose magnitude nobody has established.
    """
    entry = PROPERTY_UNITS.get(name)
    if entry is None:
        return value
    canonical, aliases = entry
    if unit == canonical:
        return value
    if unit in aliases:
        return value * aliases[unit]
    raise CandidateError(
        f"{name}: unit {unit!r} is not a unit of this property. Give it in "
        f"{canonical}" + (f" or {', '.join(sorted(aliases))}" if aliases
                          else "") +
        " — the value would otherwise reach the solver as a bare number whose "
        "magnitude nobody has established")


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
        if not isinstance(self.unit, str) or not self.unit.strip():
            # `unit = 123` was accepted before this line: not merely useless,
            # but indistinguishable from a real unit to every check downstream.
            raise CandidateError(
                f"a value of {self.value} with unit {self.unit!r} is not a "
                "property; the unit must be a non-empty string")
        if not isinstance(self.valid_range, Mapping):
            raise CandidateError(
                f"valid_range={self.valid_range!r}: must be a table of "
                "condition -> [low, high]")
        for condition, span in self.valid_range.items():
            if condition not in CONDITIONS:
                # An unenforceable range is worse than none: it looks like a
                # guarantee and is not one. Only conditions a run can actually
                # supply may be written, so the schema cannot hold a promise
                # nothing checks. Reported in review.
                raise CandidateError(
                    f"valid_range names {condition!r}, which no run can "
                    f"supply. Known conditions: {', '.join(sorted(CONDITIONS))}")
            if (not isinstance(span, (list, tuple)) or len(span) != 2
                    or not all(isinstance(v, (int, float))
                               and not isinstance(v, bool)
                               and math.isfinite(v) for v in span)
                    or span[0] > span[1]):
                raise CandidateError(
                    f"valid_range[{condition!r}]={span!r}: must be "
                    "[low, high] with low <= high")
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

    def in_canonical(self, name: str) -> "Property":
        """This property converted into ``name``'s canonical unit.

        The uncertainty goes through the SAME conversion. Converting the value
        and leaving the spread behind would turn +/- 200 cm^2/Vs into +/- 200
        m^2/Vs and silently widen a bound by four orders of magnitude, which
        would corrupt exactly the screening decisions the spread exists for.
        """
        canonical = canonical_unit(name)
        if canonical is None or self.unit == canonical:
            return self
        factor = to_canonical(name, 1.0, self.unit)
        return Property(
            value=self.value * factor, unit=canonical, basis=self.basis,
            uncertainty=(None if self.uncertainty is None
                         else self.uncertainty * factor),
            source=self.source, valid_range=self.valid_range)

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
        canonical = {}
        for name, prop in self.properties.items():
            if not isinstance(prop, Property):
                raise CandidateError(
                    f"{self.identifier}.{name} is not a Property")
            try:
                canonical[name] = prop.in_canonical(name)
            except CandidateError as exc:
                raise CandidateError(f"{self.identifier}: {exc}") from None
        # Normalised HERE, once, so that every consumer downstream — screening,
        # device overrides, the artifact lock file — is comparing numbers in
        # one unit. Converting at each use site would mean a use site that
        # forgot to convert, which is the bug this whole table exists to close.
        object.__setattr__(self, "properties", canonical)

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


def snapshot(candidate: Candidate) -> Dict[str, Any]:
    """The whole candidate as plain data, in canonical units.

    The artifact recorded the identifier and a 12-character fingerprint and
    nothing else, so composition, structure, and every property's source,
    basis, unit, uncertainty and validity range lived only in the user's file.
    Move or edit that file and the run could no longer say what it was based
    on — while still carrying a fingerprint that implied it could. Reported in
    re-review. This is what gets written next to the result and checksummed
    with it.
    """
    return {"identifier": candidate.identifier,
            "composition": candidate.composition,
            "structure": candidate.structure,
            "dimensionality": candidate.dimensionality,
            "notes": candidate.notes,
            "fingerprint": fingerprint(candidate),
            "properties": {
                name: {"value": prop.value, "unit": prop.unit,
                       "basis": prop.basis, "uncertainty": prop.uncertainty,
                       "source": prop.source,
                       "valid_range": {k: list(v)
                                       for k, v in prop.valid_range.items()}}
                for name, prop in sorted(candidate.properties.items())}}


def fingerprint(candidate: Candidate) -> str:
    """A hash of everything the candidate CLAIMS, not just what it computes to.

    Two candidates with the same four nominal numbers produced the same run id
    and the second silently overwrote the first's directory and provenance —
    measured in review: SYNTH-A and SYNTH-B both landed on 8265fbf2116f. The
    artifact id hashes the effective inputs, and the effective inputs are bare
    floats, so identity, source, basis, uncertainty and validity range all
    vanished from it.

    The identifier alone would not be enough either: the same id can be
    re-issued with a better measurement behind it, and those are two different
    pieces of evidence that must not share a directory. So the whole record is
    hashed.
    """
    import hashlib

    payload = {"identifier": candidate.identifier,
               "composition": candidate.composition,
               "structure": candidate.structure,
               "dimensionality": candidate.dimensionality,
               "properties": {
                   name: {"value": prop.value, "unit": prop.unit,
                          "basis": prop.basis,
                          "uncertainty": prop.uncertainty,
                          "source": prop.source,
                          "valid_range": {k: list(v) if isinstance(v, (list,
                                                                      tuple))
                                          else v
                                          for k, v in prop.valid_range.items()}}
                   for name, prop in sorted(candidate.properties.items())}}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# --- screening ------------------------------------------------------------

@dataclass(frozen=True)
class Threshold:
    """One hard requirement: a property, a direction, a bound."""

    prop: str
    op: str                    # ">=" or "<="
    bound: float
    unit: str = ""             # the unit the bound is expressed in, canonical

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


def judge(candidate: Candidate, threshold: Threshold,
          conditions: Optional[Mapping[str, float]] = None,
          require_conditions: bool = True) -> Judgement:
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

    if canonical_unit(threshold.prop) is None and not threshold.unit:
        # A bare number against an unregistered property means nothing: there
        # is no canonical unit for it to be implicitly IN. `hardness = 50 cm`
        # against `hardness >= 100` was a numeric FAIL that could equally have
        # been a pass in metres. Reported in re-review.
        return Judgement(
            threshold, "undecided",
            f"{threshold.prop} has no canonical unit, so a bare bound of "
            f"{threshold.bound:g} has no meaning. Write the unit explicitly, "
            f"matching the candidate's {prop.unit!r}")

    if (canonical_unit(threshold.prop) is None and threshold.unit
            and threshold.unit != prop.unit):
        # Like against like, or not at all. Without a conversion table for
        # this property the only safe comparison is an identical unit string.
        return Judgement(
            threshold, "undecided",
            f"{threshold.prop} is recorded in {prop.unit!r} and the bound is "
            f"in {threshold.unit!r}; this property has no unit table, so the "
            "two cannot be compared without inventing a conversion")

    if not conditions and prop.valid_range and require_conditions:
        return Judgement(
            threshold, "undecided",
            f"{threshold.prop} is only stated valid for "
            f"{', '.join(sorted(prop.valid_range))}, and no condition was "
            "given; a range nobody supplied a value for cannot be assumed met")

    if conditions:
        breaches = [b for b in out_of_range(
            Candidate(candidate.identifier, {threshold.prop: prop}), conditions)]
        if breaches:
            # A property outside its stated validity does not describe the
            # material under these conditions, so it cannot support a verdict
            # about them. Previously `screen` had no notion of conditions at
            # all and every such property voted unconditionally.
            return Judgement(threshold, "undecided", breaches[0])

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
              f"({prop.basis}) against {threshold.op} {threshold.bound:g}"
              f"{' ' + threshold.unit if threshold.unit else ''}")
    if verdict == "undecided":
        detail += " - the uncertainty straddles the bound"
    return Judgement(threshold, verdict, detail)


def screen(candidates, thresholds, conditions=None,
           require_conditions: bool = True) -> Dict[str, Any]:
    """Apply every threshold to every candidate. Nothing is dropped silently.

    ``require_conditions`` defaults to TRUE: a ranged property with no
    condition supplied is `undecided`, not a pass. It defaulted to False and
    only the CLI passed the safe value, so anyone calling this library
    directly got the unsafe behaviour — safe-by-default means the DEFAULT is
    safe, not that one caller remembers. Reported in re-review.

    A screen that returns only the survivors hides its own selectivity: you
    cannot tell a threshold that removed one candidate from one that removed
    forty, and you certainly cannot tell which ones were thrown out for want of
    a measurement rather than for being unsuitable. Every candidate comes back
    with its verdict and the reasons behind it.
    """
    thresholds = tuple(thresholds)
    results = []
    for candidate in candidates:
        judgements = [judge(candidate, t, conditions, require_conditions)
                      for t in thresholds]
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


def no_duplicate_keys(pairs):
    """Refuse a JSON object that names the same key twice.

    ``json.loads`` keeps the LAST value silently, so a file listing mu_n twice
    quietly discards the first. For a configuration that decides what physics
    runs, "the last one wins" is not a resolution — it is a coin flip nobody
    sees. TOML already refuses this; JSON had to be told. Reported in review.
    """
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise CandidateError(
                f"{key!r} is given twice in the same table; the second would "
                "silently replace the first")
        seen[key] = value
    return seen


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
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
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
    """``mu_n>=1000`` or ``mu_n>=0.1 m^2/Vs`` into a canonical Threshold.

    A bound may carry a unit, and if it does it is converted the same way a
    candidate's value is. A bound with NO unit is taken as already canonical —
    and the Threshold records which unit that was, so a screen report can say
    what the comparison was actually made in rather than leaving the reader to
    assume. Comparing a bound against a value in a different unit was the
    reported defect; silence about the unit is how it stayed invisible.
    """
    for op in (">=", "<="):
        if op in text:
            name, _, rest = text.partition(op)
            name = name.strip()
            rest = rest.strip()
            if not name:
                raise CandidateError(f"{text!r}: no property named")

            number, unit = rest, ""
            for cut in range(len(rest), 0, -1):
                try:
                    float(rest[:cut])
                except ValueError:
                    continue
                number, unit = rest[:cut], rest[cut:].strip()
                break
            try:
                bound = float(number)
            except ValueError:
                raise CandidateError(
                    f"{text!r}: {rest!r} is not a number") from None
            canonical = canonical_unit(name)
            if unit:
                if canonical is None:
                    # An unregistered property has no canonical unit, so there
                    # is nothing to convert INTO and no way to know that "cm"
                    # and "m" are related. Comparing them as bare numbers gave
                    # `hardness = 50 cm` a PASS against `hardness >= 1 m`.
                    # Reported in re-review. The bound keeps its unit and
                    # `judge` requires the candidate to state the same one.
                    return Threshold(name, op, bound, unit=unit)
                bound = to_canonical(name, bound, unit)
            return Threshold(name, op, bound, unit=canonical or "")
    raise CandidateError(
        f"{text!r}: expected NAME>=VALUE or NAME<=VALUE. A hard screen is a "
        "bound; anything softer belongs in ranking")

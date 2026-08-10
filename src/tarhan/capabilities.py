"""Capability registry — what the engine can actually do, and where it stops.

A command existing does not mean the physics behind it is validated. This module
is where that difference is written down, so `tarhan capabilities` can tell a
user today's limit instead of a promise.

Two design decisions worth stating, because both are load-bearing.

**Dimension and time are separate fields, and the id is DERIVED from them.**
The obvious scheme puts the dimension in the identifier and stops there —
``semiconductor.pn.drift-diffusion.1d``. That scheme cannot express a difference
this repository already contains: ``models/chronoamp1d.py`` is transient and
``models/pn1d.py`` is steady-state, and under that scheme both are "1d". The
roadmap's eventual "4D" is 3D plus time, not a fourth spatial axis, so folding it
into the same slot would have made the confusion permanent. Here ``dimension``
is an int, ``time`` is steady-or-transient, and :attr:`Capability.id` is computed
from them — an id cannot be hand-written, so it cannot disagree with its own
fields.

**Why that mattered before a single id was written.** Capability ids go into run
manifests, which are provenance. Provenance cannot be renamed retroactively: an
old run would then cite an identifier that no longer exists. The cost of getting
this wrong is paid later, by data already on disk — which is exactly the kind of
cost that never appears in the diff that caused it.

The record shape follows the roadmap's §3 list: id, status, dimension, model
family; required inputs and produced quantities; physical and numerical limits;
validation evidence with the measured error; and, for anything not runnable, the
reason plus the condition that would unlock it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

#: A capability is in exactly one of these states.
#:
#: ``validated``    an independent oracle or Layer-0 test has measured its limit
#: ``experimental`` it runs, but is not at the validation level claimed elsewhere
#: ``blocked``      a known prerequisite is missing; deliberately not runnable
#: ``planned``      the interface and data model foresee it; no engine yet
STATUSES = ("validated", "experimental", "blocked", "planned")

#: The time axis, kept separate from the spatial dimension on purpose.
#:
#: ``steady``    one solve, no time derivative
#: ``transient`` time-stepped; the large-signal time derivative is solved
#: ``ac``        small-signal harmonic response linearised about an operating
#:               point — neither of the other two, and the distinction is not
#:               cosmetic: an AC solve needs a complex-valued linear system and,
#:               for the MOS C-V case, a circuit node to hold the terminal
#:
#: ``ac`` was added while filling in the first records. The MOS capacitor stage
#: is blocked precisely because it is a small-signal AC problem, and a two-value
#: axis could not say that — it could only have said "steady", which is false,
#: or "transient", which is a different physics. A registry whose vocabulary
#: cannot express why something is blocked would defeat its own purpose.
TIME_AXES = ("steady", "transient", "ac")

#: 3 is the highest spatial dimension the roadmap contemplates. A record above
#: it is a typo, not a plan.
MAX_DIMENSION = 3

#: Statuses describing something a user cannot run today, which therefore owe an
#: explanation rather than a status word.
_NOT_RUNNABLE = ("blocked", "planned")


#: Inputs a run actually reports, and therefore the only names an envelope may
#: use. Mirrors ``candidate.CONDITIONS`` for the same reason: a bound over
#: something nothing supplies cannot be checked.
ENVELOPE_INPUTS = ("bias_v",)


#: What is known about ONE metric at ONE operating point.
#:
#: `measured-point` — a test compares this metric at exactly this value.
#: `interpolated`   — it sits between two measured points. Defensible for a
#:                    smooth I-V curve, and NOT the same claim as measured.
#: `outside`        — beyond every measured point for this metric.
#: `unverified`     — this metric has no coverage record at all.
COVERAGE_KINDS = ("measured-point", "interpolated", "outside", "unverified")


@dataclass(frozen=True)
class MetricCoverage:
    """Which biases a single metric was actually compared at.

    The envelope was metric-BLIND: one interval covered a capability whose
    potential is checked at 0.0 and 0.30 V and whose current is checked at
    0.20, 0.30 and 0.40 V. A run at 0.40 V therefore reported the same bare
    `inside` as one at 0.0 V, though they rest on different evidence — and the
    capability's own `limits` field simultaneously said the hole current is
    unsupported at 0.20 V. Reported in re-review.
    """

    metric: str
    points: Tuple[float, ...]
    evidence: str

    def __post_init__(self):
        if not self.metric or not self.metric.strip():
            raise CapabilityError("a coverage record must name its metric")
        if not self.points:
            raise CapabilityError(
                f"{self.metric}: coverage with no measured points is not "
                "coverage")
        for value in self.points:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise CapabilityError(
                    f"{self.metric}: measured point {value!r} is not a finite "
                    "number")
        if not self.evidence.strip():
            raise CapabilityError(
                f"{self.metric}: coverage must point at the evidence file")
        object.__setattr__(self, "points",
                           tuple(sorted(float(v) for v in self.points)))

    def kind_at(self, value: float) -> str:
        if any(abs(value - point) <= 1e-12 for point in self.points):
            return "measured-point"
        if self.points[0] <= value <= self.points[-1]:
            return "interpolated"
        return "outside"


class CapabilityError(ValueError):
    """A capability record that cannot be true.

    Raised at construction, not at use. A registry that accepts a nonsense
    record and reports it later is worse than one that refuses it, because the
    nonsense then reaches the user wearing the registry's authority.
    """


@dataclass(frozen=True)
class Evidence:
    """One measured result, and the test that produces it.

    ``measured`` is deliberately a string. These are heterogeneous — an absolute
    error, a ratio, an ideality range — and forcing them into a float would drop
    the units and the shape of the claim, which is the part a reader needs.
    """

    claim: str
    measured: str
    test: str          # repo-relative path to the test that produces it

    def __post_init__(self) -> None:
        for name in ("claim", "measured", "test"):
            if not getattr(self, name).strip():
                raise CapabilityError(f"Evidence.{name} must not be empty")


@dataclass(frozen=True)
class Capability:
    """One thing the engine can — or explicitly cannot — do."""

    domain: str
    family: str
    dimension: int
    time: str
    status: str
    source: str = ""                       # repo-relative module path
    inputs: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    limits: Tuple[str, ...] = ()
    evidence: Tuple[Evidence, ...] = ()
    reason: str = ""                       # why it is blocked or merely planned
    needs: str = ""                        # what would unlock it
    does_not_mean: str = ""                # the misreading to head off
    #: The input range the evidence actually covers, as name -> (low, high).
    #: MACHINE-READABLE on purpose. A limit written in prose is a limit no run
    #: can check itself against, and a 0.2 V solve was producing an artifact
    #: marked `validated` and `converged` while this registry's own prose said
    #: that very result does not converge. Reported in re-review.
    #: name -> a UNION of closed intervals. One interval per input could not
    #: describe the evidence: 2D steady is validated at equilibrium (0 V), NOT
    #: reproducible at 0.2 V, and validated again over 0.3-0.5 V. A single
    #: [0.30, 0.50] therefore reported `outside-validated-range` for the
    #: best-validated point in the capability. Reported in re-review.
    envelope: Mapping[str, Tuple[Tuple[float, float], ...]] = field(
        default_factory=dict)
    #: WHICH device the envelope describes. Required whenever an envelope is
    #: given, because the evidence identity was otherwise absent from the data
    #: model: the 2D envelope was taken from the DEVSIM-mesh stage
    #: (Na/Nd 1e18, mu 400/200, partial top contact) and applied to the CLI's
    #: generated device (1e16, 1350/480, full contacts) — a different device
    #: entirely. Reported in re-review, and a stronger finding than the
    #: mis-attributed comment I had called it.
    envelope_basis: str = ""
    #: Per-metric measured points. Without it, one status word applies the
    #: evidence for the best-covered metric to the whole artifact.
    coverage: Tuple[MetricCoverage, ...] = ()

    def coverage_report(self, inputs) -> Dict[str, str]:
        """Per metric, what this run's operating point actually rests on."""
        value = inputs.get("bias_v")
        if value is None:
            return {c.metric: "unverified" for c in self.coverage}
        return {c.metric: c.kind_at(float(value)) for c in self.coverage}

    def validation_profile(self) -> str:
        """A hash of the whole evidence claim: envelope, basis and coverage.

        `envelope_basis` alone was free text with no machine link to anything —
        an artifact could not answer "which evidence profile said inside?"
        after the fact. This id can be recorded in a run and compared later.
        """
        import hashlib

        payload = json.dumps(
            {"envelope": {k: [list(p) for p in v]
                          for k, v in sorted(self.envelope.items())},
             "basis": self.envelope_basis,
             "coverage": [{"metric": c.metric, "points": list(c.points),
                           "evidence": c.evidence}
                          for c in sorted(self.coverage,
                                          key=lambda c: c.metric)]},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _validate_envelope(self) -> None:
        """The envelope schema, enforced. It was not, and accepted an empty
        interval list, a reversed [1, 0], a NaN bound, and an input name no
        run reports — each of which silently means "everything is inside"."""
        # Frozen for real. `Capability` is a frozen dataclass but `envelope`
        # was an ordinary dict, so `cap.envelope["bias_v"] = ()` mutated the
        # global registry past every check above. Reported in re-review.
        object.__setattr__(self, "envelope", MappingProxyType(
            {name: tuple(tuple(float(v) for v in pair) for pair in intervals)
             for name, intervals in self.envelope.items()}))
        object.__setattr__(self, "envelope_basis", self.envelope_basis.strip())
        object.__setattr__(self, "coverage", tuple(self.coverage))

        if self.envelope and not self.envelope_basis:
            raise CapabilityError(
                f"{self.id} gives an envelope without naming the device it "
                "was measured on; evidence from one device says nothing about "
                "another")
        for name, intervals in self.envelope.items():
            if name not in ENVELOPE_INPUTS:
                raise CapabilityError(
                    f"{self.id}: envelope names {name!r}, which no run "
                    f"reports. Known inputs: {', '.join(ENVELOPE_INPUTS)}")
            if not intervals:
                raise CapabilityError(
                    f"{self.id}: envelope for {name} has no intervals, which "
                    "would put every value outside it while reading as a "
                    "validated range")
            for pair in intervals:
                if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                        or not all(isinstance(v, (int, float))
                                   and not isinstance(v, bool)
                                   and math.isfinite(v) for v in pair)):
                    raise CapabilityError(
                        f"{self.id}: envelope interval {pair!r} for {name} "
                        "must be two finite numbers")
                if pair[0] > pair[1]:
                    raise CapabilityError(
                        f"{self.id}: envelope interval {pair!r} for {name} is "
                        "reversed, so nothing can ever fall inside it")

    def outside_envelope(self, inputs) -> Tuple[str, ...]:
        """Which inputs fall outside the range the evidence covers.

        Empty means the run sits inside what was validated. A non-empty result
        does NOT mean the answer is wrong — it means nothing here has
        established that it is right, which is a different claim and the one
        an artifact has to be able to make.
        """
        out = []
        for name, intervals in sorted(self.envelope.items()):
            if name not in inputs:
                # "No value supplied" is not "inside". Skipping the check
                # because the record is absent is the same shape as the
                # checksum map that disabled itself when an entry was deleted
                # — the fourth appearance of it. Reported in re-review.
                out.append(f"{name} is part of the validated envelope and the "
                           "run did not report a value for it")
                continue
            value = float(inputs[name])
            if not any(low <= value <= high for low, high in intervals):
                covered = " or ".join(f"[{low:g}, {high:g}]"
                                      for low, high in intervals)
                out.append(f"{name}={value:g} is outside the validated "
                           f"{covered}")
        return tuple(out)

    def envelope_json(self):
        """The envelope as data, for a machine format.

        Emitted as prose — "bias_v in [0.3, 0.5]" — it was visible and still
        unusable: an automated consumer had to parse a sentence, which is the
        one thing this project's output contract exists to prevent. Reported
        in re-review.
        """
        return {name: {"intervals": [list(pair) for pair in intervals]}
                for name, intervals in sorted(self.envelope.items())}

    def __post_init__(self) -> None:
        for name in ("domain", "family"):
            value = getattr(self, name)
            if not value or value != value.strip().lower() or " " in value:
                raise CapabilityError(
                    f"{name} must be a non-empty lowercase token without spaces; "
                    f"got {value!r}")

        # bool is an int subclass, and True would silently become dimension 1.
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise CapabilityError(
                f"dimension must be an int, got {type(self.dimension).__name__}")
        if not 0 <= self.dimension <= MAX_DIMENSION:
            raise CapabilityError(
                f"dimension must be 0..{MAX_DIMENSION}, got {self.dimension}")

        if self.time not in TIME_AXES:
            raise CapabilityError(
                f"time must be one of {TIME_AXES}, got {self.time!r}. The time "
                "axis is separate from the spatial dimension on purpose")
        if self.status not in STATUSES:
            raise CapabilityError(
                f"status must be one of {STATUSES}, got {self.status!r}")

        if self.status == "validated":
            if not self.evidence:
                raise CapabilityError(
                    f"{self.id} is marked validated with no evidence; that is the "
                    "one claim this registry exists to make impossible")
            if not self.source:
                raise CapabilityError(
                    f"{self.id} is marked validated but names no source module")
        if self.status == "experimental" and not self.source:
            raise CapabilityError(
                f"{self.id} is experimental but names no source module")

        if self.status in _NOT_RUNNABLE:
            for name in ("reason", "needs"):
                if not getattr(self, name).strip():
                    raise CapabilityError(
                        f"{self.id} is {self.status} but has no {name}; a status "
                        "word without an explanation is how a scope decision "
                        "decays into folklore")
            if self.evidence:
                raise CapabilityError(
                    f"{self.id} is {self.status} but carries evidence; nothing "
                    "unrunnable can have been measured")
        if self.status == "planned" and self.source:
            raise CapabilityError(
                f"{self.id} is planned but names the source module {self.source!r}; "
                "planned means no engine exists")
        if self.status == "blocked" and not self.does_not_mean.strip():
            raise CapabilityError(
                f"{self.id} is blocked but does not say what that does NOT mean. "
                "Readers hear 'blocked' as 'unsupported forever'; say otherwise")

        self._validate_envelope()

    @property
    def id(self) -> str:
        """The identifier, computed rather than stored.

        Nothing accepts an id as input, so no record can carry one that
        contradicts its own dimension or time axis.
        """
        return f"{self.domain}.{self.family}.{self.dimension}d.{self.time}"

    @property
    def runnable(self) -> bool:
        return self.status not in _NOT_RUNNABLE

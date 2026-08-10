"""Layer-0 for the capability registry — including the seam where it can lie.

The registry repeats, in Python, what ``docs/DESIGN-2D.md`` §5 says in prose. Two
sources for one fact is the failure this repository keeps catching in itself: the
version number lived in two files and only one was on the release checklist; the
§5 table was updated while the paragraph above it went stale. Adding a third copy
of the 2D story without pinning it would be repeating that mistake knowingly.

So the drift check is not "both mention 2D-2". It is: the measured numbers the
registry quotes must be findable in §5, character for character. Re-measure one
and forget the other, and this fails loudly instead of quietly.

What this cannot catch, stated because the boundary matters: it cannot tell
whether either number is RIGHT. Both could be wrong together and this file would
pass. Only re-running the validation tests settles that — which is why every
piece of evidence names the test that produces it, and why those paths are
checked to exist.
"""
import pathlib

import pytest

from tarhan.capabilities import STATUSES, TIME_AXES, Capability, CapabilityError
from tarhan.capability_registry import CapabilityNotFound, all_capabilities, get

REPO = pathlib.Path(__file__).resolve().parents[2]
DESIGN_2D = REPO / "docs/DESIGN-2D.md"

#: Which registry record each §5 stage belongs to. A stage in the document with
#: no entry here fails the coverage test rather than being silently ignored, so
#: adding a stage becomes a decision instead of an omission.
STAGE_TO_CAPABILITY = {
    "2D-0": "semiconductor.pn.drift-diffusion.2d.steady",
    "2D-1": "semiconductor.pn.drift-diffusion.2d.steady",
    "2D-2": "semiconductor.pn.drift-diffusion.2d.steady",
    "2D-3′": "semiconductor.pn.drift-diffusion.2d.steady",
    "2D-3": "semiconductor.mos.capacitance.2d.ac",
    "2D-4": "semiconductor.mosfet.drift-diffusion.2d.steady",
}

#: Numbers the registry quotes that §5 must also contain, verbatim. Ranges are
#: pinned as their endpoints rather than as "a-b": the document writes them with
#: an en dash and the registry with a hyphen, and pinning punctuation would make
#: this fail for a reason that has nothing to do with the physics.
PINNED_NUMBERS = ("1.8e-15", "0.953719", "3.350171660e-12", "8281",
                  "1.000000000",
                  # The 2D-2 ideality endpoints, corrected after the first
                  # cross-oracle CI run: the old 1.0119-1.0134 / 1.0114 range
                  # included a 0.2 V point that is not converged. Both the
                  # registry and §5 now quote the 0.3-0.5 V range.
                  "1.0120", "1.0126", "1.0121",
                  # ...and the numbers behind the exclusion, so the correction
                  # cannot be quietly rolled back to a prettier range.
                  "1.063", "1.0347")


def _stage_rows():
    """(stage, status_word) for each ``| 2D-… |`` row of the §5 table."""
    text = DESIGN_2D.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("| 2D-"):
            continue
        stage = line.split("|")[1].strip().replace("~~", "")
        status = "DONE" if "**DONE**" in line else (
            "BLOCKED" if "BLOCKED" in line else "?")
        rows.append((stage, status))
    return rows


def _registry_prose():
    parts = []
    for cap in all_capabilities():
        parts.append(cap.does_not_mean)
        parts.append(cap.reason)
        for ev in cap.evidence:
            parts.extend((ev.claim, ev.measured))
    return " ".join(parts)


def test_the_table_still_looks_like_a_table():
    """If §5's format changed, every check below would pass vacuously."""
    rows = _stage_rows()
    assert len(rows) >= 5, f"only {len(rows)} stage rows found; has §5 changed?"
    assert {s for _, s in rows} <= {"DONE", "BLOCKED"}, (
        "a stage row is neither DONE nor BLOCKED; teach this test the new word "
        "rather than letting the row be ignored")


def test_every_documented_stage_maps_to_a_capability():
    """A stage added to the document must be a decision about the registry too."""
    unmapped = [s for s, _ in _stage_rows() if s not in STAGE_TO_CAPABILITY]
    assert not unmapped, (
        f"§5 documents stages with no registry mapping: {unmapped}. Add them to "
        "STAGE_TO_CAPABILITY — deciding they do not belong is fine, deciding by "
        "forgetting is not")


@pytest.mark.parametrize("stage,status", _stage_rows())
def test_stage_status_agrees_with_the_registry(stage, status):
    """DONE must not describe a capability the registry calls blocked."""
    cap = get(STAGE_TO_CAPABILITY[stage])
    if status == "DONE":
        assert cap.status == "validated", (
            f"§5 marks {stage} DONE but {cap.id} is {cap.status!r}")
    else:
        assert cap.status == "blocked", (
            f"§5 marks {stage} BLOCKED but {cap.id} is {cap.status!r}")


def test_a_blocked_record_points_at_its_stage():
    """The registry gives the short reason; §5 holds the long one. Link them."""
    for stage, status in _stage_rows():
        if status != "BLOCKED":
            continue
        cap = get(STAGE_TO_CAPABILITY[stage])
        assert stage in cap.reason, (
            f"{cap.id} is blocked but its reason never names stage {stage}, so a "
            "reader cannot find the explanation that justifies it")


@pytest.mark.parametrize("number", PINNED_NUMBERS)
def test_measured_numbers_appear_in_both_places(number):
    """The anti-drift check, and the reason the registry is safe to keep."""
    doc = DESIGN_2D.read_text(encoding="utf-8")
    assert number in doc, f"{number} is quoted by the registry but not in §5"
    assert number in _registry_prose(), (
        f"{number} is pinned but no longer appears in the registry; if it was "
        "re-measured, update both places and this list")


def test_every_evidence_names_a_test_that_exists():
    """Evidence pointing at a deleted test is a claim with no way to re-check it."""
    missing = [(c.id, e.test) for c in all_capabilities() for e in c.evidence
               if not (REPO / e.test).exists()]
    assert not missing, f"evidence names tests that do not exist: {missing}"


def test_every_source_module_exists():
    missing = [(c.id, c.source) for c in all_capabilities()
               if c.source and not (REPO / "src/tarhan" / c.source).exists()]
    assert not missing, f"capabilities name modules that do not exist: {missing}"


def test_ids_are_unique():
    ids = [c.id for c in all_capabilities()]
    assert len(ids) == len(set(ids)), "two capabilities derive the same id"


def test_the_time_axis_separates_what_the_dimension_cannot():
    """The decision this schema exists for, asserted rather than described.

    chronoamp1d is transient and pn1d is steady-state. Under a scheme that put
    only the dimension in the id, both would be '1d' and the difference would be
    unsayable.
    """
    chrono = get("electrochemistry.chronoamperometry.1d.transient")
    pn1d = get("semiconductor.pn.drift-diffusion.1d.steady")
    assert chrono.dimension == pn1d.dimension == 1
    assert chrono.time != pn1d.time
    assert chrono.id != pn1d.id


def test_the_roadmaps_4d_is_recorded_as_3d_plus_time():
    """'4D' must never appear as a dimension. It is 3D plus a time axis."""
    assert all(c.dimension <= 3 for c in all_capabilities())
    four_d = get("semiconductor.device.drift-diffusion.3d.transient")
    assert four_d.dimension == 3 and four_d.time == "transient"
    assert four_d.status == "planned"


def test_unknown_id_raises_rather_than_returning_nothing():
    with pytest.raises(CapabilityNotFound):
        get("semiconductor.nope.2d.steady")


# --- schema guards: the records the registry must refuse -------------------

def _valid(**over):
    base = dict(domain="semiconductor", family="pn.drift-diffusion", dimension=1,
                time="steady", status="planned", reason="r", needs="n")
    base.update(over)
    return base


@pytest.mark.parametrize("over,why", [
    (dict(status="validated", source="models/pn1d.py", reason="", needs=""),
     "validated with no evidence"),
    (dict(status="blocked", does_not_mean=""), "blocked without does_not_mean"),
    (dict(dimension=4), "dimension above 3"),
    (dict(dimension=True), "bool as dimension"),
    (dict(time="4d"), "4d smuggled onto the time axis"),
    (dict(source="models/pn1d.py"), "planned but names a module"),
    (dict(domain="Semiconductor"), "non-lowercase domain"),
])
def test_the_schema_refuses_a_record_that_cannot_be_true(over, why):
    with pytest.raises(CapabilityError):
        Capability(**_valid(**over))


def test_statuses_and_axes_are_what_the_registry_uses():
    assert {c.status for c in all_capabilities()} <= set(STATUSES)
    assert {c.time for c in all_capabilities()} <= set(TIME_AXES)

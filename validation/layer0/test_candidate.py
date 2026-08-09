"""Candidates: the schema, and the two behaviours that make it not a spreadsheet.

Roadmap §5.1 — "a candidate is not free text and not a single score". Most of
what is asserted here is the refusals, because a candidate schema earns its
place by what it will NOT accept: a number with no unit, a number whose origin
nobody recorded, a "measured" value with nothing to check it against.

Two behaviours carry the real weight and each has its own section below:

* **Uncertainty can make a threshold undecidable.** A screen that resolves
  every borderline case in one direction produces a shortlist whose length is
  a property of the rounding. `undecided` is a third answer, not a soft fail.
* **Applicability is stated as what is MISSING.** The missing names are the
  measurements somebody would have to make; "unsuitable" throws that away.

**Every value in this file is synthetic and obviously so.** Real property
values would be unverifiable claims — physics_verify is unavailable this
session (the physicist MCP server is disconnected) — and a test suite is the
last place a made-up band gap should acquire the look of a citation. Nothing
here asserts anything about any real material; the numbers exist only to
exercise comparisons, and the identifiers say SYNTH so nobody mistakes them.
"""
import json
from pathlib import Path

import pytest

from tarhan.candidate import (BASES, Candidate, CandidateError, Property,
                              Threshold, applicability, device_overrides,
                              fingerprint, judge, load_candidates, out_of_range,
                              parse_threshold, screen)

PN1D = "semiconductor.pn.drift-diffusion.1d.steady"


def prop(value, unit="cm^2/Vs", basis="computed", **kw):
    return Property(value=value, unit=unit, basis=basis, **kw)


def synth(identifier="SYNTH-A", **properties):
    return Candidate(identifier=identifier,
                     properties=properties or {"mu_n": prop(1000.0)})


# --- what a property refuses to be ----------------------------------------

def test_a_number_without_a_unit_is_not_a_property():
    """1350 is not a mobility. The unit is not metadata."""
    with pytest.raises(CandidateError, match="not a property"):
        Property(value=1350.0, unit="", basis="computed")


@pytest.mark.parametrize("basis", ["unknown", "guess", "", "MEASURED", None])
def test_a_value_must_say_where_it_came_from(basis):
    """A number whose origin nobody recorded cannot be weighed against one
    that was measured, and "unknown" is not a fourth basis — it is a missing
    property, which must be absent rather than present-and-vague."""
    with pytest.raises(CandidateError, match="basis"):
        Property(value=1.0, unit="eV", basis=basis)


def test_a_measured_value_must_name_its_source():
    """The one combination that actively misleads: the strongest claim
    available, with nothing behind it to check."""
    with pytest.raises(CandidateError, match="must name its source"):
        Property(value=1.12, unit="eV", basis="measured")

    Property(value=1.12, unit="eV", basis="measured", source="synthetic")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1350"])
def test_a_value_that_is_not_a_finite_number_is_refused(bad):
    with pytest.raises(CandidateError):
        Property(value=bad, unit="eV", basis="computed")


@pytest.mark.parametrize("bad", [-1.0, float("nan"), True])
def test_an_uncertainty_that_cannot_be_a_spread_is_refused(bad):
    with pytest.raises(CandidateError):
        Property(value=1.0, unit="eV", basis="computed", uncertainty=bad)


def test_a_candidate_with_no_properties_cannot_be_reasoned_about():
    with pytest.raises(CandidateError, match="no properties"):
        Candidate(identifier="SYNTH-EMPTY", properties={})


def test_a_candidate_must_be_identifiable():
    with pytest.raises(CandidateError, match="canonical identifier"):
        Candidate(identifier="  ", properties={"mu_n": prop(1.0)})


# --- uncertainty can refuse to decide -------------------------------------

def test_a_value_with_no_uncertainty_is_a_point():
    assert prop(1000.0).interval == (1000.0, 1000.0)


@pytest.mark.parametrize("value,spread,bound,expected", [
    # comfortably above, comfortably below, and the interesting middle
    (1200.0, 50.0, 1000.0, "pass"),
    (800.0, 50.0, 1000.0, "fail"),
    (1000.0, 50.0, 1000.0, "undecided"),
    (960.0, 50.0, 1000.0, "undecided"),
    (1000.0, None, 1000.0, "pass"),      # exactly on the bound, no doubt
    (999.0, None, 1000.0, "fail"),
])
def test_a_threshold_is_undecided_when_the_spread_straddles_it(
        value, spread, bound, expected):
    """The behaviour this module exists for.

    Note the fifth case against the third: the SAME nominal value, 1000
    against ">= 1000", passes when it is known exactly and is undecided when
    it carries +/- 50. That is the whole point — the verdict depends on what is
    known, not only on what was written in the middle column.
    """
    candidate = synth(mu_n=prop(value, uncertainty=spread))
    assert judge(candidate, Threshold("mu_n", ">=", bound)).verdict == expected


@pytest.mark.parametrize("value,spread,bound,expected", [
    (800.0, 50.0, 1000.0, "pass"),
    (1200.0, 50.0, 1000.0, "fail"),
    (1000.0, 50.0, 1000.0, "undecided"),
])
def test_the_upper_bound_direction_behaves_the_same_way(
        value, spread, bound, expected):
    candidate = synth(mu_n=prop(value, uncertainty=spread))
    assert judge(candidate, Threshold("mu_n", "<=", bound)).verdict == expected


def test_a_property_nobody_recorded_is_undecided_not_failed():
    """Absent is not the same as bad, and screening it out as a failure would
    quietly rank measurement effort as a material defect."""
    result = judge(synth(), Threshold("band_gap", ">=", 1.0))
    assert result.verdict == "undecided"
    assert "not recorded" in result.detail


def test_the_reason_survives_the_verdict():
    """A screen whose output is pass/fail cannot be argued with."""
    result = judge(synth(mu_n=prop(980.0, uncertainty=60.0)),
                   Threshold("mu_n", ">=", 1000.0))
    assert "980" in result.detail and "60" in result.detail
    assert "computed" in result.detail          # the basis is part of the case
    assert "straddles" in result.detail


# --- screening reports everything -----------------------------------------

def test_every_candidate_comes_back_not_only_the_survivors():
    """A shortlist alone hides the screen's own selectivity, and hides which
    candidates were dropped for want of a measurement."""
    candidates = [synth("SYNTH-HIGH", mu_n=prop(2000.0)),
                  synth("SYNTH-LOW", mu_n=prop(100.0)),
                  synth("SYNTH-VAGUE", mu_n=prop(1000.0, uncertainty=500.0)),
                  synth("SYNTH-SILENT", eps_s=prop(1e-12, unit="F/cm"))]
    report = screen(candidates, [parse_threshold("mu_n>=1000")])
    verdicts = {r["identifier"]: r["verdict"] for r in report["results"]}
    assert verdicts == {"SYNTH-HIGH": "pass", "SYNTH-LOW": "fail",
                        "SYNTH-VAGUE": "undecided", "SYNTH-SILENT": "undecided"}


def test_one_failed_threshold_fails_the_candidate():
    """Hard thresholds are hard: they do not average."""
    candidate = synth(mu_n=prop(2000.0), mu_p=prop(10.0))
    report = screen([candidate], [parse_threshold("mu_n>=1000"),
                                  parse_threshold("mu_p>=100")])
    assert report["results"][0]["verdict"] == "fail"


def test_a_fail_outranks_an_undecided():
    """A candidate that definitely fails one bound is not rescued by being
    unclear about another."""
    candidate = synth(mu_n=prop(10.0), mu_p=prop(100.0, uncertainty=50.0))
    report = screen([candidate], [parse_threshold("mu_n>=1000"),
                                  parse_threshold("mu_p>=100")])
    assert report["results"][0]["verdict"] == "fail"


@pytest.mark.parametrize("text", ["mu_n", "mu_n>1000", "mu_n>=abc",
                                  ">=1000", "mu_n~1000"])
def test_a_threshold_that_is_not_a_hard_bound_is_refused(text):
    with pytest.raises(CandidateError):
        parse_threshold(text)


def test_a_soft_operator_is_refused_by_the_threshold_itself():
    with pytest.raises(CandidateError, match="belongs in ranking"):
        Threshold("mu_n", "~=", 1000.0)


# --- applicability names what is missing ----------------------------------

def test_a_complete_candidate_can_drive_the_model():
    full = synth(ni=prop(1e10, unit="cm^-3"), eps_s=prop(1e-12, unit="F/cm"),
                 mu_n=prop(1000.0), mu_p=prop(400.0))
    fit = applicability(full, PN1D)
    assert fit.usable and fit.missing == ()
    assert device_overrides(full, PN1D) == {
        "ni": 1e10, "eps_s": 1e-12, "mu_n": 1000.0, "mu_p": 400.0}


def test_an_incomplete_candidate_names_the_measurements_it_needs():
    fit = applicability(synth(mu_n=prop(1000.0)), PN1D)
    assert not fit.usable
    assert set(fit.missing) == {"ni", "eps_s", "mu_p"}


def test_a_gap_is_never_filled_with_a_default():
    """The failure this prevents is the quiet one: defaulting a missing
    property substitutes some OTHER material's number, and the run then
    describes a material that does not exist."""
    with pytest.raises(CandidateError, match="does not exist"):
        device_overrides(synth(mu_n=prop(1000.0)), PN1D)


def test_a_capability_that_declares_nothing_is_an_error_not_a_pass():
    """Silence must not read as 'needs nothing'."""
    with pytest.raises(CandidateError, match="must declare what it needs"):
        applicability(synth(), "semiconductor.mosfet.drift-diffusion.2d.steady")


# --- loading refuses what it cannot represent -----------------------------

SYNTHETIC_FILE = """
[SYNTH-A]
composition = "SyntheticA"
[SYNTH-A.properties.mu_n]
value = 1000.0
unit = "cm^2/Vs"
basis = "computed"
uncertainty = 50.0
"""


def _write(tmp_path, text, name="candidates.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_candidate_file_round_trips(tmp_path):
    loaded = load_candidates(_write(tmp_path, SYNTHETIC_FILE))
    assert len(loaded) == 1
    assert loaded[0].identifier == "SYNTH-A"
    assert loaded[0].properties["mu_n"].uncertainty == 50.0
    assert loaded[0].properties["mu_n"].basis == "computed"


def test_json_and_toml_describe_the_same_candidate(tmp_path):
    as_json = json.dumps({"SYNTH-A": {"composition": "SyntheticA",
                                      "properties": {"mu_n": {
                                          "value": 1000.0,
                                          "unit": "cm^2/Vs",
                                          "basis": "computed",
                                          "uncertainty": 50.0}}}})
    from_toml = load_candidates(_write(tmp_path, SYNTHETIC_FILE))
    from_json = load_candidates(_write(tmp_path, as_json, "candidates.json"))
    assert from_toml == from_json


@pytest.mark.parametrize("text,expected", [
    ('[SYNTH-A]\nproperties = {}\n', "needs a properties table"),
    ('[SYNTH-A]\n[SYNTH-A.properties.mu_n]\nvalue = 1.0\nunit = "x"\n',
     "missing basis"),
    ('[SYNTH-A]\n[SYNTH-A.properties.mu_n]\nvalue = 1.0\nbasis = "computed"\n',
     "missing unit"),
    ('[SYNTH-A]\nbogus = 1\n[SYNTH-A.properties.mu_n]\nvalue = 1.0\n'
     'unit = "x"\nbasis = "computed"\n', "unknown field"),
    ('[SYNTH-A]\n[SYNTH-A.properties.mu_n]\nvalue = 1.0\nunit = "x"\n'
     'basis = "computed"\nuncertanty = 1.0\n', "unknown field"),
])
def test_a_malformed_candidate_file_is_refused_by_name(tmp_path, text,
                                                       expected):
    """A misspelt field dropped in silence would make a candidate quietly
    weaker than the file says it is — note the `uncertanty` case, which is
    exactly how a stated uncertainty goes missing."""
    with pytest.raises(CandidateError, match=expected):
        load_candidates(_write(tmp_path, text))


def test_an_unknown_file_type_is_refused(tmp_path):
    with pytest.raises(CandidateError, match="toml or .json"):
        load_candidates(_write(tmp_path, SYNTHETIC_FILE, "candidates.yaml"))


def test_no_material_database_ships_with_the_package():
    """Deliberate, and asserted so it stays deliberate.

    Real property values written from memory would be unverifiable numbers
    wearing the authority of a package. If a curated set is ever added it must
    arrive with sources per value, and this test failing is the moment to
    argue about that rather than to notice it later.
    """
    import tarhan

    package = Path(tarhan.__file__).resolve().parent
    shipped = sorted(p.name for p in package.rglob("*")
                     if p.suffix.lower() in (".toml", ".json", ".csv"))
    assert shipped == [], f"a data file appeared in the package: {shipped}"


def test_the_bases_are_the_roadmaps_three():
    assert BASES == ("measured", "computed", "inferred")


# --- units are physics, not labels ----------------------------------------
#
# Reported in review, with a real run: ni in seconds, eps_s in eV, mu_p in
# "bananas", and the candidate came back "usable" with the bare numbers handed
# straight to the solver. Units were checked only for emptiness.

@pytest.mark.parametrize("unit", ["seconds", "eV", "kg", "bananas", "m/s", ""])
def test_a_unit_that_is_not_a_unit_of_the_property_is_refused(unit):
    with pytest.raises(CandidateError):
        Candidate("SYNTH-X", {"mu_n": Property(3.0, unit, "computed")})


@pytest.mark.parametrize("unit", [123, None, 1.5, ["cm^2/Vs"]])
def test_a_unit_that_is_not_even_a_string_is_refused(unit):
    """`unit = 123` was accepted, and is indistinguishable from a real unit to
    every check downstream."""
    with pytest.raises(CandidateError, match="non-empty string"):
        Property(3.0, unit, "computed")


def test_two_spellings_of_one_mobility_are_one_mobility():
    """0.1 m^2/Vs and 1000 cm^2/Vs are the same material.

    Before the unit table the first solved a material ten thousand times
    slower than the one described, silently.
    """
    metric = Candidate("SYNTH-M", {"mu_n": Property(0.1, "m^2/Vs", "computed")})
    cgs = Candidate("SYNTH-C", {"mu_n": Property(1000.0, "cm^2/Vs", "computed")})
    assert metric.properties["mu_n"].value == cgs.properties["mu_n"].value
    assert metric.properties["mu_n"].unit == "cm^2/Vs"


def test_the_uncertainty_is_converted_with_the_value():
    """Converting the value and leaving the spread behind would turn
    +/- 0.005 m^2/Vs into +/- 0.005 cm^2/Vs — a bound narrowed by four orders
    of magnitude, corrupting exactly the decisions the spread exists for."""
    converted = Candidate("SYNTH-U", {
        "mu_n": Property(0.1, "m^2/Vs", "computed", uncertainty=0.005)})
    prop_ = converted.properties["mu_n"]
    assert (prop_.value, prop_.uncertainty) == (1000.0, 50.0)
    assert prop_.interval == (950.0, 1050.0)


@pytest.mark.parametrize("name,good,bad", [
    ("ni", "cm^-3", "F/cm"),
    ("eps_s", "F/cm", "cm^-3"),
    ("mu_p", "cm^2/Vs", "s"),
])
def test_each_property_accepts_only_its_own_dimension(name, good, bad):
    Candidate("SYNTH-OK", {name: Property(1.0, good, "computed")})
    with pytest.raises(CandidateError):
        Candidate("SYNTH-BAD", {name: Property(1.0, bad, "computed")})


def test_a_threshold_may_carry_a_unit_and_is_converted_too():
    """A bound in m^2/Vs compared against a value in cm^2/Vs was the reported
    defect; silence about which unit the comparison used is how it stayed
    invisible."""
    metric = parse_threshold("mu_n>=0.1m^2/Vs")
    cgs = parse_threshold("mu_n>=1000")
    assert metric.bound == cgs.bound == 1000.0
    assert metric.unit == "cm^2/Vs"


def test_a_threshold_in_a_nonsense_unit_is_refused():
    with pytest.raises(CandidateError, match="not a unit of this property"):
        parse_threshold("mu_n>=1000kg")


def test_the_verdict_says_which_unit_it_compared_in():
    detail = judge(synth(mu_n=prop(1200.0)), parse_threshold("mu_n>=1000")).detail
    assert "cm^2/Vs" in detail


def test_a_property_outside_the_table_is_left_alone():
    """The table constrains what can reach a SOLVER. A candidate may carry
    anything else it likes; it simply cannot be fed to a model on the strength
    of it."""
    free = Candidate("SYNTH-F", {"hardness": Property(9.0, "Mohs", "measured",
                                                      source="synthetic")})
    assert free.properties["hardness"].unit == "Mohs"


# --- valid_range must at least be a range ---------------------------------

def test_a_valid_range_that_is_not_a_table_is_refused():
    """`valid_range = "anything"` was accepted. Whether the range is ENFORCED
    is a separate open question, but storing a string where a table belongs
    guarantees it never can be."""
    with pytest.raises(CandidateError, match="valid_range"):
        Property(1.0, "cm^2/Vs", "computed", valid_range="anything")


# --- the evidence is part of the problem's identity -----------------------

def test_two_candidates_with_equal_numbers_are_still_two_candidates():
    """Measured in review: SYNTH-A and SYNTH-B both landed on 8265fbf2116f,
    and the second overwrote the first's directory and provenance."""
    shared = {"ni": Property(1e10, "cm^-3", "computed"),
              "mu_n": Property(1000.0, "cm^2/Vs", "computed")}
    a = Candidate("SYNTH-A", dict(shared))
    b = Candidate("SYNTH-B", dict(shared))
    assert fingerprint(a) != fingerprint(b)


def test_the_same_id_with_better_evidence_is_a_different_fingerprint():
    """The identifier alone would not be enough: an id can be re-issued with a
    better measurement behind it, and those are two different pieces of
    evidence that must not share a directory."""
    rough = Candidate("SYNTH-A", {"mu_n": Property(1000.0, "cm^2/Vs",
                                                   "inferred")})
    better = Candidate("SYNTH-A", {"mu_n": Property(1000.0, "cm^2/Vs",
                                                    "measured",
                                                    source="synthetic",
                                                    uncertainty=5.0)})
    assert fingerprint(rough) != fingerprint(better)


def test_an_identical_record_fingerprints_identically():
    """Otherwise re-running the same candidate would never reuse its
    directory, and the id would stop meaning anything."""
    def build():
        return Candidate("SYNTH-A", {"mu_n": Property(1000.0, "cm^2/Vs",
                                                      "computed")})
    assert fingerprint(build()) == fingerprint(build())


# --- valid_range is enforced, or it is not accepted ------------------------

def test_a_range_over_a_condition_no_run_supplies_is_refused():
    """An unenforceable range is worse than none: it reads as a guarantee and
    is not one. Reported in review as stored-and-never-consulted."""
    with pytest.raises(CandidateError, match="which no run can supply"):
        Property(1.0, "cm^2/Vs", "computed",
                 valid_range={"temperature_k": [250, 350]})


@pytest.mark.parametrize("span", ["anything", [1.0], [3.0, 1.0], [1.0, "x"],
                                  [float("nan"), 1.0], 5])
def test_a_range_that_is_not_a_range_is_refused(span):
    with pytest.raises(CandidateError):
        Property(1.0, "cm^2/Vs", "computed", valid_range={"bias_v": span})


def test_a_run_inside_the_stated_range_is_clean():
    inside = Candidate("SYNTH-R", {"mu_n": Property(
        1000.0, "cm^2/Vs", "computed", valid_range={"bias_v": [0.0, 0.5]})})
    assert out_of_range(inside, {"bias_v": 0.3}) == ()


def test_a_run_outside_the_stated_range_is_named():
    outside = Candidate("SYNTH-R", {"mu_n": Property(
        1000.0, "cm^2/Vs", "computed", valid_range={"bias_v": [0.0, 0.5]})})
    breaches = out_of_range(outside, {"bias_v": 0.9})
    assert len(breaches) == 1
    assert "mu_n" in breaches[0] and "0.9" in breaches[0]


def test_a_condition_the_run_does_not_report_is_not_invented():
    """Absence of a condition is not a breach of it."""
    candidate = Candidate("SYNTH-R", {"mu_n": Property(
        1000.0, "cm^2/Vs", "computed", valid_range={"bias_v": [0.0, 0.5]})})
    assert out_of_range(candidate, {}) == ()


# --- duplicate keys ------------------------------------------------------

def test_a_json_key_given_twice_is_refused(tmp_path):
    """json.loads keeps the LAST value silently. For a file that decides what
    physics runs, "the last one wins" is a coin flip nobody sees. TOML already
    refuses this; JSON had to be told."""
    path = tmp_path / "dup.json"
    path.write_text('{"SYNTH-A": {"properties": {'
                    '"mu_n": {"value": 1000.0, "unit": "cm^2/Vs", '
                    '"basis": "computed"}, '
                    '"mu_n": {"value": 5.0, "unit": "cm^2/Vs", '
                    '"basis": "computed"}}}}', encoding="utf-8")
    with pytest.raises(CandidateError, match="given twice"):
        load_candidates(path)

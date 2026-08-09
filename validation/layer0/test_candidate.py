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
                              judge, load_candidates, parse_threshold, screen)

PN1D = "semiconductor.pn.drift-diffusion.1d.steady"


def prop(value, unit="cm^2/Vs", basis="computed", **kw):
    return Property(value=value, unit=unit, basis=basis, **kw)


def synth(identifier="SYNTH-A", **properties):
    return Candidate(identifier=identifier,
                     properties=properties or {"mu_n": prop(1000.0)})


# --- what a property refuses to be ----------------------------------------

def test_a_number_without_a_unit_is_not_a_property():
    """1350 is not a mobility. The unit is not metadata."""
    with pytest.raises(CandidateError, match="no unit"):
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
    result = judge(synth(), Threshold("band_gap_ev", ">=", 1.0))
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

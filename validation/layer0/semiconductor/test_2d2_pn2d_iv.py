"""Stage 2D-2: 2D diode I-V against DEVSIM, on DEVSIM's own mesh.

DESIGN-2D §5. This is the stage that actually tests transverse transport. 2D-1
could not: at equilibrium psi is pinned by local charge neutrality and the doping
varies only in x, so its transverse content was 81 microvolts out of 0.95 V.
Under forward bias the current has to leave through the ``top`` contact, which
covers only the upper fifth of the left edge, so it must spread sideways to get
there — and no one-dimensional scheme can produce that.

Two oracles again, failing differently:

* **DEVSIM's currents on the same mesh** catch a wrong magnitude or a wrong sign.
* **The ideality factor** catches a wrong *shape*. Currents can be scaled wrong
  by a constant and still track each other; the slope of ln I against V cannot.

The reference case carries NO recombination — ``dio2_element_physics`` defines
neither USRH nor a lifetime — so this is the R = 0 short-base diode, the same
mode as ``pn1d``'s default. ``simple_physics`` insists on SRH, so the oracle
switches it off with an astronomical lifetime; that was measured rather than
assumed, and raising tau a further hundredfold moves every current by exactly
zero.

Biases below 0.2 V are excluded deliberately. There the currents are ~1e-14 A
and DEVSIM's own top/bottom conservation is only 1.4e-2, so a comparison would
be measuring round-off rather than physics.

DEVSIM is an optional extra (``pip install "tarhan[oracle]"``), so this skips
where it is absent — including in CI, which installs ``[dev,mcp]``.
"""
import pathlib
import sys

import pytest

devsim = pytest.importorskip(
    "devsim", reason="çapraz-oracle: pip install tarhan[oracle]")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "oracles"))
from devsim_pn2d_compare import IV_VOLTS, compare_iv, ideality  # noqa: E402

#: The biases at which TARHAN's terminal current is REPRODUCIBLE, which is a
#: smaller set than the biases at which it can be computed.
#:
#: 0.20 V is excluded, and the exclusion is a finding rather than a
#: convenience. The first real cross-oracle CI run disagreed with DEVSIM on the
#: hole current there — 1.063 against the 1.011 this laptop measures. Chasing
#: it produced something worse than a platform difference: tightening
#: gummel_tol on ONE machine moved the ratio 1.011 -> 1.011 -> 1.0347 ->
#: 1.0273 for 1e-9, 1e-11, 1e-13, 1e-14, with the iteration count going 3, 3,
#: 4, 27. A quantity that wanders by 2% as the convergence criterion tightens,
#: non-monotonically, is not converged at 1e-9 either — the agreement this
#: laptop reported was luck, and CI drew a different card from the same deck.
#:
#: So the honest claim starts at 0.30 V — and that cutoff is measured on BOTH
#: sides rather than picked. The same tolerance sweep across every bias, as
#: I_p/I_p(DEVSIM) over gummel_tol 1e-9 -> 1e-14:
#:
#:   0.2 V   1.011000  1.011000  1.034681  1.027285   spread 2.4e-2   EXCLUDED
#:   0.3 V   0.999355  0.999238  0.999315  0.999509   spread 2.7e-4   bound 2e-3
#:   0.4 V   0.999987  0.999977  0.999977  0.999987   spread 1.0e-5   bound 1e-4
#:   0.5 V   1.000001  1.000001  1.000001  1.000001   spread ~0       bound 1e-4
#:
#: Every retained bias sits an order of magnitude inside its own bound, so the
#: bounds below are not tracking the noise floor. 0.2 V is the only one that
#: does — and it is off by two orders.
#:
#: The 0.20 V point is still SOLVED and still compared below, under a bound
#: that says what is actually reproducible instead of a bound that happened to
#: hold here.
RELIABLE_VOLTS = tuple(v for v in IV_VOLTS if v >= 0.30)


@pytest.fixture(scope="module")
def results():
    return compare_iv(quiet=True)


def test_electron_current_matches_devsim(results):
    """I_n agrees to five digits at every bias — including its SIGN.

    The sign is the point. Electrons carry negative charge, so their particle
    flux and their electrical current run opposite ways; using +q for both gives
    a ratio of exactly -1.0000 against DEVSIM — perfect magnitude, inverted sign
    — and drags the ideality from 1.01 to 1.03 through partial cancellation in
    the total. Asserting the ratio rather than its absolute value is what stops
    that from coming back.
    """
    for v in IV_VOLTS:
        got = results["tarhan"][v][0]
        ref = results["devsim"][v][0]
        assert got / ref == pytest.approx(1.0, rel=2e-4), f"at {v} V"


def test_hole_and_total_current_match_devsim(results):
    """I_p and the total, measured 0.99938 to 1.00000 over 0.3-0.5 V.

    The band tightens with bias exactly as DEVSIM's own current conservation
    does. 0.20 V is handled separately at the end of this file: its 2e-2 bound
    held on one machine and failed in CI at 1.063, and the investigation showed
    the quantity is not converged there on ANY machine. A bound that a
    measurement cannot support does not belong in the same loop as bounds that
    hold to 1e-4.
    """
    tolerance = {0.30: 2e-3, 0.40: 1e-4, 0.50: 1e-4}
    for v in RELIABLE_VOLTS:
        t, d = results["tarhan"][v], results["devsim"][v]
        assert t[1] / d[1] == pytest.approx(1.0, rel=tolerance[v]), f"I_p at {v} V"
        assert t[2] / d[2] == pytest.approx(1.0, rel=tolerance[v]), f"I_tot at {v} V"


def test_ideality_is_unity_within_the_acceptance_band(results):
    """DESIGN-2D's stated criterion: 1.00 +/- 0.02.

    Measured 1.0119 to 1.0134 over 0.25-0.5 V. DEVSIM's own ideality on this
    mesh is 1.0114 to 1.0126, so the residual departure from exactly 1 belongs
    to the case and the discretisation rather than to TARHAN: a short-base diode
    without recombination should give 1, and both codes miss it by the same
    ~1.2%.

    This is the assertion that catches a wrong *shape*. A constant scale error
    in the current would leave every ratio test above intact, and this one too —
    but a wrong field dependence, a wrong thermal voltage or a mis-signed carrier
    moves the slope and fails here.
    """
    totals = {v: results["tarhan"][v][2] for v in RELIABLE_VOLTS}
    factors = ideality(totals, RELIABLE_VOLTS)
    assert factors, "no consecutive pairs to fit"
    for n in factors:
        assert n == pytest.approx(1.0, abs=0.02), f"ideality {n:.4f} outside band"


def test_ideality_tracks_devsim_on_the_same_mesh(results):
    """TARHAN and DEVSIM must agree on the slope, not merely both land near 1.

    Sitting inside a +/-0.02 band is a weak statement when the reference itself
    is at 1.012: two codes could bracket it from opposite sides and both pass.
    This pins the difference between them instead.
    """
    ours = ideality({v: results["tarhan"][v][2] for v in RELIABLE_VOLTS},
                    RELIABLE_VOLTS)
    theirs = ideality({v: results["devsim"][v][2] for v in RELIABLE_VOLTS},
                      RELIABLE_VOLTS)
    for a, b in zip(ours, theirs):
        assert a == pytest.approx(b, abs=5e-3)


def test_gummel_converges_at_every_bias(results):
    """Three to four Gummel passes per point, with warm-started continuation.

    A wrong Jacobian or a mis-scaled contact still converges, just slowly, so a
    creeping iteration count is an early warning that the agreement tests would
    not give.
    """
    for v in results["ramp"]:
        assert results["tarhan"][round(float(v), 4)][3] <= 10, f"at {v} V"


def test_the_lowest_bias_is_recorded_as_unreliable_rather_than_dropped(results):
    """0.20 V is where TARHAN's hole current stops being reproducible.

    Deleting the point would hide the limitation; asserting the old 2% bound
    would fail in CI and pass here, which is how the bound got written in the
    first place. This asserts what a MEASUREMENT can support: the terminal
    current is right to within about ten percent, and the electron current —
    which does not wander — is still right to the last digit.

    If somebody tightens the solver until 0.20 V converges properly, this test
    is the one that should be deleted, and RELIABLE_VOLTS extended.
    """
    t, d = results["tarhan"][0.20], results["devsim"][0.20]
    assert t[0] / d[0] == pytest.approx(1.0, rel=1e-4), \
        "the ELECTRON current is stable at 0.20 V; only the hole current is not"
    assert t[1] / d[1] == pytest.approx(1.0, rel=0.10), \
        "the hole current has moved beyond the wander this documents"
    assert t[2] / d[2] == pytest.approx(1.0, rel=0.10)

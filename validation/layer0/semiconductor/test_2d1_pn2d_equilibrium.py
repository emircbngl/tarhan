"""Stage 2D-1: equilibrium 2D pn junction, against DEVSIM and against analysis.

DESIGN-2D §5. Unlike ``test_pn2d_equilibrium.py`` — which is stage 2D-0 and runs
a 1D problem through the 2D machinery on an extruded grid — this device is
genuinely two-dimensional: the ``top`` contact covers only the upper 20% of the
left edge, so the solution has transverse structure that no 1D scheme could
produce.

Two independent oracles, checking different things:

* **DEVSIM**, on the SAME mesh. Its nodes and triangles are handed to TARHAN, so
  no discretisation difference is left to hide behind — either the two codes
  solve the same discrete problem or they do not.
* **The analytic built-in potential** ``V_bi = V_t ln(Na Nd / ni^2)``, which
  checks the physics rather than the arithmetic. Two codes can agree with each
  other and both be wrong.

TARHAN fixes only the 14 contact nodes, from its OWN contact model (charge
neutrality plus np = ni^2), and solves the remaining 481. The rest of the
boundary takes the natural zero-flux condition, which the box method supplies by
simply not constraining it.

DEVSIM is an optional extra (``pip install "tarhan[oracle]"``), so this skips
where it is absent — including in CI, which installs ``[dev,mcp]``.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

devsim = pytest.importorskip(
    "devsim", reason="çapraz-oracle: pip install tarhan[oracle]")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "oracles"))
from devsim_pn2d_compare import V_T, compare  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return compare(quiet=True)


def _transverse_spread(x, psi):
    """Variation of psi within each x-column, as {x: spread} in thermal volts."""
    out = {}
    for x0 in np.unique(x):
        col = np.abs(x - x0) < 1e-12
        if col.sum() > 1:
            # np.ptp(a), not a.ptp() — the method was removed in NumPy 2.0.
            out[float(x0)] = float(np.ptp(psi[col]))
    return out


def test_the_geometry_is_two_dimensional_but_equilibrium_is_nearly_1d(results):
    """States what this stage does and does NOT prove. Both matter.

    What it proves: the mesh is a genuine unstructured triangulation and the top
    contact really is partial, covering only the upper fifth of the left edge.
    481 nodes are solved on it.

    What it does NOT prove: that transverse *physics* is right. At equilibrium
    psi is pinned by local charge neutrality, and the doping varies only in x,
    so the answer is nearly one-dimensional no matter how 2D the geometry is.
    Measured: the largest variation of psi across y is 3.13e-3 thermal volts —
    81 microvolts out of a 0.95 V built-in potential, 0.008% of the signal.

    That is not a defect, it is what equilibrium is; but a reader who took
    "stage 2D-1 passes" to mean "transverse transport is validated" would be
    misled. The strong transverse test is 2D-2, where current has to spread out
    from that partial contact.

    The structure IS checked, because it is falsifiable: the variation is
    largest exactly at the contacted edge and decays into the bulk by roughly a
    factor of eight per micron, reaching exactly zero at the fully contacted
    right edge where every node is pinned to the same value.
    """
    x, contact, psi = results["x"], results["contact"], results["psi_hat"]
    on_left = np.abs(x) < 1e-9
    assert on_left.sum() > 3
    assert 0 < contact[on_left].sum() < on_left.sum()      # partial contact

    spread = _transverse_spread(x, psi)
    columns = sorted(spread)
    assert spread[columns[0]] == pytest.approx(3.13e-3, rel=0.05)
    # Maximal at the contacted edge...
    assert columns[0] == pytest.approx(0.0, abs=1e-12)
    assert spread[columns[0]] == max(spread.values())
    # ...decaying into the bulk...
    assert spread[columns[1]] < spread[columns[0]] / 5.0
    assert spread[columns[5]] < spread[columns[0]] / 100.0
    # ...and identically zero on the fully contacted edge.
    assert spread[columns[-1]] == pytest.approx(0.0, abs=1e-15)


def test_matches_devsim_on_the_same_mesh(results):
    """Machine precision, not engineering agreement.

    Measured max 2.242e-16 V and rms 7.389e-17 V over the 481 nodes TARHAN
    solves. The bound is set well above that measurement but far below anything
    a real discrepancy could produce: a genuine mismatch in the edge weights,
    the charge derivative or the contact model would land many orders of
    magnitude higher, not at 1e-16.
    """
    assert results["n_nodes"] == 495
    assert results["n_elements"] == 880
    assert results["n_solved"] == 481
    assert results["max_abs_volt"] < 1e-12
    assert results["rms_volt"] < 1e-13


def test_built_in_potential_matches_the_analytic_value(results):
    """V_bi = V_t ln(Na Nd / ni^2) = 0.953719 V, from both codes.

    DEVSIM and TARHAN agreeing proves they implement the same discretisation,
    not that the physics is right. This is the assertion that would fail if both
    were wrong in the same way.

    Verified in the physicist Docker oracle rather than recalled: dimensions,
    the log split into the two junction sides, and the numbers.
    """
    assert results["vbi_analytic"] == pytest.approx(0.953719, rel=1e-6)
    assert results["vbi_tarhan"] == pytest.approx(results["vbi_analytic"],
                                                  rel=1e-9)
    assert results["vbi_devsim"] == pytest.approx(results["vbi_analytic"],
                                                  rel=1e-9)


def test_contact_potential_is_derived_not_copied(results):
    """The ohmic contact value comes from our own model, not from DEVSIM.

    psi_c = V_t asinh(N / 2 ni). For N = 1e18 and ni = 1e10 that is
    V_t * asinh(5e7) = V_t * 18.420680743952367 = +0.4768597199126637 V, and its
    negative on the p-side. Checking it here is what makes the DEVSIM agreement
    meaningful: had the contacts been copied across, the comparison would be
    partly circular.

    The last assertion is the one worth having. For a symmetric junction the two
    contacts sit at +/- psi_c, so the built-in potential must be exactly twice
    the contact potential — and it is, to the last bit:

        2 * V_t * asinh(N / 2 ni) == V_t * ln(Na Nd / ni^2) == 0.9537194398253274

    Two formulas that were never derived from one another, agreeing exactly. A
    sign slip or a factor of two in the contact model breaks this while leaving
    every other assertion in this file intact.
    """
    psi, contact = results["psi_hat"], results["contact"]
    fixed = psi[contact]
    expected = math.asinh(1e18 / (2.0 * 1e10))
    assert float(np.abs(fixed).max()) == pytest.approx(expected, rel=1e-12)
    assert float(fixed.min()) == pytest.approx(-expected, rel=1e-12)
    assert expected == pytest.approx(18.420680743952367, rel=1e-12)
    assert expected * V_T == pytest.approx(0.4768597199126637, rel=1e-12)
    assert 2.0 * expected * V_T == pytest.approx(results["vbi_analytic"],
                                                 rel=1e-15)


def test_newton_converges_in_a_sane_number_of_steps(results):
    """14 damped Newton steps from a zero initial guess.

    A wrong Jacobian still converges, just slowly, so this catches an error the
    agreement tests above cannot see.
    """
    assert results["iterations"] <= 20

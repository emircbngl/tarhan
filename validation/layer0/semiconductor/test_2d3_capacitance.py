"""Stage 2D-3′: electrostatic contact charge, against DEVSIM `cap2d`.

Read the prime. This is NOT stage 2D-3 as the design table originally listed it,
and the substitution is recorded in DESIGN-2D rather than hidden here.
`ssac_cap_2d_edge.py` turned out to be a parallel-plate AIR capacitor measuring
displacement current through a small-signal AC solve wired to a lumped circuit —
no semiconductor, no C–V curve. It needs circuit-node coupling and a
complex-valued AC solve, and DESIGN-2D proposes neither. That row is blocked.

What this stage does instead is the same physical question stripped of the AC
machinery: one region of uniform permittivity, charge-free Laplace, two Dirichlet
contacts, and the contact charge as the answer. It is a weaker statement, which
is why it is numbered 2D-3′.

It does earn its place. The answer comes from summing the UNCONSTRAINED Poisson
residual over a contact's nodes, which asserts that the residual is a genuine
flux — Gauss's law. Stage 2D-2 established that for the continuity equation
(contact current); nothing had established it for Poisson. And no scale factor
appears, for a reason worth stating: A/L is dimensionless and the charge term is
zero, so volume never enters and the residual is already C/cm whatever units the
mesh is expressed in.

DEVSIM is an optional extra (``pip install "tarhan[oracle]"``), so this skips
where it is absent — including in CI, which installs ``[dev,mcp]``.
"""
import pathlib
import sys

import pytest

devsim = pytest.importorskip(
    "devsim", reason="çapraz-oracle: pip install tarhan[oracle]")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "oracles"))
from devsim_cap2d_compare import compare  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return compare(quiet=True)


def test_contact_charge_matches_devsim(results):
    """3.350171660e-12 C/cm from both codes — ratio 1.000000000.

    No fitted constant stands between the two numbers. The residual sum IS the
    charge, because A/L is dimensionless and the charge term is zero, so what
    gets assembled is already ``eps * (A/L) * psi`` = C/cm. A missing or invented
    scale factor would appear here as a clean numerical ratio rather than as
    noise, which is what makes this a sharp test of that claim.
    """
    assert results["q_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert results["q_top_tarhan"] == pytest.approx(3.350171660e-12, rel=1e-6)


def test_the_two_plates_carry_equal_and_opposite_charge(results):
    """Gauss's law on a closed system: the charges must cancel.

    Independent of DEVSIM. If the residual were not a true flux — if it leaked at
    the far-field boundary, or double-counted an edge — this sum would not
    vanish, and it would fail even if each plate happened to match DEVSIM on its
    own.
    """
    total = results["q_top_tarhan"] + results["q_bot_tarhan"]
    assert abs(total) < 1e-12 * abs(results["q_top_tarhan"])


def test_potential_matches_devsim(results):
    """max |dpsi| = 6.6e-13 V on a 1 V scale, over 8281 nodes.

    The charge test alone could pass on a wrong field if the errors happened to
    integrate away around the contact, so the field itself is pinned too.
    """
    assert results["n_nodes"] == 8281
    assert results["n_elements"] == 15636
    assert results["psi_max_abs_diff"] < 1e-9


def test_the_contacts_were_actually_found(results):
    """Guard the node-selection trick this oracle relies on.

    Contacts are identified as the nodes sitting at exactly 1.0 or exactly 0.0,
    which is sound for a Laplace solution between two conductors — interior nodes
    are strictly between — but it is a trick, and a silently empty or
    all-inclusive selection would make every assertion above meaningless.
    """
    assert 0 < results["n_top"] < results["n_nodes"]
    assert 0 < results["n_bot"] < results["n_nodes"]
    assert results["n_top"] + results["n_bot"] < results["n_nodes"]

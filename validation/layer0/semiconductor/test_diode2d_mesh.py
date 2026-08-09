"""The generated 2D diode mesh, and the 1D solver as its oracle.

A mesh generator is the kind of code that looks right and is wrong: it produces
plausible numbers, the solver converges on them, and the answer is quietly off
because a triangle is inside out or a contact sits one node from where it was
meant to. So this file does not check that the mesh "looks like a rectangle".
It checks the properties an assembly actually depends on, and then it checks
the only thing that settles the question — that a solver already validated
against DEVSIM gives the same answer on it.

**The oracle needs no external formula, which matters here.** ``physics_verify``
is unavailable this session (the physicist MCP server is disconnected), so any
constant or closed-form expression written from memory would be an unverifiable
claim. The comparison used instead is internal and stronger than a formula
would be: with no variation along y, this device IS the 1D device, so
:mod:`tarhan.models.pn1d` — validated against DEVSIM at stage 1D — must
reproduce the 2D answer on the same node positions. A disagreement can only
mean the mesh, the doping or the contacts are wrong, because everything else is
already pinned elsewhere.

The bounds below are set from what a DEFECT would produce, not from what this
machine measured — the lesson a red CI taught on the 1D transient, where a
bound fitted to one laptop's 8.62e-12 failed against ubuntu's 1.624e-10 for the
identical computation. An inside-out triangle, a misplaced contact or a doping
sign error all give O(1) here.
"""
import numpy as np
import pytest

from tarhan.models import pn1d, pn2d
from tarhan.models.diode2d_mesh import (MeshError, RectangularDiode2D, build,
                                        device)

#: The 1D device's constants, so the two solvers are given the SAME physics.
#: PNDiode2D's defaults differ (11.1 vs 11.7 for the permittivity, a longer
#: thermal voltage), and comparing across them would measure the constants
#: rather than the mesh.
CONST = dict(ni=1e10, ut=0.0259, eps_s=11.7 * 8.85e-14, q=1.6e-19,
             mu_n=1350.0, mu_p=480.0)

PSI_BOUND = 1e-9          # observed 1.3e-13 … 2.3e-13 (scaled potential)
Y_SPREAD_BOUND = 1e-9     # observed ~1.1e-14
CURRENT_REL_BOUND = 1e-3  # observed 1.2e-5 … 4e-7 relative


@pytest.fixture(scope="module")
def spec():
    return RectangularDiode2D()


@pytest.fixture(scope="module")
def devices(spec):
    two = device(spec, **CONST)
    one = pn1d.PNDiode1D(Na=spec.Na, Nd=spec.Nd, len_p=spec.len_p,
                         len_n=spec.len_n, h0=spec.h0, gamma=spec.gamma,
                         **CONST)
    return one, two


def _row_of_y(spec):
    return np.linspace(0.0, spec.height, int(spec.ny) + 1)


# --- the properties an assembly depends on --------------------------------

def test_every_triangle_is_wound_the_same_way(spec):
    """Signed area, not absolute.

    An inside-out triangle has negative signed area. An assembly that sums
    |area| would never notice; one that sums signed area would cancel it
    against a neighbour and report a mesh smaller than it is. Either way the
    solve converges to the wrong answer rather than failing.
    """
    kw = build(spec)
    pts = np.asarray(kw["points"])
    tri = np.asarray(kw["triangles"])
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    assert signed.min() > 0.0, "at least one triangle is wound backwards"


def test_the_triangles_tile_the_rectangle_exactly(spec):
    """Total area is the identity a gap or an overlap breaks.

    A missing triangle and a doubled one are both invisible in a picture and
    both change every volume the box method computes.
    """
    kw = build(spec)
    pts = np.asarray(kw["points"])
    tri = np.asarray(kw["triangles"])
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    expected = (spec.len_p + spec.len_n) * spec.height
    assert signed.sum() == pytest.approx(expected, rel=1e-12)


def test_the_contacts_sit_on_the_two_ends_and_nowhere_else(spec):
    """A contact one node from its edge still solves, and lies."""
    kw = build(spec)
    pts = np.asarray(kw["points"])
    for name, x_expected in (("anode", -spec.len_p), ("cathode", spec.len_n)):
        nodes = np.asarray(kw["contacts"][name], dtype=int)
        assert nodes.size == int(spec.ny) + 1
        assert np.allclose(pts[nodes, 0], x_expected, rtol=1e-12)
        assert np.allclose(np.sort(pts[nodes, 1]), _row_of_y(spec))


def test_the_doping_changes_sign_exactly_at_the_junction(spec):
    kw = build(spec)
    x = np.asarray([p[0] for p in kw["points"]])
    doping = np.asarray(kw["net_doping"])
    assert np.all(doping[x < 0] == -spec.Na)
    assert np.all(doping[x > 0] == spec.Nd)
    assert np.all(doping[x == 0.0] == 0.0), \
        "the junction column is the metallurgical junction, not one side of it"


def test_the_x_grid_is_the_one_the_1d_device_builds(spec, devices):
    """The comparison below is only meaningful on identical node positions.

    A uniform x grid would resolve the depletion region so much worse that a
    disagreement could not be blamed on the mesh rather than the spacing.
    """
    one, _ = devices
    x_1d = np.asarray(one.build_grid()) * one.L_D
    x_2d = spec.x_nodes()
    assert len(x_1d) == len(x_2d)
    assert np.abs(np.sort(x_1d) - np.sort(x_2d)).max() < 1e-15


# --- the oracle: a validated solver on the generated mesh ------------------

@pytest.mark.parametrize("v", [0.0, 0.3])
def test_nothing_varies_along_y(devices, spec, v):
    """The premise of the whole comparison, asserted rather than assumed.

    If the potential drifted along y the device would not be the 1D device and
    the agreement below would be a coincidence worth distrusting.
    """
    _, two = devices
    psi = np.asarray(pn2d.solve_bias(two, v)["psi"])
    psi = psi.reshape(len(spec.x_nodes()), int(spec.ny) + 1)
    assert np.abs(psi - psi[:, :1]).max() < Y_SPREAD_BOUND


@pytest.mark.parametrize("v", [0.0, 0.3])
def test_the_potential_matches_the_validated_1d_solver(devices, spec, v):
    """Measured max|dpsi_hat| = 1.26e-13 at equilibrium, 2.28e-13 at 0.30 V."""
    one, two = devices
    psi_1d = np.asarray(pn1d.solve_bias(one, v)["psi"])
    psi_2d = np.asarray(pn2d.solve_bias(two, v)["psi"])
    psi_2d = psi_2d.reshape(len(spec.x_nodes()), int(spec.ny) + 1)
    row = psi_2d[:, int(spec.ny) // 2]
    assert np.abs(row - psi_1d).max() < PSI_BOUND


@pytest.mark.parametrize("v", [0.2, 0.3, 0.4])
def test_the_terminal_current_matches_the_validated_1d_solver(devices, spec, v):
    """Measured ratio 0.999988 at 0.2 V and 1.000000 at 0.3 and 0.4 V.

    The 2D terminal current is integrated over the contact EDGE, so it is a
    current per unit depth and carries an extra length compared with the 1D
    current density. Dividing by the device height is what the ratio below
    tests; that reading is not asserted from a formula (physics_verify is
    unavailable), it is what the agreement with an independently validated
    solver across three decades of current establishes.
    """
    one, two = devices
    j_1d = float(pn1d.solve_bias(one, v)["j"])
    i_2d = float(pn2d.solve_bias(two, v)["i"])
    assert abs(abs(i_2d / spec.height / j_1d) - 1.0) < CURRENT_REL_BOUND


def test_a_bigger_device_still_matches():
    """The agreement must not be an artefact of one geometry."""
    wide = RectangularDiode2D(height=4e-4, ny=6, Na=2e16, Nd=5e15)
    two = device(wide, **CONST)
    one = pn1d.PNDiode1D(Na=wide.Na, Nd=wide.Nd, len_p=wide.len_p,
                         len_n=wide.len_n, h0=wide.h0, gamma=wide.gamma,
                         **CONST)
    j_1d = float(pn1d.solve_bias(one, 0.3)["j"])
    i_2d = float(pn2d.solve_bias(two, 0.3)["i"])
    assert abs(abs(i_2d / wide.height / j_1d) - 1.0) < CURRENT_REL_BOUND


# --- refusing what cannot produce a mesh ----------------------------------

@pytest.mark.parametrize("kwargs", [
    {"len_p": 0.0}, {"len_n": -1e-4}, {"height": 0.0}, {"h0": 0.0},
    {"Na": float("nan")}, {"Nd": float("inf")},
    {"ny": 0}, {"ny": -2}, {"ny": 2.5},
])
def test_a_mesh_that_cannot_exist_is_refused(kwargs):
    with pytest.raises(MeshError):
        RectangularDiode2D(**kwargs)


def test_a_shrinking_step_is_refused_rather_than_looping_forever():
    """gamma < 1 makes the walk converge short of the boundary.

    The loop condition is `while xs[-1] < length`, so the generator would hang
    rather than fail — the worst failure mode a test can be given, because a
    CI job that hangs looks like a slow one until it times out.
    """
    with pytest.raises(MeshError, match=">= 1"):
        RectangularDiode2D(gamma=0.9)


def test_the_same_scalars_give_the_same_mesh(spec):
    """The property the artifact id rests on: the lock file records these
    scalars, so re-running them must rebuild the identical mesh."""
    first, second = build(spec), build(RectangularDiode2D())
    assert np.array_equal(np.asarray(first["points"]),
                          np.asarray(second["points"]))
    assert first["triangles"] == second["triangles"]
    assert np.array_equal(first["net_doping"], second["net_doping"])
    assert first["contacts"] == second["contacts"]

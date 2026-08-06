"""Stage 2D-0, Poisson half: the pn junction equilibrium through the 2D machinery.

This is **not** stage 2D-1. 2D-1 asks for an equilibrium junction on a genuinely
two-dimensional mesh, checked against DEVSIM's ``dio2_element_2d.py``; the mesh
here is a 1D grid extruded one cell thick, deliberately, and the only physical
oracle is the analytic built-in potential. Calling it 2D-1 would let the
validation table go green on a problem that has no transverse structure at all.

DESIGN-2D §5 puts stage 2D-0 first for a reason — run a *1D* problem through the
*2D* code, so that the first genuine 2D discrepancy has one candidate cause
instead of two. Here the 1D problem is the flagship's own equilibrium Poisson
solve, and the oracle is ``models/pn1d`` itself: same device, same grid, same
scaling.

The agreement is exact rather than approximate, and that is a structural claim
worth stating. Build the strip from ``pn1d``'s grid and the box method reduces to
``pn1d``'s tridiagonal scheme *identically*:

* Every cell is a rectangle split by a diagonal. Both angles opposite that
  diagonal are the rectangle's own right angles, so they sum to exactly pi and
  the diagonal's Voronoi facet is exactly zero — for any aspect ratio, not only
  for squares. Nothing is transmitted along it.
* A horizontal edge then has facet ``h/2`` and length ``hm``, while the node's
  Voronoi volume is ``hbar * h / 2``. Their ratio is ``1 / (hm * hbar)``, which
  is character for character the ``sub`` coefficient in
  ``pn1d._poisson_newton``.

So this is not "2D reproduces 1D to plotting accuracy". It is the same linear
system, and any deviation beyond round-off means the assembly is wrong.
"""
import math

import numpy as np
import pytest

from tarhan import backend
from tarhan.models.pn1d import PNDiode1D, _neutral_psi, _poisson_newton
from tarhan.numerics.assemble import assemble_poisson, node_volumes
from tarhan.numerics.mesh import build_mesh

STRIP_HEIGHT = 1.0          # scaled; the result must not depend on it


def _strip_from_grid(x_hat, height=STRIP_HEIGHT):
    """One cell of thickness ``height`` over the 1D grid. Node (k,r) -> 2k + r."""
    pts = []
    for xv in x_hat:
        pts.append((float(xv), 0.0))
        pts.append((float(xv), float(height)))
    tris = []
    for k in range(len(x_hat) - 1):
        a, b = 2 * k, 2 * (k + 1)
        c, d = b + 1, a + 1
        tris.append((a, b, c))
        tris.append((a, c, d))
    return build_mesh(pts, tris)


def _equilibrium_1d(dev):
    """pn1d's own equilibrium: phi_n = phi_p = 0, contacts charge-neutral."""
    x = np.asarray(dev.build_grid())
    n_dop = np.asarray(dev.doping_hat(x))
    psi = np.asarray(_neutral_psi(n_dop, dev.delta)).copy()
    zero = np.zeros(len(x))
    psi, _ = _poisson_newton(dev, x, psi, zero, zero)
    return x, n_dop, psi


def _equilibrium_2d(dev, mesh, n_dop, ends, tol=1e-10, max_iter=200):
    """Damped Newton on the assembled 2D Poisson system, clamped like pn1d."""
    delta = dev.delta
    doping = np.repeat(n_dop, 2)
    psi = np.repeat(np.asarray(_neutral_psi(n_dop, delta)), 2).astype(float)
    biggest = float("inf")
    for it in range(max_iter):
        n_h = delta * np.exp(np.clip(psi, -700.0, 700.0))
        p_h = delta * np.exp(np.clip(-psi, -700.0, 700.0))
        sys_ = assemble_poisson(mesh, psi, charge=n_h - p_h - doping,
                                dcharge_dpsi=n_h + p_h, dirichlet=ends)
        step = backend.solve_sparse(sys_.rows, sys_.cols, sys_.vals,
                                    -sys_.residual, n=sys_.n_nodes)
        biggest = float(np.abs(step).max())
        if biggest > 5.0:                      # the same step clamp pn1d uses
            step = step * (5.0 / biggest)
        psi = psi + step
        if biggest < tol:
            return psi, it + 1
    raise AssertionError(
        f"2D Poisson-Newton did not converge (last step {biggest:.2e})")


@pytest.fixture(scope="module")
def solved():
    dev = PNDiode1D()
    x, n_dop, psi1 = _equilibrium_1d(dev)
    mesh = _strip_from_grid(x)
    ends = {0: psi1[0], 1: psi1[0],
            mesh.n_nodes - 2: psi1[-1], mesh.n_nodes - 1: psi1[-1]}
    psi2, iters = _equilibrium_2d(dev, mesh, n_dop, ends)
    return dev, x, n_dop, psi1, mesh, psi2, iters


def test_rectangle_diagonal_carries_nothing(solved):
    """Every cell diagonal has a zero Voronoi facet, at any aspect ratio.

    The two angles opposite it are the rectangle's right angles and sum to
    exactly pi. This is what reduces the strip to a 1D chain, so it is checked
    rather than assumed — and on the real geometric grid, whose cells are very
    far from square near the junction.
    """
    _, _, _, _, mesh, _, _ = solved
    for k in range(0, mesh.n_nodes - 2, 2):
        assert abs(mesh.edge(k, k + 3).facet) < 1e-15


def test_box_weights_reduce_to_the_1d_scheme(solved):
    """A_e / L_e divided by the node volume equals 1 / (hm * hbar).

    That expression is the off-diagonal coefficient in pn1d._poisson_newton. If
    this identity holds, the two discretisations are one discretisation — which
    is the claim the exactness below rests on.
    """
    _, x, _, _, mesh, _, _ = solved
    vol = node_volumes(mesh)
    for k in (1, 5, len(x) // 2, len(x) - 3):
        hm = float(x[k] - x[k - 1])
        hbar = 0.5 * float(x[k + 1] - x[k - 1])
        w = mesh.edge(2 * (k - 1), 2 * k).transmissibility
        assert w / vol[2 * k] == pytest.approx(1.0 / (hm * hbar), rel=1e-12)


def test_2d_equilibrium_matches_pn1d_to_round_off(solved):
    """max|psi_2D - psi_1D| must be round-off, not a discretisation difference.

    Measured 1.8e-15 over a 125-node grid spanning 27.6 thermal volts. A sign
    error, a mis-weighted volume or a wrong charge derivative would each show up
    here as something many orders of magnitude larger.
    """
    _, _, _, psi1, _, psi2, _ = solved
    assert psi2[0::2] == pytest.approx(psi1, abs=1e-12)


def test_2d_equilibrium_is_transversally_flat(solved):
    """A 1D problem must give a y-independent answer, though the rows are coupled.

    Not automatic: the vertical edges have non-zero weight and cancel only
    because the two rows carry equal values. A transverse gradient would be a
    real 2D bug, invisible to the comparison above, which looks at one row.
    """
    _, _, _, _, _, psi2, _ = solved
    assert psi2[0::2] == pytest.approx(psi2[1::2], abs=1e-12)


def test_built_in_potential_matches_the_analytic_value(solved):
    """psi spans V_bi / U_T = ln(Na*Nd/ni^2) — stage 2D-1's analytic oracle.

    For Na = Nd = 1e16 and ni = 1e10 that is ln(1e12) = 27.631021115928547,
    i.e. 0.715643 V at U_T = 0.0259 V. Verified independently in the physicist
    Docker oracle — dimensions, the ln split into the two junction sides, and
    both numbers — rather than asserted from memory.
    """
    dev, _, _, _, _, psi2, _ = solved
    span = float(psi2.max() - psi2.min())
    analytic = math.log(dev.Na * dev.Nd / dev.ni ** 2)
    assert span == pytest.approx(analytic, rel=1e-9)
    assert span * dev.ut == pytest.approx(0.715643, rel=1e-5)


def test_newton_converges_as_fast_as_in_1d(solved):
    """Same system, so the same Newton history: pn1d takes 8 iterations.

    A much larger count would mean the Jacobian is not the exact derivative of
    the residual — which the exactness tests alone would not catch, since a
    wrong Jacobian still converges, just slowly.
    """
    _, _, _, _, _, _, iters = solved
    assert iters <= 10


def test_result_is_independent_of_the_strip_height(solved):
    """Halving the strip thickness must not move psi.

    The height cancels between the edge weights and the node volumes, so a
    dependence would mean the volume rule and the transmissibility disagree
    about the geometry.
    """
    dev, x, n_dop, psi1, _, psi2, _ = solved
    mesh_thin = _strip_from_grid(x, height=0.5)
    ends = {0: psi1[0], 1: psi1[0],
            mesh_thin.n_nodes - 2: psi1[-1], mesh_thin.n_nodes - 1: psi1[-1]}
    psi_thin, _ = _equilibrium_2d(dev, mesh_thin, n_dop, ends)
    assert psi_thin[0::2] == pytest.approx(psi2[0::2], abs=1e-12)

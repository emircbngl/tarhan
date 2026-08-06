"""Layer-0: edge-loop assembly, and stage 2D-0 of the DESIGN-2D ladder.

Every expected number is derived by hand in the docstring that asserts it, or is
exact for a stated structural reason. That is what makes these Layer-0: a reader
with a pencil can check them, rather than trusting a golden file produced by the
code under test.

Stage 2D-0 (DESIGN-2D §5) is the one people skip: run a *1D* problem through the
*2D* machinery. Without it the first real 2D discrepancy has two candidate causes
— wrong assembly or wrong physics — and no way to choose between them.
"""
import numpy as np
import pytest

from tarhan import backend
from tarhan.numerics.assemble import System, assemble_continuity, node_volumes
from tarhan.numerics.flux import sg_edge_flux
from tarhan.numerics.mesh import build_mesh

# The unit square split by the diagonal 0-2, as in test_mesh_geometry.py.
#   3 (0,1) ---- 2 (1,1)
#       |  \        |
#   0 (0,0) ---- 1 (1,0)
SQUARE_PTS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
SQUARE_TRIS = [(0, 1, 2), (0, 2, 3)]


def square():
    return build_mesh(SQUARE_PTS, SQUARE_TRIS)


def strip(n_cells: int, dx: float = 1.0):
    """A one-cell-thick ladder of SQUARE cells, each split by a diagonal.

    Square cells are not incidental. With dx == h the two triangles are right
    isoceles, every angle opposite a boundary edge is 45 deg (so ``build_mesh``
    accepts it), and the diagonal's Voronoi facet is exactly zero — the same
    fact pinned for the unit square. Transport therefore runs only along the
    axis-aligned edges, which is precisely the 1D problem embedded in a 2D mesh
    that stage 2D-0 calls for.

    Node ``(k, r)`` — column k, row r in {0, 1} — has index ``2 * k + r``.
    """
    pts = []
    for k in range(n_cells + 1):
        pts.append((k * dx, 0.0))
        pts.append((k * dx, dx))
    tris = []
    for k in range(n_cells):
        a, b = 2 * k, 2 * (k + 1)          # bottom row: (k,0), (k+1,0)
        c, d = b + 1, a + 1                # top row:    (k+1,1), (k,1)
        tris.append((a, b, c))
        tris.append((a, c, d))
    return build_mesh(pts, tris), pts


def newton_step(sys_: System) -> np.ndarray:
    """One Newton update. With psi fixed the system is linear, so this is exact."""
    return backend.solve_sparse(sys_.rows, sys_.cols, sys_.vals, -sys_.residual,
                                n=sys_.n_nodes)


# --------------------------------------------------------------------------
# Node volumes
# --------------------------------------------------------------------------

def test_square_node_volumes_are_a_quarter_each():
    """By hand: vol_i = (1/4) * sum over incident edges of A_e * L_e.

    Node 0 touches side 0-1 (A=1/2, L=1), side 0-3 (A=1/2, L=1) and the
    diagonal 0-2 (A=0, so it contributes nothing however long it is):

        vol_0 = (1/2*1 + 1/2*1 + 0*sqrt2) / 4 = 1/4

    Every node is that same figure rotated, and the four quarters must tile the
    square exactly — which is the real assertion here, since a volume rule that
    did not sum to the domain area would silently mis-weight every source term.
    """
    vol = node_volumes(square())
    assert vol == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert vol.sum() == pytest.approx(1.0)


def test_strip_volumes_sum_to_its_area():
    """5 unit cells -> area 5, whatever the volume rule does per node."""
    mesh, _ = strip(5)
    assert node_volumes(mesh).sum() == pytest.approx(5.0)


def _circumcentre(a, b, c):
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


def _voronoi_areas_via_circumcentres(pts, tris):
    """Per-node Voronoi area built the other way: from actual circumcentres.

    Inside each triangle the node's share is the quadrilateral spanned by the
    node, the midpoints of its two incident edges and the circumcentre; shoelace
    it and sum over triangles. This shares no line of code and no algebra with
    the ``(1/4) sum A_e L_e`` rule, which is the point of having it.
    """
    area = np.zeros(len(pts))
    for i, j, k in tris:
        cc = np.array(_circumcentre(pts[i], pts[j], pts[k]))
        for n, p, q in ((i, j, k), (j, k, i), (k, i, j)):
            node = np.array(pts[n], dtype=float)
            poly = [node, (node + np.array(pts[p])) / 2.0, cc,
                    (node + np.array(pts[q])) / 2.0]
            s = sum(poly[t][0] * poly[(t + 1) % 4][1]
                    - poly[(t + 1) % 4][0] * poly[t][1] for t in range(4))
            area[n] += abs(s) / 2.0
    return area


@pytest.mark.parametrize("pts,tris,name", [
    (SQUARE_PTS, SQUARE_TRIS, "unit square"),
    ([(0.0, 0.0), (0.0, 1.0), (1.3, 0.0), (1.3, 1.0), (2.1, 0.0), (2.1, 1.0)],
     [(0, 2, 3), (0, 3, 1), (2, 4, 5), (2, 5, 3)], "non-uniform strip"),
    # The case the rule is most likely to get wrong, and the reason this one is
    # here: node 2's angle facing the interior edge 0-1 is 118 deg, so that
    # triangle's circumcentre falls OUTSIDE it. mesh.py deliberately permits
    # this — the neighbour's larger cotangent absorbs it — which means the
    # volume rule has to survive it too. Both constructions still agree, and
    # both still total the true mesh area 1.15 exactly.
    ([(0.0, 0.0), (1.0, 0.0), (0.5, 0.3), (0.5, -2.0)],
     [(0, 1, 2), (0, 1, 3)], "interior obtuse triangle"),
])
def test_volume_rule_agrees_with_the_circumcentre_construction(pts, tris, name):
    """Two independent constructions of the same area must agree per NODE.

    Checking only that the volumes sum to the domain area is necessary but not
    sufficient: a rule that mis-distributed area between nodes while still
    totalling correctly would pass it, and would then silently mis-weight every
    source term. So the reference here is built the other way — from real
    circumcentres and a shoelace — and compared node by node, on a mesh with
    unequal spacing so that a rule accidentally right only for uniform cells
    fails.
    """
    mesh = build_mesh(pts, tris)
    ref = _voronoi_areas_via_circumcentres(
        [(float(x), float(y)) for x, y in pts], tris)
    assert node_volumes(mesh) == pytest.approx(ref, abs=1e-14)


# --------------------------------------------------------------------------
# The assembly uses the flux law rather than re-deriving it
# --------------------------------------------------------------------------

def test_residual_equals_sg_edge_flux_on_a_single_edge():
    """The assembled residual must BE sg_edge_flux, not something like it.

    DESIGN-2D §7 lists "touching bernoulli/sg_edge_flux" as a way this fails.
    Calling it is right; quietly re-deriving an equivalent expression is how the
    two drift apart later. So: take the square, silence every edge but 0-1 with
    edge_coef, and check against an independent call.

    Edge 0-1 has A=1/2 and L=1, so w = 1/2. With psi_1 - psi_0 = 0.4 and
    u_t = 1, the flux into node 0 is w*(B(-x) n_1 - B(x) n_0), and node 0's
    residual is its negative.
    """
    mesh = square()
    idx = [i for i, e in enumerate(mesh.edges) if e.nodes == (0, 1)]
    assert len(idx) == 1
    coef = np.zeros(len(mesh.edges))
    coef[idx[0]] = 1.0

    n = np.array([2.0, 5.0, 0.0, 0.0])
    psi = np.array([0.0, 0.4, 0.0, 0.0])
    sys_ = assemble_continuity(mesh, n, psi, edge_coef=coef)

    expected_inflow = float(sg_edge_flux(psi[1] - psi[0], n[0], n[1],
                                         u_t=1.0, coef=0.5))
    assert sys_.residual[0] == pytest.approx(-expected_inflow, rel=1e-14)
    assert sys_.residual[1] == pytest.approx(+expected_inflow, rel=1e-14)


def test_conservation_is_structural():
    """The residual sums to zero over all nodes, for any state.

    Whatever leaves one node arrives at the other, so the totals cancel edge by
    edge — the two contributions are the same float with opposite signs. A
    non-zero sum would mean the assembly invents or destroys carriers.
    """
    mesh = square()
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.uniform(0.1, 10.0, mesh.n_nodes)
        psi = rng.uniform(-2.0, 2.0, mesh.n_nodes)
        sys_ = assemble_continuity(mesh, n, psi)
        assert abs(sys_.residual.sum()) < 1e-13


# --------------------------------------------------------------------------
# The M-matrix claim mesh.py had to leave open
# --------------------------------------------------------------------------

def test_jacobian_is_a_z_matrix():
    """Every off-diagonal entry is <= 0, because B > 0 and the weights are >= 0.

    This is the half of the M-matrix property the mesh guard buys: an
    off-diagonal is -w * B(...), and B is strictly positive for all real
    arguments, so only a negative Voronoi weight could flip the sign — which
    build_mesh refuses to produce.
    """
    mesh, _ = strip(4)
    rng = np.random.default_rng(1)
    a = assemble_continuity(mesh, rng.uniform(0.1, 5.0, mesh.n_nodes),
                            rng.uniform(-3.0, 3.0, mesh.n_nodes)).to_dense()
    off = a - np.diag(np.diag(a))
    assert off.max() <= 1e-15
    assert np.all(np.diag(a) > 0.0)


def test_unconstrained_columns_sum_to_zero():
    """Each edge writes +w B(-x) and -w B(-x) into the same column.

    So the unconstrained matrix is singular and weakly column-diagonally
    dominant, like a graph Laplacian. This is why a Dirichlet node is not a
    convenience but a well-posedness requirement.
    """
    mesh, _ = strip(3)
    rng = np.random.default_rng(2)
    a = assemble_continuity(mesh, rng.uniform(0.1, 5.0, mesh.n_nodes),
                            rng.uniform(-2.0, 2.0, mesh.n_nodes)).to_dense()
    assert np.abs(a.sum(axis=0)).max() < 1e-12


def test_dirichlet_makes_the_inverse_non_negative():
    """A^-1 >= 0 entrywise — the property that actually gives positivity.

    mesh.py's docstring records this as unproven "until assemble.py exists to
    produce a matrix". It exists, so here is a measurement in place of the
    citation: constrain both ends of a strip and invert.

    A^-1 >= 0 means a non-negative right-hand side cannot produce a negative
    density anywhere. That is the whole reason the Delaunay condition is
    enforced upstream.
    """
    mesh, _ = strip(4)
    ends = {0: 1.0, 1: 1.0, mesh.n_nodes - 2: 2.0, mesh.n_nodes - 1: 2.0}
    rng = np.random.default_rng(3)
    a = assemble_continuity(mesh, np.ones(mesh.n_nodes),
                            rng.uniform(-1.0, 1.0, mesh.n_nodes),
                            dirichlet=ends).to_dense()
    assert np.linalg.inv(a).min() > -1e-12


# --------------------------------------------------------------------------
# Stage 2D-0 — a 1D problem through the 2D machinery
# --------------------------------------------------------------------------

def test_2d0_equilibrium_is_exact_on_a_2d_mesh():
    """Constant field, zero current: n = C exp(-psi/U_T), to machine precision.

    Scharfetter-Gummel is built from the exact solution of the drift-diffusion
    ODE between two nodes, so its discrete equilibrium condition j_e = 0 holds
    EXACTLY at n_i = C exp(-psi_i / U_T) — for every edge, on any mesh, with no
    discretisation error to leak in. A sign error, a mis-scaled U_T or a
    mismatched edge orientation all die here.

    Take psi = -E x with E = 3 and U_T = 0.4, so n(x) = exp(3x/0.4) = exp(7.5x).
    Fix only the two ends and solve; every interior node must land on the
    analytic curve.
    """
    n_cells, u_t, field = 6, 0.4, 3.0
    mesh, pts = strip(n_cells)
    xs = np.array([p[0] for p in pts])
    psi = -field * xs
    exact = np.exp(-psi / u_t)

    ends = {0: exact[0], 1: exact[1],
            mesh.n_nodes - 2: exact[-2], mesh.n_nodes - 1: exact[-1]}
    guess = np.ones(mesh.n_nodes)
    sys_ = assemble_continuity(mesh, guess, psi, u_t=u_t, dirichlet=ends)
    got = guess + newton_step(sys_)

    assert got == pytest.approx(exact, rel=1e-10)
    # And the converged state really is a zero of the residual.
    check = assemble_continuity(mesh, got, psi, u_t=u_t, dirichlet=ends)
    assert np.abs(check.residual).max() < 1e-9 * exact.max()


def test_2d0_pure_diffusion_matches_the_1d_answer():
    """psi = 0, Dirichlet 1 and 2 at the ends -> n exactly linear in x.

    The 1D answer on a uniform grid is n(x) = 1 + x/L, and the discrete
    Laplacian reproduces a linear profile exactly. Running it through the 2D
    assembly must change nothing: this is what separates "my assembly is wrong"
    from "my physics is wrong".
    """
    n_cells = 8
    mesh, pts = strip(n_cells)
    xs = np.array([p[0] for p in pts])
    exact = 1.0 + xs / float(xs.max())

    ends = {0: exact[0], 1: exact[1],
            mesh.n_nodes - 2: exact[-2], mesh.n_nodes - 1: exact[-1]}
    guess = np.zeros(mesh.n_nodes)
    sys_ = assemble_continuity(mesh, guess, np.zeros(mesh.n_nodes),
                               dirichlet=ends)
    got = guess + newton_step(sys_)
    assert got == pytest.approx(exact, rel=1e-12)


def test_2d0_solution_does_not_vary_across_the_strip():
    """A 1D problem must produce a y-independent answer on the 2D mesh.

    The two rows are coupled by vertical edges, so this is not automatic: it
    holds only if those edges carry exactly zero flux. A y-dependence here would
    be a real 2D bug that the exactness tests above could not see.
    """
    mesh, pts = strip(5)
    xs = np.array([p[0] for p in pts])
    exact = 1.0 + xs / float(xs.max())
    ends = {0: exact[0], 1: exact[1],
            mesh.n_nodes - 2: exact[-2], mesh.n_nodes - 1: exact[-1]}
    guess = np.zeros(mesh.n_nodes)
    got = guess + newton_step(
        assemble_continuity(mesh, guess, np.zeros(mesh.n_nodes),
                            dirichlet=ends))
    assert got[0::2] == pytest.approx(got[1::2], rel=1e-12)


# --------------------------------------------------------------------------
# The matrix-free path DESIGN-2D asks to keep open
# --------------------------------------------------------------------------

def test_apply_agrees_with_the_assembled_matrix():
    """System.apply must equal to_dense() @ v, or the matrix-free path is a lie.

    Both consume the same triplets and must sum duplicate (i, j) entries the
    same way — which is also what backend.solve_sparse relies on, so this test
    covers that assumption too.
    """
    mesh, _ = strip(4)
    rng = np.random.default_rng(4)
    sys_ = assemble_continuity(mesh, rng.uniform(0.1, 3.0, mesh.n_nodes),
                               rng.uniform(-1.5, 1.5, mesh.n_nodes),
                               dirichlet={0: 1.0, 1: 1.0})
    v = rng.normal(size=mesh.n_nodes)
    assert sys_.apply(v) == pytest.approx(sys_.to_dense() @ v, rel=1e-12)


def test_source_is_weighted_by_node_volume():
    """A source enters as vol_i * s_i, so callers pass a density.

    On the unit square every volume is 1/4, so a constant source of 8 must add
    exactly 2 to every node's residual. Getting this wrong would scale every
    recombination term by the mesh spacing and look like a physics error.
    """
    mesh = square()
    n = np.ones(4)
    psi = np.zeros(4)
    without = assemble_continuity(mesh, n, psi).residual
    with_src = assemble_continuity(mesh, n, psi, source=np.full(4, 8.0)).residual
    assert (with_src - without) == pytest.approx([2.0, 2.0, 2.0, 2.0])


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"u_t": 0.0}, "u_t must be positive"),
    ({"edge_coef": [1.0]}, "one entry per edge"),
    ({"source": [1.0, 2.0]}, "source must have length"),
    ({"dirichlet": {99: 1.0}}, "outside 0"),
])
def test_malformed_input_is_refused(kwargs, match):
    mesh = square()
    with pytest.raises(ValueError, match=match):
        assemble_continuity(mesh, np.ones(4), np.zeros(4), **kwargs)


def test_mismatched_state_length_is_refused():
    mesh = square()
    with pytest.raises(ValueError, match="must both have length 4"):
        assemble_continuity(mesh, np.ones(3), np.zeros(4))


def test_singular_without_dirichlet():
    """No constrained node -> the matrix is singular, and that is correct.

    Recorded as a test rather than a comment because it is the reason the
    dirichlet argument exists. The physical statement: with only zero-flux
    boundaries, the total carrier count is not determined by the equations.
    """
    mesh = square()
    a = assemble_continuity(mesh, np.ones(4), np.zeros(4)).to_dense()
    assert abs(np.linalg.det(a)) < 1e-12

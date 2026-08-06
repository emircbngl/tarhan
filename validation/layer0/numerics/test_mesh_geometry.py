"""Layer-0: box-method mesh geometry, against hand-computed values.

Every expected number here is derived by hand in the docstring that asserts it.
That is what makes it a Layer-0 test in this project: it has to be checkable by
a reader with a pencil, not merely reproducible by the code under test.
"""
import math

import pytest

from tarhan.numerics.mesh import MeshError, build_mesh, is_delaunay

# Reference figure for most of these: the unit square, split by the diagonal
# 0-2 into triangles (0,1,2) and (0,2,3).
#
#   3 (0,1) ---- 2 (1,1)
#       |  \        |
#       |    \      |
#       |      \    |
#   0 (0,0) ---- 1 (1,0)
SQUARE_PTS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
SQUARE_TRIS = [(0, 1, 2), (0, 2, 3)]


def test_square_diagonal_carries_no_flux():
    """The diagonal's Voronoi facet is exactly zero.

    By hand: the angles opposite edge 0-2 sit at node 1 and node 3. At
    node 1 = (1,0) the vectors to nodes 0 and 2 are (-1,0) and (0,1) — a right
    angle. Same at node 3 by symmetry. cot(90 deg) = 0, so

        A_e = 0.5 * (cot 90 + cot 90) * L_e = 0

    Surprising but correct, and worth pinning: in this two-triangle square the
    diagonal transmits nothing and all transport goes around the sides.
    """
    mesh = build_mesh(SQUARE_PTS, SQUARE_TRIS)
    diag = mesh.edge(0, 2)

    assert diag.length == pytest.approx(math.sqrt(2.0))
    assert diag.facet == pytest.approx(0.0, abs=1e-15)
    assert diag.transmissibility == pytest.approx(0.0, abs=1e-15)
    assert not diag.is_boundary


@pytest.mark.parametrize("i,j", [(0, 1), (1, 2), (2, 3), (0, 3)])
def test_square_side_facets_are_half(i, j):
    """Each side has facet 1/2.

    By hand for side 0-1: its only triangle is (0,1,2), so there is a single
    opposite angle, at node 2 = (1,1). The vectors from there to nodes 0 and 1
    are (-1,-1) and (0,-1), which meet at 45 deg, and cot(45 deg) = 1. With
    L_e = 1,

        A_e = 0.5 * 1 * 1 = 0.5

    The other three sides are the same figure rotated.
    """
    mesh = build_mesh(SQUARE_PTS, SQUARE_TRIS)
    e = mesh.edge(i, j)

    assert e.length == pytest.approx(1.0)
    assert e.facet == pytest.approx(0.5)
    assert e.is_boundary


def test_square_topology():
    mesh = build_mesh(SQUARE_PTS, SQUARE_TRIS)
    assert mesh.n_nodes == 4
    assert len(mesh.edges) == 5            # 4 sides + 1 diagonal
    assert mesh.neighbours(0) == [1, 2, 3]
    assert mesh.neighbours(1) == [0, 2]    # node 1 never touches node 3


def test_orientation_does_not_change_geometry():
    """A clockwise triangle gives the same weights as its ccw twin.

    Orientation is normalised on input precisely so that a mesh file's winding
    convention cannot silently flip a sign in the assembled matrix.
    """
    ccw = build_mesh(SQUARE_PTS, [(0, 1, 2), (0, 2, 3)])
    cw = build_mesh(SQUARE_PTS, [(0, 2, 1), (0, 3, 2)])
    assert [e.facet for e in ccw.edges] == pytest.approx([e.facet for e in cw.edges])


# --------------------------------------------------------------------------
# The positivity guard: Delaunay on interior edges, non-obtuseness on the
# boundary. Conflating the two is the trap these tests exist to hold open.
# --------------------------------------------------------------------------

def _kite(h: float):
    """Two triangles sharing edge 0-1, apexes at +/- h.

    Squashing h drives the angles opposite the shared edge past 90 deg, which is
    what turns their cotangents negative. At h = 0.5 both are exactly 90 deg
    (weight 0); below that the Delaunay condition fails.
    """
    return [(0.0, 0.0), (1.0, 0.0), (0.5, h), (0.5, -h)], [(0, 1, 2), (0, 1, 3)]


def test_non_delaunay_mesh_is_refused():
    """h = 0.2 puts both opposite angles near 136 deg, so the cot sum is < 0.

    The message has to say why, because the fix is to re-mesh rather than to
    retry: a clamped negative weight would quietly assemble a matrix that is no
    longer an M-matrix, and the first symptom would be negative carrier
    densities much later in the solve.
    """
    pts, tris = _kite(0.2)
    with pytest.raises(MeshError, match="not Delaunay"):
        build_mesh(pts, tris)
    assert not is_delaunay(pts, tris)


def test_right_angle_kite_is_the_boundary_case():
    """h = 0.5 gives 90 + 90 deg: weight exactly 0, and still accepted.

    Zero is the edge of the allowed region, not outside it; refusing here would
    reject a legitimate mesh.
    """
    pts, tris = _kite(0.5)
    mesh = build_mesh(pts, tris)
    assert mesh.edge(0, 1).facet == pytest.approx(0.0, abs=1e-12)


def test_obtuse_triangle_is_allowed_when_the_obtuse_angle_faces_an_interior_edge():
    """An obtuse triangle does not by itself break positivity.

    Interior edge 0-1 with apexes at (0.5, 0.3) and (0.5, -2.0). Triangle
    (0,1,2) has angles 31/31/118 deg, so it is obtuse — and its obtuse angle is
    the one *facing* the shared edge. The opposite angles sum to 118 + 28 = 146
    deg, under 180, so Delaunay holds and the far triangle's larger cotangent
    absorbs the negative one. Measured smallest facet in this mesh: +0.2577.

    This pins the correction to docs/DESIGN-2D.md, which first claimed the whole
    mesh had to be non-obtuse.
    """
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.3), (0.5, -2.0)]
    tris = [(0, 1, 2), (0, 1, 3)]

    apex = pts[2]                                   # the obtuse angle, at node 2
    ux, uy = pts[0][0] - apex[0], pts[0][1] - apex[1]
    vx, vy = pts[1][0] - apex[0], pts[1][1] - apex[1]
    assert math.degrees(math.atan2(abs(ux * vy - uy * vx),
                                   ux * vx + uy * vy)) > 90.0

    mesh = build_mesh(pts, tris)                    # must not raise
    assert all(e.facet >= 0.0 for e in mesh.edges)
    assert is_delaunay(pts, tris)


def test_obtuse_angle_facing_a_boundary_edge_is_refused():
    """...but on a boundary edge there is nothing to compensate with.

    A boundary edge has a single opposite angle, so its facet is
    (L_e / 2) * cot(gamma), negative exactly when gamma > 90 deg — the
    circumcentre has fallen outside the domain across that edge.

    Here the angle at node 0, facing boundary edge 1-2, is obtuse. The refusal
    must NOT say "not Delaunay": there is no edge to flip, and the reader would
    go looking for one. The fix is to refine the boundary.
    """
    pts = [(0.0, 0.0), (1.0, 0.0), (-0.6, 0.9), (0.5, -1.2)]
    tris = [(0, 1, 2), (0, 1, 3)]

    with pytest.raises(MeshError, match="boundary edge") as excinfo:
        build_mesh(pts, tris)
    assert "not Delaunay" not in str(excinfo.value)
    assert "<= 90 deg" in str(excinfo.value)


# --------------------------------------------------------------------------
# Input validation: refuse loudly rather than compute nonsense
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pts,tris,match", [
    ([(0.0, 0.0), (1.0, 0.0)], [(0, 1, 0)], "at least 3 nodes"),
    (SQUARE_PTS, [(0, 1, 9)], r"outside 0\.\.3"),
    (SQUARE_PTS, [(0, 1, 1)], "repeats a node"),
    (SQUARE_PTS, [(0, 1)], "exactly 3 nodes"),
    ([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], [(0, 1, 2)], "degenerate"),
    (SQUARE_PTS, [], "no triangles"),
])
def test_malformed_input_is_refused(pts, tris, match):
    with pytest.raises(MeshError, match=match):
        build_mesh(pts, tris)


# --------------------------------------------------------------------------
# Scale invariance. Every tolerance in mesh.py is relative; these tests exist
# because the first version's were absolute, and unit-square figures cannot
# see the difference.
# --------------------------------------------------------------------------

DEVICE = 1e-5      # cm — a semiconductor device is about this wide


def test_device_scale_mesh_is_accepted():
    """The unit square shrunk to 1e-5 cm must still be a valid mesh.

    Twice its signed area is then 1e-10, and every triangle would have been
    refused as "degenerate (zero area)" by the original absolute floor of
    1e-14. Nothing about the geometry changed — only the unit — so a validator
    that changes its mind here is measuring the wrong thing.

    Regression for a real failure: DEVSIM's own 2D diode mesh (495 nodes, 880
    elements, 1e-5 cm across) had 40 elements rejected for exactly this reason,
    which blocked the entire 2D-1 comparison.
    """
    pts = [(x * DEVICE, y * DEVICE) for x, y in SQUARE_PTS]
    mesh = build_mesh(pts, SQUARE_TRIS)
    assert mesh.n_nodes == 4
    assert all(e.facet >= 0.0 for e in mesh.edges)


def test_transmissibility_is_scale_invariant():
    """A_e / L_e is unchanged by a uniform rescaling.

    Facet and length are both lengths, so their ratio is dimensionless and must
    come out identical whether the mesh is expressed in centimetres or in unit
    squares. The assembled matrix depends on the mesh ONLY through this ratio,
    so the physics is scale-free and the validator has to be too.
    """
    unit = build_mesh(SQUARE_PTS, SQUARE_TRIS)
    tiny = build_mesh([(x * DEVICE, y * DEVICE) for x, y in SQUARE_PTS],
                      SQUARE_TRIS)
    assert ([e.transmissibility for e in tiny.edges]
            == pytest.approx([e.transmissibility for e in unit.edges],
                             rel=1e-12, abs=1e-15))
    # ...while the raw lengths did scale, so this is not a no-op comparison.
    assert tiny.edge(0, 1).length == pytest.approx(DEVICE * unit.edge(0, 1).length)


def test_a_sliver_at_device_scale_is_accepted():
    """0.46 deg is thin, not degenerate — and DEVSIM's mesh really contains it.

    Reproduces the worst element of DEVSIM's diode mesh: edges of 1e-6, 1e-6 and
    8.082e-9 cm, with |2A| = 8.082e-15, which the old absolute floor rejected.
    Its scale-invariant shape measure |2A|/Lmax^2 is 8.08e-3 — small, but many
    orders of magnitude above any sane degeneracy tolerance.

    mesh.py computes cotangents through atan2 specifically so that slivers stay
    well conditioned; refusing them would make that effort pointless.
    """
    pts = [(0.0, 0.0), (1e-6, 0.0), (0.0, 8.082e-9), (1e-6, 8.082e-9)]
    mesh = build_mesh(pts, [(0, 1, 3), (0, 3, 2)])
    assert all(e.facet >= 0.0 for e in mesh.edges)
    assert 8.082e-15 / (1e-6 ** 2) == pytest.approx(8.082e-3, rel=1e-3)


def test_truly_collinear_is_still_refused_at_device_scale():
    """Relative does not mean permissive: zero area is still zero area.

    Three collinear points 1e-5 cm apart have |2A|/Lmax^2 = 0 exactly, so the
    relative test refuses them just as the absolute one did. The change was to
    stop conflating "small" with "flat", not to stop checking.
    """
    pts = [(0.0, 0.0), (DEVICE, 0.0), (2 * DEVICE, 0.0)]
    with pytest.raises(MeshError, match="collinear"):
        build_mesh(pts, [(0, 1, 2)])


def test_duplicate_nodes_are_refused_relative_to_extent():
    """A duplicated node is short compared to the DOMAIN, not compared to 1.

    At device scale an honest edge is ~1e-6 long, so an absolute "zero length"
    floor of 1e-14 could never fire and the check would be decoration.
    """
    pts = [(0.0, 0.0), (DEVICE, 0.0), (DEVICE, DEVICE), (0.0, 0.0)]
    with pytest.raises(MeshError, match="zero length|degenerate|collinear"):
        build_mesh(pts, [(0, 1, 2), (0, 2, 3)])


def test_non_manifold_edge_is_refused():
    """Three triangles on one edge is not a 2D mesh; say so rather than average."""
    pts = SQUARE_PTS + [(0.5, 2.0)]
    tris = [(0, 2, 1), (0, 2, 3), (0, 2, 4)]
    with pytest.raises(MeshError, match="shared by 3 triangles"):
        build_mesh(pts, tris)

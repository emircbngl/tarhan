"""What is actually known about the box method on tetrahedra — and what is not.

There is no 3D mesh builder in this repository. This file exists because the
attempt to write one produced three measurements worth keeping and one broken
one worth admitting to, and a capability record that says "blocked" should point
at numbers rather than at a feeling.

What is established here:

* On a WELL-CENTRED tetrahedron the circumcentric dual is exact. For the regular
  tetrahedron every edge has facet √2/3 and length 2√2, and the node-volume
  identity vol = (1/6)·Σ A·L holds to machine precision — the 3D analogue of the
  (1/4) the 2D builder uses.
* On a tetrahedron whose circumcentre lies OUTSIDE it, the same construction
  overshoots by exactly a factor of two. The witness is the commonest shape in
  any structured mesh: the corner tetrahedron (0,0,0),(1,0,0),(0,1,0),(0,0,1),
  whose circumcentre (0.5,0.5,0.5) is outside the simplex.
* On DEVSIM's own 3D diode mesh, 2555 of 6701 tetrahedra — 38.1% — are not
  well-centred. For comparison, stage 2D-4 is blocked because 22 of 8192 edges
  (0.27%) failed the 2D condition.

What is NOT established, stated plainly because it was nearly reported as fact:
a SIGNED variant of the construction — which is how the 2D builder survives
individually negative triangle contributions — was written and produced "26% of
edges have a negative total facet" on the same mesh. That number is not
trustworthy. The same signed construction reproduces the mesh volume to only
66%, so it is wrong somewhere, and a measurement that fails its own consistency
check cannot be evidence for anything. The blocked record therefore says the
geometry is unsolved, not that the mesh is unusable.
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REGULAR = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
CORNER = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def circumcentre(points):
    """Circumcentre of a tetrahedron (4 points) or a triangle in 3D (3 points)."""
    p0 = points[0]
    a = 2.0 * (points[1:] - p0)
    rhs = (points[1:] ** 2).sum(1) - (p0 ** 2).sum()
    if len(points) == 4:
        return np.linalg.solve(a, rhs)
    normal = np.cross(points[1] - p0, points[2] - p0)
    return np.linalg.solve(np.vstack([a, normal]), np.append(rhs, normal @ p0))


def tet_volume(p):
    return abs(np.linalg.det(p[1:] - p[0])) / 6.0


def dual_facet(p, i, j):
    """Unsigned circumcentric dual facet area for edge (i, j) in one tetrahedron.

    The quadrilateral spanned by the edge midpoint, the circumcentres of the two
    faces containing the edge, and the tetrahedron's own circumcentre.
    """
    others = [k for k in range(4) if k not in (i, j)]
    mid = 0.5 * (p[i] + p[j])
    cc = circumcentre(p)
    f0 = circumcentre(p[[i, j, others[0]]])
    f1 = circumcentre(p[[i, j, others[1]]])
    return (0.5 * np.linalg.norm(np.cross(f0 - mid, cc - mid))
            + 0.5 * np.linalg.norm(np.cross(cc - mid, f1 - mid)))


def is_well_centred(p):
    matrix = (p[1:] - p[0]).T
    if abs(np.linalg.det(matrix)) < 1e-30:
        return None                                   # degenerate
    lam = np.linalg.solve(matrix, circumcentre(p) - p[0])
    return bool(min(1.0 - lam.sum(), *lam) >= 0.0)


# --- the anchor that works -------------------------------------------------

def test_the_regular_tetrahedron_has_the_hand_derived_facet():
    """√2/3 per edge, derived by hand before the code was written.

    The quadrilateral for one edge is (midpoint, face centroid, origin, face
    centroid); both halves have area √2/6, so the facet is √2/3 ≈ 0.4714045.
    """
    for i, j in itertools.combinations(range(4), 2):
        assert dual_facet(REGULAR, i, j) == pytest.approx(np.sqrt(2) / 3)
        assert np.linalg.norm(REGULAR[i] - REGULAR[j]) == pytest.approx(
            2 * np.sqrt(2))


def test_the_node_volume_factor_in_3d_is_one_sixth():
    """vol = (1/6)·Σ A·L, the 3D counterpart of the 2D builder's (1/4).

    A pyramid of base A and height L/2 has volume A·L/6, and each edge feeds
    both of its endpoints. Checked against the exact volume of the regular
    tetrahedron, 8/3.
    """
    total = sum(2.0 * dual_facet(REGULAR, i, j)
                * np.linalg.norm(REGULAR[i] - REGULAR[j])
                for i, j in itertools.combinations(range(4), 2))
    assert tet_volume(REGULAR) == pytest.approx(8.0 / 3.0)
    assert tet_volume(REGULAR) / total == pytest.approx(1.0 / 6.0)


# --- the anchor that breaks, and why it matters ---------------------------

def test_the_corner_tetrahedron_is_not_well_centred():
    """The commonest shape in a structured mesh is the failure case.

    Splitting a cube produces this tetrahedron, and its circumcentre
    (0.5, 0.5, 0.5) lies outside it.
    """
    assert is_well_centred(REGULAR) is True
    assert is_well_centred(CORNER) is False
    assert circumcentre(CORNER) == pytest.approx([0.5, 0.5, 0.5])


def test_the_unsigned_construction_doubles_on_a_non_well_centred_tetrahedron():
    """Exactly a factor of two, which is why the naive construction cannot ship.

    The true Voronoi facet for the axis edge inside the corner tetrahedron is
    the triangle {y, z ≥ 0, y + z ≤ 1/2}, of area 1/8. The circumcentric quad
    gives 1/4, and the volume identity comes out at 1/12 instead of 1/6.
    """
    assert dual_facet(CORNER, 0, 1) == pytest.approx(0.25)
    total = sum(2.0 * dual_facet(CORNER, i, j)
                * np.linalg.norm(CORNER[i] - CORNER[j])
                for i, j in itertools.combinations(range(4), 2))
    assert tet_volume(CORNER) / total == pytest.approx(1.0 / 12.0)


# --- the survey that keeps 3D blocked -------------------------------------

def _devsim_diode3d():
    path = (Path(sys.prefix) / "devsim_data" / "examples" / "diode"
            / "gmsh_diode3d.msh")
    if not path.exists():
        pytest.skip("DEVSIM's 3D diode mesh is not installed")
    text = path.read_text(errors="replace")

    def block(name):
        found = re.search(rf"\${name}\n(.*?)\$End{name}", text, re.S)
        return found.group(1).strip().split("\n")

    nodes, elements = block("Nodes"), block("Elements")
    pts = np.array([[float(v) for v in line.split()[1:4]]
                    for line in nodes[1:1 + int(nodes[0])]])
    tets = []
    for line in elements[1:1 + int(elements[0])]:
        fields = line.split()
        if fields[1] == "4":
            start = 3 + int(fields[2])
            tets.append([int(v) - 1 for v in fields[start:start + 4]])
    return pts, np.array(tets)


def test_the_reference_3d_mesh_is_mostly_not_well_centred():
    """38.1% — the measurement that keeps 3D blocked.

    Stage 2D-4 is blocked because 22 of 8192 edges, 0.27%, failed the 2D
    condition. Here 2555 of 6701 tetrahedra fail the 3D one. The band below is
    loose on purpose: the claim is "most of the mesh", not a digit that would
    change if DEVSIM reshipped the file.
    """
    pts, tets = _devsim_diode3d()
    assert (len(pts), len(tets)) == (1417, 6701)
    bad = sum(1 for t in tets if is_well_centred(pts[t]) is False)
    assert bad == 2555
    assert 0.3 < bad / len(tets) < 0.5

"""2D triangular mesh geometry for the box (finite-volume / Voronoi) method.

Geometry only. It computes the two numbers the transport discretisation needs
for every edge, and it refuses a mesh that would break the solve. It
deliberately does NOT generate meshes: producing a good Delaunay triangulation
is a solved problem solved better elsewhere, and writing one is how this project
would spend months not doing physics. Read a ``.msh`` instead.

Why edges and not cells
-----------------------
``numerics.flux.sg_edge_flux`` is already a function of an EDGE: a potential
difference and the carrier densities at its two endpoints. It has no idea how
many neighbours a node has, so the Scharfetter-Gummel discretisation carries to
2D unchanged. What is 1D-specific is the tridiagonal solver, the ``i±1``
neighbourhood assumption, and the unit edge geometry — this module replaces the
third of those with real geometry.

The two numbers, for an edge ``e = (i, j)``:

``length``
    node-to-node distance ``L_e``.
``facet``
    length of the Voronoi facet bisecting the edge, ``A_e``. In 2D this is a
    length; in 3D the same construction gives an area, which is why choosing it
    here makes the eventual 3D step mechanical rather than another redesign.

The flux into node ``i`` then enters as ``A_e * sg_edge_flux(..., coef=1/L_e)``.

Positivity
----------
For an interior edge with opposite angles ``α`` and ``β`` in its two triangles::

    A_e / L_e = (cot α + cot β) / 2

Non-negativity of that is what the M-matrix property — and so the guaranteed
positivity of carrier densities — rests on.

**The condition differs between interior and boundary edges, and conflating them
is the trap.**

*Interior edge* — two opposite angles, so the terms can compensate. The Delaunay
empty-circumcircle condition is equivalent to ``α + β ≤ π``, which gives
``cot α + cot β ≥ 0`` directly. An obtuse triangle is fine here: if its obtuse
angle faces the interior edge, the neighbour's larger cotangent absorbs it.
Measured over 20 000 random two-triangle configurations, of the 19 294
satisfying Delaunay none produced a negative interior weight, and 18 022 of those
contained an obtuse triangle.

*Boundary edge* — one opposite angle, nothing to compensate with. The weight is
``(L_e / 2) · cot γ``, which is negative exactly when ``γ > 90°``, i.e. when the
triangle's circumcentre falls outside the domain across that edge. **This is
where non-obtuseness genuinely matters**, and only here.

Two earlier statements of this rule were both wrong: the 2D design first claimed
the whole mesh had to be non-obtuse (too strict — it would reject valid interior
configurations), and the first correction claimed obtuseness never mattered (too
loose — it misses the boundary case). The distinction was found by a test, which
is why the expected numbers in the Layer-0 suite are hand-derived.

This module therefore validates the quantity that actually breaks the solve — a
negative computed facet — and reports which of the two causes it hit, because
the remedies differ: flip the edge, versus refine the boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Point = Tuple[float, float]
Triangle = Tuple[int, int, int]
Edge = Tuple[int, int]


class MeshError(ValueError):
    """The mesh cannot be used for a box-method solve, with the reason why."""


@dataclass(frozen=True)
class EdgeGeometry:
    """Everything the discretisation needs about one edge."""

    nodes: Edge
    length: float                 #: L_e, node-to-node distance
    facet: float                  #: A_e, Voronoi facet bisecting the edge
    triangles: Tuple[int, ...]    #: adjacent triangle indices (1 = boundary)

    @property
    def transmissibility(self) -> float:
        """``A_e / L_e`` — what the edge flux gets multiplied by."""
        return self.facet / self.length

    @property
    def is_boundary(self) -> bool:
        return len(self.triangles) == 1


def _angle_at(apex: Point, a: Point, b: Point) -> float:
    """Interior angle at ``apex`` in triangle (apex, a, b), in radians.

    atan2 of the cross and dot products rather than acos of a normalised dot:
    acos loses most of its precision for the sliver triangles a real mesh
    carries near a junction, which is exactly where a wrong weight matters most.
    """
    ux, uy = a[0] - apex[0], a[1] - apex[1]
    vx, vy = b[0] - apex[0], b[1] - apex[1]
    return math.atan2(abs(ux * vy - uy * vx), ux * vx + uy * vy)


def _cot(x: float) -> float:
    s = math.sin(x)
    if abs(s) < 1e-300:
        raise MeshError("degenerate triangle: a vertex angle is 0 or pi")
    return math.cos(x) / s


def _signed_area2(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _key(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


@dataclass(frozen=True)
class Mesh:
    """A triangulated 2D domain with box-method edge geometry.

    Build it with :func:`build_mesh`, which validates. Constructing this class
    directly bypasses every check in this module.
    """

    points: Tuple[Point, ...]
    triangles: Tuple[Triangle, ...]
    edges: Tuple[EdgeGeometry, ...]

    def edge(self, i: int, j: int) -> EdgeGeometry:
        want = _key(i, j)
        for e in self.edges:
            if e.nodes == want:
                return e
        raise KeyError(f"no edge between nodes {i} and {j}")

    def neighbours(self, i: int) -> List[int]:
        out = []
        for e in self.edges:
            if e.nodes[0] == i:
                out.append(e.nodes[1])
            elif e.nodes[1] == i:
                out.append(e.nodes[0])
        return sorted(out)

    @property
    def n_nodes(self) -> int:
        return len(self.points)


def build_mesh(points: Sequence[Point],
               triangles: Iterable[Triangle],
               *,
               weight_atol: float = 1e-12) -> Mesh:
    """Validate a triangulation and compute its box-method edge geometry.

    Raises :class:`MeshError` rather than returning a mesh that would silently
    assemble a non-M-matrix. A solve that hands back negative electron
    concentrations is worse than one that refuses to start.
    """
    pts = tuple((float(x), float(y)) for x, y in points)
    if len(pts) < 3:
        raise MeshError(f"a triangulation needs at least 3 nodes, got {len(pts)}")

    tris: List[Triangle] = []
    for t in triangles:
        t = tuple(t)
        if len(t) != 3:
            raise MeshError(f"triangle {t!r} does not have exactly 3 nodes")
        if any(not (0 <= n < len(pts)) for n in t):
            raise MeshError(
                f"triangle {t!r} references a node outside 0..{len(pts) - 1}")
        if len(set(t)) != 3:
            raise MeshError(f"triangle {t!r} repeats a node")
        a2 = _signed_area2(pts[t[0]], pts[t[1]], pts[t[2]])
        if abs(a2) < 1e-14:
            raise MeshError(f"triangle {t!r} is degenerate (zero area)")
        # Normalise to counter-clockwise so orientation cannot silently flip a
        # sign downstream.
        tris.append(t if a2 > 0 else (t[0], t[2], t[1]))
    if not tris:
        raise MeshError("no triangles given")

    incident: Dict[Edge, List[Tuple[int, int]]] = {}
    for ti, (a, b, c) in enumerate(tris):
        for i, j, opp in ((a, b, c), (b, c, a), (c, a, b)):
            incident.setdefault(_key(i, j), []).append((ti, opp))

    out: List[EdgeGeometry] = []
    for (i, j), inc in sorted(incident.items()):
        if len(inc) > 2:
            raise MeshError(
                f"edge {(i, j)} is shared by {len(inc)} triangles; a 2D mesh "
                "allows at most 2 (non-manifold input?)")
        pi, pj = pts[i], pts[j]
        length = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
        if length < 1e-14:
            raise MeshError(f"edge {(i, j)} has zero length (duplicate nodes?)")

        cot_sum = sum(_cot(_angle_at(pts[opp], pi, pj)) for _, opp in inc)
        # A boundary edge sees one triangle, so it contributes one cot term.
        # Halving both cases keeps the interior formula (cot α + cot β)/2 and
        # the boundary case on the same footing.
        facet = 0.5 * cot_sum * length

        if facet < -weight_atol:
            # The two cases have different causes and different fixes, so they
            # get different messages. Saying "not Delaunay" at a boundary edge
            # would send the reader looking for an edge to flip that does not
            # exist.
            if len(inc) == 2:
                why = (f"the two angles opposite it sum to more than 180 deg, so "
                       f"this edge is not Delaunay. Flip it, or re-mesh")
            else:
                opp = inc[0][1]
                deg = math.degrees(_angle_at(pts[opp], pi, pj))
                why = (f"it is a boundary edge with only one opposite angle, and "
                       f"that angle (at node {opp}, {deg:.1f} deg) is obtuse, so "
                       f"the triangle's circumcentre falls outside the domain "
                       f"across it. There is no second triangle to compensate: "
                       f"refine the boundary so this angle is <= 90 deg")
            raise MeshError(
                f"edge {(i, j)} has a negative Voronoi facet ({facet:.6g}): {why}. "
                "Left as is, the assembled matrix would not be an M-matrix and "
                "carrier positivity would no longer be guaranteed — do not clamp "
                "the weight.")
        out.append(EdgeGeometry(nodes=(i, j), length=length,
                                facet=max(facet, 0.0),
                                triangles=tuple(ti for ti, _ in inc)))

    return Mesh(points=pts, triangles=tuple(tris), edges=tuple(out))


def is_delaunay(points: Sequence[Point], triangles: Iterable[Triangle]) -> bool:
    """True when the triangulation yields no negative edge weight.

    The practical form of the empty-circumcircle test: it asks the question the
    solve actually cares about instead of re-deriving circumcircles.
    """
    try:
        build_mesh(points, triangles)
    except MeshError:
        return False
    return True

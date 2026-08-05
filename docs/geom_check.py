#!/usr/bin/env python3
"""Numerical evidence for the three geometric claims numerics/mesh.py rests on.

Run it: ``python3 docs/geom_check.py``. Every number quoted in mesh.py's module
docstring and in DESIGN-2D.md comes from here, so the claims can be re-checked
rather than taken on trust. Nothing here is imported by the package.

The claims, in dependency order:

1. Empty circumcircle  <=>  alpha + beta <= pi   (the two angles opposite a
   shared edge). This is the one that had to be checked INDEPENDENTLY: the
   equivalence alpha+beta <= pi <=> cot alpha + cot beta >= 0 is an identity, so
   testing Delaunay *via* the angle sum and then checking the cotangent sum
   proves nothing. The circumcircle determinant is the independent oracle.
2. A_e = (L_e / 2) * (cot alpha + cot beta) really is the Voronoi facet, i.e.
   the distance between the two triangles' circumcentres.
3. Delaunay therefore gives non-negative INTERIOR weights, including when a
   triangle is obtuse — while a BOUNDARY edge, having one term and nothing to
   compensate with, needs its single opposite angle <= 90 deg.
"""
import math
import random

A, B = (0.0, 0.0), (1.0, 0.0)


def angle_at(p, q, r):
    u = (q[0] - p[0], q[1] - p[1])
    v = (r[0] - p[0], r[1] - p[1])
    return math.atan2(abs(u[0] * v[1] - u[1] * v[0]), u[0] * v[0] + u[1] * v[1])


def cot(x):
    return math.cos(x) / math.sin(x)


def circumcentre(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


def in_circumcircle(a, b, c, d):
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    det = ((ax * ax + ay * ay) * (bx * cy - by * cx)
           - (bx * bx + by * by) * (ax * cy - ay * cx)
           + (cx * cx + cy * cy) * (ax * by - ay * bx))
    orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return det * (1 if orient > 0 else -1) > 1e-12


def min_angle(t):
    return min(angle_at(t[i], t[(i + 1) % 3], t[(i + 2) % 3]) for i in range(3))


def claim_1_delaunay_equals_angle_sum(n=200_000, seed=7):
    """Independent oracle: circumcircle determinant vs the angle-sum test."""
    random.seed(seed)
    agree = disagree = 0
    for _ in range(n):
        C = (random.uniform(-2, 3), random.uniform(0.02, 2.0))
        D = (random.uniform(-2, 3), random.uniform(-2.0, -0.02))
        empty = not in_circumcircle(A, B, C, D)
        anglesum = angle_at(C, A, B) + angle_at(D, A, B) <= math.pi + 1e-12
        agree += empty == anglesum
        disagree += empty != anglesum
    print(f"1. empty-circumcircle <=> alpha+beta <= pi")
    print(f"   {agree:,} agree, {disagree:,} disagree over {n:,} samples"
          f"  -> {'VERIFIED' if disagree == 0 else 'FALSE'}")
    return disagree == 0


def claim_2_cotangent_formula_is_the_facet(n=200_000, seed=11, min_deg=15):
    """(L/2)(cot a + cot b) vs the distance between the two circumcentres.

    Restricted to Delaunay, well-shaped triangles. Slivers are excluded on
    purpose and that exclusion is itself the finding: below roughly 15 deg the
    circumcentres run off to infinity and their difference loses all precision,
    while the cotangent form stays well conditioned. That is why mesh.py
    computes the cotangent form and never constructs a circumcentre.
    """
    random.seed(seed)
    worst = 0.0
    used = 0
    for _ in range(n):
        C = (random.uniform(-0.5, 1.5), random.uniform(0.3, 1.5))
        D = (random.uniform(-0.5, 1.5), random.uniform(-1.5, -0.3))
        a, b = angle_at(C, A, B), angle_at(D, A, B)
        if a + b > math.pi:
            continue
        if min(min_angle((A, B, C)), min_angle((A, B, D))) < math.radians(min_deg):
            continue
        o1, o2 = circumcentre(A, B, C), circumcentre(A, B, D)
        geometric = math.hypot(o1[0] - o2[0], o1[1] - o2[1])
        formula = 0.5 * 1.0 * (cot(a) + cot(b))
        worst = max(worst, abs(geometric - formula))
        used += 1
    print(f"2. A_e = (L/2)(cot a + cot b) equals |circumcentre_1 - circumcentre_2|")
    print(f"   {used:,} Delaunay, well-shaped (min angle >= {min_deg} deg) samples")
    print(f"   worst absolute difference {worst:.3e}"
          f"  -> {'VERIFIED' if worst < 1e-9 else 'FALSE'}")
    return worst < 1e-9


def claim_3_obtuse_is_fine_on_interior_edges(n=20_000, seed=0):
    random.seed(seed)
    delaunay = negative = obtuse = obtuse_positive = 0
    for _ in range(n):
        C = (random.uniform(-1, 2), random.uniform(0.05, 1.5))
        D = (random.uniform(-1, 2), random.uniform(-1.5, -0.05))
        a, b = angle_at(C, A, B), angle_at(D, A, B)
        if a + b > math.pi + 1e-12:
            continue
        delaunay += 1
        w = 0.5 * (cot(a) + cot(b))
        negative += w < -1e-12
        is_obtuse = any(max(angle_at(t[i], t[(i + 1) % 3], t[(i + 2) % 3])
                            for i in range(3)) > math.pi / 2 + 1e-12
                        for t in ((A, B, C), (A, B, D)))
        if is_obtuse:
            obtuse += 1
            obtuse_positive += w > 1e-12
    print(f"3. Delaunay => non-negative INTERIOR weight, obtuse triangles included")
    print(f"   {delaunay:,} Delaunay samples, {negative:,} with a negative weight")
    print(f"   {obtuse:,} of them contain an obtuse triangle,"
          f" {obtuse_positive:,} of those still positive"
          f"  -> {'VERIFIED' if negative == 0 and obtuse == obtuse_positive else 'FALSE'}")
    return negative == 0 and obtuse == obtuse_positive


def claim_3b_boundary_needs_non_obtuse():
    """One term, nothing to compensate: negative exactly when gamma > 90 deg."""
    bad = 0
    for deg in (30, 60, 89, 90, 91, 120, 150):
        gamma = math.radians(deg)
        facet = 0.5 * cot(gamma)
        expect_negative = deg > 90
        got_negative = facet < -1e-15
        bad += expect_negative != got_negative
    print(f"4. boundary edge facet = (L/2) cot(gamma) < 0 exactly when gamma > 90 deg")
    print(f"   {'VERIFIED' if bad == 0 else 'FALSE'}")
    return bad == 0


if __name__ == "__main__":
    ok = all([
        claim_1_delaunay_equals_angle_sum(),
        claim_2_cotangent_formula_is_the_facet(),
        claim_3_obtuse_is_fine_on_interior_edges(),
        claim_3b_boundary_needs_non_obtuse(),
    ])
    raise SystemExit(0 if ok else 1)

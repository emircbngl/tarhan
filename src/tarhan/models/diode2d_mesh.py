"""A rectangular pn-diode mesh, generated from scalars.

``PNDiode2D`` takes points, triangles, doping and contacts. Until now the only
things that could produce those were a fixture defined inside a test file and
DEVSIM's own oracle mesh — so the 2D physics was validated while nothing in the
package could hand a device to a caller. That is why ``run solve`` reached only
1D: not because the 2D solver was unproven, but because there was no device to
give it.

**Scope, deliberately narrow.** This is not mesh generation. It is one shape —
an axis-aligned rectangle, split into right triangles on a structured grid,
doped p on the left and n on the right of a junction at x=0. It cannot mesh a
MOSFET, a curved boundary or a refined region. That narrowness is what makes it
safe to ship: every mesh it produces is describable by the handful of scalars
below, so a run's lock file can record the DEVICE rather than fifteen thousand
coordinates, and re-running the same scalars reproduces the same mesh exactly.

**The x grid mirrors PNDiode1D's.** Geometric growth away from the junction,
same ``h0`` and ``gamma``, because that is what makes a 1D-on-2D comparison
meaningful: with no variation along y, this mesh must reproduce the 1D solver's
answer on the same node positions. A uniform x grid would resolve the depletion
region so much worse that a disagreement could not be attributed to the
discretisation rather than to the mesh.

The y direction is uniform and coarse by default. Nothing in an ideal 1D-like
diode varies along it, so spending nodes there buys nothing; it exists so the
problem is genuinely two-dimensional and the contacts are edges rather than
points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from tarhan.models.pn1d import (MAX_GRID_NODES,  # noqa: E402
                                estimated_nodes)


class MeshError(ValueError):
    """A mesh request that cannot produce a usable device."""


@dataclass
class RectangularDiode2D:
    """The scalars that fully determine a rectangular pn-diode mesh.

    Every field is an input to the hash that names a run, so anything that
    changes the mesh must live here and nothing that does not may.
    """

    len_p: float = 3e-4        # cm, p side, left of the junction
    len_n: float = 3e-4        # cm, n side
    height: float = 1e-4       # cm, extent along y
    h0: float = 5e-7           # cm, first step at the junction (5 nm)
    gamma: float = 1.06        # geometric growth away from the junction
    ny: int = 4                # uniform cells along y
    Na: float = 1e16           # cm^-3, p side
    Nd: float = 1e16           # cm^-3, n side

    def __post_init__(self):
        for name in ("len_p", "len_n", "height", "h0", "Na", "Nd"):
            value = getattr(self, name)
            if not (value > 0.0 and math.isfinite(value)):
                raise MeshError(f"{name}={value}: must be positive and finite")
        if not math.isfinite(self.gamma) or self.gamma < 1.0:
            # gamma < 1 shrinks the step away from the junction, so the walk in
            # x_nodes converges to a point short of the boundary and never
            # terminates. Refusing beats hanging.
            raise MeshError(f"gamma={self.gamma}: must be >= 1")
        if int(self.ny) != self.ny or self.ny < 1:
            raise MeshError(f"ny={self.ny}: must be a positive integer")
        # The same bound as the 1D grid, applied to the PRODUCT: a modest nx
        # and a large ny multiply into a mesh nothing can build. Estimated
        # before any of it is allocated.
        columns = sum(estimated_nodes(side, self.h0, self.gamma)
                      for side in (self.len_p, self.len_n))
        total = columns * (int(self.ny) + 1)
        if not math.isfinite(total) or total > MAX_GRID_NODES:
            raise MeshError(
                f"h0={self.h0:g}, gamma={self.gamma:g} and ny={self.ny} need "
                f"about {total:.3g} nodes, over the {MAX_GRID_NODES:,} limit")

    def x_nodes(self) -> np.ndarray:
        """Junction at x=0, geometric growth outward. Mirrors PNDiode1D."""
        def side(length):
            xs, h = [0.0], self.h0
            while xs[-1] < length:
                xs.append(xs[-1] + h)
                h *= self.gamma
            xs[-1] = length            # land exactly on the boundary
            return np.asarray(xs)

        left = -side(self.len_p)[::-1]
        right = side(self.len_n)
        return np.concatenate([left, right[1:]])   # the junction node once

    def y_nodes(self) -> np.ndarray:
        return np.linspace(0.0, self.height, int(self.ny) + 1)


def build(spec: RectangularDiode2D = None) -> Dict[str, object]:
    """Return the keyword arguments ``PNDiode2D`` needs.

    Nodes are numbered column-major in x and row-major in y, so a column of
    nodes at one x is contiguous — which is what makes the contacts below
    slices rather than a search.
    """
    spec = RectangularDiode2D() if spec is None else spec
    xs, ys = spec.x_nodes(), spec.y_nodes()
    nx, ny = len(xs), len(ys)

    points: List[Tuple[float, float]] = [(float(x), float(y))
                                         for x in xs for y in ys]

    def node(i, j):                       # column i, row j
        return i * ny + j

    triangles: List[Tuple[int, int, int]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = node(i, j), node(i + 1, j)
            c, d = node(i + 1, j + 1), node(i, j + 1)
            # Both wound counter-clockwise. Orientation is asserted in the
            # tests rather than assumed: a flipped triangle has negative signed
            # area, and an assembly that sums |area| would hide it while one
            # that sums signed area would cancel it against its neighbour.
            triangles += [(a, b, c), (a, c, d)]

    x_of_node = np.asarray([p[0] for p in points])
    doping = np.where(x_of_node < 0.0, -float(spec.Na), float(spec.Nd))
    # The junction column is the metallurgical junction: net doping is zero
    # there, rather than assigning the column arbitrarily to one side.
    doping[x_of_node == 0.0] = 0.0

    contacts = {"anode": list(range(0, ny)),                     # x = -len_p
                "cathode": list(range((nx - 1) * ny, nx * ny))}  # x = +len_n
    return {"points": points, "triangles": triangles,
            "net_doping": doping, "contacts": contacts,
            "biased_contact": "anode"}


def device(spec: RectangularDiode2D = None, **overrides):
    """Build a ready ``PNDiode2D``. Physical constants pass straight through."""
    from tarhan.models.pn2d import PNDiode2D

    return PNDiode2D(**build(spec), **overrides)

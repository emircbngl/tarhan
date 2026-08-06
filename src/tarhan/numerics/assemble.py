"""Edge-loop assembly of the steady continuity residual and its Jacobian.

Dimension-agnostic by construction: it walks :class:`~tarhan.numerics.mesh.Mesh`
edges and calls ``flux.bernoulli``. Nothing here knows whether the mesh is 2D or
3D — in 3D the only change is that ``EdgeGeometry.facet`` becomes an area, which
is exactly why DESIGN-2D §2 chose the Voronoi facet as the stored quantity.

Sign convention, because everything below rests on it
-----------------------------------------------------
For an edge ``e = (i, j)`` write ``x = (psi_j - psi_i) / U_T`` and
``w = coef_e * A_e / L_e``. The Scharfetter-Gummel particle flux **into** node
``i`` from its neighbour ``j`` is::

    Jin(i<-j) = w * (B(-x) * n_j - B(x) * n_i)

Two checks that this is the right object rather than a plausible one:

* At ``psi = 0`` it is ``w * (n_j - n_i)`` — positive when the neighbour is
  denser, i.e. diffusion flows toward ``i``. So it really is an inflow.
* Swapping ``i`` and ``j`` sends ``x -> -x`` and yields exactly ``-Jin``.
  Conservation is therefore **structural**, not something the assembly has to
  arrange: whatever leaves one node arrives at the other, to the last bit.

The residual is the negated inflow sum, ``F_i = -sum_j Jin(i<-j)``, so that at
``psi = 0`` it reduces to the classical positive-definite discrete Laplacian
``sum_j w (n_i - n_j)`` rather than to its negative.

Why the matrix is an M-matrix — the claim ``mesh.py`` could not check
---------------------------------------------------------------------
``mesh.py`` had to leave this open, because there was no matrix yet. With one it
is three lines of algebra and, more usefully, a test:

* **Z-matrix.** ``dF_i/dn_j = -w * B(-x)`` and ``dF_j/dn_i = -w * B(x)``. The
  Bernoulli function is strictly positive for every real argument, so the sign
  of every off-diagonal entry is decided by ``w`` alone. Non-negative Voronoi
  weights — precisely what ``build_mesh`` refuses to hand back without — are
  therefore the whole of the condition.
* **Zero column sums.** Each edge writes ``+w B(-x)`` at ``(j, j)`` and
  ``-w B(-x)`` at ``(i, j)``; the pair cancels. Summed over edges every column
  sums to zero, so the unconstrained matrix is *singular* and weakly
  column-diagonally-dominant, exactly like a graph Laplacian.
* **Dirichlet makes it non-singular.** One constrained node per connected
  component suffices, and then ``A^{-1} >= 0`` — which is what actually
  guarantees non-negative carrier densities.

The last point is standard in the literature and, here, measured: the Layer-0
suite inverts a small mesh's matrix and checks every entry.

Keeping the matrix-free option open
-----------------------------------
The Jacobian is returned as COO triplets, never as a CSR matrix, and
:meth:`System.apply` consumes those triplets directly. DESIGN-2D §4 asks for
exactly this: a matrix-free apply is the path where a GPU could eventually pay
in 3D, and it costs nothing to keep open now. ``backend.solve_sparse`` does the
CSR conversion behind the seam.

This module is numpy-bound, like every other feeder of a scipy delegation point
(see the ``backend`` module docstring, and ``models/chronoamp1d.py`` for the
same boundary in 1D). Writing it through ``backend.xp()`` would be portability
theatre: its output goes straight into SuperLU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from tarhan.numerics.flux import bernoulli
from tarhan.numerics.mesh import Mesh


@dataclass(frozen=True)
class System:
    """A residual and its Jacobian, the latter as COO triplets.

    Deliberately not a CSR matrix — see the module docstring. ``rows``/``cols``
    may contain repeated pairs; both consumers below sum duplicates, and so does
    ``scipy``'s COO constructor, which is what ``backend.solve_sparse`` relies
    on.
    """

    residual: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    n_nodes: int

    def apply(self, vec) -> np.ndarray:
        """Matrix-vector product straight from the triplets — no matrix built."""
        vec = np.asarray(vec, dtype=float)
        out = np.zeros(self.n_nodes, dtype=float)
        np.add.at(out, self.rows, self.vals * vec[self.cols])
        return out

    def to_dense(self) -> np.ndarray:
        """Dense matrix — for tests and tiny meshes only, never for a solve."""
        a = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        np.add.at(a, (self.rows, self.cols), self.vals)
        return a


def node_volumes(mesh: Mesh) -> np.ndarray:
    """Voronoi cell measure per node, ``vol_i = (1/4) * sum_e A_e * L_e``.

    The cell around ``i`` is cut by its incident edges into quadrilaterals, each
    spanned by the node, an edge midpoint and the adjacent circumcentres. The
    piece belonging to ``i`` on edge ``e`` has area ``A_e * (L_e / 2) / 2``,
    hence the quarter.

    Taken on faith only until the tests run: the Layer-0 suite checks that these
    volumes sum to the triangulated area, on figures whose area is known by
    hand.
    """
    vol = np.zeros(mesh.n_nodes, dtype=float)
    for e in mesh.edges:
        share = 0.25 * e.facet * e.length
        vol[e.nodes[0]] += share
        vol[e.nodes[1]] += share
    return vol


def assemble_continuity(mesh: Mesh,
                        density,
                        psi,
                        *,
                        u_t: float = 1.0,
                        edge_coef: Optional[Sequence[float]] = None,
                        source: Optional[Sequence[float]] = None,
                        dirichlet: Optional[Dict[int, float]] = None) -> System:
    """Residual and Jacobian of the steady continuity equation on ``mesh``.

    Parameters
    ----------
    density, psi
        Node arrays: carrier density and electrostatic potential. ``psi`` is
        held FIXED here — this is the continuity half of a Gummel step, so the
        system is *linear in* ``density`` and a single Newton step solves it
        exactly. That is not an optimisation; it is the property stage 2D-0
        leans on.
    u_t
        Thermal voltage, in the same units as ``psi``.
    edge_coef
        Per-edge multiplier on ``A_e / L_e`` (mobility or diffusivity; a
        permittivity for a Poisson-shaped problem). Default 1 on every edge.
    source
        Per-node volumetric source, e.g. net recombination. Multiplied by the
        node's Voronoi volume, so callers pass a density rather than an already
        integrated quantity.
    dirichlet
        ``{node: value}``. The node's row becomes the identity and its residual
        becomes ``n_i - value``. At least one per connected component is
        required, or the matrix is singular by construction (see the module
        docstring) and the solve fails — deservedly, since the problem would
        then be ill-posed.
    """
    n = np.asarray(density, dtype=float)
    p = np.asarray(psi, dtype=float)
    if n.shape != (mesh.n_nodes,) or p.shape != (mesh.n_nodes,):
        raise ValueError(
            f"density and psi must both have length {mesh.n_nodes} "
            f"(got {n.shape} and {p.shape})")
    if not u_t > 0:
        raise ValueError(f"u_t must be positive, got {u_t}")

    coef = (np.ones(len(mesh.edges), dtype=float) if edge_coef is None
            else np.asarray(edge_coef, dtype=float))
    if coef.shape != (len(mesh.edges),):
        raise ValueError(
            f"edge_coef must have one entry per edge ({len(mesh.edges)}), "
            f"got {coef.shape}")

    residual = np.zeros(mesh.n_nodes, dtype=float)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for k, e in enumerate(mesh.edges):
        i, j = e.nodes
        w = float(coef[k]) * e.transmissibility          # coef * A_e / L_e
        if w == 0.0:
            # A zero Voronoi facet is legitimate — the diagonal of the unit
            # square has one — and it transmits nothing. Skipping keeps
            # structural zeros out of the sparsity pattern rather than storing
            # them and hoping the solver drops them.
            continue
        x = (p[j] - p[i]) / u_t
        b_plus = float(bernoulli(x))                      # B(x)
        b_minus = float(bernoulli(-x))                    # B(-x)

        # Flux INTO i from j; the flux into j from i is its exact negative, so
        # conservation costs nothing to enforce.
        inflow_i = w * (b_minus * n[j] - b_plus * n[i])
        residual[i] -= inflow_i
        residual[j] += inflow_i

        rows.extend((i, i, j, j))
        cols.extend((i, j, j, i))
        vals.extend((w * b_plus, -w * b_minus, w * b_minus, -w * b_plus))

    if source is not None:
        s = np.asarray(source, dtype=float)
        if s.shape != (mesh.n_nodes,):
            raise ValueError(
                f"source must have length {mesh.n_nodes}, got {s.shape}")
        residual += node_volumes(mesh) * s

    rows_a = np.asarray(rows, dtype=int)
    cols_a = np.asarray(cols, dtype=int)
    vals_a = np.asarray(vals, dtype=float)

    if dirichlet:
        for node in dirichlet:
            if not 0 <= node < mesh.n_nodes:
                raise ValueError(
                    f"dirichlet node {node} outside 0..{mesh.n_nodes - 1}")
        fixed = np.zeros(mesh.n_nodes, dtype=bool)
        fixed[list(dirichlet)] = True
        # Drop the constrained rows, do not zero them: a duplicate-summing
        # consumer would otherwise add the old entries straight back on top of
        # the identity.
        keep = ~fixed[rows_a]
        rows_a, cols_a, vals_a = rows_a[keep], cols_a[keep], vals_a[keep]
        idx = np.asarray(sorted(dirichlet), dtype=int)
        rows_a = np.concatenate([rows_a, idx])
        cols_a = np.concatenate([cols_a, idx])
        vals_a = np.concatenate([vals_a, np.ones(idx.size, dtype=float)])
        for node, value in dirichlet.items():
            residual[node] = n[node] - float(value)

    return System(residual=residual, rows=rows_a, cols=cols_a, vals=vals_a,
                  n_nodes=mesh.n_nodes)


__all__ = ["System", "assemble_continuity", "node_volumes"]

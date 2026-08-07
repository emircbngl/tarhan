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


CARRIERS = ("electron", "hole")


def assemble_continuity(mesh: Mesh,
                        density,
                        psi,
                        *,
                        carrier: str,
                        u_t: float = 1.0,
                        edge_coef: Optional[Sequence[float]] = None,
                        source: Optional[Sequence[float]] = None,
                        subdomain: Optional[Sequence[int]] = None,
                        dirichlet: Optional[Dict[int, float]] = None) -> System:
    """Residual and Jacobian of the steady continuity equation on ``mesh``.

    Parameters
    ----------
    subdomain
        Node indices where this carrier EXISTS. Default ``None`` means the whole
        mesh. Outside it the density is pinned to zero and every edge with one
        foot outside is dropped, which is exactly the zero-flux condition a
        silicon/oxide boundary imposes on carriers.

        Zeroing ``edge_coef`` on those edges instead does not work and fails in
        a way worth naming: a zero-weight edge is skipped by the assembly, so
        the outside nodes end up with no equation at all and the matrix is
        singular. The carrier has to be declared absent, not merely immobile.
    carrier
        ``"electron"`` or ``"hole"``. Required, with no default, because the
        sign of ``psi`` differs between the two and getting it wrong fails
        SILENTLY — see below.
    density, psi
        Node arrays: the carrier density and the ELECTROSTATIC potential
        (always the physical psi; this function negates it for electrons
        itself). ``psi`` is held FIXED here — this is the continuity half of a
        Gummel step, so the system is *linear in* ``density`` and a single
        Newton step solves it exactly. That is not an optimisation; it is the
        property stage 2D-0 leans on.

    Why ``carrier`` is mandatory
    ----------------------------
    The edge flux vanishes at ``n_i / n_j = exp((psi_j - psi_i) / U_T)``, so the
    operator's null space is ``density ∝ exp(-psi / U_T)``. That is the HOLE
    relation. Electrons obey ``n = n_i exp(+psi / U_T)``, so their transport
    needs ``-psi`` — and passing the raw ``psi`` for electrons is both the
    natural thing to write and wrong.

    It is wrong in the worst way. The solve still converges, the matrix is still
    an M-matrix, every density comes out positive, and the profile looks
    reasonable. Measured on DEVSIM's 495-node diode mesh, the residual at the
    exact equilibrium is 9.4e-16 with the correct sign and 8.9e-01 with the
    wrong one — and nothing except that residual complains.
    ``pn1d._continuity_solve`` already takes a carrier for this reason; this is
    the same guard, made unskippable by having no default.
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
    if carrier not in CARRIERS:
        raise ValueError(
            f"carrier must be one of {CARRIERS}, got {carrier!r}. It has no "
            "default on purpose: electrons transport in -psi and holes in "
            "+psi, and the wrong choice converges to a plausible wrong answer "
            "instead of failing.")

    n = np.asarray(density, dtype=float)
    p = np.asarray(psi, dtype=float)
    if n.shape != (mesh.n_nodes,) or p.shape != (mesh.n_nodes,):
        raise ValueError(
            f"density and psi must both have length {mesh.n_nodes} "
            f"(got {n.shape} and {p.shape})")
    if not u_t > 0:
        raise ValueError(f"u_t must be positive, got {u_t}")

    # The one line the carrier argument exists for.
    if carrier == "electron":
        p = -p

    coef = (np.ones(len(mesh.edges), dtype=float) if edge_coef is None
            else np.asarray(edge_coef, dtype=float))
    if coef.shape != (len(mesh.edges),):
        raise ValueError(
            f"edge_coef must have one entry per edge ({len(mesh.edges)}), "
            f"got {coef.shape}")
    # The geometry guard in mesh.py refuses a negative Voronoi weight precisely
    # to keep every off-diagonal non-positive. A negative coefficient puts it
    # straight back: the entry becomes +|coef| * B(...), the Z-matrix property
    # is gone, and with it the guarantee of positive carrier densities —
    # measured at +5.0e-01 on a unit square. NaN propagates instead, which is
    # worse only in being harder to notice.
    if not np.all(np.isfinite(coef)) or float(coef.min()) < 0.0:
        raise ValueError(
            "edge_coef must be finite and non-negative; a negative or NaN "
            "coefficient reintroduces exactly the positive off-diagonal that "
            "mesh.py refuses to let the geometry produce, so the matrix is no "
            "longer an M-matrix and carrier positivity is no longer guaranteed")

    residual = np.zeros(mesh.n_nodes, dtype=float)
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    inside = None
    if subdomain is not None:
        inside = np.zeros(mesh.n_nodes, dtype=bool)
        idx_sub = np.asarray(subdomain, dtype=int)
        if idx_sub.size and (idx_sub.min() < 0 or idx_sub.max() >= mesh.n_nodes):
            raise ValueError(
                f"subdomain references a node outside 0..{mesh.n_nodes - 1}")
        inside[idx_sub] = True
        if dirichlet and not all(inside[i] for i in dirichlet):
            raise ValueError(
                "every dirichlet node must lie inside the subdomain; pinning a "
                "density where the carrier does not exist is meaningless")

    for k, e in enumerate(mesh.edges):
        i, j = e.nodes
        if inside is not None and not (inside[i] and inside[j]):
            # An edge with one foot outside carries nothing: carriers do not
            # cross into a region where they are not a variable. Skipping it IS
            # the zero-flux condition on the subdomain boundary, which is what a
            # silicon/oxide boundary is for electrons and holes.
            continue
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

    if inside is not None:
        # Nodes where this carrier does not exist get an identity row pinning
        # them to zero, so the system stays non-singular and the absence is
        # explicit in the solution rather than implied by a missing equation.
        outside = np.nonzero(~inside)[0]
        if outside.size:
            residual[outside] = n[outside]
            rows_a = np.concatenate([rows_a, outside])
            cols_a = np.concatenate([cols_a, outside])
            vals_a = np.concatenate([vals_a,
                                     np.ones(outside.size, dtype=float)])

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


def assemble_poisson(mesh: Mesh,
                     psi,
                     *,
                     charge,
                     dcharge_dpsi,
                     edge_coef: Optional[Sequence[float]] = None,
                     dirichlet: Optional[Dict[int, float]] = None) -> System:
    """Residual and Jacobian of the (possibly nonlinear) Poisson equation.

    In the scaled variables ``pn1d`` already uses (De Mari) the equation is
    ``lap(psi) = n - p - N``, so on the box method::

        F_i = sum_e w_e (psi_i - psi_j) + vol_i * charge_i

    with ``w_e = coef_e * A_e / L_e``. The sign matches
    :func:`assemble_continuity`: the Laplacian term is the positive-definite
    one.

    Parameters
    ----------
    charge, dcharge_dpsi
        Node arrays for ``n - p - N`` and its derivative with respect to
        ``psi``. They are passed in rather than computed here because the
        statistics are the caller's business — Boltzmann today, Fermi-Dirac
        later — and baking ``delta * exp(psi - phi)`` into the assembly would
        turn that into a rewrite instead of an argument. Under Boltzmann the
        derivative is ``n + p``, which is non-negative and so only strengthens
        the diagonal.
    dirichlet
        ``{node: value}``; the row becomes the identity and the residual
        ``psi_i - value``. Required for the same reason as in the continuity
        assembly: the pure Laplacian has zero column sums.

    The Jacobian is again a Z-matrix — off-diagonals are ``-w_e <= 0`` — and
    here it is strictly diagonally dominant wherever ``dcharge_dpsi > 0``, which
    is what makes the damped Newton inside a Gummel loop converge globally.
    """
    p = np.asarray(psi, dtype=float)
    q = np.asarray(charge, dtype=float)
    dq = np.asarray(dcharge_dpsi, dtype=float)
    for name, arr in (("psi", p), ("charge", q), ("dcharge_dpsi", dq)):
        if arr.shape != (mesh.n_nodes,):
            raise ValueError(
                f"{name} must have length {mesh.n_nodes}, got {arr.shape}")

    coef = (np.ones(len(mesh.edges), dtype=float) if edge_coef is None
            else np.asarray(edge_coef, dtype=float))
    if coef.shape != (len(mesh.edges),):
        raise ValueError(
            f"edge_coef must have one entry per edge ({len(mesh.edges)}), "
            f"got {coef.shape}")
    # The geometry guard in mesh.py refuses a negative Voronoi weight precisely
    # to keep every off-diagonal non-positive. A negative coefficient puts it
    # straight back: the entry becomes +|coef| * B(...), the Z-matrix property
    # is gone, and with it the guarantee of positive carrier densities —
    # measured at +5.0e-01 on a unit square. NaN propagates instead, which is
    # worse only in being harder to notice.
    if not np.all(np.isfinite(coef)) or float(coef.min()) < 0.0:
        raise ValueError(
            "edge_coef must be finite and non-negative; a negative or NaN "
            "coefficient reintroduces exactly the positive off-diagonal that "
            "mesh.py refuses to let the geometry produce, so the matrix is no "
            "longer an M-matrix and carrier positivity is no longer guaranteed")

    vol = node_volumes(mesh)
    residual = vol * q
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for k, e in enumerate(mesh.edges):
        i, j = e.nodes
        w = float(coef[k]) * e.transmissibility
        if w == 0.0:
            continue
        flow = w * (p[i] - p[j])
        residual[i] += flow
        residual[j] -= flow
        rows.extend((i, i, j, j))
        cols.extend((i, j, j, i))
        vals.extend((w, -w, w, -w))

    idx_all = np.arange(mesh.n_nodes)
    rows.extend(idx_all.tolist())
    cols.extend(idx_all.tolist())
    vals.extend((vol * dq).tolist())

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
        keep = ~fixed[rows_a]
        rows_a, cols_a, vals_a = rows_a[keep], cols_a[keep], vals_a[keep]
        idx = np.asarray(sorted(dirichlet), dtype=int)
        rows_a = np.concatenate([rows_a, idx])
        cols_a = np.concatenate([cols_a, idx])
        vals_a = np.concatenate([vals_a, np.ones(idx.size, dtype=float)])
        for node, value in dirichlet.items():
            residual[node] = p[node] - float(value)

    return System(residual=residual, rows=rows_a, cols=cols_a, vals=vals_a,
                  n_nodes=mesh.n_nodes)


__all__ = ["System", "assemble_continuity", "assemble_poisson", "node_volumes"]

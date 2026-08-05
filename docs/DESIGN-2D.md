# Extending TARHAN to 2D — a design, not an implementation

Status: **proposal**. No 2D code exists in this repository today, and nothing
here has been built or measured. Read it as a set of decisions to argue with
before any of it is written.

## 0. Where we actually are

`src/tarhan/` is 0D and 1D, and says so in its own module names: `pemfc0d`,
`pn1d`, `sofc1d`, `chronoamp1d`, `diffusion1d`. There is no mesh abstraction, no
2D anything.

(The `*_2d.py` and `*_3d.py` files under `.venv/devsim_data/testing/` belong to
**DEVSIM**, the reference simulator this project validates against. They are not
TARHAN capability. They are, however, exactly the validation corpus §5 uses.)

## 1. The one thing that does not have to change

The transport physics is already dimension-independent, and this is the whole
reason a 2D extension is tractable rather than a rewrite.

`numerics/flux.py` exposes:

```python
sg_edge_flux(dpsi, n_left, n_right, u_t=1.0, coef=1.0)
```

That is a function **of an edge**: a potential difference and the carrier
densities at its two endpoints. It knows nothing about `x`, about ordering, or
about how many neighbours a node has. The Scharfetter–Gummel discretisation is
formulated on edges in any dimension; going to 2D does not change the flux law,
the Bernoulli function, or the M-matrix property that guarantees positivity.

**What is 1D-specific is not the physics. It is three things:**

1. **The solver.** `backend.solve_tridiag` — a line has a tridiagonal Jacobian.
   A 2D mesh does not.
2. **Neighbourhood.** 1D assembly can assume node `i` couples to `i±1`.
3. **Geometry weights.** In 1D an edge's "area" is 1 and its length is `dx`.

A 2D port that touches `bernoulli` or `sg_edge_flux` has gone wrong.

## 2. Discretisation: box method on a Delaunay mesh

Recommended: **finite-volume (box / Voronoi) method on a Delaunay triangulation**,
which is what DEVSIM, Sentaurus and Minimos use for the same equations.

Each node owns a Voronoi cell. For every edge `e = (i, j)`:

- `L_e` — the edge length (node-to-node distance)
- `A_e` — the length of the Voronoi facet bisecting that edge (in 2D this is a
  length; in 3D it becomes an area, which is why this choice makes the eventual
  3D step mechanical rather than another redesign)

The current flux then enters the balance for node `i` as `A_e * sg_edge_flux(...)`
with `coef = 1/L_e`. Poisson's equation gets the same treatment with the
permittivity on the edge.

Why not finite differences on a structured grid? It is easier for one week and
wrong forever: real device geometry (a MOS corner, a junction that is not
axis-aligned) is what 2D is *for*, and a structured grid cannot represent it.
The Delaunay/Voronoi pair is also what makes the M-matrix property survive —
**provided the mesh is Delaunay and non-obtuse**. That is a real constraint, not
a footnote: an obtuse triangle produces a negative `A_e` and destroys positivity.
The mesh layer must check it and refuse.

## 3. Proposed module layout

```
numerics/mesh.py       Mesh: nodes, edges, Voronoi weights (A_e, L_e), regions,
                       contacts. Delaunay/non-obtuse validation lives here.
numerics/assemble.py   Edge-loop assembly of residual + sparse Jacobian.
                       Dimension-agnostic; takes a Mesh, calls flux.py.
backend.py            (extend) solve_sparse() alongside solve_tridiag()
models/pn2d.py         The 2D device, mirroring pn1d.py's public shape
```

`mesh.py` is deliberately not a mesh *generator*. Generating a good Delaunay
mesh is a solved problem someone else has solved better; read `.msh` (Gmsh)
files, as DEVSIM does. Writing a mesher is how this project would spend six
months not doing physics.

## 4. Solver strategy

Keep the existing **Gummel outer loop** (Poisson ↔ continuity) as the first
target, because it is what `pn1d.py` already does and it isolates the change to
assembly plus the linear solve. Gummel converges slowly at high injection, so
full Newton on the coupled `(ψ, n, p)` system is the eventual answer — but it
should be a second step, after 2D is correct, not bundled into the same change.

Linear algebra: `scipy.sparse` + SuperLU (`spsolve`) is the honest starting
point — it is already an implicit dependency via scipy and needs no new native
build. Expect it to become the bottleneck around 10⁵ nodes; that is the moment
to consider UMFPACK or an iterative scheme with an ILU preconditioner, and not
before, because a preconditioner chosen without a profile is a guess.

**Not GPU.** Measured on this machine (M4), MLX at TARHAN's 1D working size is
~50× *slower* than NumPy — transfer overhead dominates. Sparse LU is also
branch-heavy and latency-bound rather than throughput-bound. GPU is the wrong
tool here until there is a measured profile that says otherwise.

## 5. Validation — the part that makes it TARHAN

This project's rule is that physics enters the kernel only with a pinned,
source-cited test. 2D must not be the exception, and it does not have to be:
**DEVSIM is already installed in the venv and ships the reference cases.**

Staged, each stage gated on the previous:

| Stage | Case | Oracle | Passes when |
|---|---|---|---|
| 2D-0 | 1D problem on a 2D mesh (one cell thick) | existing `pn1d` results | matches 1D to solver tolerance — proves assembly, not physics |
| 2D-1 | Equilibrium 2D pn junction, no bias | analytic depletion width, DEVSIM `dio2_element_2d.py` | ψ profile and W agree |
| 2D-2 | 2D diode I–V | DEVSIM `dio2_element_2d.py` | ideality 1.00±0.02, currents agree |
| 2D-3 | MOS capacitor C–V | DEVSIM `ssac_cap_2d_edge.py` | C–V curve agrees |
| 2D-4 | MOSFET I–V | DEVSIM `mos_2d.py` | drain current agrees |

Stage 2D-0 is the one people skip and should not: running a 1D problem through
the 2D machinery separates "my assembly is wrong" from "my physics is wrong".
Without it, the first 2D discrepancy has two candidate causes and no way to
choose between them.

Textbook cross-checks (Sze, Selberherr) belong on top of these, not instead of
them: a textbook gives a formula, DEVSIM gives numbers on the same mesh.

## 6. Milestones

1. `mesh.py` + Gmsh reader + Delaunay/non-obtuse validation, with unit tests on
   hand-computed Voronoi weights for a 4-node square.
2. `assemble.py` + `solve_sparse`, gated on **stage 2D-0**.
3. Equilibrium 2D (2D-1), then I–V (2D-2). This is the first point at which the
   phrase "TARHAN does 2D" becomes true.
4. MOS cases (2D-3, 2D-4). These need oxide regions and a second material —
   check whether the region/interface model in `mesh.py` survived step 1.
5. Only then: full Newton, and only then 3D — which under this design is mostly
   `A_e` becoming an area and the Gmsh reader learning tetrahedra.

## 7. Things that would make this fail

- Writing a mesh generator.
- Reaching for GPU or full Newton before 2D-2 passes.
- Adding 2D to the MCP tool surface before the validation table is green — the
  tools are described to agents as oracle-verified, and an unvalidated 2D tool
  would make that description false.
- Skipping 2D-0.
- Touching `bernoulli`/`sg_edge_flux`. If a 2D change seems to need it, the
  geometry weights are wrong, not the flux law.

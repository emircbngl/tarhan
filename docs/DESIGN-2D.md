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

The Delaunay/Voronoi pair is also what keeps the edge weights non-negative, which
is what the M-matrix property (and therefore guaranteed positivity of carrier
densities) rests on. The interior-edge weight is

    A_e / L_e = (cot α + cot β) / 2

with α, β the angles *opposite* the shared edge in its two triangles.

**The rule differs between interior and boundary edges, and both earlier
statements of it in this document were wrong.**

*Interior edge* — two opposite angles, so the terms compensate. Delaunay's
empty-circumcircle test is equivalent to α + β ≤ π, which gives
cot α + cot β ≥ 0. An obtuse triangle is fine here: when its obtuse angle faces
the interior edge, the neighbour's larger cotangent absorbs it. Measured over
20 000 random two-triangle configurations, of the 19 294 satisfying Delaunay,
**none** produced a negative interior weight, and **18 022 of those contained an
obtuse triangle**.

*Boundary edge* — one opposite angle, nothing to compensate with. The weight is
(L_e / 2)·cot γ, negative exactly when γ > 90°, i.e. when the triangle's
circumcentre falls outside the domain across that edge. **This is where
non-obtuseness genuinely matters, and only here.**

The first draft of this section said the whole mesh had to be non-obtuse (too
strict: it rejects valid interior configurations). The first correction said
obtuseness never mattered (too loose: it misses the boundary). The distinction
surfaced when a hand-computed Layer-0 test failed on a boundary edge — which is
the argument for hand-derived expectations rather than golden files.

`numerics/mesh.py` implements this and reports which of the two causes it hit,
because the remedies differ: flip the edge, versus refine the boundary.

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

**GPU: decided by the solver choice, not yet decided.** Measured on an M4, MLX
at TARHAN's *current* 1D size (n=2e4) is ~50x slower than NumPy — transfer
overhead dominates. That verdict does not carry to 2D/3D, where node counts go
from 2e4 to 6.5e4 (256^2) and 2.1e6 (128^3) and the arithmetic finally dwarfs
the transfer.

The split matters. **Assembly** — the edge loop — is a dependence-free
gather/scatter over 10^5-10^6 edges and is a good GPU workload. **The linear
solve** is the dominant cost and the blocker: MLX has no sparse support today
(`mx.sparse` absent, `mx.linalg` dense-only as of 0.31.1), and dense is not an
option (65k nodes dense in float64 is ~34 GB).

So the two solver paths differ in their GPU story, and this should be weighed
when choosing one:

- **Sparse direct (SuperLU/UMFPACK)** — recommended to start. CPU-only, and
  fine: it is branch-heavy and latency-bound, a poor GPU fit regardless.
- **Matrix-free iterative (CG/GMRES + preconditioner)** — the path where a GPU
  could actually pay, because the stencil apply is exactly the kind of dense
  elementwise work MLX is good at. Also the path that scales to 3D.

Recommendation unchanged for stage 1 (start sparse direct, on CPU), but do not
write the assembly layer in a way that assumes an assembled CSR matrix is the
only consumer — keeping a matrix-free apply possible costs nothing now and is
what keeps the GPU option open for 3D.

## 5. Validation — the part that makes it TARHAN

This project's rule is that physics enters the kernel only with a pinned,
source-cited test. 2D must not be the exception, and it does not have to be:
**DEVSIM is already installed in the venv and ships the reference cases.**

Staged, each stage gated on the previous:

| Stage | Case | Oracle | Passes when | Status |
|---|---|---|---|---|
| 2D-0 | 1D problem on a 2D mesh (one cell thick) | existing `pn1d` results | matches 1D to solver tolerance — proves assembly, not physics | **DONE**, and exactly rather than approximately: on a strip built from `pn1d`'s own grid the box method reduces to its tridiagonal scheme identically (`w/vol == 1/(hm·h̄)` to 12 digits), so ψ agrees to 1.8e-15 |
| 2D-1 | Equilibrium 2D pn junction, no bias | analytic depletion width, DEVSIM `dio2_element_2d.py` | ψ profile and W agree | **DONE** — on DEVSIM's *own* mesh (495 nodes, 880 unstructured triangles): max ∣Δψ∣ = 2.24e-16 V over the 481 nodes TARHAN solves, rms 7.39e-17 V. V_bi = 0.953719 V from TARHAN, from DEVSIM **and** from the analytic ln, differences exactly zero. Read the caveat below before quoting this. |
| 2D-2 | 2D diode I–V | DEVSIM `dio2_element_2d.py` | ideality 1.00±0.02, currents agree | **DONE** — `models/pn2d.py` on DEVSIM's own mesh: I_n ratio 1.00000 at every bias, I_p and total 0.99938→1.00000 over 0.2–0.5 V, ideality **1.0119–1.0134** against DEVSIM's own 1.0114–1.0126. Biases ≤0.1 V excluded: currents ~1e-14 A where DEVSIM's own conservation is already 1.4e-2. This is the first stage that tests transverse transport — current has to spread to reach the partial top contact. |
| 2D-3 | ~~MOS capacitor C–V~~ | ~~DEVSIM `ssac_cap_2d_edge.py`~~ | | **BLOCKED — and the row was wrong.** See below. |
| 2D-3′ | Electrostatic capacitance, contact charge | DEVSIM `examples/capacitance/cap2d.py` | contact charge agrees | **DONE** — on DEVSIM's own 8281-node mesh: contact charge ratio **1.000000000** (3.350171660e-12 C/cm both), ψ within 6.6e-13 V on a 1 V scale, and the two plates cancel to −2.9e-25. No scale factor stands between the two numbers. |
| 2D-4 | MOSFET I–V | DEVSIM `mos_2d.py` | drain current agrees | needs regions + interfaces in `mesh.py` first |

### 2D-3 is blocked, and this table described it incorrectly

`ssac_cap_2d_edge.py` is not a MOS capacitor and produces no C–V curve. It is a
parallel-plate air-gap capacitor — its regions are `air` (material `gas`) and two
`metal` plates, with no semiconductor anywhere — and what it measures is the
displacement current at a contact. It needs two things TARHAN does not have:

- **Circuit-node coupling.** `topbias` is a circuit node, not a parameter, and
  the device is wired to a lumped `V1` source through an `R1` resistor.
- **A small-signal AC solve.** `solve(type="ac", frequency=…)` at 1e-3, 1e10 and
  1e15 Hz, i.e. a complex-valued linearisation about the DC operating point. It
  prints complex circuit node voltages.

Neither is difficulty; both are a different capability. §3 proposes no circuit or
AC layer and §4 commits to DC — Gummel first, Newton later. Building a complex
solver and a circuit-coupling layer to tick this row would be scope invented
after the fact, so the row is recorded as blocked rather than quietly redefined.

**What replaces it.** `examples/capacitance/cap2d.py` is the same physical
question without the AC machinery: one region of uniform permittivity, pure
Laplace with no charge, two Dirichlet contacts, and `get_contact_charge` as the
answer. It is reachable with what already exists — `assemble_poisson` with zero
charge and permittivity as the edge coefficient — and it tests something no
stage has yet: that the Poisson residual summed over a contact is a genuine
**flux**, the electrostatic twin of the contact-current extraction 2D-2
validated. It is a weaker statement than 2D-3 would have been, and saying so is
the point of numbering it 2D-3′ rather than 2D-3.

**The 2D-1 caveat, because the number above is easy to over-read.** Equilibrium
is a weak test of *transverse* physics, however two-dimensional the geometry is.
ψ is pinned by local charge neutrality and the doping varies only in x, so the
answer is nearly one-dimensional by construction: the measured variation of ψ
across y is 3.13e-3 thermal volts — 81 µV against a 0.95 V built-in potential,
0.008% of the signal. It is real and it has the right shape (largest exactly at
the partially contacted edge, decaying about eightfold per micron into the bulk,
identically zero on the fully contacted edge), and the Layer-0 test asserts that
shape rather than a bare threshold. But "2D-1 passes" means the assembly and the
contact model are right on an unstructured mesh. It does not mean transverse
transport is validated. **2D-2 is where that gets tested**, because current has
to spread out from the partial contact.

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

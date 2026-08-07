# Extending TARHAN to 2D — the design, and what became of it

Status: **partly built**. Stages 2D-0, 2D-1, 2D-2 and 2D-3′ are implemented and
validated against DEVSIM; 2D-3 and 2D-4 are blocked, for reasons recorded in §5.
**The §5 table is the authority** on what is measured and what is not.

This began as a proposal and its argument sections still read that way, on
purpose: the reasoning behind each decision is worth more than a tidied account
of the outcome. Where a decision turned out wrong — and several did — the
correction sits next to it rather than replacing it.

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
| 2D-4 | MOSFET I–V | ~~`testing/mos_2d.py`~~ → DEVSIM `examples/mobility/gmsh_mos2d.py` | drain current agrees | **BLOCKED** — the reference mesh is not Delaunay. See below; every machinery piece it needed is built and tested. |

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

### 2D-4's oracle was also named wrong, and the right one exists

Same mistake, same cause: this table was written from filenames rather than from
reading the files. `testing/mos_2d.py` performs **no bias sweep at all** — it
sets every one of its four contacts to 0 V, solves once with a loose
`relative_error=1e-5`, and then writes mesh and parameter files. It is a
setup/regression script, not an I–V characteristic. The contact currents it
leaves behind (drain 3.5e-12, source −9.0e-12, body −8.4e-12) are residual noise
from that loose solve at zero bias, which is why they do not sum to zero, as
currents at a set of terminals must.

`examples/mobility/gmsh_mos2d.py` is the real oracle: it ramps the gate to 0.5 V
and then the drain to 0.5 V in 0.1 V steps, reporting every terminal current at
each point. It reads a Gmsh mesh, but that does not resurrect the Gmsh-reader
milestone — the mesh is extracted through the same DEVSIM API used for every
other stage, once the case has built it.

### 2D-4 is blocked: the reference mesh is not Delaunay

Everything 2D-4 needed was built and each piece is tested: the node subdomain so
carriers exist only in silicon, per-triangle facet shares so a material
interface can be weighted, and the region-merge with its connectivity proof
— on `testing/mos_2d.py`'s structure, 2539 raw nodes to 2517, fusing only the
two interfaces, single connected component; on `gmsh_mos2d`'s, 2954 to 2847,
107 pairs. The MOSFET port fell over on the mesh itself.

(An earlier revision of this paragraph gave the second merge as "2954 to 2517",
carrying the first mesh's result across to a mesh whose merge had never been
counted, because `build_mesh` refused it before the count mattered. Two
different meshes cannot merge to the same total; the number was copied, not
measured, and it is corrected above. Caught in review by Codex.)

`build_mesh` refuses `gmsh_mos2d`'s geometry: 22 of its 8192 interior edges have
a negative Voronoi weight. The first suspicion was that merging three
independently meshed regions had created bad triangle pairs across the
interfaces. Measured, that is wrong — **all 22 lie strictly inside a single
region** (`bulk` and `oxide`), none across. The merge is innocent; the
Gmsh-generated triangulations are simply not Delaunay in places.

DEVSIM solves it regardless because these cases use its element formulation,
which does not lean on the empty-circumcircle condition the way a pure edge/box
method does. The two codes make different discretisation assumptions, and this
mesh satisfies DEVSIM's and not TARHAN's. That is a real difference, not a bug
in either.

Every way out costs more than the row is worth:

- **Clamping the weight** is forbidden here in writing, and for a reason —
  a negative weight breaks the M-matrix property and with it the guarantee of
  positive carrier densities. `mesh.py` says "do not clamp the weight" at the
  point of refusal.
- **Re-meshing** contradicts §3 ("`mesh.py` is deliberately not a mesh
  generator") and §7 lists writing one as a way this project fails.
- **Flipping the 22 edges** is the tempting middle path, and it is the one that
  quietly destroys the evidence: it changes the discretisation, so the result
  would no longer be a comparison *on the same mesh*, which is the entire basis
  on which 2D-1, 2D-2 and 2D-3′ mean anything.

So the honest position is that a MOSFET is reachable with what now exists — on a
Delaunay mesh. What is blocked is this particular comparison, because its
reference geometry violates a precondition the design chose deliberately. Fixing
it means either a Delaunay MOSFET oracle, or an element-based assembly to sit
beside the box method, and the second is a new discretisation rather than a
missing feature.

**The gap that was blocking 2D-4 before that, found by testing the earlier claim
rather than trusting it.** The series-capacitor result above says multi-region
electrostatics needs no region machinery, and for the bulk it does. But the
edges lying ALONG an interface were never exercised by it: in that geometry they
sit perpendicular to the field, so reassigning them from one material to the
other moves the answer by 3.4e-15 relative — nothing. Their assignment was
untested, not validated.

It stops being harmless in a MOSFET. An edge whose two triangles are different
materials needs the facet-weighted combination `(cot α · ε₁ + cot β · ε₂)·L/2`,
not either material's value, and the channel of a MOSFET runs along precisely
such a line of edges. `EdgeGeometry` cannot express this today: `build_mesh`
sums the two cotangents before anything can weight them, so only one coefficient
per edge is available downstream.

This is DESIGN-2D §6 milestone 4's "check whether the region model survived"
arriving, in a narrower and more precise form than feared — not a region and
interface subsystem, but per-triangle facet contributions on each edge so a
material-weighted coefficient can be formed. Until that exists, a MOSFET result
would rest on an assumption this repository has explicitly measured as untested.

**Mesh merging, which the earlier stages did not need.** DEVSIM numbers nodes
per region, so a MOSFET arrives as three separate meshes (`bulk`, `oxide`,
`gate`) that happen to share coordinates along two interfaces. Building one
global mesh means fusing coincident nodes and remapping the triangles. Verified
on the shipped structure: 2539 raw nodes merge to 2517, fusing exactly 22 pairs
— 11 on `bulk`∩`oxide` and 11 on `gate`∩`oxide`, which is the two interfaces and
nothing else. `build_mesh` accepts the result (7308 edges, zero negative facets),
and a breadth-first walk reaches all 2517 nodes from one, so it is a single
connected component rather than three problems that would each solve happily on
their own. That last check is the one that matters: an unmerged interface does
not raise, it just quietly stops conducting.

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

1. **DONE**, except the Gmsh reader, which was never needed and never written:
   meshes come out of DEVSIM through its own API, so there was nothing to parse.
   `mesh.py` plus the positivity validation shipped, with hand-computed Voronoi
   weights for a 4-node square — and every tolerance in it turned out to need to
   be *relative*, since a device is 1e-5 cm across.
2. **DONE** — `assemble.py` + `solve_sparse`, and stage 2D-0 passed on both
   halves, exactly rather than approximately: on a strip built from `pn1d`'s own
   grid the box method reduces to its tridiagonal scheme identically.
3. **DONE** — 2D-1 and 2D-2 both green against DEVSIM on DEVSIM's own mesh. This
   is the point at which "TARHAN does 2D" became true, and it is true.
4. **PARTLY.** 2D-3 turned out to be an AC + circuit-coupling case and is
   blocked; 2D-3′ replaced it and passed. 2D-4's machinery is all built and
   tested — node subdomains, per-triangle facet shares for material interfaces,
   region merging with a connectivity proof — but the stage is blocked on its
   reference mesh not being Delaunay. The milestone's own instruction ("check
   whether the region/interface model survived step 1") was the right question;
   the answer was that no region subsystem was needed, only per-triangle facet
   shares, and that the obstacle lay somewhere else entirely.
5. Not started: full Newton, then 3D. 3D remains mostly `A_e` becoming an area —
   and, on the evidence above, a mesh source that guarantees Delaunay.

**What the 2D work does not cover, stated so nobody has to infer it:** no
small-signal or AC solve, no circuit-node coupling, no mesh generation and no
mesh repair, and no 2D on the MCP tool surface while any row above is not green.

## 7. Things that would make this fail

- Writing a mesh generator.
- Reaching for GPU or full Newton before 2D-2 passes.
- Adding 2D to the MCP tool surface before the validation table is green — the
  tools are described to agents as oracle-verified, and an unvalidated 2D tool
  would make that description false.
- Skipping 2D-0.
- Touching `bernoulli`/`sg_edge_flux`. If a 2D change seems to need it, the
  geometry weights are wrong, not the flux law.

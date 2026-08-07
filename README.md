# TARHAN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21761218.svg)](https://doi.org/10.5281/zenodo.21761218)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**Physics-first, extensible materials simulator** — galvanic cells, hydrogen fuel
cells and semiconductor devices, built on a single charged-species transport kernel
(finite-volume + Scharfetter–Gummel + damped Newton).

> **Status: pre-alpha.** There is no solver product here yet — deliberately. TARHAN is
> being built *validation-first*: the Layer-0 reproduction catalog below comes before
> the engine, so every piece of physics that later enters the kernel already has a
> pinned, source-cited test.

Named after **Tarhan**, protector of forges and metalworkers in Turkic mythology
(depicted with hammer and anvil).

## Quickstart

```bash
pip install -e ".[dev]"
tarhan demo                 # zero-config: converged Cottrell reproduction
tarhan demo --case diode    # flagship drift-diffusion diode: I-V + band diagram
tarhan demo --show          # ...and open the plot window (interactive terminals)
tarhan demo --save iv.png   # ...or write the plot to a file (CI, servers)

tarhan capabilities list                 # what the engine can do, and what it cannot
tarhan capabilities show <id>            # limits, measured evidence, or why it is blocked
tarhan capabilities list --format json   # stdout is ONLY json; notes go to stderr
pytest                      # run the Layer-0 validation catalog
```

Optional extras: `tarhan[mcp]` — MCP server (`tarhan-mcp`) exposing 9 guarded
tools (diode I-V/band diagram, cyclic voltammetry incl. Butler-Volmer, Nicholson
working curve, SOFC polarization, PEMFC V(i) polarization, fuel-cell loss ladder,
honesty-tier formula catalog) so AI agents can drive the engine; `tarhan[oracle]` — DEVSIM
cross-validation tests. Known-good pins: `pip install tarhan -c constraints.txt`.

## Web service

[**tarhan-web**](https://github.com/emircbngl/tarhan-web) is a separate
repository: a FastAPI service and React front end that call this solver and
archive the runs. It adds no physics — it pins this package at `v0.1.0` and
displays what comes back.

It is worth a look for one design decision. The cross-validation figures below
were measured at a single operating point, and a web form lets anyone leave that
point. So every response the service returns carries a flag saying whether *that
run* sits inside the validated regime, together with the measured numbers and a
link to the test file. The agreement is a measurement, not a property that
travels with the code.

Not deployed yet — it runs locally today.

## The honesty model

No formula enters this codebase without all three of:

1. a **source citation** (book/paper, section, printed expected numbers),
2. an **oracle check** where applicable (dimensional + symbolic + numeric verification;
   e.g. the Cottrell equation passed 3/3 before being added to `tarhan.physics`),
3. a **Layer-0 reproduction test** whose acceptance is pinned to published numbers, a
   measured convergence order, or a Richardson-extrapolated limit — not to a tolerance
   invented to make the test pass.

Every function in `tarhan.physics` carries an explicit honesty tier:
`first-principles (oracle-verified)` / `textbook (reproduced)` / `empirical fit`
(a few first-principles entries are labelled with their basis instead, e.g.
`first-principles (closed form)`).
Reduced-precision backends can never be the truth path (see `tarhan/backend.py`).

**Where this discipline is not yet absolute** (kept honest rather than advertised away).
An independent review (2026-07-15) refuted an earlier "never an arbitrary tolerance"
claim. This section then said "a small number of loose sanity bounds remain underived —
e.g. `test_rank10_sg_flux.py`". A full audit (2026-07-19) classified **all 192**
acceptance thresholds in the suite and showed that phrasing still understated it:

| class | n | meaning |
|---|---|---|
| source-pinned | 32 | the bound is the last printed digit of a published number |
| order / limit | 13 | the asserted quantity *is* a measured convergence order or a Richardson-extrapolated limit |
| exact structural | 76 | machine-precision identity, conservation law, monotonicity, sign, domain guard — not a tolerance at all |
| measured-margin | 52 | regression guard set with deliberate margin over a value we measured **and recorded in the code** |
| qualitative band | 5 | the band encodes the source's own prose (e.g. "~1.0 V near OCV") |
| **underived** | **14** | no printed number, no measured order, no recorded measurement |

So 121 of 192 are pinned to printed digits, a measured order, or are exact structural
invariants, and 52 more are a recorded measurement plus a chosen margin — but **14 are
underived**, not one example. The complete list is in
[`validation/CATALOG.md`](validation/CATALOG.md). Eleven are loose by 15×–1e11× over the
quantity they guard (safe, but the number is tied to nothing). Two are thin and genuinely
fragile: `test_rank04_semiintegral.py:43` accepts 3e-3 against a measured 2.9312e-3 (2.3%
headroom), and `test_robertson_stiff.py:135` accepts 2e-3 against a measured 1.28e-3.

**One was worse than loose — it was false, and it has been fixed.**
`test_chronoamp_transient.py` asserted "≥100× fewer steps than the explicit CFL
requirement", but `numerics/transient.py` reported `nsteps` as the *output-point* count
(`len(t_eval)` = 1), so the assertion reduced to `1 < 88.9` and measured nothing. The
field is now split into `n_output_points` and an opt-in, genuinely-measured
`n_accepted_steps`; the real ratio is **18.4×** (483 accepted BDF steps vs 8889 explicit
steps), and every "≥100×" claim in the docs has been corrected.

**The four `strict-xfail`s are not all the same kind of thing:**

| kind | count | meaning |
|---|---|---|
| **contradiction within our transcription** | 2 | the transcribed printed inputs do not reproduce the transcribed printed answer — shown arithmetically (O'Hayre Ex. 5.1 `D=0.1`-vs-`0.2`; Hu Ex. 4-2 substituting `1e-8` for `A²`). **Limit:** the source files are *not* in this repo (copyright), so the contradiction is demonstrated over *our* transcription — a clean checkout cannot re-verify that the book prints those values |
| **unresolved provenance** | 2 | our catalog transcription and the correlation disagree, and we have **not** yet confirmed which side is wrong from the printed source (Springer `λ(0.3)`; Pierret `ε_r`) |

The second kind must not be read as "the source is wrong". Precedent: the rank-12
`(0.085, 1.1)` constants were carried as a suspected source error until a primary-source
check showed the mis-attribution was **ours** (they are Spiegel's α₁/k, not Kim's m,n).
The same may be true of the remaining two.

## Layer-0 validation catalog (15/15 — COMPLETE, incl. the original rank-12 parametric-PEMFC target) + flagship device solver

| # | Case | Source | Status |
|---|------|--------|--------|
| 0 | Cottrell chronoamperometry, explicit FD | Britz & Strutwolf 4e | ✅ max err 0.016%, observed order 1.97 |
| 1 | Nafion membrane correlations λ(a), κ(λ,T) | Springer et al., JES 138 (1991) | ✅ 5/6 printed digits exact |
| 2 | P+N step-junction electrostatics | Hu, *Modern Semiconductor Devices*, Ex. 4-1 | ✅ 3/3 printed values |
| 3 | 1D diffusion MMS, exact numerical solution | Linge & Langtangen §3.6.5 (CC BY) | ✅ machine precision (≤8e-15) |
| 4 | Semi-integral of Cottrell current = const | Oldham, Myland & Bond | ✅ measured order 0.50, Richardson → 1.000000 |
| 5 | C-V doping extraction (inverse problem) | Hu, Ch. 4 Ex. 4-2 (open PDF, read visually) | ✅ N_l=6e15, N_h=1.8e18, 60 mV→one-decade sensitivity; in-book input slip documented |
| 6 | PEMFC loss ladder (4 worked examples, boxed answers) | O'Hayre et al., *Fuel Cell Fundamentals* 3e | ✅ E⁰=1.199 V, 59 mV/dec, η_ohm 0.15/0.10 V, j_L=2.26, η_conc=0.22 V |
| 8 | Reversible CV peak ψ_p = 0.4463 (solver-level) | Compton ⊕ Britz ⊕ Bard & Faulkner | ✅ J_p=0.44636, θ_p=−1.109, half-width 2.202 |
| 7 | Observed convergence-rate harness | Linge & Langtangen §1.1.4, §3.6.6 | ✅ 2.0000 / 1.0000 / 2.0002 |
| 9 | Pierret step junction + Shockley log-slopes | Pierret, *Semiconductor Device Fundamentals* Ch. 5-6 | ✅ V_bi=0.716 V, W=0.972 μm (pins ε_r=11.8), 59.6/119.3 mV/dec |
| 11 | Levich RDE limiting current (first convection term) | Newman & Thomas-Alyea 3e | ✅ two independent routes agree to 5e-7 on 0.620450; FD order ≈2; √ω scaling exact |
| 12 | End-to-end 1D SOFC cell voltage (first Layer-3 domain model) | O'Hayre 3e §6.2, Table 6.4 | ✅ ASR=0.176 Ω·cm², η_ohm=0.088 V, η_cat=0.158 V, V=0.754 V — all printed |
| 10 | Scharfetter–Gummel vs central-difference flux | Farrell et al. (WIAS 2263); Selberherr | ✅ equilibrium at U_T·ln10 to 9e-16 |

| ⚑ | **Flagship: 1D pn-diode drift-diffusion (Gummel + Scharfetter-Gummel)** | Selberherr; Farrell (WIAS 2263); short-base analytics | ✅ V_bi to 0.6 µV; ideality 1.000-1.002; absolute J vs closed form 0.3%; **independently reproduces Sze's ψ_bi−2kT/q correction to 0.03%**; discrete conservation ~1e-8 |
| 13 | Solar-cell FF/V_oc anchors (dual-route: Green 1981 vs exact maximization) | Green, Solid-State Electronics 24 (1981); PVEducation | ✅ agreement ≤1.4e-4 across v_oc 10.5-30; both oracle-VERIFIED |
| 14 | Nicholson (1965) ΔEp-ψ working table, quasi-reversible CV (Butler-Volmer boundary) | Nicholson, Anal. Chem. 37, 1351 — primary-source PDF transcription | ✅ 13/13 pairs within ±2 mV (our values grid-converged; residual = 1965-era table granularity); BV→Nernst limit 0.08 mV |
| 12 | Full parametric PEMFC polarization curve V(i) — the original rank-12 target (row above at #12 was its SOFC substitution) | **Parameters: Spiegel (2008)/FuelCellStore** (attribution corrected 2026-07-09; the earlier "Kim/Barbir" was wrong). Kim et al. JES 142(8):2670 (1995, DOI 10.1149/1.2050072) is a *separate* model (m·exp(n·i), no i_L) | ✅ assembly of oracle-verified loss ladder: ~1.0 V near OCV (0.997 @ 1 mA/cm²), ~0.6 V @ 1 A/cm² (0.582), measured roll-off toward i_L=1.4; Kim-form core identical to ladder ≤1e-12. **Provenance resolved (source research):** circulated (0.085, 1.1) are NOT Kim's m,n — they are Spiegel's α₁ (V) / k (dimensionless), used with a *separate* i_L; the genuine Kim A/cm² constants are m≈3e-5 V, n≈8. Former strict-xfail → passing provenance test |
| ⚑ | **Flagship + SRH recombination** | Shockley-Read/Hall 1952; Sah-Noyce-Shockley 1957 | ✅ τ→∞ regression exact; conservation with R active 1e-7; **minority diffusion length emerges from profile: L_p/√(Dτ)=0.99**; two-regime ideality ~1.8→1.08 |
| ⏱ | **Transient/BDF capability** (`numerics/transient.py`, scipy-delegated stiff integrator) validated on the Robertson stiff kinetics benchmark | Robertson 1966 / Hairer-Wanner II; cross-code: SUNDIALS CVODE cvRoberts (tolerances verified from source) | ✅ structural conservation Σ(RHS)≡0 → invariant to machine precision over 16 time-decades; analytic Jacobian vs FD ~1e-11; **3 independent methods (Radau/BDF/LSODA) agree ~1e-9**; SUNDIALS cross-code pin y3 all 12 decades ≤4e-4. **Honest finding:** the SUNDIALS printed table is a loose-tolerance demo — its own late-time values drift ~1–33% (confirmed by high-accuracy Radau); stiffness real (RK45 ~740× the BDF f-evals) |

See [`validation/CATALOG.md`](validation/CATALOG.md) for details, provenance and
license flags. Code is never copied from sources — algorithms are reimplemented and
compared against published numbers.

## 2D — partly built, and the boundary is the point

The box (finite-volume / Voronoi) method on triangular meshes, validated stage by
stage against **DEVSIM** — an independent production TCAD code — on *DEVSIM's own
meshes*, so no difference in mesh or discretisation can hide inside the
comparison. The ladder and its reasoning live in
[`docs/DESIGN-2D.md`](docs/DESIGN-2D.md) §5, which is the authority if that table
and this one ever drift apart.

| Stage | Case | Oracle | Status |
|---|------|--------|--------|
| 2D-0 | A 1D problem run through the 2D machinery | `pn1d` itself | ✅ exact rather than approximate: on a strip built from `pn1d`'s own grid the box method reduces to its tridiagonal scheme identically, so ψ agrees to **1.8e-15** |
| 2D-1 | Equilibrium 2D pn junction | DEVSIM + analytic V_bi | ✅ max \|Δψ\| = **2.24e-16 V**; V_bi identical from TARHAN, from DEVSIM, and from V_t·ln(N_a·N_d/n_i²) |
| 2D-2 | 2D diode I–V | DEVSIM, same mesh | ✅ I_n ratio **1.00000** at every bias; ideality **1.0119–1.0134** against DEVSIM's own **1.0114–1.0126**. The first stage that exercises transverse transport |
| 2D-3′ | Electrostatic contact charge | DEVSIM `cap2d.py` | ✅ contact charge ratio **1.000000000** |
| 2D-3 | MOS capacitor C–V | — | ⛔ **BLOCKED** — the reference case is a small-signal AC solve wired to a lumped circuit, and this code has neither an AC nor a circuit layer |
| 2D-4 | MOSFET I–V | — | ⛔ **BLOCKED** — the reference mesh is not Delaunay on 22 of its interior edges, and the box method's positivity guarantee rests on exactly that property |

**What 2D is not.** No AC or small-signal solve, no circuit coupling, no mesh
generation and no mesh repair — meshes are read, never made — no 3D, and no 2D on
the MCP tool surface. Two of the six rows above are blocked and say why; that is
the honest shape of it.

## License

**Apache-2.0** (see `LICENSE` and `NOTICE`). Copyright © 2026 Muhammet Emir Çobanoğlu.

You may use, study, modify and redistribute TARHAN freely, including inside
closed-source and commercial work. Keep the copyright notice, the licence text
and the `NOTICE` file with the code, and state any changes you made.

Apache-2.0 also grants an explicit patent licence from every contributor, which
matters for numerical methods that may later be patentable.

> Relicensed from AGPL-3.0-or-later on 2026-08-05, to prioritise academic reuse
> and citation. Versions published before that date remain available under
> AGPL-3.0-or-later; that grant cannot be withdrawn retroactively.

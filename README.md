# TARHAN

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
tarhan demo                 # zero-config: converged Cottrell reproduction + plot
tarhan demo --case diode    # flagship drift-diffusion diode: I-V + band diagram
pytest                      # run the Layer-0 validation catalog
```

Optional extras: `tarhan[mcp]` — MCP server (`tarhan-mcp`) exposing 8 guarded
tools (diode I-V/band diagram, cyclic voltammetry incl. Butler-Volmer, Nicholson
working curve, SOFC polarization, fuel-cell loss ladder, honesty-tier formula
catalog) so AI agents can drive the engine; `tarhan[oracle]` — DEVSIM
cross-validation tests. Known-good pins: `pip install tarhan -c constraints.txt`.

## The honesty model

No formula enters this codebase without all three of:

1. a **source citation** (book/paper, section, printed expected numbers),
2. an **oracle check** where applicable (dimensional + symbolic + numeric verification;
   e.g. the Cottrell equation passed 3/3 before being added to `tarhan.physics`),
3. a **Layer-0 reproduction test** pinned to published numbers or measured convergence
   order (never an arbitrary tolerance).

Every function in `tarhan.physics` carries an explicit honesty tier:
`first-principles (oracle-verified)` / `textbook (reproduced)` / `empirical fit`.
Reduced-precision backends can never be the truth path (see `tarhan/backend.py`).

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

## License

BSD-3-Clause (see `LICENSE`).

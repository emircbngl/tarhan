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
tarhan demo          # zero-config: runs a converged Cottrell chronoamperometry
                     # reproduction and plots simulated vs analytic current
pytest               # run the Layer-0 validation catalog
```

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

## Layer-0 validation catalog (8/13 reproduced so far)

| # | Case | Source | Status |
|---|------|--------|--------|
| 0 | Cottrell chronoamperometry, explicit FD | Britz & Strutwolf 4e | ✅ max err 0.016%, observed order 1.97 |
| 1 | Nafion membrane correlations λ(a), κ(λ,T) | Springer et al., JES 138 (1991) | ✅ 5/6 printed digits exact |
| 2 | P+N step-junction electrostatics | Hu, *Modern Semiconductor Devices*, Ex. 4-1 | ✅ 3/3 printed values |
| 3 | 1D diffusion MMS, exact numerical solution | Linge & Langtangen §3.6.5 (CC BY) | ✅ machine precision (≤8e-15) |
| 4 | Semi-integral of Cottrell current = const | Oldham, Myland & Bond | ✅ measured order 0.50, Richardson → 1.000000 |
| 8 | Reversible CV peak ψ_p = 0.4463 (solver-level) | Compton ⊕ Britz ⊕ Bard & Faulkner | ✅ J_p=0.44636, θ_p=−1.109, half-width 2.202 |
| 7 | Observed convergence-rate harness | Linge & Langtangen §1.1.4, §3.6.6 | ✅ 2.0000 / 1.0000 / 2.0002 |
| 10 | Scharfetter–Gummel vs central-difference flux | Farrell et al. (WIAS 2263); Selberherr | ✅ equilibrium at U_T·ln10 to 9e-16 |

See [`validation/CATALOG.md`](validation/CATALOG.md) for details, provenance and
license flags. Code is never copied from sources — algorithms are reimplemented and
compared against published numbers.

## License

BSD-3-Clause (see `LICENSE`).

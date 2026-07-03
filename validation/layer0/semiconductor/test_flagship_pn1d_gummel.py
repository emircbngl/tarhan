"""AMİRAL GEMİSİ — 1D pn-diyot Gummel/SG drift-diffusion çözücüsü doğrulaması.

Roadmap Faz-1 kabul kriterleri (Materials Sim — Roadmap): denge V_bi analitiğe
<%1 (ölçüm: 0.57 µV!), ideal Shockley ideality 1.00±0.02 düşük enjeksiyonda,
korunum her çözümde. Ek: R=0 kısa-taban mimarisi mutlak akımı da kapalı-formla
kilitler (J0 = q·ni²·[Dp/(Nd·Wqn)+Dn/(Na·Wqp)]) — ölçüm: %0.32 mutabakat.

Yapıtaşları: rank-9 analitikleri (builtin_potential, depletion_width) + rank-10
SG akısı (bernoulli) + backend.solve_tridiag dikişi + Playbook De Mari ölçekleme.
Düşük-V ideality penceresi 0.15-0.40 V: (e^{V/UT}−1)'in −1'i 0.05-0.10 V'ta
ihmal edilemez (rank-9 pencere dersinin aynısı — orada öğrenildi, burada uygulandı).
"""
import math

import numpy as np
import pytest

from tarhan.models.pn1d import PNDiode1D, band_diagram, iv_sweep, solve_bias
from tarhan.physics import builtin_potential, depletion_width_step_junction

DEV = PNDiode1D()          # Na=Nd=1e16, Si-tipik, 3+3 um, açık vaka girdileri
VOLTS = [0.05 * k for k in range(1, 9)]           # 0.05..0.40


@pytest.fixture(scope="module")
def sweep():
    js, state = iv_sweep(DEV, VOLTS)
    return js, state


def test_equilibrium_builtin_potential_microvolt():
    eq = solve_bias(DEV, 0.0)
    vbi_sim = DEV.ut * float(eq["psi"][-1] - eq["psi"][0])
    vbi_ana = builtin_potential(DEV.Na, DEV.Nd, DEV.ni, DEV.ut)
    assert abs(vbi_sim - vbi_ana) < 5e-5          # ölçüm: 5.7e-7 V


def test_equilibrium_current_is_zero():
    eq = solve_bias(DEV, 0.0)
    assert abs(eq["j"]) < 1e-9                     # ölçüm: ~2e-12 A/cm²


def test_equilibrium_depletion_width_debye_tails():
    """Fizik-temelli kriter (ilk sürümdeki %10 keyfî toleransı rank-4 dersi uyarınca
    değiştirildi): %10-eşikli genişlik Debye kuyruklarıyla depletion-yaklaşımından
    GENİŞTİR (ölçüm: +2.91·L_D); izin payı fiziksel — 4·L_D."""
    eq = solve_bias(DEV, 0.0)
    rho = np.asarray(eq["n_hat"] - eq["p_hat"] - DEV.doping_hat(eq["x_hat"]))
    x_cm = np.asarray(eq["x_hat"]) * DEV.L_D
    mask = np.abs(rho) > 0.1
    w_sim = x_cm[mask][-1] - x_cm[mask][0]
    w_ana = depletion_width_step_junction(
        DEV.eps_s, builtin_potential(DEV.Na, DEV.Nd, DEV.ni, DEV.ut), DEV.q, DEV.Na, DEV.Nd)
    assert w_ana < w_sim < w_ana + 4.0 * DEV.L_D


def _e_max(h0, gamma):
    eq = solve_bias(PNDiode1D(h0=h0, gamma=gamma), 0.0)
    x = np.asarray(eq["x_hat"]) * DEV.L_D
    psi_v = DEV.ut * np.asarray(eq["psi"])
    return float(np.max(np.abs(np.diff(psi_v) / np.diff(x))))


def test_peak_field_reproduces_sze_correction():
    """Çözücü, Sze'nin ψ_bi−2kT/q düzeltmesini BAĞIMSIZ üretiyor.

    TARİHÇE (dürüstlük): ilk sürümde sabit-grid E_max, Sze'yle %0.03 uyumluydu —
    FV eklem-doping düzeltmesi (2026-07-04) sonrası bunun TESADÜFİ hata-iptali
    olduğu anlaşıldı. Gerçek iddia daha güçlü: E_max temiz O(h) yakınsar (farklar
    tam yarılanır) ve order-1 Richardson limiti Sze'yi ~4e-5 içinde vurur; sabit
    baseline-grid değeri ~%1.2 grid-nicemlemesi taşır (belgeli)."""
    e_c = _e_max(2.5e-7, 1.03)
    e_f = _e_max(1.25e-7, 1.0149)
    e_ext = 2.0 * e_f - e_c                       # order-1 Richardson (h yarılanır)

    vbi = builtin_potential(DEV.Na, DEV.Nd, DEV.ni, DEV.ut)
    v_sze = vbi - 2.0 * DEV.ut
    w_sze = depletion_width_step_junction(DEV.eps_s, v_sze, DEV.q, DEV.Na, DEV.Nd)
    e_sze = 2.0 * v_sze / w_sze
    assert abs(e_ext / e_sze - 1.0) < 5e-4        # ölçüm: 4e-5
    assert abs(_e_max(DEV.h0, DEV.gamma) / e_sze - 1.0) < 0.02   # baseline: O(h) bandı
    w0 = depletion_width_step_junction(DEV.eps_s, vbi, DEV.q, DEV.Na, DEV.Nd)
    assert e_ext / (2.0 * vbi / w0) < 0.985       # naif depletion ölçülebilir sapar


def test_e_max_first_order_grid_convergence():
    """E_max kink-noktası edge-max'ı: farklar yarılanmalı (gözlenen mertebe ~1)."""
    e1, e2, e3 = _e_max(2.5e-7, 1.03), _e_max(1.25e-7, 1.0149), _e_max(6.25e-8, 1.00744)
    p = math.log2((e2 - e1) / (e3 - e2)) if (e3 - e2) != 0 else 0.0
    assert 0.8 < abs(p) < 1.3                     # ölçüm: 0.98


def test_shockley_ideality_low_injection(sweep):
    js, _ = sweep
    for i in range(3, len(VOLTS)):                 # 0.15 V'tan itibaren
        nid = (VOLTS[i] - VOLTS[i - 1]) / (DEV.ut * math.log(js[i] / js[i - 1]))
        assert abs(nid - 1.0) < 0.02, f"{VOLTS[i-1]}->{VOLTS[i]}: n_id={nid:.4f}"


def test_short_base_absolute_current():
    v = 0.30
    st = solve_bias(DEV, v)
    vbi = builtin_potential(DEV.Na, DEV.Nd, DEV.ni, DEV.ut)
    w = depletion_width_step_junction(DEV.eps_s, vbi - v, DEV.q, DEV.Na, DEV.Nd)
    xn = w * DEV.Na / (DEV.Na + DEV.Nd)
    wq_n, wq_p = DEV.len_n - xn, DEV.len_p - (w - xn)
    j0 = DEV.q * DEV.ni ** 2 * (DEV.mu_p * DEV.ut / (DEV.Nd * wq_n)
                                + DEV.mu_n * DEV.ut / (DEV.Na * wq_p))
    j_ana = j0 * (math.exp(v / DEV.ut) - 1.0)
    assert abs(st["j"] / j_ana - 1.0) < 0.02       # ölçüm: 0.0032


def test_current_spatially_constant(sweep):
    _, state = sweep                                # V=0.40 durumu
    spread = float(np.max(np.abs(np.asarray(state["j_edges"]) - state["j"]))) / abs(state["j"])
    assert spread < 1e-6                            # ölçüm: ~1e-8 (ayrık korunum)


def test_gummel_converges_fast_with_warm_start(sweep):
    _, state = sweep
    assert state["gummel_iters"] <= 10              # ölçüm: 1-3


def test_quasi_fermi_splitting_equals_bias():
    v = 0.30
    st = solve_bias(DEV, v)
    split = DEV.ut * float(np.max(np.asarray(st["phi_p"] - st["phi_n"])))
    assert abs(split - v) < 0.01


def test_band_diagram_consistency():
    st = solve_bias(DEV, 0.2)
    bd = band_diagram(DEV, st, e_gap=1.12)
    gap = np.asarray(bd["Ec"] - bd["Ev"])
    assert float(np.max(np.abs(gap - 1.12))) < 1e-12


def test_input_guards():
    with pytest.raises(ValueError):
        PNDiode1D(Na=-1e16)
    with pytest.raises(ValueError):
        PNDiode1D(ni=float("nan"))

"""Rank 8 — reversible LSV/CV tepe sabiti ψ_p = 0.4463 (üç-kitap çapraz benchmark).

Kaynaklar: Compton, Laborda & Ward (fully-implicit + genişleyen grid — bizim şema);
Britz & Strutwolf CV bölümü; Bard & Faulkner Böl. 6 (Nicholson-Shain 1964 tablosu).
Tablolanmış çapalar: J_p = 0.4463; θ_p = −1.109 (E_p = E_1/2 − 28.5/n mV);
|θ_{p/2} − θ_p| = 2.20 (E_p − E_{p/2} = 56.5/n mV, 25 °C).
Ölçüm (2026-07-03): J_p = 0.44636 üç çözünürlükte de (dθ = 2e-3 → 5e-4 kararlı).
İlk çözücü-düzeyi hedef: zamana bağlı Nernst BC + Thomas (backend.solve_tridiag dikişi).
"""
from tarhan.numerics.voltammetry import find_peak, half_peak_theta, reversible_lsv

PSI_P, THETA_P, HALF_WIDTH = 0.4463, -1.109, 2.20


def _solve(d_theta, h0):
    thetas, J = reversible_lsv(d_theta=d_theta, h0=h0)
    theta_p, j_p = find_peak(thetas, J)
    return thetas, J, theta_p, j_p


def test_peak_current_three_book_benchmark():
    _, _, _, j_p = _solve(2e-3, 2e-3)
    assert abs(j_p - PSI_P) < 5e-4, f"J_p = {j_p:.5f}, tablolanmış {PSI_P}"


def test_peak_potential():
    _, _, theta_p, _ = _solve(2e-3, 2e-3)
    assert abs(theta_p - THETA_P) < 0.02


def test_half_peak_width():
    thetas, J, theta_p, j_p = _solve(2e-3, 2e-3)
    assert abs(abs(half_peak_theta(thetas, J, j_p) - theta_p) - HALF_WIDTH) < 0.02


def test_resolution_stability():
    _, _, _, j_coarse = _solve(2e-3, 2e-3)
    _, _, _, j_fine = _solve(1e-3, 1e-3)
    assert abs(j_fine - j_coarse) < 2e-4

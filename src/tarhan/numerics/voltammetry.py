"""Voltametri çözücüleri — Rank 8: reversible linear-sweep/CV (boyutsuz).

Yöntem kaynağı (aktarım; kod bizim): Compton, Laborda & Ward, "Understanding
Voltammetry: Simulation of Electrode Processes" — fully-implicit FD + genişleyen
uzaysal grid + Thomas çözümü; çapraz kaynaklar: Britz & Strutwolf CV bölümü,
Bard & Faulkner Böl. 6 (Nicholson-Shain 1964).

Model (eşit difüzyon katsayılı O + e⁻ ⇌ R, yarı-sonsuz):
    ∂a/∂s = ∂²a/∂X²,  a(X,0)=1,  a(∞)=1
    θ(s) = θ_başlangıç − s  (indirgeme taraması),  Nernst: a(0) = 1/(1+e^{−θ})
    (eşit D ⇒ a+b=1 her yerde ⇒ tek denklem yeter)
Boyutsuz akım J(s) = ∂a/∂X|₀ ; klasik sonuç: J_p = 0.4463, θ_p = −1.109,
|θ_{p/2} − θ_p| = 2.20 (E_p−E_{p/2} = 56.5/n mV, 25 °C).

Sayısal şema: backward-Euler zaman + 3-nokta düzensiz-grid Laplasyeni,
X_j+1 = X_j + h₀·γ^j genişleyen grid; her adım backend.solve_tridiag (dikiş).
"""
from __future__ import annotations

import math

from tarhan import backend


def _expanding_grid(h0: float, gamma: float, x_max: float):
    xp = backend.xp()
    xs = [0.0, h0]
    h = h0
    while xs[-1] < x_max:
        h *= gamma
        xs.append(xs[-1] + h)
    return xp.asarray(xs)


def reversible_lsv(theta_start: float = 20.0, theta_end: float = -20.0,
                   d_theta: float = 1e-3, h0: float = 1e-3,
                   gamma: float = 1.05, x_max_factor: float = 6.0):
    """Reversible LSV; döner: (thetas, J) — boyutsuz potansiyel ve akım dizileri."""
    xp = backend.xp()
    s_total = theta_start - theta_end
    X = _expanding_grid(h0, gamma, x_max_factor * math.sqrt(s_total))
    n = len(X)

    hm = X[1:-1] - X[:-2]    # h_{i-1},  i = 1..n-2 (iç düğümler)
    hp = X[2:] - X[1:-1]     # h_i
    sub = -d_theta * 2.0 / (hm * (hm + hp))
    sup = -d_theta * 2.0 / (hp * (hm + hp))
    diag = 1.0 - (sub + sup)          # BE: (I − Δs·L)

    # 3-nokta tek-yönlü türev katsayıları (X0=0, X1, X2 düzensiz)
    A_, B_ = float(X[1] - X[0]), float(X[2] - X[1])
    d0 = -(2 * A_ + B_) / (A_ * (A_ + B_))
    d1 = (A_ + B_) / (A_ * B_)
    d2 = -A_ / (B_ * (A_ + B_))

    n_steps = int(round(s_total / d_theta))
    a = xp.ones(n)
    thetas = xp.empty(n_steps)
    J = xp.empty(n_steps)
    for m in range(1, n_steps + 1):
        theta = theta_start - m * d_theta
        a0 = 1.0 / (1.0 + math.exp(-theta))
        rhs = xp.array(a[1:-1], dtype=float, copy=True)
        rhs[0] -= sub[0] * a0
        rhs[-1] -= sup[-1] * 1.0            # uzak sınır: a = 1
        a[1:-1] = backend.solve_tridiag(sub, diag, sup, rhs)
        a[0] = a0
        thetas[m - 1] = theta
        J[m - 1] = d0 * a0 + d1 * float(a[1]) + d2 * float(a[2])
    return thetas, J


def find_peak(thetas, J):
    """Parabolik interpolasyonla tepe: (theta_p, J_p). Grid-arası tepeyi yakalar."""
    xp = backend.xp()
    m0 = int(xp.argmax(J))
    if m0 == 0 or m0 == len(J) - 1:
        return float(thetas[m0]), float(J[m0])
    jm, j0, jp = float(J[m0 - 1]), float(J[m0]), float(J[m0 + 1])
    denom = jm - 2.0 * j0 + jp
    delta = 0.5 * (jm - jp) / denom if denom != 0.0 else 0.0
    d_th = float(thetas[m0 + 1] - thetas[m0])     # negatif (indirgeme taraması)
    theta_p = float(thetas[m0]) + delta * d_th
    j_p = j0 - 0.25 * (jm - jp) * delta
    return theta_p, j_p


def half_peak_theta(thetas, J, j_p: float):
    """Tepeden ÖNCE (yükselen kolda) J = J_p/2 kesişimi — lineer interpolasyon."""
    xp = backend.xp()
    m0 = int(xp.argmax(J))
    target = 0.5 * j_p
    for m in range(m0, 0, -1):
        if J[m - 1] <= target <= J[m]:
            w = (target - float(J[m - 1])) / (float(J[m]) - float(J[m - 1]))
            return float(thetas[m - 1]) + w * (float(thetas[m]) - float(thetas[m - 1]))
    raise RuntimeError("yarı-tepe kesişimi bulunamadı (tarama aralığını kontrol et)")

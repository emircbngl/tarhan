"""Konvektif difüzyon — Rank 11: dönen disk elektrot (Levich) sınır tabakası.

Yöntem kaynağı (aktarım; kod bizim): Newman & Thomas-Alyea, Electrochemical
Systems 3e (RDE bölümü); Cochran yakın-yüzey eksenel hız profili
v_z = −0.51023·ω^{3/2}·ν^{−1/2}·z². Kararlı konvektif-difüzyon
D·c'' = v_z·c' boyutsuzlaştırılınca (ζ = z/(3D/a)^{1/3}, a = 0.51023·ω^{3/2}ν^{−1/2}):

    c''(ζ) + 3ζ²·c'(ζ) = 0,   c(0)=0 (limit akımı),  c(∞)=1

Analitik: c'(ζ) ∝ exp(−ζ³) ⇒ c'(0) = 1/∫₀^∞e^{−u³}du = 1/Γ(4/3).
Levich sabiti = (0.51023/3)^{1/3} / Γ(4/3) = 0.620450 ("0.620").
NOT (rank-11 dersi): elle ara-hesap 0.620469 vermişti — iki bağımsız sayısal
yol 0.6204500'de buluşup kaymayı yakaladı; literatürdeki 0.62048 türevleri
Cochran katsayısının (0.51023, kendisi kesik) hassasiyetine iner.
Sınır tabakası: δ = (3/0.51023)^{1/3}·Γ(4/3) = 1.612 → klasik 1.61 çarpanı.
"""
from __future__ import annotations

import math

from tarhan import backend

#: (0.51023/3)^(1/3) — Cochran katsayısından gelen geometrik çarpan
_COCHRAN_CUBE = (0.51023 / 3.0) ** (1.0 / 3.0)


def levich_constant_quadrature(n_panels: int = 20000, u_max: float = 6.0) -> float:
    """Yol 1 — saf kuadratür: I = ∫₀^∞ e^{−u³} du (bileşik Simpson), sabit = küp/I.

    Γ(4/3)'e hiç başvurmaz; I'nın kendisi Γ(4/3)=0.892980'in bağımsız
    reprodüksiyonudur (test bunu math.gamma ile çapraz-kontrol eder).
    """
    if n_panels % 2:
        n_panels += 1
    h = u_max / n_panels
    total = 0.0
    for i in range(n_panels + 1):
        u = i * h
        w = 1.0 if i in (0, n_panels) else (4.0 if i % 2 else 2.0)
        total += w * math.exp(-u * u * u)
    integral = total * h / 3.0
    return _COCHRAN_CUBE / integral


def levich_constant_fd(n_nodes: int = 1000, zeta_max: float = 6.0) -> float:
    """Yol 2 — FD/BVP: c'' + 3ζ²c' = 0 merkezi farklarla, Thomas (dikiş) çözümü.

    c(0)=0, c(ζ_max)=1; boyutsuz yüzey akısı c'(0) 3-nokta türevle; sabit =
    küp·c'(0). Merkezi fark 2. mertebe — test gözlenen mertebeyi ölçer.
    (Hücre-Peclet: 3ζ²h/2 < 1 gerekir → n_nodes ≥ ~700 @ ζ_max=6.)
    """
    xp = backend.xp()
    h = zeta_max / n_nodes
    zeta = xp.arange(1, n_nodes) * h              # iç düğümler
    conv = 1.5 * zeta * zeta * h                  # (3ζ²)·h/2
    sub = xp.ones(n_nodes - 1) - conv             # c_{i-1} katsayısı
    diag = -2.0 * xp.ones(n_nodes - 1)
    sup = xp.ones(n_nodes - 1) + conv             # c_{i+1} katsayısı
    rhs = xp.zeros(n_nodes - 1)
    rhs[-1] = -(1.0 + float(conv[-1])) * 1.0      # c(ζ_max) = 1
    c_in = backend.solve_tridiag(sub, diag, sup, rhs)
    c0, c1, c2 = 0.0, float(c_in[0]), float(c_in[1])
    dcdz0 = (-3.0 * c0 + 4.0 * c1 - c2) / (2.0 * h)
    return _COCHRAN_CUBE * dcdz0

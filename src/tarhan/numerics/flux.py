"""Kenar akıları: Bernoulli fonksiyonu ve Scharfetter-Gummel / central-difference.

Kaynak (yöntem aktarımı): Farrell ve ark., WIAS Preprint 2263, §2; Selberherr
(1984) Böl. 6. Layer-0 testi: test_rank10_sg_flux.py — SG dengesi TAM
ΔΨ = U_T·ln(n_sol/n_sağ) noktasında (B(-x) = e^x·B(x) kimliği).
"""
from __future__ import annotations

from tarhan import backend


def bernoulli(x):
    """B(x) = x / (e^x − 1); |x| < 1e-8 için kararlı seri 1 − x/2 + x²/12.

    Büyük |x| overflow koruması: expm1 argümanı ±700'e kırpılır (x>700'de
    gerçek B ~ x·e^(−x) ≈ 0; kırpılmış payda da ~0 verir — belgeli yaklaşıklık).
    """
    xp = backend.xp()
    x = xp.asarray(x, dtype=float)
    small = xp.abs(x) < 1e-8
    safe = xp.where(small, 1.0, x)
    return xp.where(small,
                    1.0 - x / 2.0 + x * x / 12.0,
                    safe / xp.expm1(xp.clip(safe, -700.0, 700.0)))


def sg_edge_flux(dpsi, n_left, n_right, u_t: float = 1.0, coef: float = 1.0):
    """Scharfetter-Gummel kenar akısı (normalize).

    j = coef · [B(−x)·n_sağ − B(x)·n_sol],  x = ΔΨ/U_T  (ΔΨ = ψ_sağ − ψ_sol)
    Denge: j = 0 tam olarak x = ln(n_sol/n_sağ) noktasında.
    """
    xp = backend.xp()
    x = xp.asarray(dpsi, dtype=float) / u_t
    return coef * (bernoulli(-x) * n_right - bernoulli(x) * n_left)


def cd_edge_flux(dpsi, n_left, n_right, u_t: float = 1.0, coef: float = 1.0):
    """Merkezi-fark drift-difüzyon akısı — KARŞILAŞTIRMA için (kullanma!).

    Dengeyi yanlış noktaya koyar ve büyük mesh-Peclet'te yanlış işarete kaçar;
    Layer-0 testi bunu sayısal olarak gösterir (SG'nin varlık sebebi).
    """
    xp = backend.xp()
    x = xp.asarray(dpsi, dtype=float) / u_t
    return coef * (-(xp.asarray(n_right, dtype=float) - n_left)
                   - 0.5 * (xp.asarray(n_left, dtype=float) + n_right) * x)

"""Rank 4 — Cottrell akımının yarı-integrali sabittir: M(T) = 1 (boyutsuz).

Kaynak: Oldham, Myland & Bond (Grünwald yarı-integrasyonu). Ders (2026-07-03):
keyfî tolerans koyma — operatörün mertebesini ÖLÇ (t^-1/2 tekilliğinde 0.50) ve
limiti Richardson ile ekstrapole et (ölçüm: M_ext = 1.000000).
"""
import math

import numpy as np

from tarhan.numerics.diffusion1d import cottrell_fd_series
from tarhan.numerics.fraccalc import semi_integrate
from tarhan.numerics.verify import richardson_extrapolate


def _m_analytic(T: float, dt: float) -> float:
    k = int(round(T / dt))
    t = dt * np.arange(1, k + 1)
    return semi_integrate(1.0 / np.sqrt(math.pi * t), dt)


def test_observed_order_is_half():
    dts = [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4]
    errs = [abs(_m_analytic(1.0, dt) - 1.0) for dt in dts]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), "monoton değil"
    p = math.log(errs[0] / errs[-1]) / math.log(dts[0] / dts[-1])
    assert abs(p - 0.5) < 0.05, f"gözlenen mertebe {p:.3f}, teori 0.5"


def test_richardson_limit_hits_closed_form():
    m_c = _m_analytic(1.0, 1e-3)
    m_f = _m_analytic(1.0, 2.5e-4)
    m_ext = richardson_extrapolate(m_c, m_f, ratio=4.0, order=0.5)
    assert abs(m_ext - 1.0) < 5e-4, f"M_ext = {m_ext:.6f}"


def test_fd_and_analytic_routes_agree():
    G, dt, _ = cottrell_fd_series(n_x=200)
    for T in (0.5, 1.0):
        k = int(round(T / dt))
        m_fd = semi_integrate(np.asarray(G[:k]), dt)
        m_an = _m_analytic(T, dt)
        assert abs(m_fd - m_an) < 3e-3, f"T={T}: yol farkı {abs(m_fd-m_an):.5f}"

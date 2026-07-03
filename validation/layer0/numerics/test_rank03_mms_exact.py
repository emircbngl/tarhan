"""Rank 3 — 1D difüzyon MMS: makine-hassasiyetli tam çözüm (Langtangen §3.6.5).

u(x,t) = 5tx(L−x) seçimiyle FE + merkezi fark TAM olur (t'de lineer, x'te kuadratik);
f = 5x(L−x) + 10at kaynaklı çözüm her düğümde ~1e-14 doğrulukta olmalı — grid'den
bağımsız. İndeks/BC/kaynak-terim bug'ları için binary tuzak.
"""
import numpy as np
import pytest

from tarhan.numerics.diffusion1d import fe_step

L, A, T_END, F = 1.5, 0.5, 0.4, 0.5


def _max_err(nx: int) -> float:
    dx = L / nx
    dt = F * dx * dx / A
    nt = int(round(T_END / dt))
    x = np.linspace(0.0, L, nx + 1)
    u = np.zeros(nx + 1)
    f_num = A * dt / dx**2
    worst = 0.0
    for n in range(nt):
        t = n * dt
        src = 5.0 * x * (L - x) + 10.0 * A * t
        u = fe_step(u, f_num, dt, src)
        u[0] = u[-1] = 0.0
        exact = 5.0 * (n + 1) * dt * x * (L - x)
        worst = max(worst, float(np.max(np.abs(u - exact))))
    return worst


@pytest.mark.parametrize("nx", [10, 40, 160])
def test_machine_precision_any_grid(nx):
    assert _max_err(nx) < 1e-12

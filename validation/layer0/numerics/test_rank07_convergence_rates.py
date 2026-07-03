"""Rank 7 — gözlenen yakınsama mertebeleri (Langtangen §1.1.4, §3.6.6).

Harness ölçümleri (2026-07-03): titreşim/merkezi 2.0000, difüzyon FE 1.0000,
Crank-Nicolson 2.0002. CN, backend.solve_tridiag dikişini de test eder.
"""
import math

import numpy as np

from tarhan import backend
from tarhan.numerics.diffusion1d import fe_step
from tarhan.numerics.verify import convergence_rates


def _vibration_error(dt, w=2.0, amp=1.0, t_end=8.0):
    nt = int(round(t_end / dt))
    u_prev, u = amp, amp - 0.5 * (w * dt) ** 2 * amp
    worst = abs(u - amp * math.cos(w * dt))
    for n in range(1, nt):
        u_prev, u = u, 2.0 * u - u_prev - (w * dt) ** 2 * u
        worst = max(worst, abs(u - amp * math.cos(w * (n + 1) * dt)))
    return worst


def _diffusion_error(scheme, nx, a=0.7, L=1.0, t_end=0.4):
    dx = L / nx
    dt = 0.4 * dx * dx / a if scheme == "FE" else 0.25 * dx
    nt = int(round(t_end / dt))
    dt = t_end / nt
    x = np.linspace(0.0, L, nx + 1)
    u = np.sin(math.pi * x / L)
    if scheme == "CN":
        r = a * dt / (2 * dx * dx)
        n_in = nx - 1
        sub = np.full(n_in, -r)
        diag = np.full(n_in, 1 + 2 * r)
        sup = np.full(n_in, -r)
    for _ in range(nt):
        if scheme == "FE":
            u = fe_step(u, a * dt / dx**2)
        else:
            rhs = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
            u[1:-1] = backend.solve_tridiag(sub, diag, sup, rhs)
        u[0] = u[-1] = 0.0
    exact = math.exp(-a * (math.pi / L) ** 2 * t_end) * np.sin(math.pi * x / L)
    return float(np.max(np.abs(u - exact))), dt


def test_vibration_central_order_two():
    dts = [0.1 / 2**i for i in range(6)]
    errs = [_vibration_error(dt) for dt in dts]
    assert abs(convergence_rates(errs, dts)[-1] - 2.0) < 0.05


def test_diffusion_fe_order_one():
    errs, dts = zip(*[_diffusion_error("FE", nx) for nx in (10, 20, 40, 80, 160)])
    assert abs(convergence_rates(list(errs), list(dts))[-1] - 1.0) < 0.05


def test_diffusion_cn_order_two():
    errs, dts = zip(*[_diffusion_error("CN", nx) for nx in (10, 20, 40, 80, 160)])
    assert abs(convergence_rates(list(errs), list(dts))[-1] - 2.0) < 0.05

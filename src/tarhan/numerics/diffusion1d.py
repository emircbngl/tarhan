"""1D difüzyon primitifleri: explicit (FE) adım + boyutsuz Cottrell simülatörü.

Yöntem kaynakları (aktarım; kod bizim): Britz & Strutwolf, Digital Simulation in
Electrochemistry, 4e (explicit yöntem, λ<0.5); Linge & Langtangen (FE/CN şemaları).
Layer-0: test_rank00_cottrell_fd.py, test_rank03_mms_exact.py.
"""
from __future__ import annotations

import math

from tarhan import backend


def fe_step(u, f_num: float, dt: float = 0.0, source=None):
    """Bir Forward-Euler difüzyon adımı (Dirichlet uçlar dokunulmaz).

    u_yeni[i] = u[i] + f_num·(u[i+1] − 2u[i] + u[i−1]) (+ dt·source[i])
    """
    xp = backend.xp()
    u_new = xp.array(u, dtype=float, copy=True)
    lap = u[2:] - 2.0 * u[1:-1] + u[:-2]
    u_new[1:-1] = u[1:-1] + f_num * lap
    if source is not None:
        u_new[1:-1] = u_new[1:-1] + dt * source[1:-1]
    return u_new


def cottrell_fd_series(n_x: int = 200, t_max: float = 1.0, lam: float = 0.45,
                       x_max_factor: float = 6.0):
    """Boyutsuz Cottrell (potansiyel adımı) explicit-FD; her adımda G kaydeder.

    dC/dT = d²C/dX², C(X,0)=1, C(0,T>0)=0; G = ∂C/∂X|₀ (3-nokta, C[0]=0).
    Döner: (G_series [adım başına], dt, dx). G_series[k] = G((k+1)·dt).
    """
    xp = backend.xp()
    x_max = x_max_factor * math.sqrt(t_max)
    dx = x_max / n_x
    dt = lam * dx * dx
    n_t = int(math.ceil(t_max / dt))
    C = xp.ones(n_x + 1)
    C[0] = 0.0
    G = xp.empty(n_t)
    for step in range(n_t):
        C[1:-1] = C[1:-1] + lam * (C[2:] - 2.0 * C[1:-1] + C[:-2])
        G[step] = (4.0 * C[1] - C[2]) / (2.0 * dx)
    return G, dt, dx


def cottrell_fd_samples(targets, n_x: int = 200, t_max: float = 1.0,
                        lam: float = 0.45, x_max_factor: float = 6.0):
    """Hedef T'lerde G — adımlar arası lineer interpolasyonla (zaman-kayması
    O(dT)'yi O(dT²)'ye indirir; rank-0 dersinin kalıcı hali)."""
    G, dt, dx = cottrell_fd_series(n_x, t_max, lam, x_max_factor)
    out = {}
    for tgt in targets:
        k = int(math.ceil(tgt / dt - 1e-12))          # ilk t_k >= tgt (t_k = k·dt)
        k = min(max(k, 1), len(G))
        t_k = k * dt
        if k == 1:
            out[tgt] = float(G[0])
        else:
            w = (tgt - (t_k - dt)) / dt
            out[tgt] = float(G[k - 2] + w * (G[k - 1] - G[k - 2]))
    return out, dt, dx

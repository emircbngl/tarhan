"""1D potansiyel-basamak kronoamperometrisi — transient/BDF difüzyon (method-of-lines).

`numerics.transient` primitifinin İLK domain uygulaması. Cottrell deneyi: t=0'da
elektrot potansiyeli basamak yapar, yüzey derişimi sıfırlanır (difüzyon-limit), c(x,t)
difüzyonla evrilir; akım yüzey akısıyla orantılıdır → Cottrell i(t)=nFAc*·√(D/(π·t)).

Method-of-lines: ∂c/∂t = D·∂²c/∂x² uzamsal merkezi-farkla ODE sistemine iner. Bu sistem
STIFF'tir (Jacobian özdeğerleri 0 … −4D/dx²; oran (L/dx)²) → implicit BDF doğru yöntem
(`integrate_stiff`). rank-0 aynı fiziği explicit-FD ile üretti; bu implicit yol explicit
CFL kısıtından (dt < dx²/(2D)) MUAF ve primitifin gerçek-PDE uygulamasıdır.

Doğrulama (test_chronoamp_transient): yüzey akısı Cottrell analitiğine uzamsal olarak
temiz O(h²) yakınsar (ölçülen mertebe 2.000); derişim profili erf çözümüyle eşleşir;
implicit avantaj (BDF adım sayısı explicit CFL gereksiniminin çok altında).
"""
from __future__ import annotations

import numpy as _np

from ..numerics.transient import integrate_stiff

__all__ = ["chronoamperogram", "cottrell_current"]

_F_DEFAULT = 96485.33212  # C/mol — CODATA (açık argüman; motor gömmez)


def cottrell_current(t, D: float, c_bulk: float, *, n: float = 1.0,
                     F: float = _F_DEFAULT, area: float = 1.0) -> float:
    """Cottrell analitik akımı i(t) = n·F·A·c*·√(D/(π·t)) — karşılaştırma referansı.

    Katman: textbook (reproduced) — rank-0 Cottrell'in kapalı-form akım biçimi.
    Birimler tutarlı olmalı (ör. D [cm²/s], c* [mol/cm³], A [cm²] ⇒ i [A])."""
    if t <= 0.0:
        raise ValueError(f"t={t} <= 0: Cottrell akımı t→0'da ıraksar")
    return n * F * area * c_bulk * _np.sqrt(D / (_np.pi * t))


def chronoamperogram(D: float, c_bulk: float, t_eval, *, n: float = 1.0,
                     F: float = _F_DEFAULT, area: float = 1.0,
                     n_diff_lengths: float = 6.0, n_x: int = 400,
                     rtol: float = 1e-9, atol: float = 1e-11) -> dict:
    """Potansiyel-basamak kronoamperogramı i(t) — method-of-lines difüzyon + BDF.

    Yarı-sonsuz domain [0, L], L = n_diff_lengths·√(D·t_max) (difüzyon tabakası kesme
    sınırına ulaşmaz); BC c(0,t>0)=0 (basamak), c(L,t)=c*, başlangıç c(x,0)=c*. Akım
    i = n·F·A·D·∂c/∂x|₀ (2. mertebe tek-yanlı türev). Döner: t, current, profil ve
    çözücü tanıkları (nsteps/njev — implicit avantajın kanıtı).
    """
    if not (D > 0.0 and c_bulk > 0.0):
        raise ValueError(f"D={D}, c_bulk={c_bulk}: pozitif olmalı")
    t_eval = [float(t) for t in t_eval]
    if not t_eval or min(t_eval) <= 0.0:
        raise ValueError("t_eval boş olamaz ve t>0 olmalı (Cottrell t→0'da ıraksar)")
    if int(n_x) < 8:
        raise ValueError("n_x >= 8 olmalı")
    t_max = max(t_eval)
    L = n_diff_lengths * _np.sqrt(D * t_max)
    x = _np.linspace(0.0, L, int(n_x) + 1)
    dx = x[1] - x[0]
    inv_dx2 = 1.0 / dx ** 2
    m = int(n_x) - 1

    def rhs(t, u):
        c = _np.empty(int(n_x) + 1)
        c[0], c[-1], c[1:-1] = 0.0, c_bulk, u
        return D * (c[2:] - 2.0 * c[1:-1] + c[:-2]) * inv_dx2

    # sabit tridiag Jacobian (lineer difüzyon operatörü) — bir kez kurulur
    jc = _np.zeros((m, m))
    di = _np.arange(m)
    jc[di, di] = -2.0 * D * inv_dx2
    jc[di[1:], di[1:] - 1] = D * inv_dx2
    jc[di[:-1], di[:-1] + 1] = D * inv_dx2

    def jac(t, u):
        return jc

    u0 = _np.full(m, c_bulk)
    sol = integrate_stiff(rhs, u0, (0.0, t_max), jac=jac, t_eval=t_eval,
                          rtol=rtol, atol=atol, method="BDF")

    current = []
    for k in range(len(t_eval)):
        c1, c2 = sol.y[0, k], sol.y[1, k]           # c[0]=0 (BC)
        dcdx0 = (-3.0 * 0.0 + 4.0 * c1 - c2) / (2.0 * dx)
        current.append(float(n * F * area * D * dcdx0))

    return {
        "t": t_eval,
        "current": current,
        "x": x,
        "profile_final": _np.concatenate([[0.0], sol.y[:, -1], [c_bulk]]),
        "domain_len": L,
        "nsteps": sol.nsteps,
        "njev": sol.njev,
        "cfl_explicit_dt": dx ** 2 / (2.0 * D),      # explicit'in ihtiyaç duyacağı dt
    }

"""1D pn-diyot drift-diffusion çözücüsü — Gummel + Scharfetter-Gummel (amiral gemisi).

Yöntem kaynakları (aktarım; kod bizim): Selberherr (1984) Böl. 5-7 (van Roosbroeck
sistemi, Gummel ayrışımı, De Mari ölçekleme); Farrell ve ark. WIAS 2263 (SG akısı);
Vasileska & Goodnick ders notları (algoritmadan — kod CC BY-NC-SA, taşınmadı).

Model (v1, bilinçli minimal-güvenilir): kararlı hal, Boltzmann istatistiği,
R = 0 (rekombinasyonsuz) ⇒ KISA-TABAN diyot — azınlık taşıyıcılar kontakta
rekombine olur; ideal Shockley davranışı analitik I0 ile birlikte doğrulanabilir:
    J0 = q·ni²·[Dp/(Nd·W_qn,n) + Dn/(Na·W_qn,p)]   (kısa-taban satürasyonu)

Ölçekleme (De Mari, Numerics Playbook reçetesi):
    C0 = max(Na, Nd);  δ = ni/C0;  L_D = sqrt(εs·UT/(q·C0));  x̂ = x/L_D
    ψ̂ = ψ/UT;  n̂,p̂ = n,p/C0;  Poisson tam olarak  ψ̂'' = n̂ − p̂ − N̂
    n̂ = δ·e^{ψ̂−φ̂n},  p̂ = δ·e^{φ̂p−ψ̂}
    J_ölçek = q·UT·μ·C0/L_D  →  J [A/cm²]

Sayısal iskelet:
  - simetrik geometrik grid (eklemde ince: h0, oran γ — Debye çözünürlüğü)
  - denge/bias Poisson: damped Newton (adım kelepçesi |Δψ̂|≤5), M-matris ⇒ global
  - süreklilik: SG kenar akıları (numerics.flux.bernoulli), R=0 ⇒ lineer tridiag
    (backend.solve_tridiag dikişi), M-matris ⇒ pozitiflik garantili
  - Gummel dış döngüsü + 50 mV bias-continuation (warm start)
  - ohmik kontaklar: yük-nötr + np = ni² (ψ̂_c = φ̂ + asinh-nötrlük)

Akı işaret konvansiyonu (denge-özdeşliğiyle türetildi, testler fizikle kilitler):
    Ĵn_kenar = (μ̂n/ĥ)[n̂_sağ·B(Δψ̂) − n̂_sol·B(−Δψ̂)]     (e^Δ·B(Δ)=B(−Δ) ⇒ dengede 0)
    Ĵp_kenar = (μ̂p/ĥ)[p̂_sol·B(Δψ̂) − p̂_sağ·B(−Δψ̂)]     (dengede 0; V>0'da J>0)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from tarhan import backend
from tarhan.numerics.flux import bernoulli


@dataclass
class PNDiode1D:
    """Vaka girdileri — sabitler ASLA hardcode edilmez (konvansiyon-tuzağı kuralı)."""
    Na: float = 1e16          # cm^-3 (p-taraf, solda)
    Nd: float = 1e16          # cm^-3 (n-taraf, sağda)
    ni: float = 1e10          # cm^-3
    ut: float = 0.0259        # V
    eps_s: float = 11.7 * 8.85e-14   # F/cm
    q: float = 1.6e-19        # C
    mu_n: float = 1350.0      # cm^2/Vs
    mu_p: float = 480.0       # cm^2/Vs
    len_p: float = 3e-4       # cm (p-taraf uzunluğu, 3 um)
    len_n: float = 3e-4       # cm
    h0: float = 5e-7          # cm (eklemde ilk adım, 5 nm)
    gamma: float = 1.06       # grid büyüme oranı
    # türetilenler
    C0: float = field(init=False)
    delta: float = field(init=False)
    L_D: float = field(init=False)

    def __post_init__(self):
        for name in ("Na", "Nd", "ni", "ut", "eps_s", "q", "mu_n", "mu_p",
                     "len_p", "len_n", "h0"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"{name}={v}: pozitif ve sonlu olmalı")
        self.C0 = max(self.Na, self.Nd)
        self.delta = self.ni / self.C0
        self.L_D = math.sqrt(self.eps_s * self.ut / (self.q * self.C0))

    # ------------------------------------------------------------------ grid
    def build_grid(self):
        """Eklem x=0'da; iki yana geometrik büyüyen adımlar. Döner: x̂ dizisi."""
        xp = backend.xp()

        def side(length):
            xs, h = [0.0], self.h0
            while xs[-1] < length:
                xs.append(min(xs[-1] + h, length))
                h *= self.gamma
            return xs

        left = [-v for v in reversed(side(self.len_p))][:-1]   # eklem tek kez
        x = left + side(self.len_n)
        return xp.asarray(x) / self.L_D

    def doping_hat(self, x_hat):
        xp = backend.xp()
        return xp.where(x_hat < 0.0, -self.Na / self.C0, self.Nd / self.C0)


def _neutral_psi(n_hat_doping, delta):
    """Yük-nötr denge potansiyeli: ψ̂ = asinh(N̂/(2δ)) (nokta-bazında tam)."""
    xp = backend.xp()
    return xp.arcsinh(n_hat_doping / (2.0 * delta))


def _poisson_newton(dev, x_hat, psi, phi_n, phi_p, tol=1e-10, max_iter=100):
    """Gummel-içi nonlineer Poisson: sabit quasi-Fermi'lerle damped Newton.

    F_i = L(ψ̂)_i − (n̂_i − p̂_i − N̂_i);  n̂=δe^{ψ̂−φ̂n}, p̂=δe^{φ̂p−ψ̂}
    M-matris + adım kelepçesi (|Δψ̂|≤5) ⇒ gürbüz yakınsama.
    """
    xp = backend.xp()
    n_dop = dev.doping_hat(x_hat)
    hm = x_hat[1:-1] - x_hat[:-2]
    hp = x_hat[2:] - x_hat[1:-1]
    hbar = 0.5 * (hm + hp)
    delta = dev.delta

    for it in range(max_iter):
        n_h = delta * xp.exp(xp.clip(psi - phi_n, -700, 700))
        p_h = delta * xp.exp(xp.clip(phi_p - psi, -700, 700))
        lap = ((psi[2:] - psi[1:-1]) / hp - (psi[1:-1] - psi[:-2]) / hm) / hbar
        F = lap - (n_h[1:-1] - p_h[1:-1] - n_dop[1:-1])
        # Jacobian (tridiag): d(lap)/dψ ± ; d(n+p)/dψ = n̂+p̂ (diag)
        sub = 1.0 / (hm * hbar)
        sup = 1.0 / (hp * hbar)
        diag = -(sub + sup) - (n_h[1:-1] + p_h[1:-1])
        dpsi = backend.solve_tridiag(sub, diag, sup, -F)
        step = float(xp.max(xp.abs(dpsi)))
        if step > 5.0:                      # adım kelepçesi
            dpsi = dpsi * (5.0 / step)
        psi[1:-1] = psi[1:-1] + dpsi
        if step < tol:
            return psi, it + 1
    raise RuntimeError(f"Poisson-Newton yakınsamadı (son adım {step:.2e})")


def _continuity_solve(dev, x_hat, psi, carrier, bc_left, bc_right):
    """R=0 süreklilik: SG kenar akıları, lineer tridiag (M-matris ⇒ pozitif).

    carrier='n': Ĵ ∝ n̂_r·B(Δψ̂) − n̂_l·B(−Δψ̂)
    carrier='p': Ĵ ∝ p̂_l·B(Δψ̂) − p̂_r·B(−Δψ̂)  (ayrıklaştırma aynası)
    """
    xp = backend.xp()
    h = x_hat[1:] - x_hat[:-1]
    dpsi = psi[1:] - psi[:-1]
    Bp = bernoulli(dpsi)      # B(+Δψ̂) kenarlarda
    Bm = bernoulli(-dpsi)
    if carrier == "n":
        e_lo, e_hi = Bm / h, Bp / h        # kenar katsayıları: (sol dugum, sag dugum)
    else:
        e_lo, e_hi = Bp / h, Bm / h
    n_in = len(x_hat) - 2
    sub = xp.empty(n_in); diag = xp.empty(n_in); sup = xp.empty(n_in)
    sub[:] = e_lo[:-1]
    sup[:] = e_hi[1:]
    diag[:] = -(e_hi[:-1] + e_lo[1:])
    rhs = xp.zeros(n_in)
    rhs[0] -= sub[0] * bc_left
    rhs[-1] -= sup[-1] * bc_right
    u = backend.solve_tridiag(sub, diag, sup, rhs)
    out = xp.empty(len(x_hat))
    out[0], out[-1], out[1:-1] = bc_left, bc_right, u
    if float(xp.min(out)) <= 0.0:
        raise RuntimeError("süreklilik çözümü pozitifliği kaybetti (grid/bias kontrol)")
    return out


def _contact_densities(n_dop_val, delta):
    s = math.sqrt(n_dop_val * n_dop_val + 4.0 * delta * delta)
    return 0.5 * (n_dop_val + s), 0.5 * (-n_dop_val + s)     # (n̂_c, p̂_c)


def solve_bias(dev: PNDiode1D, v_applied: float, state=None,
               gummel_tol: float = 1e-9, max_gummel: int = 60):
    """Tek bias noktası; state=None ise dengeden başlar. Döner: durum sözlüğü."""
    xp = backend.xp()
    x_hat = state["x_hat"] if state else dev.build_grid()
    n_dop = dev.doping_hat(x_hat)
    v_hat = v_applied / dev.ut
    delta = dev.delta

    # kontaklar: sol = p-taraf (bias V), sağ = n-taraf (0)
    nL, pL = _contact_densities(float(n_dop[0]), delta)
    nR, pR = _contact_densities(float(n_dop[-1]), delta)
    phi_n = xp.full(len(x_hat), 0.0)
    phi_p = xp.full(len(x_hat), 0.0)
    phi_n[:], phi_p[:] = v_hat, v_hat          # başlangıç alanı: lineerden iyi — bölgesel
    mid = int(xp.argmin(xp.abs(x_hat)))
    phi_n[mid:] = 0.0
    phi_p[mid:] = 0.0

    psi = xp.array(state["psi"], copy=True) if state else _neutral_psi(n_dop, delta)
    if state is not None:
        phi_n = xp.array(state["phi_n"], copy=True)
        phi_p = xp.array(state["phi_p"], copy=True)
        scale = v_hat - state["v_hat"]
        phi_n[x_hat < 0] += scale                      # warm start: p-tarafını kaydır
        phi_p[x_hat < 0] += scale
    psi[0] = v_hat + math.log(nL / delta)
    psi[-1] = math.log(nR / delta)

    for g in range(max_gummel):
        psi_old = xp.array(psi, copy=True)
        psi, _ = _poisson_newton(dev, x_hat, psi, phi_n, phi_p)
        n_h = _continuity_solve(dev, x_hat, psi, "n", nL, nR)
        p_h = _continuity_solve(dev, x_hat, psi, "p", pL, pR)
        phi_n = psi - xp.log(n_h / delta)
        phi_p = psi + xp.log(p_h / delta)
        if float(xp.max(xp.abs(psi - psi_old))) < gummel_tol:
            break
    else:
        raise RuntimeError(f"Gummel yakınsamadı @ V={v_applied}")

    # kenar akıları (ölçekli) + boyutlandırma
    h = x_hat[1:] - x_hat[:-1]
    dpsi = psi[1:] - psi[:-1]
    Bp, Bm = bernoulli(dpsi), bernoulli(-dpsi)
    mu_scale = max(dev.mu_n, dev.mu_p)
    jn = (dev.mu_n / mu_scale) * (n_h[1:] * Bp - n_h[:-1] * Bm) / h
    jp = (dev.mu_p / mu_scale) * (p_h[:-1] * Bp - p_h[1:] * Bm) / h
    j_hat = jn + jp
    j_scale = dev.q * dev.ut * mu_scale * dev.C0 / dev.L_D     # A/cm^2
    return {"x_hat": x_hat, "psi": psi, "phi_n": phi_n, "phi_p": phi_p,
            "n_hat": n_h, "p_hat": p_h, "j_edges": j_hat * j_scale,
            "j": float(xp.mean(j_hat)) * j_scale, "v_hat": v_hat,
            "gummel_iters": g + 1}


def iv_sweep(dev: PNDiode1D, voltages, **kw):
    """Bias-continuation'lı I-V taraması (warm start)."""
    state, out = None, []
    for v in voltages:
        state = solve_bias(dev, v, state=state, **kw)
        out.append(state["j"])
    return out, state


def band_diagram(dev: PNDiode1D, state, e_gap: float = 1.12):
    """Band diyagramı post-processing (eV; Ec referansı keyfî sabit)."""
    xp = backend.xp()
    ec = -dev.ut * state["psi"] + e_gap / 2.0
    return {"x_cm": state["x_hat"] * dev.L_D, "Ec": ec, "Ev": ec - e_gap,
            "EFn": -dev.ut * state["phi_n"], "EFp": -dev.ut * state["phi_p"]}

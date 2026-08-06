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
    tau_n: float | None = None   # s — SRH elektron ömrü (None ⇒ R=0, kısa-taban v1)
    tau_p: float | None = None   # s — SRH deşik ömrü (midgap tuzak: n1=p1=ni)
    # türetilenler
    C0: float = field(init=False)
    delta: float = field(init=False)
    L_D: float = field(init=False)

    def __post_init__(self):
        for name in ("Na", "Nd", "ni", "ut", "eps_s", "q", "mu_n", "mu_p",
                     "len_p", "len_n", "h0"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"{name}={v}: must be positive and finite")
        if (self.tau_n is None) != (self.tau_p is None):
            raise ValueError("tau_n ve tau_p birlikte verilmeli (ya ikisi ya hiçbiri)")
        for name in ("tau_n", "tau_p"):
            v = getattr(self, name)
            if v is not None and not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"{name}={v}: must be positive and finite")
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
        """Ölçekli net doping. FV-doğruluğu: eklem DÜĞÜMÜNÜN kontrol hacmi iki
        yana yayıldığından yarı-hücre ortalaması alır — (Nd−Na)/2. (Grid-yakınsama
        çalışması 2026-07-04: tam-Nd ataması baskın O(h) hata terimiydi; yarı-hücre
        düzeltmesi seviye-farklarını ~10× küçülttü.)"""
        xp = backend.xp()
        nd = xp.where(x_hat < 0.0, -self.Na / self.C0, self.Nd / self.C0)
        j = int(xp.argmin(xp.abs(x_hat)))
        nd[j] = 0.5 * (self.Nd - self.Na) / self.C0
        return nd


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


def _continuity_solve(dev, x_hat, psi, carrier, bc_left, bc_right,
                      mu_hat=1.0, src_lin=None, src_const=None):
    """Süreklilik: SG kenar akıları + (ops.) SRH lineerize kaynak; tridiag M-matris.

    carrier='n': dĴn/dx̂ = +R̂ ;  carrier='p': dĴp/dx̂ = −R̂  (toplam J korunur).
    Gummel-içi lineerizasyon: R̂ ≈ (v_prev·u − δ²)/den_prev ⇒ her iki taşıyıcıda
    köşegeni BÜYÜTEN kayıp terimi (src_lin = h̄·v_prev/den ≥ 0, src_const =
    h̄·δ²/den ≥ 0) → M-matris + pozitif RHS ⇒ pozitiflik korunur.
    """
    xp = backend.xp()
    h = x_hat[1:] - x_hat[:-1]
    dpsi = psi[1:] - psi[:-1]
    Bp = bernoulli(dpsi)
    Bm = bernoulli(-dpsi)
    if carrier == "n":
        w_lo, w_hi = mu_hat * Bm / h, mu_hat * Bp / h   # (sol-düğüm, sağ-düğüm) kenar ağırlıkları
    else:
        w_lo, w_hi = mu_hat * Bp / h, mu_hat * Bm / h
    n_in = len(x_hat) - 2
    sub = xp.empty(n_in); diag = xp.empty(n_in); sup = xp.empty(n_in)
    sub[:] = -w_lo[:-1]
    sup[:] = -w_hi[1:]
    diag[:] = w_hi[:-1] + w_lo[1:]
    rhs = xp.zeros(n_in)
    if src_lin is not None:
        diag[:] = diag + src_lin
        rhs[:] = rhs + src_const
    rhs[0] -= sub[0] * bc_left
    rhs[-1] -= sup[-1] * bc_right
    u = backend.solve_tridiag(sub, diag, sup, rhs)
    out = xp.empty(len(x_hat))
    out[0], out[-1], out[1:-1] = bc_left, bc_right, u
    if float(xp.min(out)) <= 0.0:
        raise RuntimeError("süreklilik çözümü pozitifliği kaybetti (grid/bias kontrol)")
    return out


def _contact_densities(n_dop_val, delta):
    """Ohmik kontak: yük-nötr (n̂−p̂=N̂) + np=δ² ⇒ (n̂_c, p̂_c).

    ÇOĞUNLUK toplamayla, AZINLIK bölmeyle. İkisini de ½(±N̂+s) diye yazmak
    cebirsel olarak doğru ama sayısal olarak yanlıştır: azınlık tarafında bu,
    birbirine çok yakın iki sayının farkıdır (yıkıcı sadeleşme) ve δ küçüldükçe
    çöker. Ölçüldü (|N̂| = 1):

        δ=1e-6 → np/δ² = 0.999978   (pn1d varsayılanı; zararsız)
        δ=1e-7 → 0.999201
        δ=1e-8 → 1.110223           (%11 hata — DEVSIM'in 1e18 doping vakası)
        δ=1e-9 → 0.000000           (azınlık yoğunluğu TAM sıfır)

    Son satır sessiz bir felakettir: süreklilik çözümü sıfır Dirichlet değeri
    alır, kütle etkisi yasası tamamen kaybolur ve hiçbir şey şikâyet etmez.
    Kararlı biçim her ölçekte np = δ²'yi TAM verir. (2D-2 prototipinde
    yakalandı: dengede np/δ² her düğümde 1.110223 çıkıyordu ve hata düğüme
    değil, kontak formülüne bağlıydı.)
    """
    s = math.sqrt(n_dop_val * n_dop_val + 4.0 * delta * delta)
    if n_dop_val >= 0.0:                       # n-tarafı: elektronlar çoğunluk
        n_hat = 0.5 * (n_dop_val + s)
        return n_hat, delta * delta / n_hat
    p_hat = 0.5 * (-n_dop_val + s)             # p-tarafı: deşikler çoğunluk
    return delta * delta / p_hat, p_hat


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

    mu_scale0 = max(dev.mu_n, dev.mu_p)
    mun_hat, mup_hat = dev.mu_n / mu_scale0, dev.mu_p / mu_scale0
    t0 = dev.L_D ** 2 / (dev.ut * mu_scale0)          # zaman ölçeği [s]
    hm_all = x_hat[1:-1] - x_hat[:-2]
    hp_all = x_hat[2:] - x_hat[1:-1]
    hbar = 0.5 * (hm_all + hp_all)
    n_h = delta * xp.exp(xp.clip(psi - phi_n, -700, 700))
    p_h = delta * xp.exp(xp.clip(phi_p - psi, -700, 700))

    for g in range(max_gummel):
        psi_old = xp.array(psi, copy=True)
        psi, _ = _poisson_newton(dev, x_hat, psi, phi_n, phi_p)
        if dev.tau_n is not None:
            tau_n_hat, tau_p_hat = dev.tau_n / t0, dev.tau_p / t0
            den = (tau_p_hat * (n_h[1:-1] + delta)
                   + tau_n_hat * (p_h[1:-1] + delta))
            lin_n = hbar * p_h[1:-1] / den
            lin_p = hbar * n_h[1:-1] / den
            const = hbar * delta * delta / den
            n_h = _continuity_solve(dev, x_hat, psi, "n", nL, nR,
                                    mun_hat, lin_n, const)
            p_h = _continuity_solve(dev, x_hat, psi, "p", pL, pR,
                                    mup_hat, lin_p, const)
        else:
            n_h = _continuity_solve(dev, x_hat, psi, "n", nL, nR, mun_hat)
            p_h = _continuity_solve(dev, x_hat, psi, "p", pL, pR, mup_hat)
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

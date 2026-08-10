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
from tarhan.numerics.transient import integrate_stiff


#: A grid this size is not a fine mesh, it is a hang. The bound is arbitrary in
#: value and not in kind: a 1D device runs at ~125 nodes and a deliberately
#: fine one at ~1200, so anything approaching this is a typo in h0 rather than
#: an intention. Reported in re-review, where h0=1e-12 with gamma=1 walked
#: toward hundreds of millions of nodes and never returned.
MAX_GRID_NODES = 2_000_000


def estimated_nodes(length: float, h0: float, gamma: float) -> float:
    """How many steps the geometric walk will take, WITHOUT taking them.

    Arithmetic on the geometric series, not physics: the walk covers
    h0*(gamma^n - 1)/(gamma - 1) after n steps, so reaching `length` needs
    n >= log(1 + (gamma-1)*length/h0) / log(gamma), and n = length/h0 when
    gamma is exactly 1. Computing it up front is the difference between an
    error message and a process nobody can interrupt.
    """
    if gamma == 1.0:
        steps = length / h0
    else:
        steps = math.log1p((gamma - 1.0) * length / h0) / math.log(gamma)
    # +1 for the node at the junction itself, and a ceiling because a partial
    # step still costs a node. Measured against build_grid across six (h0,
    # gamma) pairs: without these the estimate under-counted by one or two,
    # which is nothing at the limit but is still arithmetic that was wrong.
    if not math.isfinite(steps):
        # gamma=1e308 makes the intermediate infinite and math.ceil then raises
        # OverflowError, which the CLI reports as exit 5 — OUR bug — for what
        # is a user's number. Reported in re-review. Returning infinity lets
        # the caller's own limit refuse it as input.
        return math.inf
    return math.ceil(steps) + 1


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
        # gamma < 1 SHRINKS the step away from the junction, so build_grid's
        # `while xs[-1] < length` walk converges to a point short of the
        # boundary and never terminates. Measured: PNDiode1D(gamma=0.5)
        # constructed without complaint and build_grid did not return within a
        # 2 s alarm. The identical guard was written into the 2D mesh
        # generator and never backported here, which is exactly the kind of
        # gap a second pair of eyes finds.
        if not math.isfinite(self.gamma) or self.gamma < 1.0:
            raise ValueError(
                f"gamma={self.gamma}: must be finite and >= 1, or the grid "
                "never reaches the contact")
        estimate = sum(estimated_nodes(side, self.h0, self.gamma)
                       for side in (self.len_p, self.len_n))
        if not math.isfinite(estimate) or estimate > MAX_GRID_NODES:
            raise ValueError(
                f"h0={self.h0:g} with gamma={self.gamma:g} needs about "
                f"{estimate:.3g} nodes for this device, over the "
                f"{MAX_GRID_NODES:,} limit. The progress guard below only "
                "catches a walk that stops entirely; this catches one that "
                "merely never finishes.")
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
                nxt = min(xs[-1] + h, length)
                # Belt and braces. gamma >= 1 is checked at construction, but
                # floating point can still stall the walk if h underflows
                # relative to xs[-1]. A loop that stops making progress must
                # fail, not spin: a CI job that hangs looks like a slow one
                # until it times out.
                if not (nxt > xs[-1]):
                    raise ValueError(
                        f"grid step stopped making progress at x={xs[-1]:g} "
                        f"with h={h:g}; h0 is too small for this length")
                xs.append(nxt)
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


def _poisson_linear(dev, x_hat, n_hat, p_hat, psi_left, psi_right):
    """Poisson with the carrier densities held fixed — one tridiagonal solve.

    The steady-state path solves Poisson with a Newton loop because there the
    unknowns are the quasi-Fermi levels, so n̂ = δe^{ψ̂−φ̂n} depends on ψ̂ and the
    equation is nonlinear. In the transient formulation n̂ and p̂ are the state
    variables themselves, so the charge n̂ − p̂ − N̂ contains no ψ̂ at all and the
    equation is LINEAR. That is what makes the scheme cheap: what would
    otherwise be an index-1 DAE reduces to an ODE in (n̂, p̂), with ψ̂ recovered
    by one tridiagonal solve per right-hand-side evaluation.

    UNVERIFIED (physics_verify is unavailable this session — the physicist MCP
    server is disconnected): the reduction argument is reasoning about the
    structure of the equations, not a checked derivation. It is put on trial by
    the test asserting that :func:`transient_rhs` vanishes at the independently
    computed steady state — if the reduction were wrong, that fixed point would
    not be a fixed point.
    """
    xp = backend.xp()
    hm = x_hat[1:-1] - x_hat[:-2]
    hp = x_hat[2:] - x_hat[1:-1]
    hbar = 0.5 * (hm + hp)
    sub = 1.0 / (hm * hbar)
    sup = 1.0 / (hp * hbar)
    diag = -(sub + sup)
    rhs = xp.array(n_hat[1:-1] - p_hat[1:-1] - dev.doping_hat(x_hat)[1:-1],
                   dtype=float)
    rhs[0] = rhs[0] - sub[0] * psi_left
    rhs[-1] = rhs[-1] - sup[-1] * psi_right
    inner = backend.solve_tridiag(sub, diag, sup, rhs)
    psi = xp.empty(len(x_hat))
    psi[0], psi[-1], psi[1:-1] = psi_left, psi_right, inner
    return psi


def _sg_edge_weights(dpsi, h, mu_hat, carrier):
    """Scharfetter–Gummel edge weights, in the same convention the steady-state
    solver uses, so the two operators cannot silently disagree."""
    Bp = bernoulli(dpsi)
    Bm = bernoulli(-dpsi)
    if carrier == "n":
        return mu_hat * Bm / h, mu_hat * Bp / h
    return mu_hat * Bp / h, mu_hat * Bm / h


def time_scale_seconds(dev) -> float:
    """Seconds per unit of scaled time t̂.

    t0 = L_D²/(U_T·μ_scale), and substituting L_D² = εs·U_T/(q·C0) gives
    t0 = εs/(q·C0·μ_scale) — the dielectric relaxation time at the reference
    doping and mobility. μ_scale is max(μ_n, μ_p), the same reference the
    steady-state current scale already uses, so the two cannot drift apart.

    UNVERIFIED by physics_verify (server unavailable this session). The
    dimensions were checked by hand: (F/cm) / (C·cm⁻³·cm²/(V·s))
    = (F/cm)·(cm·V·s)/C = F·V·s/C = C·s/C = s.
    """
    mu_scale = max(dev.mu_n, dev.mu_p)
    return dev.eps_s / (dev.q * dev.C0 * mu_scale)


def transient_rhs(dev, x_hat, y, psi_left, psi_right, contacts):
    """d(n̂, p̂)/dt̂ at the interior nodes, with ψ̂ solved from the state.

    The sign convention is tied to the steady-state operator rather than
    re-derived: ``_continuity_solve`` builds an M-matrix A with A·u = 0 at
    steady state, and A·u is the net OUTFLOW at each node. So the accumulation
    is −(A·u)/h̄, and a state that solves the steady problem must give exactly
    zero here. That identity is the test, and it is the reason this function
    was written against the existing operator instead of from the paper.
    """
    xp = backend.xp()
    n_in = len(x_hat) - 2
    (nL, nR), (pL, pR) = contacts
    n_hat = xp.empty(len(x_hat))
    p_hat = xp.empty(len(x_hat))
    n_hat[0], n_hat[-1], n_hat[1:-1] = nL, nR, y[:n_in]
    p_hat[0], p_hat[-1], p_hat[1:-1] = pL, pR, y[n_in:]

    psi = _poisson_linear(dev, x_hat, n_hat, p_hat, psi_left, psi_right)

    h = x_hat[1:] - x_hat[:-1]
    hbar = 0.5 * ((x_hat[1:-1] - x_hat[:-2]) + (x_hat[2:] - x_hat[1:-1]))
    dpsi = psi[1:] - psi[:-1]
    mu_scale = max(dev.mu_n, dev.mu_p)

    out = xp.empty(2 * n_in)
    for slot, (carrier, u, mu) in enumerate(
            (("n", n_hat, dev.mu_n / mu_scale),
             ("p", p_hat, dev.mu_p / mu_scale))):
        w_lo, w_hi = _sg_edge_weights(dpsi, h, mu, carrier)
        flux = w_hi * u[1:] - w_lo * u[:-1]          # on edge e: node e -> e+1
        out[slot * n_in:(slot + 1) * n_in] = (flux[1:] - flux[:-1]) / hbar
    return out


def transient_setup(dev, v_applied: float, state=None):
    """Everything the time integrator needs, taken from the steady-state path.

    Deliberately built by calling ``solve_bias`` rather than by re-deriving the
    contact densities and the boundary potentials: those two have their own
    subtleties — the minority carrier is formed by division to avoid a
    cancellation that once cost 1.1 significant figures — and a second copy of
    that reasoning is a second place for it to go wrong.
    """
    xp = backend.xp()
    st = solve_bias(dev, v_applied, state=state)
    x_hat = st["x_hat"]
    n_dop = dev.doping_hat(x_hat)
    delta = dev.delta
    nL, pL = _contact_densities(float(n_dop[0]), delta)
    nR, pR = _contact_densities(float(n_dop[-1]), delta)
    return {
        "state": st,
        "x_hat": x_hat,
        "psi_left": v_applied / dev.ut + math.log(nL / delta),
        "psi_right": math.log(nR / delta),
        "contacts": ((nL, nR), (pL, pR)),
        "y_steady": xp.concatenate([st["n_hat"][1:-1], st["p_hat"][1:-1]]),
    }


def transient_solve(dev, v_applied: float, *, y0=None, t_span_hat=(0.0, 40.0),
                    t_eval=None, rtol: float = 1e-8, atol: float = 1e-14,
                    method: str = "BDF", state=None):
    """Integrate (n̂, p̂) in scaled time. Returns the raw solution plus the setup.

    ``t_span_hat`` is in units of :func:`time_scale_seconds`. The default state
    is the steady solution itself, which should not move — that is the cheapest
    possible regression on the whole coupling.
    """
    setup = transient_setup(dev, v_applied, state=state)
    y_start = setup["y_steady"] if y0 is None else y0

    def rhs(_t, y):
        return transient_rhs(dev, setup["x_hat"], y, setup["psi_left"],
                             setup["psi_right"], setup["contacts"])

    sol = integrate_stiff(rhs, y_start, t_span_hat, t_eval=t_eval,
                          rtol=rtol, atol=atol, method=method)
    setup["solution"] = sol
    setup["t_seconds"] = sol.t * time_scale_seconds(dev)
    return setup


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


#: See pn2d.CURRENT_TOL — one contract, one number.
CURRENT_TOL = 1e-6


def _edge_current(dev: PNDiode1D, x_hat, psi, n_h, p_h,
                  with_scale: bool = False):
    """Scaled edge current density. Extracted so the convergence check and the
    reported result cannot drift apart: measuring convergence on a quantity
    the caller never sees would be measuring the wrong thing."""
    xp = backend.xp()
    h = x_hat[1:] - x_hat[:-1]
    dpsi = psi[1:] - psi[:-1]
    Bp, Bm = bernoulli(dpsi), bernoulli(-dpsi)
    mu_scale = max(dev.mu_n, dev.mu_p)
    jn = (dev.mu_n / mu_scale) * (n_h[1:] * Bp - n_h[:-1] * Bm) / h
    jp = (dev.mu_p / mu_scale) * (p_h[:-1] * Bp - p_h[1:] * Bm) / h
    if with_scale:
        # The drift and diffusion halves, before they cancel. Their size is
        # the scale a change in the net current has to be judged against.
        drift_n = (dev.mu_n / mu_scale) * (n_h[1:] * Bp + n_h[:-1] * Bm) / h
        drift_p = (dev.mu_p / mu_scale) * (p_h[:-1] * Bp + p_h[1:] * Bm) / h
        return jn + jp, xp.abs(drift_n) + xp.abs(drift_p)
    return jn + jp


def solve_bias(dev: PNDiode1D, v_applied: float, state=None,
               gummel_tol: float = 1e-9, max_gummel: int = 60,
               current_tol: float = CURRENT_TOL, on_iteration=None):
    """Tek bias noktası; state=None ise dengeden başlar. Döner: durum sözlüğü.

    ``on_iteration(index, total)`` her Gummel dış adımının başında çağrılır.
    Bir ilerleme göstergesi bunsuz yalancı olur: çağıran, bloklayan tek bir
    çağrı boyunca hiçbir şey çizemez, gösterge de ancak iş BİTTİKTEN sonra
    belirir. Ölçüldü — bloklayan çağrı süresince yazılan bayt sayısı sıfırdı.
    """
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

    previous_current = None
    current_change = float("inf")
    psi_step = float("inf")
    for g in range(max_gummel):
        if on_iteration is not None:
            on_iteration(g, max_gummel)
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
        psi_step = float(xp.max(xp.abs(psi - psi_old)))

        # The same contract as pn2d: the step is not the answer. Measured on
        # the mean edge current, which is what `j` reports.
        total = float(xp.mean(_edge_current(dev, x_hat, psi, n_h, p_h)))
        # See pn2d for why the cancelling case is named rather than normalised.
        # See pn2d: raw, and meaningless at equilibrium by construction.
        if previous_current is not None and total != 0.0:
            current_change = abs(total - previous_current) / abs(total)
        previous_current = total

        if psi_step < gummel_tol:
            break
    else:
        # See pn2d: only an unsettled POTENTIAL is a failure.
        if psi_step >= gummel_tol:
            raise RuntimeError(
                f"Gummel yakinsamadi @ V={v_applied}: "
                f"max|dpsi|={psi_step:.3e} (tol {gummel_tol:.0e})")

    j_hat = _edge_current(dev, x_hat, psi, n_h, p_h)
    mu_scale = max(dev.mu_n, dev.mu_p)
    j_scale = dev.q * dev.ut * mu_scale * dev.C0 / dev.L_D     # A/cm^2
    return {"x_hat": x_hat, "psi": psi, "phi_n": phi_n, "phi_p": phi_p,
            "n_hat": n_h, "p_hat": p_h, "j_edges": j_hat * j_scale,
            "j": float(xp.mean(j_hat)) * j_scale, "v_hat": v_hat,
            "gummel_iters": g + 1,
            "psi_step": psi_step, "current_rel_change": current_change,
}


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

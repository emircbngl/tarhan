"""Voltametri çözücüleri — Rank 8: reversible linear-sweep/CV (boyutsuz).

Yöntem kaynağı (aktarım; kod bizim): Compton, Laborda & Ward, "Understanding
Voltammetry: Simulation of Electrode Processes" — fully-implicit FD + genişleyen
uzaysal grid + Thomas çözümü; çapraz kaynaklar: Britz & Strutwolf CV bölümü,
Bard & Faulkner Böl. 6 (Nicholson-Shain 1964).

Model (eşit difüzyon katsayılı O + e⁻ ⇌ R, yarı-sonsuz):
    ∂a/∂s = ∂²a/∂X²,  a(X,0)=1,  a(∞)=1
    θ(s) = θ_başlangıç − s  (indirgeme taraması),  Nernst: a(0) = 1/(1+e^{−θ})
    (eşit D ⇒ a+b=1 her yerde ⇒ tek denklem yeter)
Boyutsuz akım J(s) = ∂a/∂X|₀ ; klasik sonuç: J_p = 0.4463, θ_p = −1.109,
|θ_{p/2} − θ_p| = 2.20 (E_p−E_{p/2} = 56.5/n mV, 25 °C).

Sayısal şema: backward-Euler zaman + 3-nokta düzensiz-grid Laplasyeni,
X_j+1 = X_j + h₀·γ^j genişleyen grid; her adım backend.solve_tridiag (dikiş).
"""
from __future__ import annotations

import math

from tarhan import backend


def _expanding_grid(h0: float, gamma: float, x_max: float):
    xp = backend.xp()
    xs = [0.0, h0]
    h = h0
    while xs[-1] < x_max:
        h *= gamma
        xs.append(xs[-1] + h)
    return xp.asarray(xs)


def reversible_lsv(theta_start: float = 20.0, theta_end: float = -20.0,
                   d_theta: float = 1e-3, h0: float = 1e-3,
                   gamma: float = 1.05, x_max_factor: float = 6.0):
    """Reversible LSV; döner: (thetas, J) — boyutsuz potansiyel ve akım dizileri."""
    xp = backend.xp()
    s_total = theta_start - theta_end
    X = _expanding_grid(h0, gamma, x_max_factor * math.sqrt(s_total))
    n = len(X)

    hm = X[1:-1] - X[:-2]    # h_{i-1},  i = 1..n-2 (iç düğümler)
    hp = X[2:] - X[1:-1]     # h_i
    sub = -d_theta * 2.0 / (hm * (hm + hp))
    sup = -d_theta * 2.0 / (hp * (hm + hp))
    diag = 1.0 - (sub + sup)          # BE: (I − Δs·L)

    # 3-nokta tek-yönlü türev katsayıları (X0=0, X1, X2 düzensiz)
    A_, B_ = float(X[1] - X[0]), float(X[2] - X[1])
    d0 = -(2 * A_ + B_) / (A_ * (A_ + B_))
    d1 = (A_ + B_) / (A_ * B_)
    d2 = -A_ / (B_ * (A_ + B_))

    n_steps = int(round(s_total / d_theta))
    a = xp.ones(n)
    thetas = xp.empty(n_steps)
    J = xp.empty(n_steps)
    for m in range(1, n_steps + 1):
        theta = theta_start - m * d_theta
        a0 = 1.0 / (1.0 + math.exp(-theta))
        rhs = xp.array(a[1:-1], dtype=float, copy=True)
        rhs[0] -= sub[0] * a0
        rhs[-1] -= sup[-1] * 1.0            # uzak sınır: a = 1
        a[1:-1] = backend.solve_tridiag(sub, diag, sup, rhs)
        a[0] = a0
        thetas[m - 1] = theta
        J[m - 1] = d0 * a0 + d1 * float(a[1]) + d2 * float(a[2])
    return thetas, J


def cv_sweep(theta_start: float = 6.5, theta_switch: float = -5.4879,
             d_theta: float = 2e-3, K0: float | None = None, alpha: float = 0.5,
             h0: float = 1e-3, gamma: float = 1.05, x_max_factor: float = 6.0):
    """Tam CV (ileri + geri tarama); K0=None ⇒ Nernst (reversible), değilse
    Butler-Volmer Robin sınırı (quasi-reversible — Nicholson 1965 problemi).

    BV sınırı (boyutsuz, eşit D ⇒ a+b=1): ∂a/∂X|₀ = K0·[a₀e^{−αθ} − (1−a₀)e^{(1−α)θ}],
    K0 = k⁰/sqrt(D·Fv/RT) = ψ·sqrt(π) (Nicholson Eq. 17, eşit D). Sınır satırı
    2-nokta akıyla tridiagonal kalır; rapor edilen J, çözülen a₀ ile BV ifadesinin
    kendisidir (sınırda tam). Varsayılan koşullar Nicholson'ınki: başlangıç
    C_O/C_R = e^{6.5}; (E_λ − E_{1/2})·n = 141 mV ⇒ θ_switch = −5.4879 (25°C).
    """
    xp = backend.xp()
    span = 2.0 * (theta_start - theta_switch)
    X = _expanding_grid(h0, gamma, x_max_factor * math.sqrt(span))
    n = len(X)

    hm = X[1:-1] - X[:-2]
    hp = X[2:] - X[1:-1]
    sub_i = -d_theta * 2.0 / (hm * (hm + hp))          # iç düğüm katsayıları (BE)
    sup_i = -d_theta * 2.0 / (hp * (hm + hp))
    diag_i = 1.0 - (sub_i + sup_i)
    h0_ = float(X[1] - X[0])
    A_, B_ = h0_, float(X[2] - X[1])
    d0 = -(2 * A_ + B_) / (A_ * (A_ + B_))             # 3-nokta türev (Nernst akımı)
    d1 = (A_ + B_) / (A_ * B_)
    d2 = -A_ / (B_ * (A_ + B_))

    n_f = int(round((theta_start - theta_switch) / d_theta))
    thetas = [theta_start - m * d_theta for m in range(1, n_f + 1)] + \
             [theta_switch + m * d_theta for m in range(1, n_f + 1)]

    a = xp.ones(n)
    out_theta = xp.empty(len(thetas))
    out_J = xp.empty(len(thetas))
    n_unk = n - 1                                       # a[0..n-2]; a[n-1]=1 uzak sınır
    sub = xp.zeros(n_unk); diag = xp.zeros(n_unk); sup = xp.zeros(n_unk)

    for m, theta in enumerate(thetas):
        rhs = xp.empty(n_unk)
        # iç düğümler (unknown indeksleri 1..n-2)
        sub[1:] = sub_i
        diag[1:] = diag_i
        sup[1:] = sup_i
        rhs[1:] = a[1:-1]
        rhs[-1] -= sup_i[-1] * 1.0                      # a[n-1] = 1
        if K0 is None:                                  # Nernst Dirichlet satırı
            diag[0], sup[0] = 1.0, 0.0
            rhs[0] = 1.0 / (1.0 + math.exp(-theta))
        else:                                           # BV Robin satırı (2-nokta akı)
            ef = math.exp(-alpha * theta)
            eb = math.exp((1.0 - alpha) * theta)
            diag[0] = -1.0 / h0_ - K0 * (ef + eb)
            sup[0] = 1.0 / h0_
            rhs[0] = -K0 * eb
        a_new = backend.solve_tridiag(sub, diag, sup, rhs)
        a = xp.concatenate([a_new, a[-1:]])
        out_theta[m] = theta
        if K0 is None:
            out_J[m] = d0 * float(a[0]) + d1 * float(a[1]) + d2 * float(a[2])
        else:
            out_J[m] = K0 * (float(a[0]) * ef - (1.0 - float(a[0])) * eb)
    return out_theta, out_J, n_f


def nicholson_peak_separation(psi: float, alpha: float = 0.5,
                              d_theta: float = 2e-3, **kw) -> float:
    """Nicholson ΔE_p·n [mV, 25°C]: ileri (katodik) ve geri (anodik) tepe ayrımı."""
    K0 = psi * math.sqrt(math.pi)
    th, J, n_f = cv_sweep(K0=K0, alpha=alpha, d_theta=d_theta, **kw)
    th_c, _ = find_peak(th[:n_f], J[:n_f])              # katodik: ileri kolda max
    th_a, _ = find_peak(th[n_f:], -J[n_f:])             # anodik: geri kolda min(J)
    return (th_a - th_c) * 25.693                       # RT/F @ 25°C [mV]


def find_peak(thetas, J):
    """Parabolik interpolasyonla tepe: (theta_p, J_p). Grid-arası tepeyi yakalar."""
    xp = backend.xp()
    m0 = int(xp.argmax(J))
    if m0 == 0 or m0 == len(J) - 1:
        return float(thetas[m0]), float(J[m0])
    jm, j0, jp = float(J[m0 - 1]), float(J[m0]), float(J[m0 + 1])
    denom = jm - 2.0 * j0 + jp
    delta = 0.5 * (jm - jp) / denom if denom != 0.0 else 0.0
    d_th = float(thetas[m0 + 1] - thetas[m0])     # negatif (indirgeme taraması)
    theta_p = float(thetas[m0]) + delta * d_th
    j_p = j0 - 0.25 * (jm - jp) * delta
    return theta_p, j_p


def half_peak_theta(thetas, J, j_p: float):
    """Tepeden ÖNCE (yükselen kolda) J = J_p/2 kesişimi — lineer interpolasyon."""
    xp = backend.xp()
    m0 = int(xp.argmax(J))
    target = 0.5 * j_p
    for m in range(m0, 0, -1):
        if J[m - 1] <= target <= J[m]:
            w = (target - float(J[m - 1])) / (float(J[m]) - float(J[m - 1]))
            return float(thetas[m - 1]) + w * (float(thetas[m]) - float(thetas[m - 1]))
    raise RuntimeError("yarı-tepe kesişimi bulunamadı (tarama aralığını kontrol et)")

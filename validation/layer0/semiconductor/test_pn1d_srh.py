"""Amiral gemisi uzantısı — SRH rekombinasyonu: uzun-taban fiziği + n≈2 rejimi.

Kaynaklar (aktarım): Shockley & Read 1952 / Hall 1952 (SRH hızı, midgap tuzak
n1=p1=ni); Sah, Noyce & Shockley 1957 (depletion-rekombinasyon akımı, n≈2 rejimi);
Selberherr Böl. 5 (Gummel-içi lineerizasyon).

Doğrulama felsefesi: mutlak akım yerine REJİM ve TAŞINIM-UZUNLUĞU çapaları —
(a) τ→∞ limitinde R=0 amiral gemisiyle birebir regresyon; (b) ±R̂ eşleşmesinin
ayrık korunumu; (c) azınlık-taşıyıcı profil bozunumundan L_p ölçümü vs √(Dp·τp)
(çözücüden fizik ÇIKARMA — ölçüm: oran 0.9905); (d) Sah-Noyce-Shockley iki-rejim
ideality'si: düşük V'de ~1.8 (saf 2.0 idealize simetrik-maksimum durumu; gerçek
analizlerde 1.7-1.9 tipiktir), difüzyon devraldıkça 1'e iner (ölçüm: 1.076 @0.55V).
"""
import math

import numpy as np
import pytest

from tarhan.models.pn1d import PNDiode1D, iv_sweep, solve_bias

DEV_SRH = PNDiode1D(len_p=25e-4, len_n=25e-4, tau_n=1e-8, tau_p=1e-8)


@pytest.fixture(scope="module")
def state_03():
    return solve_bias(DEV_SRH, 0.30)


def test_tau_infinite_regression_matches_r0():
    j_r0 = solve_bias(PNDiode1D(), 0.30)["j"]
    j_inf = solve_bias(PNDiode1D(tau_n=1.0, tau_p=1.0), 0.30)["j"]
    assert abs(j_inf / j_r0 - 1.0) < 1e-4          # ölçüm: 1e-6 içi


def test_conservation_with_recombination_active(state_03):
    spread = float(np.max(np.abs(np.asarray(state_03["j_edges"]) - state_03["j"]))
                   ) / abs(state_03["j"])
    assert spread < 1e-5                            # ölçüm: 1.25e-7 (±R̂ ayrık-tam)


def test_minority_diffusion_length_emerges(state_03):
    """Δp(x) ∝ exp(−x/L_p) bozunumu: L_p profilden ölçülür, √(Dp·τp) ile karşılaştırılır."""
    x_cm = np.asarray(state_03["x_hat"]) * DEV_SRH.L_D
    p = np.asarray(state_03["p_hat"]) * DEV_SRH.C0
    mask = (x_cm > 2e-4) & (x_cm < 15e-4)           # eklem ve kontaktan uzak
    dp = p[mask] - p[-2]
    a_mat = np.vstack([x_cm[mask], np.ones(int(mask.sum()))]).T
    slope = np.linalg.lstsq(a_mat, np.log(dp), rcond=None)[0][0]
    l_sim = -1.0 / slope
    l_ana = math.sqrt(DEV_SRH.mu_p * DEV_SRH.ut * DEV_SRH.tau_p)
    assert abs(l_sim / l_ana - 1.0) < 0.03          # ölçüm: 0.0095


def test_sah_noyce_shockley_two_regimes():
    volts = [0.05 * k for k in range(2, 12)]        # 0.10..0.55
    js, _ = iv_sweep(DEV_SRH, volts)
    nids = [(volts[i] - volts[i - 1]) / (DEV_SRH.ut * math.log(js[i] / js[i - 1]))
            for i in range(1, len(volts))]
    low = nids[:3]                                   # 0.10→0.25 pencereleri
    assert all(1.6 < n < 2.05 for n in low), f"düşük-V rejimi: {low}"
    assert nids[-1] < 1.2, f"yüksek-V rejimi: {nids[-1]:.3f}"   # ölçüm: 1.076
    assert max(nids) == max(low)                     # tepe rekombinasyon penceresinde
    assert all(nids[i] >= nids[i + 1] for i in range(1, len(nids) - 1))  # geçiş monoton


def test_srh_rate_nonnegative_in_forward(state_03):
    excess = (np.asarray(state_03["n_hat"]) * np.asarray(state_03["p_hat"])
              - DEV_SRH.delta ** 2)
    assert float(excess.min()) > -1e-3 * DEV_SRH.delta ** 2   # yuvarlama payıyla ≥ 0


def test_lifetime_input_guards():
    with pytest.raises(ValueError):
        PNDiode1D(tau_n=1e-8)                        # tek başına verilemez
    with pytest.raises(ValueError):
        PNDiode1D(tau_n=-1e-8, tau_p=1e-8)

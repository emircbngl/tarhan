"""TARHAN Layer-0 formül kütüphanesi.

Her fonksiyon açık bir DÜRÜSTLÜK KATMANI taşır (kuruluş ilkesi #4):
  - "first-principles (oracle-verified)": physics_verify oracle'ından geçti
  - "textbook (reproduced)": basılı kitap değerleri birebir yeniden üretildi
  - "empirical fit": ampirik korelasyon — boyut analizi anlamsız, doğrulama
    yalnız basılı-rakam karşılaştırması

Konvansiyon-tuzaĝı kuralı: fiziksel sabitler (ni, eps_r, kT/q, T) ASLA hardcode
edilmez — her çağrıda açık argümandır (Hu: eps_r=12, kT/q=0.026 V; Pierret:
eps_r=11.7, ni=1e10, kT/q=0.0259 V; Sze: ni=9.65e9 — kitaplar uyuşmaz).
"""
from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# Elektrokimya
# --------------------------------------------------------------------------- #

def cottrell_current(n: float, F: float, A: float, c0: float, D: float, t: float) -> float:
    """Cottrell akımı i(t) = n·F·A·c0·sqrt(D/(pi·t)).

    Katman: first-principles (oracle-verified) — physics_verify 3/3
    (DIMENSIONAL + NUMERIC + LIMIT), 2026-07-03; KB kartı
    electrochemistry/cottrell-equation (VERIFIED).
    Kaynak: Cottrell 1903; Bard & Faulkner, Electrochemical Methods.
    Geçerlilik: yarı-sonsuz lineer difüzyon, anlık tam tüketim, konveksiyon/
    migrasyon yok. Birimler SI (A [m^2], c0 [mol/m^3], D [m^2/s]) → i [A].
    """
    return n * F * A * c0 * math.sqrt(D / (math.pi * t))


def cottrell_dimensionless(T: float) -> float:
    """Boyutsuz Cottrell akımı G(T) = 1/sqrt(pi·T).

    Katman: first-principles (kapalı form; dC/dT = d2C/dX2'nin tam çözümü).
    Layer-0 test: validation/layer0/electrochem/test_rank00_cottrell_fd.py.
    """
    return 1.0 / math.sqrt(math.pi * T)


#: Reversible LSV/CV boyutsuz tepe akımı (tablolanmış; Nicholson-Shain 1964).
#: Epistemik temel: (a) üç kitap çapraz (Compton / Britz / Bard&Faulkner),
#: (b) TARHAN çözücüsünün BAĞIMSIZ reprodüksiyonu J_p = 0.44636 (rank 8,
#: üç çözünürlükte kararlı). Boyutlu form: randles_sevcik_peak_current.
REVERSIBLE_LSV_PEAK = 0.4463


def randles_sevcik_peak_current(n: float, F: float, A: float, c0: float,
                                D: float, v: float, R: float, T: float) -> float:
    """Randles–Ševčík tepe akımı i_p = 0.4463·n·F·A·c0·sqrt(n·F·v·D/(R·T)) [A].

    Katman: first-principles (oracle-verified) — physics_verify 2/2
    (DIMENSIONAL + NUMERIC: i_p(1 mM-eşdeğeri, 0.1 V/s) = 2.6865e-4 A), 2026-07-03;
    KB kartı electrochemistry/randles-sevcik. 0.4463 sabiti tablolanmış +
    TARHAN çözücüsüyle bağımsız yeniden üretilmiş (test_rank08_cv_peak.py).
    Geçerlilik: reversible (Nernstian) tek-elektron çifti, planar yarı-sonsuz
    difüzyon, 25°C-tipi rejimde eşit D. Birimler SI; v [V/s].
    """
    return 0.4463 * n * F * A * c0 * math.sqrt(n * F * v * D / (R * T))


# --------------------------------------------------------------------------- #
# PEM yakıt pili — Nafion membran (Springer 1991)
# --------------------------------------------------------------------------- #

def springer_lambda_sorption(a: float) -> float:
    """Nafion su alımı λ(a) = 0.043 + 17.81a − 39.85a² + 36.0a³ (0 ≤ a ≤ 1, 30 °C).

    Katman: empirical fit — Springer, Zawodzinski & Gottesfeld, JES 138(8),
    2334 (1991). Layer-0: 5/6 basılı rakam birebir (λ(0.3) plan-aktarımı kaynak
    teyidi bekliyor). Test: test_rank01_springer_membrane.py.
    """
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"su aktivitesi a={a}: korelasyon 0<=a<=1 için fit edildi")
    return 0.043 + 17.81 * a - 39.85 * a * a + 36.0 * a ** 3


def springer_conductivity(lam: float, T_kelvin: float) -> float:
    """Nafion proton iletkenliği κ(λ,T) [S/cm].

    κ = (0.005139·λ − 0.00326) · exp[1268·(1/303.15 − 1/T)]
    Katman: empirical fit — Springer 1991. Basılı çapalar: κ(14, 30°C)=0.0687,
    κ(14, 80°C)=0.124, κ(22, 30°C)=0.110 S/cm (Layer-0'da birebir).
    """
    return (0.005139 * lam - 0.00326) * math.exp(1268.0 * (1.0 / 303.15 - 1.0 / T_kelvin))


# --------------------------------------------------------------------------- #
# Yarı iletken — depletion approximation (step junction)
# --------------------------------------------------------------------------- #

def builtin_potential(Na: float, Nd: float, ni: float, kT_q: float) -> float:
    """Step-junction built-in potansiyeli φ_bi = (kT/q)·ln(Na·Nd/ni²) [V].

    Katman: textbook (reproduced) — Hu, Modern Semiconductor Devices, Örnek 4-1
    basılı değerleri birebir (φ_bi≈1 V @ Na=1e20, Nd=1e17, ni=1e10, kT/q=0.026).
    Oracle kartı: sıradaki physics_verify oturumunda. ni/kT_q AÇIK argüman
    (kitap-konvansiyonu tuzağı). Test: test_rank02_hu_junction.py.
    """
    return kT_q * math.log(Na * Nd / ni ** 2)


def depletion_width_one_sided(eps_s: float, phi_bi: float, q: float, N_light: float) -> float:
    """Tek-yanlı depletion genişliği W ≈ sqrt(2·εs·φ_bi/(q·N_hafif)).

    Katman: textbook (reproduced) — Hu Örnek 4-1 (W=0.12 μm birebir).
    Geçerlilik: N_ağır >> N_hafif (tek-yanlı adım eklem), sıfır dış gerilim
    (V uygulanacaksa φ_bi yerine φ_bi−V geçilir). Birim: cgs-benzeri kitap
    konvansiyonunda [cm] (εs [F/cm], N [cm^-3]).
    """
    return math.sqrt(2.0 * eps_s * phi_bi / (q * N_light))

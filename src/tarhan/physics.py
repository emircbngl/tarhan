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
# Yakıt pili kayıp-merdiveni (O'Hayre, Fuel Cell Fundamentals 3e — rank 6)
# --------------------------------------------------------------------------- #

def standard_cell_potential(e_red_cathode: float, e_red_anode: float) -> float:
    """Standart hücre potansiyeli E⁰_hücre = E⁰_katot − E⁰_anot (indirgeme potansiyelleri).

    Katman: textbook (reproduced) — O'Hayre 3e Örnek 2.3 (DMFC): katot +1.229 V,
    anot (CO2/CH3OH, indirgeme yönü) +0.03 V → E⁰ = +1.199 V (basılı; kitap anot
    yarı-tepkimesini oksidasyon yönünde −0.03 olarak yazar). Kural: E⁰ değerleri
    stokiyometrik çarpanla ÖLÇEKLENMEZ (yoğunluk-değil-miktar bağımsızlığı).
    """
    return e_red_cathode - e_red_anode


def activation_overpotential_tafel(j: float, j0: float, alpha: float, n: float,
                                   F: float, R: float, T: float) -> float:
    """Tafel aktivasyon aşırı-gerilimi η_act = (R·T/(α·n·F))·ln(j/j0) [V].

    Katman: first-principles (oracle-verified) — physics_verify 3/3 (DIMENSIONAL
    + dekad-eğimi 0.0591563 V + 6-dekad 0.354938 V), 2026-07-03; KB kartı
    electrochemistry/tafel-overpotential. O'Hayre 3e Örnek 3.3 çapası: α=0.5,
    n=2, 298.15 K → ~0.059 V/dekad (basılı "yaklaşık 60 mV"); j0=1e-6→1 A/cm²
    için basılı yuvarlak 6×60 mV = 0.36 V (hassas 0.3549).
    Geçerlilik: Tafel rejimi (j >> j0'ın ters-terim katkısı ihmal).
    """
    return (R * T / (alpha * n * F)) * math.log(j / j0)


def ionic_resistance(thickness: float, conductivity: float, area: float) -> float:
    """Elektrolit direnci R_ionic = L/(σ·A) [Ω].

    Katman: first-principles (oracle-verified) — physics_verify (ohmic-loss spec,
    4/4: R=0.01 Ω @ L=100 μm, σ=0.10 S/cm, A=10 cm²), 2026-07-03. O'Hayre 3e
    Örnek 4.1: (a) 100 μm → 0.01 Ω, (b) 50 μm → 0.005 Ω (basılı). Birimler tutarlı
    olmalı (kitap cgs: cm, S/cm, cm²).
    """
    return thickness / (conductivity * area)


def ohmic_overpotential(current: float, r_total: float) -> float:
    """Ohmik kayıp η_ohmic = i·R_toplam [V].

    Katman: first-principles (oracle-verified, aynı ohmic-loss spec'i) —
    O'Hayre 3e Örnek 4.1 basılı: 10 A × (0.005+0.01) Ω = 0.15 V (a);
    10 A × (0.005+0.005) Ω = 0.10 V (b).
    """
    return current * r_total


def effective_diffusivity_bruggeman(porosity: float, d_bulk: float) -> float:
    """Bruggeman etkin difüzyon D_eff = ε^1.5 · D.

    Katman: empirical fit (Bruggeman korelasyonu) — O'Hayre 3e Örnek 5.1 basılı
    zincir: ε=0.4, D=0.2 cm²/s → 0.0506 cm²/s. NOT: örneğin problem METNİ
    D=0.1 cm²/s der, basılı ÇÖZÜM 0.2 ile ilerler — kitap-içi tutarsızlık
    (muhtemel errata); test_rank06'da strict-xfail ile belgeli.
    """
    return porosity ** 1.5 * d_bulk


def limiting_current_density(n: float, F: float, d_eff: float, c_bulk: float,
                             delta: float) -> float:
    """Limit akım yoğunluğu j_L = n·F·D_eff·c⁰/δ.

    Katman: first-principles (oracle-verified) — physics_verify 2/2 (DIMENSIONAL
    + NUMERIC 22653 A/m² = 2.2653 A/cm²), 2026-07-03; KB kartı
    electrochemistry/limiting-current-density. O'Hayre 3e Örnek 5.1 basılı: 2.26
    A/cm² (n=4, D_eff=0.0506 cm²/s, c=5.8e-6 mol/cm³, δ=0.05 cm). Birimler tutarlı.
    """
    return n * F * d_eff * c_bulk / delta


def concentration_overpotential(c_coeff: float, j: float, j_l: float) -> float:
    """Konsantrasyon aşırı-gerilimi η_conc = c·ln(j_L/(j_L − j)) [V].

    Katman: first-principles (oracle-verified) — physics_verify 3/3 (DIMENSIONAL
    [çıkarmasız eşdeğer formla; harness aynı-boyut çıkarmasında 1−1=0 artefaktı
    veriyor] + NUMERIC 0.216244 V + LIMIT j→0 ⇒ 0), 2026-07-03; KB kartı
    electrochemistry/concentration-overpotential. O'Hayre 3e Örnek 5.1 basılı:
    c=0.1 V, j=2, j_L=2.26 → 0.22 V. Geçerlilik: j < j_L (j→j_L'de ıraksar).
    """
    if not j < j_l:
        raise ValueError(f"j={j} >= j_L={j_l}: konsantrasyon limiti aşıldı (model ıraksar)")
    return c_coeff * math.log(j_l / (j_l - j))


#: Levich sabiti (0.51023/3)^(1/3)/Γ(4/3) — Cochran a=0.51023 ile. Epistemik temel:
#: iki bağımsız TARHAN sayısal yolu (Γ'sız kuadratür + FD/BVP) 5e-7 içinde mutabık +
#: oracle NUMERIC; literatür "0.620" (3 hane) ✓, 0.62048 varyantı Cochran-hassasiyeti.
#: Rank-11 dersi: elle ara-hesap 0.620469 vermişti — makine-değerlendirme + çift-yol şart.
LEVICH_CONSTANT = 0.620450


def levich_limiting_current(n: float, F: float, D: float, nu: float,
                            omega: float, c_bulk: float) -> float:
    """Levich RDE limit akım yoğunluğu j_L = 0.62045·n·F·D^(2/3)·ν^(−1/6)·ω^(1/2)·c.

    Katman: first-principles (oracle-verified) — physics_verify 2/2 (DIMENSIONAL
    + NUMERIC 6.12606 A/m² @ 1000 rpm, sulu-tipik parametreler), 2026-07-03;
    KB kartı electrochemistry/levich-equation. Sabitin türetimi ve çift-yol
    reprodüksiyonu: numerics/convection.py + test_rank11_levich.py. j_L ∝ √ω
    (Koutecký-Levich analizi bunun üstüne kurulur). Birimler SI; ω [rad/s].
    Geçerlilik: laminer RDE akışı, Cochran yakın-yüzey profili, Sc >> 1.
    """
    return LEVICH_CONSTANT * n * F * D ** (2.0 / 3.0) * nu ** (-1.0 / 6.0) \
        * math.sqrt(omega) * c_bulk


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

    Katman: first-principles (oracle-verified) — physics_verify 3/3 (DIMENSIONAL
    + Hu ve Pierret çift NUMERIC çapası: 1.01774 V / 0.715643 V), 2026-07-03;
    KB kartı semiconductors/builtin-potential. ni/kT_q AÇIK argüman
    (kitap-konvansiyonu tuzağı). Testler: test_rank02 + test_rank09.
    """
    return kT_q * math.log(Na * Nd / ni ** 2)


def depletion_width_one_sided(eps_s: float, phi_bi: float, q: float, N_light: float) -> float:
    """Tek-yanlı depletion genişliği W ≈ sqrt(2·εs·φ_bi/(q·N_hafif)).

    Katman: first-principles (oracle-verified) — physics_verify 2/2 (DIMENSIONAL
    + NUMERIC Hu vakası 1.16232e-7 m), 2026-07-03; ayrıca adım-eklem formunun
    Na→∞ LIMIT'i olarak sembolik tutarlılığı oracle'da kanıtlı. Hu Örnek 4-1
    basılı W=0.12 μm birebir. Geçerlilik: N_ağır >> N_hafif (tek-yanlı adım eklem), sıfır dış gerilim
    (V uygulanacaksa φ_bi yerine φ_bi−V geçilir). Birim: cgs-benzeri kitap
    konvansiyonunda [cm] (εs [F/cm], N [cm^-3]).
    """
    return math.sqrt(2.0 * eps_s * phi_bi / (q * N_light))


def depletion_width_step_junction(eps_s: float, phi_bi: float, q: float,
                                  Na: float, Nd: float) -> float:
    """İki-yanlı adım eklem depletion genişliği W = sqrt(2·εs·φ_bi·(1/Na+1/Nd)/q).

    Katman: first-principles (oracle-verified) — physics_verify 3/3 (DIMENSIONAL
    + NUMERIC Pierret 9.7135e-7 m + LIMIT Na→∞ ⇒ tek-yanlı form), 2026-07-03;
    KB kartı semiconductors/depletion-width-step-junction. Pierret Böl. 5 çapası
    (Na=1e17, Nd=1e15: V_bi=0.716 V, W=0.972 μm; rank 9).
    NOT: çapa aritmetiği ε_r≈11.8'i sabitler (katalog aktarımı 11.7 idi — 0.967 μm
    verir, çapayı üretemez; xfail testi belgeliyor). Birim: [cm] (εs [F/cm]).
    """
    return math.sqrt(2.0 * eps_s * phi_bi * (1.0 / Na + 1.0 / Nd) / q)


def open_circuit_voltage(j_sc: float, j0: float, n_ideality: float, kT_q: float) -> float:
    """Açık-devre gerilimi V_oc = n·(kT/q)·ln(J_sc/J_0 + 1) (ideal diyot + ışık akımı).

    Katman: first-principles (oracle-verified) — physics_verify 2/2 (DIMENSIONAL
    + NUMERIC 0.6288 V @ J_sc=350 A/m², J_0=1e-8 A/m²), 2026-07-04; KB kartı
    semiconductors/open-circuit-voltage (prereq: shockley-diode-equation).
    Kaynak: PVEducation (open) / Green, Solar Cells (1982). Superposition varsayımı.
    """
    return n_ideality * kT_q * math.log(j_sc / j0 + 1.0)


def ideal_fill_factor(voc_norm: float) -> float:
    """İdeal diyottan TAM dolum faktörü — sayısal güç-maksimizasyonu (bağımsız yol).

    Normalize: v = V/(n·kT/q), voc_norm = V_oc/(n·kT/q); J_sc = J_0(e^voc − 1) ⇒
    p(v) = v·[1 − (e^v − 1)/(e^voc − 1)];  FF = max_v p(v) / voc.
    Katman: first-principles (kapalı-form ideal diyottan sayısal maksimizasyon;
    green_fill_factor'ın doğruluk-hakemi — çift-yol deseni, rank-13).
    """
    if voc_norm <= 1.0:
        raise ValueError(f"voc_norm={voc_norm}: normalize V_oc > 1 olmalı")
    from scipy.optimize import minimize_scalar

    em1 = math.expm1(voc_norm)

    def neg_p(v):
        return -v * (1.0 - math.expm1(v) / em1)

    res = minimize_scalar(neg_p, bounds=(0.5 * voc_norm, voc_norm), method="bounded",
                          options={"xatol": 1e-12})
    return -res.fun / voc_norm


def green_fill_factor(voc_norm: float) -> float:
    """Green (1981) ampirik dolum faktörü FF₀ = (v_oc − ln(v_oc + 0.72))/(v_oc + 1).

    Katman: empirical fit — Green, Solid-State Electronics 24, 788 (1981);
    PVEducation'ın standart FF ifadesi. Geçerlilik: v_oc > 10 (normalize).
    Doğrulama: ideal_fill_factor'la çift-yol karşılaştırması (rank-13 testi;
    ölçülen sapma bandı test dosyasında) + oracle NUMERIC (FF₀(20)=0.808043).
    """
    if voc_norm <= 10.0:
        raise ValueError(f"voc_norm={voc_norm}: Green ifadesi v_oc > 10 için geçerli")
    return (voc_norm - math.log(voc_norm + 0.72)) / (voc_norm + 1.0)


def cv_doping_from_slope(slope: float, q: float, eps_s: float, area: float) -> float:
    """C-V eğiminden hafif-taraf doping: N_l = 2/(slope·q·εs·A²)  [Hu Eq. 4.4.2].

    slope = d(1/C²_dep)/dV_r. Katman: first-principles (oracle-verified) —
    physics_verify 2/2 (DIMENSIONAL [A² ile kapanır] + NUMERIC 5.885e21 m⁻³),
    2026-07-03; KB kartı semiconductors/cv-doping-extraction. TERS-PROBLEM yolu
    (parametre çıkarımı). DİKKAT (rank 5 bulgusu): Hu Örnek 4-2'nin BASILI
    girdileri (slope=2e23, A=1 μm²) kendi basılı cevabını üretmez — kitabın
    yerine-koyması A² için 1e-8 kullanır (1e-16 değil); fiziksel öz-tutarlı
    eğim 2e31'dir. test_rank05'te strict-xfail ile belgeli.
    """
    return 2.0 / (slope * q * eps_s * area ** 2)


def cv_heavy_doping_from_intercept(ni: float, n_light: float, phi_bi: float,
                                   kT_q: float) -> float:
    """C-V kesişiminden ağır-taraf doping: N_h = (ni²/N_l)·e^(φ_bi/(kT/q)).

    (builtin_potential'ın N_h için tersine çevrilmişi.) Katman: first-principles
    (oracle-verified) — physics_verify 3/3 (DIMENSIONAL + Hu Örnek 4-2 çift
    NUMERIC: 0.84 V → 1.790e18 cm⁻³ [basılı 1.8e18]; duyarlılık 0.78 V →
    1.781e17 [basılı 1.8e17]), 2026-07-03; KB kartı
    semiconductors/cv-intercept-heavy-doping. Hu'nun dersi test olarak kodlu:
    kesişimdeki 60 mV'lik hata N_h'ı BİR DEKAD kaydırır (üstel duyarlılık).
    """
    return (ni ** 2 / n_light) * math.exp(phi_bi / kT_q)


def shockley_diode_current(I0: float, V: float, n_ideality: float, kT_q: float) -> float:
    """Shockley diyot denklemi i = I0·(exp(V/(n·kT/q)) − 1) [A].

    Katman: first-principles (oracle-verified) — physics_verify (DIMENSIONAL +
    NUMERIC + LIMIT V→−∞ ⇒ i→−I0), 2026-07-03; KB kartı
    semiconductors/shockley-diode-equation. Log-eğim: dV/dlog10(i) =
    n·(kT/q)·ln10 — 300 K'de 59.6 mV/dekad (n=1, "60"), 119.3 (n=2, "120");
    rank-9 testi eğimleri sayısal I-V'den ölçerek doğrular.
    Geçerlilik: ideal diyot (seri direnç yok, yüksek-enjeksiyon yok).
    """
    return I0 * (math.exp(V / (n_ideality * kT_q)) - 1.0)

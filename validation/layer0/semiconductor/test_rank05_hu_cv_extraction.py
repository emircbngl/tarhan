"""Rank 5 — Hu Örnek 4-2: C-V ölçümünden doping çıkarımı (ters-problem yolu).

Kaynak: Hu, Modern Semiconductor Devices, Böl. 4, s. 99-100 — açık Berkeley PDF'i
indirilip sayfalar GÖRSEL okundu (2026-07-03). KAYNAK DOSYASI REPO'DA DEĞİLDİR
(telif; `~/malzeme simulasyon/sources/hu/hu-ch4.pdf` — repo DIŞI). Sonuç: aşağıdaki
transkripsiyon temiz bir checkout'tan YENİDEN DOĞRULANAMAZ; xfail'in gösterdiği şey
"transkribe edilen değerler kendi içinde çelişkili" — "kitap kanıtlanmış biçimde
yanlış" DEĞİL (2026-07-15 review düzeltmesi).
Eq. 4.4.2: 1/C²_dep = 2(φ_bi+V_r)/(q·N·εs·A²)  →  slope = 2/(q·N·εs·A²).

Basılı girdiler: slope=2×10²³ F⁻²V⁻¹, kesişim 0.84 V, A=1 μm²; Hu sabitleri
εr=12, ni=1e10, kT/q=0.026. Basılı cevaplar: N_l=6×10¹⁵, N_h=1.8×10¹⁸ cm⁻³;
duyarlılık: kesişim 0.78 V olsaydı N_h=1.8×10¹⁷ (bir dekad!).

KİTAP-İÇİ TUTARSIZLIK (rank-5'in ana bulgusu, görsel-okumayla kesinleşti):
basılı yerine-koyma A² yerine 10⁻⁸ kullanır (A=1 μm² ⇒ A²=10⁻¹⁶ cm⁴ olmalı);
literal girdiler N=5.9×10²³ verir (katı yoğunluğu üstü — saçma). Fiziksel
öz-tutarlılık basılı cevabın tarafında: N=6e15 & A=1 μm² ⇒ slope 2×10³¹ olmalı.
Testler basılı CEVAPLARA sabitlenir; girdi-tutarsızlığı strict-xfail'le belgeli.
"""
import pytest

from tarhan.physics import cv_doping_from_slope, cv_heavy_doping_from_intercept

# Hu vaka sabitleri (cgs, kitap konvansiyonu)
Q, EPS_S = 1.6e-19, 12.0 * 8.85e-14      # C, F/cm
NI, KT_Q = 1e10, 0.026                    # cm^-3, V
AREA_CM2 = 1e-8                           # 1 μm²


def test_nl_self_consistent_slope_reproduces_printed():
    # fiziksel öz-tutarlı eğim (2e31, A=1 μm²) → basılı 6e15
    n_l = cv_doping_from_slope(2e31, Q, EPS_S, AREA_CM2)
    assert abs(n_l - 6e15) / 6e15 < 0.02


@pytest.mark.xfail(strict=True, reason=(
    "KİTAP-İÇİ TUTARSIZLIK: Hu Örnek 4-2'nin literal girdileri (slope=2e23, "
    "A=1 μm² ⇒ A²=1e-16 cm⁴) basılı 6e15'i üretmez — 5.9e23 verir (fiziksel "
    "saçmalık). Basılı yerine-koyma A² için 1e-8 kullanmış. Görsel-okumayla "
    "kesin (s.100); muhtemel errata."))
def test_printed_inputs_reproduce_printed_answer():
    n_l = cv_doping_from_slope(2e23, Q, EPS_S, AREA_CM2)
    assert abs(n_l - 6e15) / 6e15 < 0.02


def test_nh_from_intercept_printed():
    n_h = cv_heavy_doping_from_intercept(NI, 6e15, 0.84, KT_Q)
    assert abs(n_h - 1.7902e18) / 1.7902e18 < 1e-3    # hassas (oracle ile aynı)
    assert abs(n_h - 1.8e18) / 1.8e18 < 0.02          # basılı, 2 anlamlı hane


def test_nh_sensitivity_one_decade_printed():
    # Hu'nun dersi: kesişimde 60 mV hata → N_h bir dekad kayar.
    # (İlk koşum dersi: hesap 1.781e17, basılı yuvarlak 1.8e17 → fark %1.04;
    #  2-anlamlı-haneli basılı değere %1 tolerans fazla sıkı — hane-bilinçli tolerans.)
    n_h = cv_heavy_doping_from_intercept(NI, 6e15, 0.78, KT_Q)
    assert abs(n_h - 1.7811e17) / 1.7811e17 < 1e-3    # hassas (oracle ile aynı)
    assert abs(n_h - 1.8e17) / 1.8e17 < 0.02          # basılı, 2 anlamlı hane

"""Rank-12: Tam PEMFC polarizasyon eğrisi V(i) — Spiegel/FuelCellStore parametre seti.

ATIF DÜZELTMESİ (2026-07-09, kaynak-araştırması sonucu — önceki "Kim/Barbir"
atfı YANLIŞTI):
  - Parametre seti (E_r=1.229 V, i0=10^-6.912 A/cm², α=0.5, R_i=0.19 Ω·cm²,
    i_L=1.4 A/cm²) → **Colleen Spiegel, "PEM Fuel Cell Modeling and Simulation
    Using MATLAB" (2008)** / FuelCellStore tutorial. Kim ya da Barbir DEĞİL.
    Buradaki merdiven standart ders-kitabı biçimidir (O'Hayre-tarzı Tafel + ohmik
    + ln(i_L/(i_L−i)) konsantrasyon), Spiegel-kaynaklı parametre değerleriyle.
  - Kim, Lee, Srinivasan & Chamberlin, JES 142(8), 2670 (1995),
    DOI 10.1149/1.2050072: V = E₀ − b·log₁₀(i) − R·i − m·exp(n·i). Kim'in modeli
    i_L KULLANMAZ (üstel terim TÜM kütle-taşınım modelidir) ve NATİF birimleri
    i[mA/cm²], m[mV], n[cm²/mA]'dir. A/cm²'ye çevrilince temsili m≈3e-5–5e-4 V,
    n≈5–8 cm²/A (birincil Tablo I: m=0.125 mV, n=0.00945 cm²/mA). `kim_cell_voltage`
    bu GERÇEK Kim biçimini uygular (sabitler zorunlu girdi).
  - **(0.085 V, 1.1) Kim'in m,n'i DEĞİL** — bunlar Spiegel'in α₁ (amplifikasyon, V)
    ve k (BOYUTSUZ) sabitleridir; Spiegel'de AYRI bir i_L terimiyle BİRLİKTE
    kullanılır. Kim'in standalone m·exp(n·i)'sine sokulunca fiziksel-saçma
    (i=0'da 85 mV offset; kütle-taşınım terimi i→0'da sıfırlanmalı). Bu, açık
    kalem strict-xfail'i değil, artık ÇÖZÜLMÜŞ bir provenans bulgusudur.

Yayınlanmış nitel çapalar: OCV-yakınında ~1.0 V, ~1 A/cm² civarında ~0.6 V,
i_L=1.4'e roll-off.

Katman: MONTAJ — üç kayıp terimi tek tek oracle-doğrulamalı (tafel 3/3,
ohmic 4/4, concentration 3/3; rank-6). Bu test yeni formül pinlemez;
bileşen-özdeşlikleri + nitel çapalar + ıraksama guard'ları + Kim-provenans.
"""
import math

import pytest

from tarhan.models.pemfc0d import (
    Pemfc0DParams,
    cell_voltage,
    kim_cell_voltage,
    polarization_curve,
)
from tarhan.physics import activation_overpotential_tafel

R_GAS = 8.314462618
FARADAY = 96485.33212

# Spiegel (2008)/FuelCellStore parametre seti (önceden yanlışlıkla "Kim/Barbir" idi).
SPIEGEL_SET = dict(e_r=1.229, j0=10.0 ** -6.912, alpha=0.5, n_e=2.0,
                   r_i=0.19, j_l=1.4, T=298.15)


@pytest.fixture()
def params():
    return Pemfc0DParams(**SPIEGEL_SET)


def test_tafel_term_identity_with_oracle_verified_function(params):
    """Merdivenin η_act'ı, oracle-doğrulamalı fonksiyonun BİREBİR kendisi."""
    for j in (0.01, 0.3, 1.0):
        _, parts = cell_voltage(params, j)
        ref = activation_overpotential_tafel(j, params.j0, params.alpha,
                                             params.n_e, FARADAY, R_GAS, params.T)
        assert parts["eta_act"] == ref


def test_tafel_slope_matches_ohayre_anchor(params):
    """b = RT·ln10/(α·n·F): O'Hayre Örn. 3.3 basılı çapası 'yaklaşık 60 mV/dekad'
    (α=0.5, n=2, 298 K → hassas 0.0592 sınıfı). Hane-bilinçli: basılı ~2 hane."""
    b = params.tafel_slope_decade()
    assert abs(b - 0.059) < 1e-3
    # öz-tutarlılık: kapalı-form == oracle-doğrulamalı fonksiyonun bir-dekad farkı
    ref = (activation_overpotential_tafel(1.0, params.j0, params.alpha,
                                          params.n_e, FARADAY, R_GAS, params.T)
           - activation_overpotential_tafel(0.1, params.j0, params.alpha,
                                            params.n_e, FARADAY, R_GAS, params.T))
    assert abs(b - ref) < 1e-12


def test_published_anchor_near_ocv(params):
    """Yayınlanmış nitel çapa: OCV-yakınında ~1.0 V (parantez = kaynağın ~1 hanesi)."""
    v, _ = cell_voltage(params, 1e-3)
    assert 0.95 < v < 1.05


def test_published_anchor_mid_current(params):
    """Yayınlanmış nitel çapa: ~0.6 V @ ≈1 A/cm²."""
    v, _ = cell_voltage(params, 1.0)
    assert 0.55 < v < 0.65


def test_monotone_decreasing_and_rolloff(params):
    """V(i) kesin azalan; i_L'ye yaklaşırken eğri DİKLEŞİR (roll-off)."""
    grid = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.35, 1.39]
    curve = polarization_curve(params, grid)
    vs = [v for _, v in curve]
    assert all(v2 < v1 for v1, v2 in zip(vs, vs[1:]))
    # aynı Δj=0.15 için düşüş, limit yakınında orta-bölgedekinden büyük
    drop_mid = cell_voltage(params, 0.60)[0] - cell_voltage(params, 0.75)[0]
    drop_lim = cell_voltage(params, 1.20)[0] - cell_voltage(params, 1.35)[0]
    assert drop_lim > drop_mid


def test_divergence_guards(params):
    """j→i_L ıraksaması sessiz kırpılmaz; j<=0 Tafel'de tanımsız."""
    with pytest.raises(ValueError):
        cell_voltage(params, params.j_l)
    with pytest.raises(ValueError):
        cell_voltage(params, params.j_l + 0.1)
    with pytest.raises(ValueError):
        cell_voltage(params, 0.0)
    with pytest.raises(ValueError):
        kim_cell_voltage(params, 0.0, m=1e-4, n_exp=8.0)
    with pytest.raises(ValueError):
        polarization_curve(params, [0.5, 1.5])


def test_kim_form_core_identity(params):
    """Kim biçiminin Tafel+ohmik çekirdeği (m=0) merdivenle özdeş (≤1e-12):
    b·log10(j/j0) ≡ (RT/(α·n·F))·ln(j/j0) taban-değişimi burada kanıtlanır."""
    for j in (0.05, 0.7, 1.3):
        v, parts = cell_voltage(params, j)
        ladder_core = v + parts["eta_conc"]  # E_r − η_act − η_ohm
        assert abs(kim_cell_voltage(params, j, m=0.0, n_exp=0.0) - ladder_core) < 1e-12


def test_circulated_constants_are_spiegel_not_kim(params):
    """ÇÖZÜLDÜ (2026-07-09, kaynak-araştırması): (0.085, 1.1) Kim'in m,n'i DEĞİL —
    Spiegel'in α₁ (V) ve k (boyutsuz) sabitleridir. Kanıt fizikte:

    (a) Standalone Kim m·exp(n·i) olarak (0.085, 1.1) SAÇMA — i=0'da 85 mV offset
        (kütle-taşınım terimi i→0'da sıfırlanmalı) ve i_L=1.4'e kadar dik duvar YOK
        (yalnız ~4× artış).
    (b) GERÇEK Kim A/cm² sabitleri (temsili m=3e-5 V, n=8 cm²/A) fiziksel: i→0'da
        sıfır, i≈1.3-1.4'te öz-üreten limit-akım duvarı.
    """
    m_circ, n_circ = 0.085, 1.1          # Spiegel α₁, k (Kim'in m,n'i DEĞİL)
    m_kim, n_kim = 3e-5, 8.0             # temsili GERÇEK Kim (A/cm²)

    # (a) dolaşımdaki set standalone Kim terimi olarak fiziksel değil
    assert m_circ * math.exp(n_circ * 0.0) > 0.05          # i=0'da ~85 mV offset (saçma)
    wall_circ = (m_circ * math.exp(n_circ * 1.3)) / (m_circ * math.exp(n_circ * 0.1))
    assert wall_circ < 5.0                                  # i_L'ye kadar dik duvar yok

    # (b) gerçek Kim sabitleri i→0'da sıfırlanır ve dik duvar üretir
    assert m_kim * math.exp(n_kim * 0.1) < 1e-3            # düşük i'de ~0
    wall_kim = (m_kim * math.exp(n_kim * 1.3)) / (m_kim * math.exp(n_kim * 0.1))
    assert wall_kim > 1e3                                   # i≈1.3'te öz-üreten duvar

    # kim_cell_voltage GERÇEK Kim sabitleriyle monoton-azalan makul eğri verir
    # (düşük i'de OCV-yakını ~0.86 V; kütle-taşınım duvarı yüksek i'de voltajı düşürür)
    vs = [kim_cell_voltage(params, j, m=m_kim, n_exp=n_kim) for j in (0.1, 0.5, 1.0, 1.3)]
    assert all(v2 < v1 for v1, v2 in zip(vs, vs[1:]))     # kesin azalan
    assert vs[0] > 0.8                                     # düşük i'de makul (ölçülen 0.86)

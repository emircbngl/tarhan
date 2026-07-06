"""Rank-12: Tam PEMFC polarizasyon eğrisi V(i) — Barbir 2e / Kim-1995 parametre seti.

Kaynak kaydı:
  - Eğri biçimi: Kim, Lee, Srinivasan & Chamberlin, J. Electrochem. Soc. 142,
    2670 (1995): V = E_r − b·log(i/i0) − R_i·i − m·exp(n·i). Barbir (PEM Fuel
    Cells, 2e 2013, Ch.3) bu denklemi SUNAR — yazarlık Kim'indir (wiki katalog
    doğrulama düzeltmesi, 2026-07-03).
  - Parametre seti (dolaşımdaki "Barbir-tarzı" set, wiki kataloğunda kayıtlı):
    E_r=1.229 V, i0=10^-6.912 A/cm², α=0.5, R_i=0.19 Ω·cm², i_L=1.4 A/cm².
  - Yayınlanmış nitel çapalar (aynı katalog kaydı): eğri OCV-yakınında ~1.0 V,
    ~1 A/cm² civarında ~0.6 V, i_L=1.4'e doğru roll-off.

Katman: MONTAJ — üç kayıp terimi tek tek oracle-doğrulamalı (tafel 3/3,
ohmic 4/4, concentration 3/3; rank-6); bu test yeni formül pinlemez,
(a) bileşen-özdeşliklerini makine hassasiyetinde, (b) yayınlanmış nitel
çapaları parantez olarak, (c) ıraksama guard'larını doğrular. Parantezler
keyfî tolerans DEĞİL, kaynağın kendi nitel eğri tarifidir (~1 anlamlı hane).

Dürüst açık kalem (strict-xfail): wiki katalog notundaki "kütle-taşınım
sabitleri 0.085, 1.1" aktarımı (m·exp(n·i) biçimi için) aynı i_L=1.4
fiziğiyle TUTARSIZ — j=1.0'da ln-biçiminin 5.3 katı kayıp verir (ölçüldü;
Kim-1995 tipik m~3e-5 V, n~8 cm²/A sınıfıdır). Kaynak-sayfa teyidi bekliyor.
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

KIM_SET = dict(e_r=1.229, j0=10.0 ** -6.912, alpha=0.5, n_e=2.0,
               r_i=0.19, j_l=1.4, T=298.15)


@pytest.fixture()
def params():
    return Pemfc0DParams(**KIM_SET)


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


@pytest.mark.xfail(
    strict=True,
    reason="Wiki katalog aktarımı m=0.085 V, n=1.1 cm²/A: aynı i_L=1.4 fiziğiyle "
           "tutarsız — j=1.0'da m·exp(n·j)=0.2554 V, ln-biçimi 0.0483 V (5.3×; "
           "aynı büyüklük mertebesinde değil). Kim-1995 tipik m~3e-5 V, n~8 cm²/A "
           "sınıfı. Kaynak-sayfa (Kim JES 142:2670 / Barbir 2e Ch.3) teyidi açık kalem.",
)
def test_transcribed_mass_transport_constants_consistency(params):
    """(0.085, 1.1) aktarımı ln-biçimiyle aynı büyüklük mertebesinde olmalıydı."""
    j = 1.0
    _, parts = cell_voltage(params, j)
    kim_conc = 0.085 * math.exp(1.1 * j)
    ratio = kim_conc / parts["eta_conc"]
    assert 0.5 < ratio < 2.0

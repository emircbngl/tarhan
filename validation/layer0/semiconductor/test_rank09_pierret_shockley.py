"""Rank 9 — Pierret adım eklem çapaları + Shockley log-eğimleri.

Kaynak: Pierret, Semiconductor Device Fundamentals (1996), Böl. 5 (elektrostatik)
+ Böl. 6 (ideal diyot). Çapalar (UC Davis ders örneğiyle doğrulanmış; basılı
kopyadan yeniden-teyit açık kalem): Na=1e17, Nd=1e15 → V_bi=0.716 V, W=0.972 μm.
Pierret konvansiyonu: ni=1e10 cm⁻³, kT/q=0.0259 V.

KONVANSİYON DÜZELTMESİ (2026-07-03): katalog aktarımı ε_r=11.7 demişti; W=0.972 μm
çapası ε_r≈11.8'i sabitliyor (geri-hesap 11.82; 11.7 → 0.967 μm, çapayı üretemez).
strict-xfail testi bunu belgeliyor — rank-1 λ(0.3) deseninin aynısı.

Shockley eğimleri: dV/dlog10(i) = n·(kT/q)·ln10 = 59.64 mV/dekad (n=1, nominal
"60") / 119.27 (n=2, nominal "120"). Oracle: shockley-diode-equation 3/3 VERIFIED.
"""
import math

import pytest

from tarhan.physics import (builtin_potential, depletion_width_step_junction,
                            shockley_diode_current)

# Pierret vaka girdileri (açık deklarasyon — motor hardcode etmez)
NA, ND = 1e17, 1e15           # cm^-3
NI = 1e10                      # cm^-3
KT_Q = 0.0259                  # V
EPS0 = 8.85e-14                # F/cm
Q = 1.6e-19                    # C
K_S = 11.8                     # çapa-sabitlenmiş (bkz. modül docstring)


def test_builtin_potential_printed():
    assert abs(builtin_potential(NA, ND, NI, KT_Q) - 0.716) < 0.001   # hesap 0.7156


def test_depletion_width_printed_pins_ks_118():
    phi = builtin_potential(NA, ND, NI, KT_Q)
    w_um = depletion_width_step_junction(K_S * EPS0, phi, Q, NA, ND) * 1e4
    assert abs(w_um - 0.972) < 0.002                                   # hesap 0.9716


@pytest.mark.xfail(strict=True, reason=(
    "TÜR-2 xfail — TEYİT EDİLMEMİŞ PROVENANS ('kaynak hatası' diye OKUNMAMALI): "
    "katalog aktarımı ε_r=11.7, W=0.972 μm çapasını üretemiyor (0.967 μm verir). "
    "Çapa ε_r≈11.8'i sabitliyor — ama hangisinin (aktarımımızın mı kitabın mı) "
    "yanlış olduğu basılı Pierret'ten TEYİT EDİLMEDİ; hata BİZDE olabilir. EMSAL: "
    "rank-12'nin (0.085,1.1) sabitleri de kaynak hatası sanılıyordu; birincil-kaynak "
    "araştırması mis-attribution'ın BİZİM olduğunu gösterdi."))
def test_catalog_transcription_ks_117():
    phi = builtin_potential(NA, ND, NI, KT_Q)
    w_um = depletion_width_step_junction(11.7 * EPS0, phi, Q, NA, ND) * 1e4
    assert abs(w_um - 0.972) < 0.002


def _slope_mv_per_decade(n_ideality: float) -> float:
    """Sayısal I-V'den log-eğim. Pencere n ile ölçeklenir (V = 0.4n → 0.6n) ki
    exp argümanı ≥ 15 olsun ve −1 terimi gerçekten ihmal edilebilir kalsın.
    (İlk deneme sabit 0.25-0.45 V penceresiyle n=2'de FAIL etti: e^4.8≈125'te
    −1 ihmal edilemez, eğim 118.93 çıkar — testin kendisi rejim sınırını öğretti.)"""
    v1, v2 = 0.4 * n_ideality, 0.6 * n_ideality
    i1 = shockley_diode_current(1e-12, v1, n_ideality, KT_Q)
    i2 = shockley_diode_current(1e-12, v2, n_ideality, KT_Q)
    return (v2 - v1) / (math.log10(i2) - math.log10(i1)) * 1e3


@pytest.mark.parametrize("n_ideality, exact_mv, nominal_mv", [
    (1.0, 59.64, 60.0),
    (2.0, 119.27, 120.0),
])
def test_shockley_log_slopes(n_ideality, exact_mv, nominal_mv):
    slope = _slope_mv_per_decade(n_ideality)
    assert abs(slope - n_ideality * KT_Q * math.log(10) * 1e3) < 0.01  # formülle birebir
    assert abs(slope - exact_mv) < 0.05                                # hassas değer
    assert abs(slope - nominal_mv) / nominal_mv < 0.01                 # nominal "60/120" %1 içinde

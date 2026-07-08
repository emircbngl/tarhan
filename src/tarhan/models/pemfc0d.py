"""0D PEMFC polarizasyon eğrisi — kayıp-merdiveni montajı (rank-12).

V(i) = E_r − η_act(i) − η_ohm(i) − η_conc(i)

Merdiven biçimi: standart ders-kitabı polarizasyonu (O'Hayre-tarzı Tafel + ohmik
+ ln(i_L/(i_L−i)) konsantrasyon). Rank-12 parametre seti (E_r=1.229, i0=10^-6.912
A/cm², α=0.5, R_i=0.19 Ω·cm², i_L=1.4 A/cm²) → **Spiegel, "PEM Fuel Cell Modeling
and Simulation Using MATLAB" (2008)** / FuelCellStore. (ATIF DÜZELTMESİ 2026-07-09:
önceki "Kim/Barbir" atfı yanlıştı; kaynak-araştırması Spiegel/FuelCellStore olarak
belirledi.)

Katman: montaj — her kayıp terimi tek tek oracle-doğrulamalı
(tarhan.physics: activation_overpotential_tafel 3/3, ohmic_overpotential 4/4,
concentration_overpotential 3/3); bu modül yalnız birleştirir, yeni formül
getirmez. Parametre seti vaka GİRDİSİDİR (motor hiçbir sabiti gömmez).

Konsantrasyon kaybı iki biçimde:
  - ln-biçimi  η_conc = c·ln(i_L/(i_L−i))  (varsayılan; oracle-doğrulamalı)
  - `kim_cell_voltage`: GERÇEK Kim, Lee, Srinivasan & Chamberlin (JES 142(8),
    2670 (1995), DOI 10.1149/1.2050072) biçimi V=E₀−b·log₁₀(i)−R·i−m·exp(n·i) —
    i_L KULLANMAZ (üstel terim tüm kütle-taşınım modelidir). Natif birimler
    i[mA/cm²]/m[mV]/n[cm²/mA]; A/cm²'de temsili m≈3e-5 V, n≈8 cm²/A. m ve n_exp
    vaka girdisi olarak ZORUNLU. NOT: dolaşımdaki (0.085 V, 1.1) Kim'in m,n'i
    DEĞİL — Spiegel'in α₁/k'sidir (bkz. test_rank12 provenans testi).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..physics import (
    activation_overpotential_tafel,
    concentration_overpotential,
    ohmic_overpotential,
)

__all__ = ["Pemfc0DParams", "cell_voltage", "polarization_curve", "kim_cell_voltage"]

# Fiziksel sabitler DEĞİL vaka-varsayılanları da değil: çağıran her şeyi verir.
_R_GAS = 8.314462618  # J/(mol·K) — CODATA; yalnız Tafel eğimi türetiminde
_F = 96485.33212      # C/mol — CODATA


@dataclass(frozen=True)
class Pemfc0DParams:
    """0D PEMFC hücre parametreleri (hepsi açık girdi; birimler A/cm² dünyası).

    e_r      : tersinir hücre voltajı [V] (ör. 1.229 @ 25°C sıvı-su)
    j0       : değişim akım yoğunluğu [A/cm²] (Spiegel seti: 10^-6.912)
    alpha    : yük-transfer katsayısı (Spiegel seti: 0.5)
    n_e      : elektron sayısı Tafel eğiminde (PEMFC ORR pratiği n=2 ile
               b=RT·ln10/(α·n·F) ≈ 0.059 V/dekad @ 298 K — O'Hayre Örn. 3.3 çapası)
    r_i      : alan-özgül direnç [Ω·cm²] (Spiegel seti: 0.19)
    j_l      : limit akım yoğunluğu [A/cm²] (Spiegel seti: 1.4)
    c_conc   : ln-biçimi konsantrasyon katsayısı [V] (None ⇒ RT/(n_e·F)·(1+1/α),
               O'Hayre'nin c≈0.1 V sınıfındaki geleneksel seçimi)
    T        : sıcaklık [K]
    """

    e_r: float
    j0: float
    alpha: float
    n_e: float
    r_i: float
    j_l: float
    T: float
    c_conc: float | None = None

    def tafel_slope_decade(self) -> float:
        """b = RT·ln10/(α·n·F) [V/dekad] — activation_overpotential_tafel ile özdeş."""
        return _R_GAS * self.T * math.log(10.0) / (self.alpha * self.n_e * _F)

    def conc_coeff(self) -> float:
        if self.c_conc is not None:
            return self.c_conc
        return (_R_GAS * self.T / (self.n_e * _F)) * (1.0 + 1.0 / self.alpha)


def cell_voltage(p: Pemfc0DParams, j: float) -> tuple[float, dict]:
    """Hücre voltajı V(j) ve kayıp dökümü (ladder biçimi, ln-konsantrasyon).

    Geçerlilik: j0 < j < j_L (Tafel rejimi + konsantrasyon-limiti altı).
    j <= 0 veya j >= j_L ValueError (dürüst ıraksama; sessiz kırpma yok).
    """
    if j <= 0.0:
        raise ValueError(f"j={j} <= 0: Tafel logaritması tanımsız")
    eta_act = activation_overpotential_tafel(j, p.j0, p.alpha, p.n_e, _F, _R_GAS, p.T)
    eta_ohm = ohmic_overpotential(j, p.r_i)  # j·R_i, alan-özgül dünyada [V]
    eta_conc = concentration_overpotential(p.conc_coeff(), j, p.j_l)  # j>=j_L raise eder
    v = p.e_r - eta_act - eta_ohm - eta_conc
    return v, {
        "eta_act": eta_act,
        "eta_ohmic": eta_ohm,
        "eta_conc": eta_conc,
        "tafel_slope_decade": p.tafel_slope_decade(),
    }


def kim_cell_voltage(p: Pemfc0DParams, j: float, m: float, n_exp: float) -> float:
    """Kim-1995 ampirik biçim: V = E_r − b·log10(j/j0) − R_i·j − m·exp(n·j).

    GERÇEK Kim (JES 142(8):2670, DOI 10.1149/1.2050072); i_L YOK, üstel terim tüm
    kütle-taşınım modelidir. m [V] ve n_exp [cm²/A] fit sabitleri ZORUNLU girdi —
    TARHAN varsayılan gömmez. A/cm²'de temsili m≈3e-5 V, n≈8 cm²/A (birincil natif
    Tablo I: m=0.125 mV, n=0.00945 cm²/mA). Dolaşımdaki (0.085, 1.1) Kim'in DEĞİL,
    Spiegel'in α₁/k'sidir — buraya sokulmaz (bkz. test_rank12 provenans testi).
    b, ladder'daki Tafel teriminin birebir kendisi (log10 tabanına çevrilmiş).
    """
    if j <= 0.0:
        raise ValueError(f"j={j} <= 0: Tafel logaritması tanımsız")
    b = p.tafel_slope_decade()
    return p.e_r - b * math.log10(j / p.j0) - p.r_i * j - m * math.exp(n_exp * j)


def polarization_curve(p: Pemfc0DParams, j_grid) -> list[tuple[float, float]]:
    """(j, V) çiftleri; j_grid (0, j_L) içinde olmalı — aksi ValueError."""
    return [(float(j), cell_voltage(p, float(j))[0]) for j in j_grid]

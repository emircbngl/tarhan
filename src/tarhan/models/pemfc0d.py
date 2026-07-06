"""0D PEMFC polarizasyon eğrisi — kayıp-merdiveni montajı (rank-12).

V(i) = E_r − η_act(i) − η_ohm(i) − η_conc(i)

Kaynak: Barbir, *PEM Fuel Cells* 2e (2013) Ch.3'ün sunduğu parametrik eğri;
denklemin kökeni Kim, Lee, Srinivasan & Chamberlin, J. Electrochem. Soc.
142, 2670 (1995): V = E_r − b·log(i/i0) − R_i·i − m·exp(n·i).

Katman: montaj — her kayıp terimi tek tek oracle-doğrulamalı
(tarhan.physics: activation_overpotential_tafel 3/3, ohmic_overpotential 4/4,
concentration_overpotential 3/3); bu modül yalnız birleştirir, yeni formül
getirmez. Parametre seti vaka GİRDİSİDİR (motor hiçbir sabiti gömmez).

Konsantrasyon kaybı iki biçimde:
  - ln-biçimi  η_conc = c·ln(i_L/(i_L−i))  (varsayılan; oracle-doğrulamalı)
  - Kim m·exp(n·i) ampirik biçimi (opsiyonel; m ve n_exp vaka girdisi olarak
    ZORUNLU — yaygın dolaşımdaki değerlerin kaynak-sayfa teyidi yok, bkz.
    validation/layer0/test_rank12_pemfc_polarization.py strict-xfail)
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
    """Kim/Barbir 0D hücre parametreleri (hepsi açık girdi; birimler A/cm² dünyası).

    e_r      : tersinir hücre voltajı [V] (ör. 1.229 @ 25°C sıvı-su)
    j0       : değişim akım yoğunluğu [A/cm²] (Kim seti: 10^-6.912)
    alpha    : yük-transfer katsayısı (Kim seti: 0.5)
    n_e      : elektron sayısı Tafel eğiminde (PEMFC ORR pratiği n=2 ile
               b=RT·ln10/(α·n·F) ≈ 0.059 V/dekad @ 298 K — O'Hayre Örn. 3.3 çapası)
    r_i      : alan-özgül direnç [Ω·cm²] (Kim seti: 0.19)
    j_l      : limit akım yoğunluğu [A/cm²] (Kim seti: 1.4)
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

    m [V] ve n_exp [cm²/A] fit sabitleri ZORUNLU girdi — TARHAN varsayılan
    gömmez (dolaşımdaki değer çiftlerinin kaynak-sayfa teyidi açık kalem).
    b, ladder'daki Tafel teriminin birebir kendisi (log10 tabanına çevrilmiş).
    """
    if j <= 0.0:
        raise ValueError(f"j={j} <= 0: Tafel logaritması tanımsız")
    b = p.tafel_slope_decade()
    return p.e_r - b * math.log10(j / p.j0) - p.r_i * j - m * math.exp(n_exp * j)


def polarization_curve(p: Pemfc0DParams, j_grid) -> list[tuple[float, float]]:
    """(j, V) çiftleri; j_grid (0, j_L) içinde olmalı — aksi ValueError."""
    return [(float(j), cell_voltage(p, float(j))[0]) for j in j_grid]

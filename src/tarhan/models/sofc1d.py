"""1D SOFC hücre modeli — TARHAN'ın ilk Layer-3 domain modeli (rank 12).

Kaynak (yöntem aktarımı; kod bizim): O'Hayre, Cha, Colella & Prinz, "Fuel Cell
Fundamentals" 3e (Wiley 2016), §6.2 "A 1D Fuel Cell Model" — Denk. 6.23-6.32,
Tablo 6.4. Basılı uçtan-uca çapa (j = 0.5 A/cm²):
    ASR = 0.176 Ω·cm² → η_ohmic = 0.088 V;  η_cathode = 0.158 V;  V = 0.754 V

Model: V(j) = E_thermo − η_ohmic(j) − η_cathode(j)
  - η_ohmic  = j · ASR,  ASR = t_M·T / (A_SOFC·e^(−ΔG_act/RT))   [oracle 2/2]
  - x_O2|c   = x_O2|d − t_C·j·R·T/(4·F·p_C·D_eff)                 [oracle NUMERIC]
  - η_cathode= (R·T/(4·α·F))·ln[ j / (j0·p_C_atm·x_O2|c) ]        [oracle 3/3]
  (anot kinetiği ve anot konsantrasyon kaybı bu modelde ihmal — kitap kabulü)

BİRİM REÇETESİ (kitabın basılı aritmetiğiyle birebir; karıştırma!):
  - katot kinetiği: j ve j0 aynı birimde (A/cm²), p_C atm cinsinden
  - difüzyon-tükenme terimi: SI (j A/m², t_C m, p Pa, D m²/s)
Geçerlilik: Tafel rejimi (j >> j0·x'in ters terimi); j, x_O2|c > 0 sınırına kadar
(limit akım ~2.30 A/cm² bu parametre setinde) — aşımda dürüst ValueError.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_ATM_PA = 101325.0


@dataclass(frozen=True)
class Sofc1DParams:
    """O'Hayre Tablo 6.4 parametre seti (default'lar tablonun kendisi)."""
    e_thermo: float = 1.0        # V (termodinamik hücre gerilimi, tabloda verili)
    T: float = 1073.0            # K
    x_o2_inlet: float = 0.21     # katot girişi O2 mol kesri
    p_c_atm: float = 1.0         # katot basıncı [atm]
    d_eff_o2: float = 2e-5       # m²/s (O2-N2 etkin difüzyon)
    alpha: float = 0.5           # transfer katsayısı
    j0_acm2: float = 0.1         # A/cm² (katot değişim akım yoğunluğu)
    a_sofc: float = 9e7          # K/(Ω·m) (elektrolit sabiti)
    dg_act: float = 100e3        # J/mol (elektrolit aktivasyon enerjisi)
    t_electrolyte: float = 20e-6 # m
    t_cathode: float = 800e-6    # m
    R: float = 8.314             # J/(mol·K)
    F: float = 96485.0           # C/mol


def electrolyte_asr(p: Sofc1DParams) -> float:
    """Elektrolit alan-özgül direnci [Ω·m²]. Oracle: sofc-electrolyte-asr 2/2."""
    sigma_inv = p.T / (p.a_sofc * math.exp(-p.dg_act / (p.R * p.T)))
    return p.t_electrolyte * sigma_inv


def x_o2_cathode_cl(j_acm2: float, p: Sofc1DParams) -> float:
    """Katot katalizör katmanında O2 mol kesri (lineer tükenme profili, SI)."""
    j_si = j_acm2 * 1e4
    depletion = p.t_cathode * j_si * p.R * p.T / (4.0 * p.F * p.p_c_atm * _ATM_PA * p.d_eff_o2)
    x_c = p.x_o2_inlet - depletion
    if x_c <= 0.0:
        j_lim = p.x_o2_inlet * 4.0 * p.F * p.p_c_atm * _ATM_PA * p.d_eff_o2 / (
            p.t_cathode * p.R * p.T) / 1e4
        raise ValueError(
            f"j={j_acm2} A/cm² limit akımı aşıyor (j_lim≈{j_lim:.3f} A/cm²): "
            "katotta O2 tükendi — model bu bölgede tanımsız")
    return x_c


def cathode_overpotential(j_acm2: float, p: Sofc1DParams) -> float:
    """Birleşik katot aktivasyon+konsantrasyon kaybı [V]. Oracle: 3/3."""
    x_c = x_o2_cathode_cl(j_acm2, p)
    return (p.R * p.T / (4.0 * p.alpha * p.F)) * math.log(
        j_acm2 / (p.j0_acm2 * p.p_c_atm * x_c))


def cell_voltage(j_acm2: float, p: Sofc1DParams | None = None):
    """Hücre gerilimi ve kayıp ayrıştırması: (V, {'eta_ohmic':…, 'eta_cathode':…})."""
    p = p or Sofc1DParams()
    eta_ohm = j_acm2 * electrolyte_asr(p) * 1e4          # Ω·m² → Ω·cm²
    eta_cat = cathode_overpotential(j_acm2, p)
    v = p.e_thermo - eta_ohm - eta_cat
    return v, {"eta_ohmic": eta_ohm, "eta_cathode": eta_cat,
               "asr_ohm_cm2": electrolyte_asr(p) * 1e4}


def polarization_curve(j_grid, p: Sofc1DParams | None = None):
    """j-V eğrisi; limit-akım-üstü noktalar dürüstçe atlanmaz, hata verir."""
    p = p or Sofc1DParams()
    return [(j, cell_voltage(j, p)[0]) for j in j_grid]

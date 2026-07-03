"""Rank 6 — O'Hayre PEMFC kayıp-merdiveni: 4 işlenmiş örnek, kutulu basılı cevaplar.

Kaynak: O'Hayre, Cha, Colella & Prinz, "Fuel Cell Fundamentals", 3. baskı (Wiley
2016) — kullanıcının EPUB kopyasından birebir transkripsiyon (2026-07-03; denklem
görselleri kontakt-sayfa olarak okundu, sources/ohayre/). Dört çapa:

  Örnek 2.3: DMFC E⁰ = 1.229 − 0.03 = +1.199 V (E⁰ stokiyometriyle ölçeklenmez)
  Örnek 3.3: α=0.5, n=2, 298.15 K → Δη/dekad = 0.059 V; j0=1e-6→1 A/cm²: "6×60 mV=0.36 V"
  Örnek 4.1: A=10 cm², σ=0.10 S/cm, R_elec=0.005 Ω, j=1 A/cm²:
             (a) L=100 μm → R_ionic=0.01 Ω, η=0.15 V; (b) L=50 μm → 0.005 Ω, 0.10 V
  Örnek 5.1: 80°C, 1 atm, x_H2O=0.2 → p_O2=0.168 atm → c=5.8 mol/m³;
             D_eff=0.4^1.5·0.2=0.0506 cm²/s; j_L=2.26 A/cm²; η_conc(c=0.1, j=2)=0.22 V

KİTAP-İÇİ TUTARSIZLIK (belgeli): Örnek 5.1 problem metni D=0.1 cm²/s der; basılı
çözüm zinciri D=0.2 ile ilerler (0.4^1.5×0.2=0.0506 → 2.26). Test basılı çözüm
zincirine sabitlenir; metin-değeriyle xfail muhtemel erratayı işaretler.
"""
import math

import pytest

from tarhan.physics import (activation_overpotential_tafel, concentration_overpotential,
                            effective_diffusivity_bruggeman, ionic_resistance,
                            limiting_current_density, ohmic_overpotential,
                            standard_cell_potential)

R_GAS, F_CONST = 8.314, 96485.0


# --- Örnek 2.3: DMFC standart potansiyeli ---------------------------------- #

def test_ex23_dmfc_standard_potential():
    assert abs(standard_cell_potential(1.229, 0.03) - 1.199) < 1e-12


# --- Örnek 3.3: Tafel dekad-eğimi + 6 dekad -------------------------------- #

def test_ex33_tafel_decade_slope():
    slope = activation_overpotential_tafel(10.0, 1.0, 0.5, 2.0, F_CONST, R_GAS, 298.15)
    assert abs(slope - 0.059) < 5e-4            # basılı "0.059 V"


def test_ex33_six_decades_rounded_036():
    eta = activation_overpotential_tafel(1.0, 1e-6, 0.5, 2.0, F_CONST, R_GAS, 298.15)
    assert abs(eta - 0.35494) < 1e-4            # hassas değer
    assert abs(eta - 0.36) < 0.006              # basılı yuvarlak "6×60 mV = 0.36 V"


# --- Örnek 4.1: ohmik kayıplar (cgs birimleri, kitaptaki gibi) -------------- #

AREA, SIGMA, R_ELEC, CURRENT = 10.0, 0.10, 0.005, 10.0   # cm², S/cm, Ω, A

@pytest.mark.parametrize("thickness_cm, r_printed, eta_printed", [
    (0.01, 0.01, 0.15),     # (a) 100 μm
    (0.005, 0.005, 0.10),   # (b) 50 μm
])
def test_ex41_ohmic_losses(thickness_cm, r_printed, eta_printed):
    r_ion = ionic_resistance(thickness_cm, SIGMA, AREA)
    assert abs(r_ion - r_printed) < 1e-9
    assert abs(ohmic_overpotential(CURRENT, R_ELEC + r_ion) - eta_printed) < 1e-9


# --- Örnek 5.1: limit akımı + konsantrasyon kaybı --------------------------- #

def test_ex51_oxygen_concentration():
    p_o2_atm = 0.8 * 0.21                                   # 0.168 (basılı)
    c = p_o2_atm * 101300.0 / (R_GAS * 353.0)               # mol/m³ (kitap 353 K kullanır)
    assert abs(p_o2_atm - 0.168) < 1e-12
    assert abs(c - 5.8) < 0.05                              # basılı 5.8 mol/m³


def test_ex51_bruggeman_and_limiting_current():
    d_eff = effective_diffusivity_bruggeman(0.4, 0.2)       # basılı çözüm zinciri (D=0.2)
    assert abs(d_eff - 0.0506) < 2e-4
    j_l = limiting_current_density(4.0, F_CONST, d_eff, 5.8e-6, 0.05)   # cgs → A/cm²
    assert abs(j_l - 2.26) < 0.01                           # basılı 2.26 A/cm²


def test_ex51_concentration_overpotential():
    eta = concentration_overpotential(0.1, 2.0, 2.26)
    assert abs(eta - 0.22) < 0.005                          # basılı 0.22 V


def test_ex51_guard_beyond_limit():
    with pytest.raises(ValueError):
        concentration_overpotential(0.1, 2.3, 2.26)


@pytest.mark.xfail(strict=True, reason=(
    "KİTAP-İÇİ TUTARSIZLIK (muhtemel errata): Örnek 5.1 problem metni D=0.1 cm²/s "
    "der; basılı çözüm D=0.2 ile 2.26 A/cm² verir. Metin-değeri basılı cevabı "
    "üretemez (1.13 verir). Wiley errata listesinden teyit açık kalem."))
def test_ex51_statement_diffusivity_reproduces_printed_answer():
    d_eff = effective_diffusivity_bruggeman(0.4, 0.1)       # problem METNİNDEKİ D
    j_l = limiting_current_density(4.0, F_CONST, d_eff, 5.8e-6, 0.05)
    assert abs(j_l - 2.26) < 0.01

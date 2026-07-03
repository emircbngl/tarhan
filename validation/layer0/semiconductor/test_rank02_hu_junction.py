"""Rank 2 — Hu Örnek 4-1: P+N step junction (depletion approximation).

Kaynak: Chenming Hu, Modern Semiconductor Devices for Integrated Circuits,
Böl. 4, Örnek 4-1 (Berkeley açık PDF). Basılı: φ_bi ~ 1 V; W = 0.12 μm; x_P ~ 1.2 Å.
KONVANSİYON-TUZAĞI kuralı: Hu sabitleri (ni=1e10, eps_r=12, kT/q=0.026) burada
açık vaka girdisidir — motor hiçbirini hardcode etmez.
"""
from tarhan.physics import builtin_potential, depletion_width_one_sided

# Hu Örnek 4-1 vaka girdileri
NA, ND = 1e20, 1e17          # cm^-3
KT_Q = 0.026                  # V
NI = 1e10                     # cm^-3
EPS_S = 12.0 * 8.85e-14       # F/cm
Q = 1.6e-19                   # C


def test_builtin_potential_printed():
    phi = builtin_potential(NA, ND, NI, KT_Q)
    assert abs(phi - 1.0) < 0.05          # basılı "~1 V" (hesap 1.018)


def test_depletion_width_printed():
    phi = builtin_potential(NA, ND, NI, KT_Q)
    w_um = depletion_width_one_sided(EPS_S, phi, Q, ND) * 1e4
    assert abs(w_um - 0.12) < 0.006       # basılı 1.2e-5 cm (hesap 0.1164 μm)


def test_xp_charge_balance_printed():
    phi = builtin_potential(NA, ND, NI, KT_Q)
    x_n = depletion_width_one_sided(EPS_S, phi, Q, ND)
    x_p_angstrom = x_n * ND / NA * 1e8
    assert abs(x_p_angstrom - 1.2) < 0.1  # basılı "~1.2 Å ≈ 0" (hesap 1.16)

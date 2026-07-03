"""Rank 12 — uçtan-uca hücre gerilimi: 1D SOFC modeli, basılı hesap zinciri.

Kaynak: O'Hayre 3e §6.2 (Denk. 6.29-6.32, Tablo 6.4) — kullanıcının EPUB'undan
birebir transkripsiyon. KAYNAK NOTU: katalog planı bu rank için Barbir/Kim 1995
parametre setini öngörmüştü (kitap elde yok); elde-var O'Hayre'nin TAM BELİRLENMİŞ
ve cevabı BASILI 1D SOFC hesabıyla ikame edildi — daha güçlü çapa (uçtan-uca V
basılı). Barbir seti erişilince ek çapa olarak eklenebilir.

Basılı zincir (j = 0.5 A/cm², Tablo 6.4):
    ASR = 0.176 Ω·cm²  →  η_ohmic = 0.088 V
    x_O2|c = 0.210 − 0.0456 = 0.1644  →  η_cathode = 0.158 V
    V = 1.0 − 0.088 − 0.158 = 0.754 V
Oracle: sofc-electrolyte-asr 2/2 + sofc-cathode-overpotential 3/3 (2026-07-03).
"""
import pytest

from tarhan.models.sofc1d import (Sofc1DParams, cathode_overpotential, cell_voltage,
                                  electrolyte_asr, polarization_curve, x_o2_cathode_cl)

P = Sofc1DParams()   # Tablo 6.4 default'ları


def test_electrolyte_asr_printed():
    assert abs(electrolyte_asr(P) * 1e4 - 0.176) < 1e-3          # basılı 0.176 Ω·cm²


def test_oxygen_depletion_printed():
    assert abs(x_o2_cathode_cl(0.5, P) - 0.1644) < 3e-4          # basılı 0.210−0.0456


def test_cathode_overpotential_printed():
    assert abs(cathode_overpotential(0.5, P) - 0.158) < 1e-3     # basılı 0.158 V


def test_cell_voltage_printed_end_to_end():
    v, parts = cell_voltage(0.5, P)
    assert abs(parts["eta_ohmic"] - 0.088) < 1e-3                # basılı 0.088 V
    assert abs(v - 0.754) < 1e-3                                 # basılı 0.754 V


def test_polarization_curve_monotone_decreasing():
    js = [0.05 * k for k in range(1, 45)]                        # 0.05 … 2.20 A/cm²
    curve = polarization_curve(js, P)
    vs = [v for _, v in curve]
    assert all(vs[i] > vs[i + 1] for i in range(len(vs) - 1))


def test_limiting_current_guard():
    # x_O2 → 0: j_lim = x_d·4F·p·D/(t_C·R·T) ≈ 2.301 A/cm² (analitik)
    x_o2_cathode_cl(2.29, P)                                     # hâlâ tanımlı
    with pytest.raises(ValueError):
        x_o2_cathode_cl(2.31, P)


def test_tafel_regime_note_low_j_invalid():
    # Model Tafel-temelli: j → j0·x civarında η→0'ın altına iner (fiziksel değil);
    # bu bilinen geçerlilik sınırı — j0·p·x'te η tam 0 olmalı (ln 1).
    j_zero = P.j0_acm2 * P.p_c_atm * x_o2_cathode_cl(P.j0_acm2 * 0.21, P)
    assert abs(cathode_overpotential(j_zero, P)) < 5e-3

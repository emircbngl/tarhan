"""Rank 13 — güneş hücresi FF/V_oc çapaları (solar-cell ipliğinin açılışı).

Kaynaklar: Green, Solid-State Electronics 24, 788 (1981) — ampirik FF₀ ifadesi;
PVEducation (open) sunumu; V_oc ideal-diyot + süperpozisyondan.

Çift-yol deseni (rank-4/11'in devamı): Green'in ampirik FF₀'ı, ideal diyottan
SAYISAL güç-maksimizasyonuyla (bağımsız yol) karşılaştırılır. Ölçüm (2026-07-04):
|FF₀ − FF_exact| ≤ 1.4e-4 (v_oc=10.5) ve ≤ 9e-5 (v_oc ≥ 12) — Green'in beyan
ettiği ~1e-4 doğruluk sınıfı bağımsız olarak doğrulandı. Oracle: V_oc 2/2,
FF₀ aritmetiği NUMERIC (0.808043 @ v_oc=20).
"""
import math

import pytest

from tarhan.physics import green_fill_factor, ideal_fill_factor, open_circuit_voltage


def test_voc_oracle_point():
    voc = open_circuit_voltage(350.0, 1e-8, 1.0, 0.0259)   # 35 mA/cm², J0=1e-12 A/cm²
    assert abs(voc - 0.62882) < 1e-4                        # oracle NUMERIC ile aynı


@pytest.mark.parametrize("voc_norm, band", [
    (10.5, 2e-4), (12.0, 1e-4), (15.0, 1e-4), (20.0, 1e-4), (25.0, 1e-4), (30.0, 1e-4),
])
def test_green_vs_exact_dual_route(voc_norm, band):
    ff_exact = ideal_fill_factor(voc_norm)
    ff_green = green_fill_factor(voc_norm)
    assert abs(ff_exact - ff_green) < band
    assert 0.65 < ff_exact < 0.90


def test_green_oracle_arithmetic_point():
    assert abs(green_fill_factor(20.0) - 0.808043) < 1e-5


def test_ff_monotone_in_voc():
    ffs = [ideal_fill_factor(v) for v in (10.5, 15.0, 20.0, 30.0)]
    assert all(ffs[i] < ffs[i + 1] for i in range(len(ffs) - 1))


def test_validity_guards():
    with pytest.raises(ValueError):
        green_fill_factor(8.0)          # Green: v_oc > 10
    with pytest.raises(ValueError):
        ideal_fill_factor(0.5)

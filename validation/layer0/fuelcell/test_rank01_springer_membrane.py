"""Rank 1 — Springer 1991 Nafion membran korelasyonları (ampirik fit).

Kaynak: Springer, Zawodzinski & Gottesfeld, JES 138(8), 2334 (1991).
Doğrulama = basılı-rakam karşılaştırması (5/6 birebir, 2026-07-03).
"""
import pytest

from tarhan.physics import springer_conductivity, springer_lambda_sorption


@pytest.mark.parametrize("a, expected, tol", [
    (1.0, 14.00, 0.01),
    (0.5, 3.49, 0.01),
])
def test_lambda_sorption_printed_digits(a, expected, tol):
    assert abs(springer_lambda_sorption(a) - expected) <= tol


@pytest.mark.parametrize("lam, T, expected, tol", [
    (14.0, 303.15, 0.0687, 5e-4),
    (14.0, 353.15, 0.124, 1e-3),
    (22.0, 303.15, 0.110, 1e-3),
])
def test_conductivity_printed_digits(lam, T, expected, tol):
    assert abs(springer_conductivity(lam, T) - expected) <= tol


@pytest.mark.xfail(strict=True, reason=(
    "AÇIK KALEM: katalog-planı λ(0.3)=2.32 aktarımı kaynak-sayfa teyidi bekliyor; "
    "korelasyon 2.7715 veriyor. Hipotez: aktarım a=0.2 içindi (λ(0.2)=2.299≈2.30). "
    "Springer 1991 orijinalinden teyit edilince bu test düzeltilecek."))
def test_lambda_a03_plan_transcription():
    assert abs(springer_lambda_sorption(0.3) - 2.32) <= 0.01


def test_activity_domain_guard():
    with pytest.raises(ValueError):
        springer_lambda_sorption(1.5)

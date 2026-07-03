"""Rank 11 — Levich RDE limit akımı: 0.620 sabiti, iki bağımsız sayısal yol.

Kaynak: Newman & Thomas-Alyea, Electrochemical Systems 3e (RDE); Cochran hız
profili a=0.51023. Motorun İLK konveksiyon terimi.

İki bağımsız yol AYNI sabitte buluşur (rank-4 çift-yol deseninin konveksiyonlusu):
  Yol 1 — Γ'sız Simpson kuadratürü: ∫₀^∞ e^(−u³) du (Γ(4/3)=0.892980'in bağımsız
          reprodüksiyonu) → sabit = (0.51023/3)^(1/3)/I
  Yol 2 — FD/BVP: c'' + 3ζ²c' = 0, Thomas (backend.solve_tridiag dikişi),
          3-nokta yüzey akısı → sabit = küp·c'(0)

Ölçüm (2026-07-03): yol-1 = 0.62044995, yol-2(n→∞) = 0.62045002 — 5e-7 mutabakat;
a=0.51023'ten türeyen kesin değer 0.620450. RANK-11 DERSİ (physics_learn'de):
elle ara-hesap 0.620469 vermişti — bileşik sabitler makine-değerlendirilir,
çift-yol çapraz-doğrulanır. Literatür "0.620" (3 hane) ✓; 0.62048 varyantı
Cochran katsayısı hassasiyetine iner (katalog kriteri <%0.5 fazlasıyla sağlanır).
"""
import math

from tarhan.numerics.convection import levich_constant_fd, levich_constant_quadrature
from tarhan.physics import LEVICH_CONSTANT, levich_limiting_current

TARGET = 0.620450        # a=0.51023'ten; modül docstring'ine bakınız


def test_quadrature_route_and_gamma_reproduction():
    const = levich_constant_quadrature()
    assert abs(const - TARGET) < 5e-5
    # ara ürün Γ(4/3) bağımsız reprodüksiyonu: sabit = küp/I ⇒ I = küp/sabit
    integral = (0.51023 / 3.0) ** (1.0 / 3.0) / const
    assert abs(integral - math.gamma(4.0 / 3.0)) < 5e-5


def test_fd_route_agrees_with_quadrature():
    assert abs(levich_constant_fd(2000) - levich_constant_quadrature()) < 5e-6


def test_fd_observed_order_about_two():
    f = {n: levich_constant_fd(n) for n in (500, 1000, 2000)}
    p = math.log2(abs(f[500] - f[1000]) / abs(f[1000] - f[2000]))
    assert 1.7 < p < 2.6, f"gözlenen mertebe {p:.2f}"


def test_tabulated_three_sig_figs():
    # katalog kriteri: 0.620 sabiti <%0.5 — üç anlamlı haneyle birebir
    assert abs(LEVICH_CONSTANT - 0.620) < 5e-4
    assert abs(levich_constant_fd(2000) - 0.620) < 5e-4


def test_dimensional_form_and_sqrt_omega_scaling():
    # oracle-çapraz nokta: 1000 rpm, sulu-tipik parametreler → 6.12606 A/m²
    j1 = levich_limiting_current(1.0, 96485.0, 1e-9, 1e-6, 104.7198, 1.0)
    assert abs(j1 - 6.12606) < 1e-3
    # j_L ∝ √ω: ω 4 katına çıkınca akım tam 2 katı
    j4 = levich_limiting_current(1.0, 96485.0, 1e-9, 1e-6, 4 * 104.7198, 1.0)
    assert abs(j4 / j1 - 2.0) < 1e-12

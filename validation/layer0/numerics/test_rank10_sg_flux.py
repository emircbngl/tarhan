"""Rank 10 — Scharfetter-Gummel vs central-difference kenar akısı.

Kaynak: Farrell ve ark. (WIAS 2263) + Selberherr (1984). Kapalı-form kimlik
B(−x) = e^x·B(x) ⇒ SG dengesi TAM x = ln(n_sol/n_sağ) noktasında. CD ise dengeyi
yanlış noktaya koyar ve büyük mesh-Peclet'te yanlış işarete kaçar — SG'nin
varlık sebebi. Çekirdeğin yük-taşıyan akı testi.
"""
import math

from tarhan.numerics.flux import bernoulli, cd_edge_flux, sg_edge_flux

N_L, N_R = 10.0, 1.0


def test_sg_equilibrium_exact_point():
    x_eq = math.log(N_L / N_R)
    assert abs(float(sg_edge_flux(x_eq, N_L, N_R))) < 1e-14


def test_cd_equilibrium_wrong_point():
    x_cd = -2.0 * (N_R - N_L) / (N_R + N_L)          # 18/11
    assert abs(float(cd_edge_flux(x_cd, N_L, N_R))) < 1e-14   # CD'nin kendi sıfırı
    assert abs(x_cd - math.log(N_L / N_R)) > 0.5              # ...ama yanlış yerde


def test_large_peclet_sg_bounded_cd_wrong_sign():
    for x in (10.0, 20.0, 40.0):
        j_sg = float(sg_edge_flux(x, N_L, N_R))
        j_cd = float(cd_edge_flux(x, N_L, N_R))
        assert abs(j_sg - x * N_R) / (x * N_R) < 0.01   # drift asimptotu x·n_sağ
        assert j_cd < 0.0 < j_sg                          # CD yanlış işarette


def test_bernoulli_small_x_series_stable():
    assert abs(float(bernoulli(1e-12)) - (1.0 - 5e-13)) < 1e-15


def test_bernoulli_large_x_no_overflow():
    assert 0.0 <= float(bernoulli(1000.0)) < 1e-290
    assert abs(float(bernoulli(-1000.0)) - 1000.0) < 1e-9

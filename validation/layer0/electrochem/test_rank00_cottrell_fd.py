"""Rank 0 — Cottrell kronoamperometri, explicit FD (Britz & Strutwolf 4e).

Beklenen: G(T) = 1/sqrt(pi*T); ölçülmüş performans (2026-07-03): n_x=200'de maks
bağıl hata %0.0155, gözlenen uzaysal mertebe p=1.97. Ders: hedef-T'de lineer
interpolasyon şart (O(dT) zaman-kayması mertebe ölçümünü kirletir).
Oracle: boyutlu Cottrell physics_verify 3/3 (KB: electrochemistry/cottrell-equation).
"""
import math

from tarhan.numerics.diffusion1d import cottrell_fd_samples
from tarhan.physics import cottrell_dimensionless

TARGETS = [0.1 * k for k in range(1, 11)]


def _max_rel_err(n_x: int) -> float:
    samples, _, _ = cottrell_fd_samples(TARGETS, n_x=n_x)
    return max(abs(samples[t] - cottrell_dimensionless(t)) / cottrell_dimensionless(t)
               for t in TARGETS)


def _err_at(T: float, n_x: int) -> float:
    samples, _, _ = cottrell_fd_samples([T], n_x=n_x)
    return abs(samples[T] - cottrell_dimensionless(T))


def test_accuracy_nx200():
    # ölçülen 1.55e-4; tolerans 2x pay ile
    assert _max_rel_err(200) < 3e-4


def test_spatial_order_is_two():
    p = math.log2(_err_at(1.0, 100) / _err_at(1.0, 200))
    assert 1.6 < p < 2.5, f"gözlenen mertebe {p:.2f}, beklenen ~2"

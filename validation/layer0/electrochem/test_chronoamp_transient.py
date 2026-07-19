"""Transient kronoamperometri (method-of-lines difüzyon + BDF) — Cottrell/erf doğrulaması.

`numerics.transient` primitifinin ilk domain uygulaması (`models.chronoamp1d`).
rank-0 Cottrell'i explicit-FD ile üretti; bu, aynı fiziğin implicit-BDF yolu:
- yüzey akısı Cottrell analitiğine (i=nFAc*√(D/πt)) uzamsal olarak temiz O(h²) yakınsar
  (ölçülen mertebe 2.000; keyfî tolerans yok — rank-4 dersi);
- derişim profili yarı-sonsuz erf çözümüyle eşleşir;
- implicit avantaj: BDF adım sayısı explicit CFL gereksiniminin (dt<dx²/2D) çok altında
  (sabit Jacobian bir kez değerlendirilir).
"""
import math

import numpy as np
import pytest

from tarhan.models.chronoamp1d import chronoamperogram, cottrell_current
from tarhan.numerics.verify import convergence_rates

D, C_BULK = 1.0, 1.0                 # normalize (D=1, c*=1) — mertebe/erf temiz görünür


def _current_relerr(n_x, t=0.3):
    r = chronoamperogram(D, C_BULK, [t], n_x=n_x, rtol=1e-11, atol=1e-13)
    ref = cottrell_current(t, D, C_BULK)
    return abs(r["current"][0] - ref) / ref


def test_cottrell_current_spatial_order_two():
    """Yüzey akısı Cottrell analitiğine uzamsal O(h²) yakınsar (ölçülen 2.000)."""
    grids = [100, 200, 400, 800]
    errs = [_current_relerr(n) for n in grids]
    orders = convergence_rates(errs, [1.0 / n for n in grids])
    assert errs[-1] < errs[0]
    assert abs(orders[-1] - 2.0) < 0.03, f"orders={orders}"


def test_concentration_profile_matches_erf():
    """Derişim profili yarı-sonsuz analitik c(x,t)=c*·erf(x/2√(Dt)) ile eşleşir."""
    t = 0.5
    r = chronoamperogram(D, C_BULK, [t], n_x=400, rtol=1e-11, atol=1e-13)
    x = r["x"]
    c_exact = np.array([C_BULK * math.erf(xi / (2.0 * math.sqrt(D * t))) for xi in x])
    assert float(np.max(np.abs(r["profile_final"] - c_exact))) < 1e-4


def test_current_matches_cottrell_across_times():
    """Birden çok t'de akım Cottrell'e yakın (n_x=400'de ölçülen ≤1e-3, t≥0.05)."""
    ts = [0.05, 0.1, 0.3, 0.6, 1.0]
    r = chronoamperogram(D, C_BULK, ts, n_x=400, rtol=1e-10, atol=1e-12)
    for t, i_num in zip(ts, r["current"]):
        i_ref = cottrell_current(t, D, C_BULK)
        assert abs(i_num - i_ref) / i_ref < 1e-3
    # akım monoton azalır (t^{-1/2})
    assert all(i2 < i1 for i1, i2 in zip(r["current"], r["current"][1:]))


def test_implicit_advantage_over_explicit_cfl():
    """BDF'in KABUL ETTİĞİ adım sayısı, explicit'in CFL kısıtıyla gereksineceğinin
    altında — stiff MOL sisteminde implicit yöntemin varlık nedeni.

    DÜZELTME (2026-07-19 tam tolerans denetimi): bu test eskiden `r["nsteps"]`
    kullanıyordu, ama o alan ÇIKTI noktası sayısıydı (t_eval tek nokta) → assertion
    `1 < 88.9`'a indirgeniyordu, yani BOŞTU ve "≥100×" iddiası hiçbir şey ölçmüyordu.
    Artık gerçek kabul-edilen adım sayısı ölçülüyor (count_steps=True) ve eşik
    ÖLÇÜLEN orana bağlı: 8889/483 = 18.4×.
    """
    r = chronoamperogram(D, C_BULK, [1.0], n_x=400, rtol=1e-9, atol=1e-11)
    steps = r["n_accepted_steps"]
    assert steps is not None and steps > 1          # gerçekten adım sayısı (çıktı değil)
    explicit_steps_needed = 1.0 / r["cfl_explicit_dt"]      # t_max / (dx²/2D)
    ratio = explicit_steps_needed / steps
    assert ratio > 10.0, f"ölçülen oran {ratio:.1f}× (referans ölçüm 18.4×)"
    assert r["njev"] <= 3                                     # sabit Jacobian ~1 kez


def test_input_guards():
    with pytest.raises(ValueError):
        chronoamperogram(-1.0, C_BULK, [0.5])               # D<=0
    with pytest.raises(ValueError):
        chronoamperogram(D, C_BULK, [0.0])                  # t<=0 (Cottrell ıraksar)
    with pytest.raises(ValueError):
        chronoamperogram(D, C_BULK, [])                     # boş t_eval
    with pytest.raises(ValueError):
        cottrell_current(0.0, D, C_BULK)                    # t=0

"""Çapraz-oracle: TARHAN pn1d vs DEVSIM (bağımsız TCAD, Apache-2.0, pip).

Faz-1 kabul kriteri "çapraz-kod oracle'a 1e-3 bağıl" (roadmap'te Sesame'ye karşı
yazılmıştı; usnistgov/sesame uzun süredir güncellenmiyor — son commit 2025-08-29,
arşivlenmemiş — DEVSIM canlı ve üretim-sınıfı, daha güçlü oracle).

Ölçümler (2026-07-31, DEVSIM 2.10.0, yeniden koşuldu):
  V_bi farkı 0.5730 µV (0.7156440199 vs 0.7156434469)
  akım oranları R=0 : 0.999916 / 0.999994 / 1.000004 / 1.000008  (|sapma| <= 8.4e-5)
  akım oranları SRH : 1.002728 / 1.002479 / 1.001909 / 1.001008  (|sapma| <= 2.7e-3)

DÜZELTME (2026-07-31): bu docstring önceki hâlinde SRH oranlarını "1.0005-1.0016"
diye veriyordu. Testler yalnızca sınır doğruladığı için sayılar hiç basılmıyordu ve
hata görülmemişti; compare() doğrudan koşulunca gerçek sapmanın 2.7e-3 olduğu
görüldü — eski değer gerçek sapmayı OLDUĞUNDAN KÜÇÜK gösteriyordu. R=0 tarafı ise
tersine, iddia edilenden daha iyi çıktı.

Kurulum: pip install "tarhan[oracle]".
DEVSIM kurulu değilse test atlanır (CI'da opsiyonel iş).
"""
import pytest

devsim = pytest.importorskip("devsim", reason="çapraz-oracle: pip install tarhan[oracle]")

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "oracles"))
from devsim_pn1d_compare import compare  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return compare(quiet=True)


def test_builtin_potential_cross_code(results):
    for cfg in results.values():
        assert abs(cfg["vbi_tarhan"] - cfg["vbi_devsim"]) < 5e-6      # ölçüm: 0.5730 µV


def test_r0_currents_within_1e3(results):
    for v, r in results["R0_kisa_taban"]["ratios"].items():
        assert abs(r - 1.0) < 1e-3, f"V={v}: oran {r:.5f}"            # ölçüm: ≤8.4e-5 (12× pay)


def test_srh_currents_within_3e3(results):
    for v, r in results["SRH_iki_rejim"]["ratios"].items():
        # UYARI: ölçüm 2.7e-3, sınır 3e-3 — yalnızca %9 pay. Bu eşik İNCE.
        # README'nin "measured-margin" sınıfına girer ama payı dar; DEVSIM sürümü
        # veya BLAS değişirse kırılabilir. Genişletilmeden önce sapmanın nereden
        # geldiği anlaşılmalı (ağ mı, SRH ayrıklaştırması mı) — körlemesine gevşetme.
        assert abs(r - 1.0) < 3e-3, f"V={v}: oran {r:.5f}"            # ölçüm: ≤2.7e-3

"""Çapraz-oracle: TARHAN pn1d vs DEVSIM (bağımsız TCAD, Apache-2.0, pip).

Faz-1 kabul kriteri "çapraz-kod oracle'a 1e-3 bağıl" (roadmap'te Sesame'ye karşı
yazılmıştı; Sesame ölü — DEVSIM canlı ve üretim-sınıfı, daha güçlü oracle).
Ölçümler (2026-07-03, DEVSIM 2.10.0): V_bi farkı 0.6 µV; akım oranları
R=0: 1.0002-1.0004, SRH: 1.0005-1.0016. Kurulum: pip install "tarhan[oracle]".
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
        assert abs(cfg["vbi_tarhan"] - cfg["vbi_devsim"]) < 5e-6      # ölçüm: 0.6 µV


def test_r0_currents_within_1e3(results):
    for v, r in results["R0_kisa_taban"]["ratios"].items():
        assert abs(r - 1.0) < 1e-3, f"V={v}: oran {r:.5f}"            # ölçüm: ≤4e-4


def test_srh_currents_within_3e3(results):
    for v, r in results["SRH_iki_rejim"]["ratios"].items():
        assert abs(r - 1.0) < 3e-3, f"V={v}: oran {r:.5f}"            # ölçüm: ≤1.6e-3

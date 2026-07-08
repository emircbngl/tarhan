"""pn1d grid-sağlamlığı: mesh-bağımsızlık bandı + FV eklem-doping regresyon koruması.

GRID-YAKINSAMA ÇALIŞMASI BULGULARI (2026-07-04, dürüst kayıt):
1. Eklem DÜĞÜMÜNE tam-Nd doping atamak baskın O(h) hata terimiydi — FV yarı-hücre
   düzeltmesi ((Nd−Na)/2) uniform 10→5 nm J-farkını ~10× küçülttü (1.3e-8→1.7e-9).
2. J-tabanlı öz-yakınsama MERTEBE ölçümü ~1e-4-bağıl altında Gummel sabit-nokta
   gürültü tabanına çarpar (tol-yolu değişince J ~1e-4-bağıl oynar; tol'u sıkmak
   ince gridde yuvarlama platosuna takılır) — bu bir DIŞ-İTERASYON artefaktı,
   ayrıklaştırma özelliği DEĞİL. TEMİZ MERTEBE MMS ile ölçüldü (2026-07-09,
   `test_pn1d_mms_order.py`): izole `_poisson_newton` ve `_continuity_solve`
   operatörleri her ikisi de O(h²) (2.000) — makine-hassasiyetine çözüldüklerinden
   Gummel tabanı atlatıldı. KALAN dar kalem: tam-kuplajlı SOLVE mertebesi coupled-Newton
   ister. (Keyfî mertebe İDDİA EDİLMEDİ; rank-4 dersinin devamı.)
3. Bugün dürüstçe sabitlenen: MESH-BAĞIMSIZLIK BANDI — pratik grid ailesi boyunca
   J'nin bağıl yayılımı küçük (ölçüm: graded-baseline vs uniform 10/5 nm ≤ ~6e-5).
"""
import numpy as np
import pytest

from tarhan.models.pn1d import PNDiode1D, iv_sweep


def _j03(h0, gamma):
    seq, _ = iv_sweep(PNDiode1D(h0=h0, gamma=gamma), [0.1, 0.2, 0.3],
                      gummel_tol=1e-10, max_gummel=200)
    return seq[-1]


@pytest.fixture(scope="module")
def j_family():
    return {
        "graded_baseline": _j03(5e-7, 1.06),
        "uniform_10nm": _j03(1e-6, 1.0),
        "uniform_5nm": _j03(5e-7, 1.0),
    }


def test_mesh_independence_band(j_family):
    js = np.array(list(j_family.values()))
    spread = float((js.max() - js.min()) / js.mean())
    assert spread < 3e-4, f"mesh-bağımsızlık bandı aşıldı: {spread:.2e} ({j_family})"


def test_fv_junction_doping_regression_guard(j_family):
    """Yarı-hücre düzeltmesinin geri kaçmasına karşı koruma: uniform 10→5 nm
    J-farkı düzeltme-öncesi seviyeye (1.3e-8) dönerse FAIL."""
    d = abs(j_family["uniform_10nm"] - j_family["uniform_5nm"])
    assert d < 6e-9                                   # ölçüm: 1.7e-9; eski kod: 1.3e-8

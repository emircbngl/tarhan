"""Kesirli hesap: Grünwald-Letnikov yarı-integrasyon.

Kaynak (yöntem aktarımı): Oldham, Myland & Bond, Electrochemical Science and
Technology (Wiley 2012). Ölçülmüş davranış (Layer-0, rank-4): t^(-1/2) tekilliği
altında yakınsama mertebesi 0.50 — bu yüzden sonuçlar keyfî eşikle değil,
mertebe-ölçümü + Richardson ekstrapolasyonuyla değerlendirilir.
"""
from __future__ import annotations

import math

from tarhan import backend


def grunwald_weights(k: int):
    """g_0 = 1, g_j = g_{j-1}·(j − 1/2)/j — yarı-integral konvolüsyon ağırlıkları."""
    xp = backend.xp()
    g = xp.empty(k)
    g[0] = 1.0
    for j in range(1, k):
        g[j] = g[j - 1] * (j - 0.5) / j
    return g


def semi_integrate(samples, dt: float) -> float:
    """d^{-1/2} operatörü: M_k = sqrt(dt) · Σ_{j} g_j · i_{k−j}.

    ``samples[j]`` = i(t_{j+1}) (uniform dt). Döner: M(t_k), k = len(samples).
    """
    xp = backend.xp()
    s = xp.asarray(samples, dtype=float)
    g = grunwald_weights(len(s))
    return math.sqrt(dt) * float(xp.dot(g, s[::-1]))

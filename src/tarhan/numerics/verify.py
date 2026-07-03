"""Doğrulama yardımcıları: gözlenen yakınsama mertebesi.

Kaynak (yöntem aktarımı): Linge & Langtangen, Finite Difference Computing with
PDEs (CC BY 4.0), §1.1.4 / §3.6.6. Kural: "yakınsamış görünüyor" yok — mertebe
TABLOLANIR (kuruluş ilkesi #4-5 ile aynı ruh; rank-4 dersi: keyfî eşik yerine
mertebeyi ölç, limiti ekstrapole et).
"""
from __future__ import annotations

import math


def convergence_rates(errors, steps):
    """r_i = ln(E_{i−1}/E_i) / ln(h_{i−1}/h_i) — ardışık gözlenen mertebeler."""
    return [math.log(errors[i - 1] / errors[i]) / math.log(steps[i - 1] / steps[i])
            for i in range(1, len(errors))]


def richardson_extrapolate(m_coarse: float, m_fine: float, ratio: float, order: float) -> float:
    """Bilinen mertebeyle limit kestirimi: M_ext = (r^p·M_f − M_c)/(r^p − 1)."""
    rp = ratio ** order
    return (rp * m_fine - m_coarse) / (rp - 1.0)

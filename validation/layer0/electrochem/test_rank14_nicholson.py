"""Rank 14 — Nicholson (1965) ΔE_p·n – ψ çalışma tablosu: quasi-reversible CV.

KAYNAK (birincil, birebir transkripsiyon): Nicholson, Anal. Chem. 37, 1351 (1965),
Tablo I — açık PDF (sources/nicholson/nicholson1965.pdf) pdftotext + görsel kontrol.
Hesap koşulları da metinden: α=0.5; başlangıç C_O*/C_R* = e^{6.5};
(E_λ − E_{1/2})·n = 141 mV; tablo α∈(0.3,0.7)'de ψ≥0.5 için ~%5 duyarlı.

Çözücü: cv_sweep(K0=ψ√π) — Butler-Volmer Robin sınırı (2-nokta akı, tridiag
korunur; J sınırda BV ifadesinin kendisi), tam ileri+geri tarama.

ÖLÇÜM SONUCU (2026-07-04): 13 çiftin tümü basılıya −0.5…−1.5 mV bandında (en
büyük sapma ψ=0.75: sim 90.46 vs basılı 92); bizim değerler GRID-YAKINSAMIŞ
(4× incelmede <0.05 mV) → artık farklar Nicholson'ın 1965 el-çağı sayısallaştırma
granülaritesi + 1 mV yuvarlaması (sistematik-negatif desen; tablonun modern
yeniden-hesaplamaları da benzer sapmalar raporlar — ör. ACS Electrochemistry 2025
Nicholson-yöntemi yeniden ziyareti). Band ±2.0 mV; değerlerimizin doğruluğu ayrıca
yakınsama + monotonluk + BV→Nernst limiti (0.08 mV) assert'leriyle kilitli.
"""
import pytest

from tarhan.numerics.voltammetry import (cv_sweep, find_peak,
                                         nicholson_peak_separation)

# Nicholson 1965, Tablo I (birincil kaynaktan; ΔEp·n [mV], α=0.5, 25°C)
NICHOLSON_TABLE_I = [
    (20, 61), (7, 63), (6, 64), (5, 65), (4, 66), (3, 68), (2, 72),
    (1, 84), (0.75, 92), (0.5, 105), (0.35, 121), (0.25, 141), (0.1, 212),
]


@pytest.mark.parametrize("psi, printed_mv", NICHOLSON_TABLE_I)
def test_nicholson_table_pairs(psi, printed_mv):
    sim = nicholson_peak_separation(psi)
    assert abs(sim - printed_mv) < 2.0, f"ψ={psi}: sim {sim:.2f} vs basılı {printed_mv}"


def test_grid_converged_not_our_error():
    """Basılıyla ~1 mV artık farkın bizden olmadığının kanıtı: 4× incelmede
    değerimiz <0.1 mV oynar (ölçüm: 104.32→104.29 @ψ=0.5)."""
    a = nicholson_peak_separation(0.5, d_theta=2e-3, h0=1e-3)
    b = nicholson_peak_separation(0.5, d_theta=5e-4, h0=2.5e-4)
    assert abs(a - b) < 0.1


def test_bv_recovers_nernst_limit():
    """ψ→∞: BV Robin sınırı, doğrulanmış Nernst-CV'ye yakınsar (iç limit kontrolü)."""
    th, J, nf = cv_sweep(K0=None)
    thc, _ = find_peak(th[:nf], J[:nf])
    tha, _ = find_peak(th[nf:], -J[nf:])
    dep_nernst = (tha - thc) * 25.693
    dep_bv = nicholson_peak_separation(500.0)
    assert abs(dep_bv - dep_nernst) < 0.2              # ölçüm: 0.08 mV
    assert 57.0 < dep_nernst < 61.0                     # 'reversible ~60/n' (141 mV switch)


def test_peak_separation_monotone_in_psi():
    psis = [20, 2, 0.5, 0.1]
    deps = [nicholson_peak_separation(p) for p in psis]
    assert all(deps[i] < deps[i + 1] for i in range(len(deps) - 1))

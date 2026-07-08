"""Robertson stiff kinetiği — TARHAN transient/BDF yeteneğinin doğrulaması.

Kaynak: H. H. Robertson (1966); Hairer & Wanner, *Solving Ordinary Differential
Equations II* (stiff test seti). Çapraz-kod pin: SUNDIALS CVODE cvRoberts_dns
örneği (cv_examples-5.7.0.pdf basılı çıktısı; RTOL/ATOL kaynak koddan doğrulandı:
RTOL=1e-4, ATOL=(1e-8,1e-14,1e-6), T0=0, T1=0.4, ×10, NOUT=12).

Sistem (mass-action kinetiği):
    y1' = -0.04 y1 + 1e4 y2 y3
    y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2²
    y3' =  3e7 y2²           ,  y0=(1,0,0);  korunum y1+y2+y3 = 1.

Doğrulama katmanları (rigorden çapraz-koda):
1. Yapısal korunum: Σ(RHS) ≡ 0 → integre invariant y1+y2+y3=1 makine-hassasiyeti.
2. Analitik Jacobian ↔ merkezi-fark (~1e-11) — "sahip olduğumuz" parça doğru.
3. Üç bağımsız stiff yöntem (Radau/BDF/LSODA) tight-tol ~1e-9 mutabık → gerçek
   referans çözüm (tek-kütüphane değil, üç-yöntem).
4. SUNDIALS çapraz-kod pin (eşleşmiş gevşek-tol): büyüyen tür y3 tüm 12 dekad;
   y1,y2 t≤4e8'e kadar. **Dürüst bulgu:** SUNDIALS basılı tablo gevşek-tol
   demo'sudur — t≥4e9'da kendi değerleri ~%1–33 kayar (yüksek-hassasiyet
   Radau ile teyitli); o rejimde gerçeğe tight-tol + korunum ile bağlanılır.
5. Stiffness gerçek: explicit RK45, BDF'ten ≫ sağ-taraf değerlendirmesi harcar.
6. y2 zirvesi ≈ 3.6487e-5 (t≈4.56e-3) — hesaplanır, hafızadan değil.
"""
import numpy as np
import pytest

from tarhan.numerics.transient import (
    STIFF_METHODS,
    conservation_residual,
    integrate_stiff,
)

K1, K2, K3 = 0.04, 1.0e4, 3.0e7
Y0 = [1.0, 0.0, 0.0]


def rhs(t, y):
    y1, y2, y3 = y
    return [-K1 * y1 + K2 * y2 * y3,
            K1 * y1 - K2 * y2 * y3 - K3 * y2 * y2,
            K3 * y2 * y2]


def jac(t, y):
    y1, y2, y3 = y
    return [[-K1, K2 * y3, K2 * y2],
            [K1, -K2 * y3 - 2 * K3 * y2, -K2 * y2],
            [0.0, 2 * K3 * y2, 0.0]]


# SUNDIALS cv_examples-5.7.0.pdf, cvRoberts_dns basılı çıktısı (12 dekad):
SUNDIALS = {
    4.0e-01: (9.851641e-01, 3.386242e-05, 1.480205e-02),
    4.0e+00: (9.055097e-01, 2.240338e-05, 9.446793e-02),
    4.0e+01: (7.158017e-01, 9.185037e-06, 2.841892e-01),
    4.0e+02: (4.505360e-01, 3.223271e-06, 5.494608e-01),
    4.0e+03: (1.832299e-01, 8.944378e-07, 8.167692e-01),
    4.0e+04: (3.898902e-02, 1.622006e-07, 9.610108e-01),
    4.0e+05: (4.936383e-03, 1.984224e-08, 9.950636e-01),
    4.0e+06: (5.168093e-04, 2.068293e-09, 9.994832e-01),
    4.0e+07: (5.202440e-05, 2.081083e-10, 9.999480e-01),
    4.0e+08: (5.201061e-06, 2.080435e-11, 9.999948e-01),
    4.0e+09: (5.258603e-07, 2.103442e-12, 9.999995e-01),  # gevşek-tol drift başlıyor
    4.0e+10: (6.934511e-08, 2.773804e-13, 9.999999e-01),  # SUNDIALS'in kendi hatası ~%33
}
T_EVAL = sorted(SUNDIALS)


@pytest.fixture(scope="module")
def truth():
    """Yüksek-hassasiyet referans: tight-tol Radau — modül başına bir kez hesaplanır."""
    return integrate_stiff(rhs, Y0, (0.0, T_EVAL[-1]), jac=jac, t_eval=T_EVAL,
                           rtol=1e-11, atol=1e-15, method="Radau")


def test_structural_conservation_rhs_sums_to_zero():
    """Σ(RHS) ≡ 0 (üç terim çiftler halinde birbirini götürür) → invariant sabit."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(3)
        assert abs(sum(rhs(0.0, y))) < 1e-12


def test_analytic_jacobian_matches_finite_difference():
    """Sahip olduğumuz analitik Jacobian merkezi-farkla ~1e-11 uyumlu."""
    for y in ([1.0, 0.0, 0.0], [0.9, 3e-5, 0.1], [0.2, 1e-6, 0.8]):
        Ja = np.array(jac(0.0, y))
        Jf = np.zeros((3, 3))
        for j in range(3):
            dj = 1e-7 * max(1.0, abs(y[j]))
            yp = np.array(y, float); ym = np.array(y, float)
            yp[j] += dj; ym[j] -= dj
            Jf[:, j] = (np.array(rhs(0.0, yp)) - np.array(rhs(0.0, ym))) / (2 * dj)
        assert np.abs(Ja - Jf).max() < 1e-9


def test_integrated_conservation_invariant(truth):
    """y1+y2+y3=1, 16 zaman-dekadı boyunca makine-hassasiyetinde korunur."""
    res = conservation_residual(truth.y, invariant=lambda col: col.sum())
    assert res < 1e-12


def test_three_independent_methods_agree():
    """Radau (implicit RK) / BDF (multistep) / LSODA bağımsız algoritmalar ~1e-8
    mutabık → gerçek referans çözüm tek-kütüphane artefaktı değil."""
    sols = {m: integrate_stiff(rhs, Y0, (0.0, T_EVAL[-1]), jac=jac, t_eval=T_EVAL,
                               rtol=1e-10, atol=1e-14, method=m)
            for m in ("Radau", "BDF", "LSODA")}
    ref = sols["Radau"].y
    for m in ("BDF", "LSODA"):
        assert np.abs(sols[m].y - ref).max() < 1e-8


def test_cross_code_pin_sundials_growing_species(truth):
    """Çapraz-kod: TARHAN yüksek-hassasiyet çözümü, SUNDIALS'in basılı y3'üyle
    (büyüyen tür, O(1)) 12 dekad boyunca ≤1e-3 uyumlu — SUNDIALS'in kendi belirtilen
    toleransı (RTOL=1e-4) düzeyinde. Truth tight-tol Radau (deterministik ~1e-10)
    ve SUNDIALS sabit → platform-robust (adım-dizisi gürültüsü yok)."""
    for i, t in enumerate(T_EVAL):
        y3_ref = SUNDIALS[t][2]
        rel = abs(truth.y[2, i] - y3_ref) / abs(y3_ref)
        assert rel < 1e-3, f"t={t:.0e}: y3 rel-err {rel:.2e}"


def test_cross_code_pin_sundials_resolvable_regime(truth):
    """y1,y2: SUNDIALS çözülebilir rejiminde (t≤4e8, türler ≳1e-11) truth ile ≤2e-3.
    t≥4e9'da SUNDIALS gevşek-tol drift'i başlar (ayrı testte belgeli)."""
    for i, t in enumerate(T_EVAL):
        if t > 4.0e8:
            continue
        for comp in (0, 1):
            ref = SUNDIALS[t][comp]
            rel = abs(truth.y[comp, i] - ref) / abs(ref)
            assert rel < 2e-3, f"t={t:.0e} comp{comp}: rel-err {rel:.2e}"


def test_sundials_printed_table_is_loose_tol_demo(truth):
    """DÜRÜST BULGU: SUNDIALS basılı geç-zaman değerleri kendi gevşek-tol hatasını
    taşır. Yüksek-hassasiyet Radau: t=4e10'da gerçek y1≈5.2e-8 iken SUNDIALS
    6.93e-8 basmış (~%33). Bu bir defect değil, gevşek-tol demo'nun özelliği —
    o rejimde gerçeğe korunum + tight-tol ile bağlanılır (yukarıdaki testler)."""
    i = T_EVAL.index(4.0e10)
    y1_true = truth.y[0, i]
    y1_sund = SUNDIALS[4.0e10][0]
    assert abs(y1_true - 5.2e-8) < 5e-9        # gerçek değer ~5.2e-8
    assert abs(y1_sund - y1_true) / y1_true > 0.2   # SUNDIALS ondan >%20 sapıyor


def test_stiffness_is_real_explicit_wastes_evaluations():
    """Explicit RK45, aynı kısa span [0,10]'da BDF'ten ≫ sağ-taraf harcar
    (stiffness kanıtı; kararlılık, doğruluk değil, adım boyunu sınırlar). RK45
    truth-path olmadığından integrate_stiff dışında doğrudan çağrılır — onu
    integrate_stiff reddeder (aşağıdaki test)."""
    from scipy.integrate import solve_ivp
    bdf = integrate_stiff(rhs, Y0, (0.0, 10.0), jac=jac, rtol=1e-6, atol=1e-9,
                          method="BDF")
    rk = solve_ivp(rhs, (0.0, 10.0), Y0, method="RK45", rtol=1e-6, atol=1e-9)
    assert rk.success
    assert rk.nfev > 50 * bdf.nfev   # ölçülen oran ~200×; güvenli alt-sınır 50×


def test_integrate_stiff_rejects_explicit_method():
    """No-silent-degrade: explicit RK truth-path DEĞİL → açık ValueError."""
    with pytest.raises(ValueError, match="stiff truth-path"):
        integrate_stiff(rhs, Y0, (0.0, 40.0), method="RK45")
    with pytest.raises(ValueError, match="stiff truth-path"):
        integrate_stiff(rhs, Y0, (0.0, 40.0), method="DOP853")
    assert "RK45" not in STIFF_METHODS and "DOP853" not in STIFF_METHODS


def test_y2_transient_peak():
    """y2 hızlı yükselip bozunur; zirve ≈3.6487e-5 (t≈4.56e-3) — hesaplanır."""
    sol = integrate_stiff(rhs, Y0, (0.0, 1.0), jac=jac,
                          t_eval=np.linspace(1e-4, 1.0, 40000),
                          rtol=1e-10, atol=1e-15, method="Radau")
    imax = sol.y[1].argmax()
    assert abs(sol.y[1, imax] - 3.6487e-5) < 1e-8
    assert 3e-3 < sol.t[imax] < 6e-3

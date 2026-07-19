"""pn1d — MMS uzamsal-mertebe doğrulaması (Faz-1 açık kalemini kapatır).

Bağlam (grid-yakınsama çalışması, 2026-07-04): amiral geminin J-tabanlı öz-yakınsama
MERTEBE ölçümü ~1e-4-bağıl altında **Gummel sabit-nokta gürültü tabanına** çarpıyordu
(dış iterasyon artefaktı, ayrıklaştırma özelliği DEĞİL) → "temiz mertebe = coupled-Newton
veya gerçek MMS" açık kalemi bırakılmıştı, keyfî mertebe iddia edilmemişti.

Bu test o kalemi **MMS yoluyla** kapatır. İçgörü: Gummel gürültü tabanı KUPLAJ
iterasyonundan gelir; iki çekirdek uzamsal operatör tek başına makine-hassasiyetine
çözülür (Poisson-Newton damped-Newton ile tol=1e-13'e; SG-süreklilik DOĞRUDAN lineer
tridiag). Her operatöre ayrı ayrı MMS uygulamak gürültü tabanını tamamen atlatıp
temiz mertebe verir. GERÇEK kod operatörleri (`_poisson_newton`, `_continuity_solve`)
test edilir — reimplementasyon değil.

Yöntem (Langtangen MMS disiplini, rank-3 ile aynı ruh): düz uniform grid'de smooth
manufactured alan seç, SÜREKLİ operatörden kaynağı hesapla, ayrık sistemi çöz,
‖sayısal − manufactured‖∞'yi grid incelmesinde ölç → mertebe `convergence_rates` ile.

SONUÇ (ölçülen, 6 grid 20→640): her iki operatör de TEMİZ **2. mertebe** (son oran
2.000; hata her h-yarılanmasında çeyreğe iner). SG akı şeması ve nonlineer Poisson
ayrıklaştırması O(h²).

KUPLAJLI mertebe (2026-07-09) — KAPSAMI DÜRÜSTÇE SINIRLI (söylem 2026-07-15 triad
review'unda düzeltildi; eski hâli fazla iddialıydı):

  NE ÖLÇÜLÜYOR: TEST-YEREL bir (ψ,n,p) DD rezidüeli — üretimden yalnız `bernoulli` SG
  kernel'i DOĞRUDAN paylaşılır; Laplacian stencil'i, diverjans, ölçekleme ve akı
  işaretleri üretim konvansiyonlarının TEST-İÇİNDE YENİDEN İFADESİDİR (aynı fonksiyon
  DEĞİL). Buna manufactured kaynaklar eklenir ve coupled-Newton (scipy.optimize.root,
  makine-hassasiyetine) ile çözülür → temiz O(h²), son oran 2.000. Yani: üretim
  konvansiyonlarını yeniden üreten bir ayrıklaştırma, kuplajlı ve makine-hassasiyetli
  çözüldüğünde O(h²). ("Motorun AYRIKLAŞTIRMASI" demek FAZLA GÜÇLÜ olurdu — bunu
  2026-07-15 bağımsız review yakaladı.)

  NE ÖLÇÜLMÜYOR: motorun kendi kuplajlı SOLVE'u — çünkü `solve_bias`'ta kuplajlı çözüm
  yolu YOKTUR (Gummel sırası koşar). Buradaki `_coupled_residual` bir TEST KURGUSUDUR;
  kaynak-enjeksiyon yolunun (`sn`/`sp`) motorda karşılığı yoktur.

  KANIT GÜCÜ: aşağıdaki cross-check rezidüeli motorun gerçek Gummel çözümünde
  değerlendirir (‖F‖~1e-14) — bu, işaret/ölçek konvansiyonlarının eşleştiğine dair
  GÜÇLÜ KANIT ama KİMLİK İSPATI DEĞİL: tek bir durumda (denge, V=0, kaynak=0) yapılır;
  iki farklı fonksiyon bir noktada uyuşabilir.

  Gummel gürültü tabanı bu yolda atlatılır (root rezidüeli ~1e-13'e indirir) — ama bu,
  üretimdeki Gummel'in tabanını kaldırmaz. Mimari: rezidüel bizim, Newton çözümü scipy'a
  delege (transient/BDF ile aynı çizgi). Üretim-sınıfı coupled-Newton MOTOR MODU ayrı ve
  açık bir özelliktir.
"""
import numpy as np
import pytest
from scipy.optimize import root

from tarhan.models.pn1d import (
    PNDiode1D,
    _continuity_solve,
    _poisson_newton,
    solve_bias,
)
from tarhan.numerics.flux import bernoulli
from tarhan.numerics.verify import convergence_rates

GRIDS = [20, 40, 80, 160, 320, 640]


# ----------------------------------------------------------- SG süreklilik MMS
_L, _K, _A, _B, _C, _D, _MU = 1.0, 2.0 * np.pi, 0.3, 0.5, 2.0, 0.5, 1.0


def _psi(x):    return _A * np.sin(_K * x) + _B * x
def _n(x):      return _C + _D * np.cos(_K * x)               # >0: [1.5, 2.5]
def _dn(x):     return -_D * _K * np.sin(_K * x)
def _d2n(x):    return -_D * _K * _K * np.cos(_K * x)
def _dpsi(x):   return _A * _K * np.cos(_K * x) + _B
def _d2psi(x):  return -_A * _K * _K * np.sin(_K * x)


def _sg_source(x):
    """Sürekli ölçekli elektron akısı Ĵn = μ(n' − n·ψ'); kaynak S = Ĵn'."""
    return _MU * (_d2n(x) - _dn(x) * _dpsi(x) - _n(x) * _d2psi(x))


def _sg_continuity_error(N):
    x = np.linspace(0.0, _L, N + 1)
    hbar = 0.5 * ((x[1:-1] - x[:-2]) + (x[2:] - x[1:-1]))     # kontrol-hacmi genişliği
    # operatör A = −div; sürekli denklem div(Ĵn)=S ⇒ rhs = −S·h̄ (işaret ampirik kilitli)
    src_const = -_sg_source(x[1:-1]) * hbar
    n_h = _continuity_solve(None, x, _psi(x), "n", _n(x[0]), _n(x[-1]),
                            mu_hat=_MU, src_lin=np.zeros(N - 1), src_const=src_const)
    return float(np.max(np.abs(n_h - _n(x))))


def test_sg_continuity_is_second_order():
    """SG-ayrık süreklilik operatörü uniform grid'de temiz O(h²) — doğrudan lineer
    çözüm, Gummel gürültü tabanı YOK."""
    errs = [_sg_continuity_error(N) for N in GRIDS]
    orders = convergence_rates(errs, [_L / N for N in GRIDS])
    assert errs[-1] < errs[0]                      # gerçekten yakınsıyor
    assert abs(orders[-1] - 2.0) < 0.02, f"orders={orders}"


# ----------------------------------------------------------- Poisson-Newton MMS
_PK, _DELTA = 2.0 * np.pi, 1e-4


def _psi_star(x):    return 0.5 * np.sin(_PK * x) + 0.2 * x
def _d2psi_star(x):  return -0.5 * _PK * _PK * np.sin(_PK * x)


class _MMSDev:
    """MMS-doping stub: ψ''=n̂−p̂−N̂ ⇒ N̂ = n̂*−p̂*−ψ*'' seçilir (φn=φp=0). Böylece
    manufactured ψ* nonlineer Poisson'un TAM çözümü olur; ayrık ψ_h ondan O(h²) sapar.
    `_poisson_newton` yalnız dev.doping_hat ve dev.delta kullanır."""

    def __init__(self, xnodes):
        self.delta = _DELTA
        nh = _DELTA * np.exp(_psi_star(xnodes))
        ph = _DELTA * np.exp(-_psi_star(xnodes))
        self._N = nh - ph - _d2psi_star(xnodes)

    def doping_hat(self, x_hat):
        return self._N


def _poisson_error(N):
    x = np.linspace(0.0, _L, N + 1)
    dev = _MMSDev(x)
    phi = np.zeros(N + 1)
    psi = np.zeros(N + 1)
    psi[0], psi[-1] = _psi_star(x[0]), _psi_star(x[-1])       # Dirichlet BC = tam
    psi, _ = _poisson_newton(dev, x, psi, phi, phi, tol=1e-13, max_iter=200)
    return float(np.max(np.abs(psi - _psi_star(x))))


def test_poisson_newton_is_second_order():
    """Tam nonlineer Poisson-Newton (gerçek `_poisson_newton`) manufactured çözüme
    temiz O(h²) yakınsar; Newton makine-hassasiyetine iner (gürültü tabanı YOK)."""
    errs = [_poisson_error(N) for N in GRIDS]
    orders = convergence_rates(errs, [_L / N for N in GRIDS])
    assert errs[-1] < errs[0]
    assert abs(orders[-1] - 2.0) < 0.02, f"orders={orders}"


def test_mms_source_sign_is_load_bearing():
    """Regresyon bekçisi: kaynak işareti YANLIŞSA yakınsama BOZULUR (bu bir
    gürültü tabanı değil gerçek doğrulama — de-risk'te +işaret sabit ~3.2 hata verdi)."""
    N = 160
    x = np.linspace(0.0, _L, N + 1)
    hbar = 0.5 * ((x[1:-1] - x[:-2]) + (x[2:] - x[1:-1]))
    wrong = +_sg_source(x[1:-1]) * hbar                       # ters işaret
    n_h = _continuity_solve(None, x, _psi(x), "n", _n(x[0]), _n(x[-1]),
                            mu_hat=_MU, src_lin=np.zeros(N - 1), src_const=wrong)
    assert float(np.max(np.abs(n_h - _n(x)))) > 1.0           # yakınsamıyor


# --------------------------------------------------- Tam-kuplajlı (ψ,n,p) MMS
# Manufactured smooth kuplajlı alanlar (primitif değişkenler — Boltzmann/δ gerekmez;
# n,p süreklilik denklemlerinin bağımsız bilinmeyenleridir).
def _cpsi(x):    return 0.3 * np.sin(np.pi * x) + 0.2 * x
def _cdpsi(x):   return 0.3 * np.pi * np.cos(np.pi * x) + 0.2
def _cd2psi(x):  return -0.3 * np.pi ** 2 * np.sin(np.pi * x)
def _cn(x):      return 1.5 + 0.4 * np.cos(np.pi * x)           # >0
def _cdn(x):     return -0.4 * np.pi * np.sin(np.pi * x)
def _cd2n(x):    return -0.4 * np.pi ** 2 * np.cos(np.pi * x)
def _cp(x):      return 1.2 + 0.3 * np.sin(np.pi * x + 0.5)     # >0
def _cdp(x):     return 0.3 * np.pi * np.cos(np.pi * x + 0.5)
def _cd2p(x):    return -0.3 * np.pi ** 2 * np.sin(np.pi * x + 0.5)

_CMUN, _CMUP = 1.0, 0.5


def _coupled_residual(u, x, mun, mup, ndop_int, sn_int, sp_int, bc):
    """(ψ,n,p) primitif DD rezidüeli — MOTOR konvansiyonlarıyla birebir: `bernoulli`
    SG kernel'i, `_poisson_newton` Laplacian'ı, `solve_bias` akı işaretleri
    (Ĵn=μ̂n(n_r·B(Δψ)−n_l·B(−Δψ))/h, Ĵp=μ̂p(p_l·B(Δψ)−p_r·B(−Δψ))/h). Süreklilik
    rezidüeli h̄'ye bölünerek Poisson'la aynı (pointwise-PDE) ölçeğe getirilir.
    bc=((ψL,ψR),(nL,nR),(pL,pR)); ndop/sn/sp iç düğüm değerleri."""
    N = len(x)
    ni = N - 2
    h = x[1:] - x[:-1]
    hm, hp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    hbar = 0.5 * (hm + hp)
    (psiL, psiR), (nL, nR), (pL, pR) = bc
    ps = np.empty(N); nn = np.empty(N); pp = np.empty(N)
    ps[0], ps[-1] = psiL, psiR
    nn[0], nn[-1] = nL, nR
    pp[0], pp[-1] = pL, pR
    ps[1:-1], nn[1:-1], pp[1:-1] = u[:ni], u[ni:2 * ni], u[2 * ni:]
    lap = ((ps[2:] - ps[1:-1]) / hp - (ps[1:-1] - ps[:-2]) / hm) / hbar
    f_psi = lap - (nn[1:-1] - pp[1:-1] - ndop_int)
    dps = ps[1:] - ps[:-1]
    Bp, Bm = bernoulli(dps), bernoulli(-dps)
    jn = mun * (nn[1:] * Bp - nn[:-1] * Bm) / h
    jp = mup * (pp[:-1] * Bp - pp[1:] * Bm) / h
    f_n = (jn[1:] - jn[:-1]) / hbar - sn_int
    f_p = (jp[1:] - jp[:-1]) / hbar - sp_int
    return np.concatenate([f_psi, f_n, f_p])


def test_coupled_residual_consistent_with_engine_at_equilibrium():
    """Kuplajlı rezidüel motorun DENGE çözümünde makine-sıfırı vermeli (S=0, R=0).

    KAPSAM (dürüst): bu, işaret/ölçek konvansiyonlarının motorunkiyle eşleştiğine dair
    GÜÇLÜ KANIT — ama KİMLİK İSPATI DEĞİL. Tek bir durumda (denge, V=0, kaynak=0)
    değerlendirilir; iki farklı fonksiyon bir noktada uyuşabilir. Kaynak-enjeksiyon
    yolu (sn/sp) burada hiç sınanmaz (motorda karşılığı yok)."""
    dev = PNDiode1D()
    st = solve_bias(dev, 0.0)                       # denge, en iyi-koşullu
    x, psi, n, p = st["x_hat"], st["psi"], st["n_hat"], st["p_hat"]
    mu_scale = max(dev.mu_n, dev.mu_p)
    ndop = dev.doping_hat(x)
    ni = len(x) - 2
    u = np.concatenate([psi[1:-1], n[1:-1], p[1:-1]])
    bc = ((psi[0], psi[-1]), (n[0], n[-1]), (p[0], p[-1]))
    F = _coupled_residual(u, x, dev.mu_n / mu_scale, dev.mu_p / mu_scale,
                          ndop[1:-1], np.zeros(ni), np.zeros(ni), bc)
    assert float(np.max(np.abs(F))) < 1e-8         # ölçülen ~1e-13


def _coupled_error(N):
    x = np.linspace(0.0, _L, N + 1)
    ni = N - 1
    ndop = _cn(x[1:-1]) - _cp(x[1:-1]) - _cd2psi(x[1:-1])       # S_ψ=0 seçimi
    sn = _CMUN * (_cd2n(x[1:-1]) - _cdn(x[1:-1]) * _cdpsi(x[1:-1])
                  - _cn(x[1:-1]) * _cd2psi(x[1:-1]))            # Ĵn'
    sp = -_CMUP * (_cd2p(x[1:-1]) + _cdp(x[1:-1]) * _cdpsi(x[1:-1])
                   + _cp(x[1:-1]) * _cd2psi(x[1:-1]))           # Ĵp'
    bc = ((_cpsi(x[0]), _cpsi(x[-1])), (_cn(x[0]), _cn(x[-1])), (_cp(x[0]), _cp(x[-1])))
    args = (x, _CMUN, _CMUP, ndop, sn, sp, bc)
    u0 = np.concatenate([_cpsi(x[1:-1]), _cn(x[1:-1]), _cp(x[1:-1])])
    sol = root(_coupled_residual, u0, args=args, method="hybr", tol=1e-13)
    assert sol.success and float(np.max(np.abs(sol.fun))) < 1e-8   # kuplajlı Newton yakınsadı
    ps = np.concatenate([[_cpsi(x[0])], sol.x[:ni], [_cpsi(x[-1])]])
    nn = np.concatenate([[_cn(x[0])], sol.x[ni:2 * ni], [_cn(x[-1])]])
    pp = np.concatenate([[_cp(x[0])], sol.x[2 * ni:], [_cp(x[-1])]])
    return max(float(np.max(np.abs(ps - _cpsi(x)))),
               float(np.max(np.abs(nn - _cn(x)))),
               float(np.max(np.abs(pp - _cp(x)))))


def test_coupled_discretization_order_under_coupled_newton():
    """Üretim konvansiyonlarını yeniden üreten TEST-YEREL (ψ,n,p) ayrıklaştırma, kuplajlı
    ve makine-hassasiyetli çözüldüğünde (coupled-Newton = scipy.root) MMS'te temiz O(h²)
    — kuplaj operatör-mertebesini bozmaz.

    KAPSAM (dürüst): bu, motorun kendi SOLVE'unun mertebesi DEĞİLDİR — `solve_bias`'ta
    kuplajlı çözüm yolu yoktur (Gummel koşar) ve üretimdeki Gummel gürültü tabanı bu
    testle kalkmaz. Ölçülen şey ayrıklaştırmanın mertebesi; test-içi bir rezidüel
    üzerinden (bkz. modül docstring'i)."""
    grids = [20, 40, 80, 160]
    errs = [_coupled_error(N) for N in grids]
    orders = convergence_rates(errs, [_L / N for N in grids])
    assert errs[-1] < errs[0]
    assert abs(orders[-1] - 2.0) < 0.03, f"orders={orders}"

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

Kalan (dürüst): TAM KUPLAJLI çözümün (Poisson⊕süreklilik, Gummel ile bağlı) mertebesi
min(bunlar)=2 ile SINIRLIDIR ama pratikte gerçek-cihaz parametrelerinde Gummel gürültü
tabanıyla (~1e-4-bağıl J) kısıtlı; kuplajlı SOLVE mertebesini o tabanın altında ölçmek
tam-kuplajlı Newton ister — daha dar tanımlı açık kalem.
"""
import numpy as np
import pytest

from tarhan.models.pn1d import _continuity_solve, _poisson_newton
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

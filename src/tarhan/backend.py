"""Array-backend dikişi (seam).

Karar (Product Form Decision + MLX eki, 2026-07-03): hızlandırma backend'i ŞİMDİ
kurulmaz; bu DİKİŞ kurulur. Kurallar:

- **Array-taşınabilir kod** numpy'ı doğrudan import etmez; ``xp = backend.xp()``
  üzerinden aktif array modülüne erişir (flux, diffusion1d, fraccalc, convection,
  voltammetry, models/pn1d böyledir).
- **scipy çağrıları koda saçılmaz**: yalnız ADLANDIRILMIŞ delegasyon noktalarında
  bulunur — backend değişiminin maliyeti modüldür, rewrite değil. Bugünkü noktalar:
    * ``backend.solve_tridiag``              → scipy.linalg.solve_banded (lineer)
    * ``backend.solve_sparse``               → scipy.sparse.linalg.spsolve (2D)
    * ``numerics.transient.integrate_stiff`` → scipy.integrate.solve_ivp (stiff ODE)
- **Delegasyon noktaları numpy/f64-BAĞLIDIR** (scipy öyle). Bu yüzden hem bu modül
  hem `numerics/transient.py` numpy'ı doğrudan import eder — bu bir sızıntı DEĞİL,
  belgelenmiş sınırdır: scipy'a delege edilen kernel'ler bir array-backend'in
  ARDINDA duramaz, dikişin ÜSTÜNDE durur. Bir scipy-delege kernel'i besleyen kod
  (ör. `models/chronoamp1d.py`) da aynı nedenle numpy-bağlıdır; onu xp() üzerinden
  yazmak taşınabilirlik tiyatrosu olurdu (mlx dizisi solve_ivp'ye giremez).
- Doğruluk yolu HER ZAMAN float64. Düşük-hassasiyetli bir backend (ör. MLX: GPU'da
  float64 yok, f64 girdiyi sessizce f32'ye çevirir — ölçüldü, 2026-07-03) yalnız
  "preview" katmanı olabilir; asla truth-path. Sessiz hassasiyet düşüşü yasaktır.

(Kural metni 2026-07-15 triad review'unda DÜZELTİLDİ: eski hâli "çekirdek/numerics
kodu numpy'ı doğrudan import etmez" diyordu, ama transient.py/chronoamp1d.py bunu
çiğniyordu → mimari cümle yanlıştı. Sınır artık olduğu gibi yazılı.)
"""
from __future__ import annotations

import numpy as _np
from scipy.linalg import solve_banded as _solve_banded

_BACKENDS: dict[str, object] = {"numpy": _np}
_ACTIVE = "numpy"


def xp():
    """Aktif array modülü (bugün: numpy; f64 truth-path)."""
    return _BACKENDS[_ACTIVE]


def active_backend() -> str:
    return _ACTIVE


def set_backend(name: str) -> None:
    global _ACTIVE
    if name not in _BACKENDS:
        raise ValueError(
            f"bilinmeyen backend {name!r}; kayıtlı: {sorted(_BACKENDS)}. "
            "Hızlandırma backend'leri (mlx/cupy) v0.2 kapısında değerlendirilecek "
            "— bkz. TARHAN Product Form Decision, MLX eki."
        )
    _ACTIVE = name


def solve_sparse(rows, cols, vals, rhs, n=None):
    """Seyrek lineer sistem çözümü (dikiş noktası, 2D için).

    Girdi bir COO ÜÇLÜSÜ'dür (``rows``/``cols``/``vals``), assembled bir matris
    değil. Sebep DESIGN-2D §4'te yazılı: assembly katmanı, tek tüketicisinin CSR
    bir matris olduğunu varsaymamalı — aynı üçlüler matrix-free bir apply'ı da
    besleyebilir, ve 3D'de GPU seçeneğini açık tutan tam olarak budur. Üçlü
    burada, dikişin ARDINDA CSR'a çevrilir; çağıran bunu bilmez.

    Tekrar eden ``(i, j)`` girdileri scipy tarafından TOPLANIR. Bu bir ayrıntı
    değil: kenar döngüsü aynı düğüm çiftine her komşu kenar için ayrı ayrı
    yazar, ve assembly'nin doğruluğu bu toplama davranışına dayanır.

    Bugünkü implementasyon: scipy.sparse.linalg.spsolve (SuperLU, CPU, float64).
    ~10^5 düğümde darboğaz olması beklenir; UMFPACK ya da ILU'lu iteratif şema O
    ZAMAN, elde profil varken değerlendirilir — profilsiz seçilen bir
    önkoşullayıcı tahmindir.
    """
    from scipy.sparse import coo_matrix as _coo
    from scipy.sparse.linalg import spsolve as _spsolve

    rhs = _np.asarray(rhs, dtype=float)
    if n is None:
        n = int(rhs.shape[0])
    a = _coo((_np.asarray(vals, dtype=float),
              (_np.asarray(rows, dtype=int), _np.asarray(cols, dtype=int))),
             shape=(n, n)).tocsr()
    return _spsolve(a, rhs)


def solve_tridiag(sub, diag, sup, rhs):
    """Tridiagonal sistem çözümü (dikiş noktası).

    ``sub``/``diag``/``sup``/``rhs`` uzunluğu n; konvansiyon: ``sub[0]`` ve
    ``sup[-1]`` kullanılmaz (0 verilebilir). Bugünkü implementasyon:
    scipy.linalg.solve_banded (CPU, float64).
    """
    n = len(diag)
    ab = _np.zeros((3, n))
    ab[0, 1:] = _np.asarray(sup, dtype=float)[:-1]
    ab[1, :] = _np.asarray(diag, dtype=float)
    ab[2, :-1] = _np.asarray(sub, dtype=float)[1:]
    return _solve_banded((1, 1), ab, _np.asarray(rhs, dtype=float))

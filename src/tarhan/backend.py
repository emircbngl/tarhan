"""Array-backend dikişi (seam).

Karar (Product Form Decision + MLX eki, 2026-07-03): hızlandırma backend'i ŞİMDİ
kurulmaz; bu DİKİŞ kurulur. Kurallar:

- Çekirdek/numerics kodu numpy'ı doğrudan import etmez; ``xp = backend.xp()``
  üzerinden aktif array modülüne erişir.
- Lineer çözücüler dikişi adlandırılmış fonksiyonlardan geçer (``solve_tridiag``);
  scipy çağrıları koda saçılmaz — backend değişiminin maliyeti modüldür, rewrite değil.
- Doğruluk yolu HER ZAMAN float64. Düşük-hassasiyetli bir backend (ör. MLX: GPU'da
  float64 yok, f64 girdiyi sessizce f32'ye çevirir — ölçüldü, 2026-07-03) yalnız
  "preview" katmanı olabilir; asla truth-path. Sessiz hassasiyet düşüşü yasaktır.
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

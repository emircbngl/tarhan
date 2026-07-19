"""Stiff zaman-entegrasyonu primitifi (backend dikişi felsefesi).

Karar (backend.py ile aynı ruh): üretim-sınıfı adaptif stiff entegratör —
adım/mertebe kontrolü + implicit-adım Newton'u + hata testi — sıfırdan yazılmaz;
kanıtlanmış kernel **scipy.integrate.solve_ivp**'ye delege edilir (tıpkı
``solve_tridiag``'ın ``scipy.linalg.solve_banded``'e delege etmesi gibi). TARHAN'ın
SAHİP OLDUĞU = sağ-taraf (fizik) + analitik Jacobian + doğrulama disiplini.

Neden delege: Robertson gibi 16 zaman-dekadını (1e-5…1e11) kapsayan stiff bir
problemde robust adaptif BDF/Radau'yu yeniden yazmak orantısız VE dürüstlük
kazancı yok — scipy'ın on yıllardır test edilmiş çözücüsünü out-valide edemeyiz.
Reimplemente ETTİĞİMİZ şeyler fizik ayrıklaştırmaları (SG akısı, Gummel kaynak
lineerizasyonu, sınır kinetiği); bir stiff ODE entegratörü "kanıtlanmış düşük-
seviye kernel" sınıfına girer.

Dürüstlük kuralları (kuruluş ilkeleri):
- f64 truth-path; ``method`` yalnız implicit/stiff-uygun ('BDF','Radau','LSODA').
  Explicit RK (RK45/DOP853) stiff truth-path DEĞİLDİR — istenirse açıkça reddedilir.
- Sessiz degrade yasak: çözücü başarısızsa ``RuntimeError`` fırlatılır, kısmi/
  sessiz sonuç dönülmez.
- Analitik Jacobian öncelikli: verilirse iletilir (stiff çözümde kritik).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as _np
from scipy.integrate import solve_ivp as _solve_ivp

__all__ = ["StiffSolution", "integrate_stiff", "conservation_residual"]

#: Stiff-uygun (implicit) yöntemler — truth-path yalnız bunlar olabilir.
STIFF_METHODS = ("BDF", "Radau", "LSODA")


@dataclass(frozen=True)
class StiffSolution:
    """Entegrasyon sonucu (scipy sonucunun ince, isimlendirilmiş sarımı)."""

    t: _np.ndarray          # (n_t,) değerlendirme zamanları
    y: _np.ndarray          # (n_species, n_t) çözüm
    method: str
    nfev: int               # sağ-taraf değerlendirme sayısı (stiffness tanığı)
    njev: int               # Jacobian değerlendirme sayısı
    nlu: int                # LU ayrıştırma sayısı (implicit çözücü eforu)
    n_output_points: int    # ÇIKTI noktası sayısı (t_eval verildiyse = len(t_eval))
    n_accepted_steps: int | None = None
    #: Çözücünün KABUL ETTİĞİ dahili adım sayısı — yalnız ``count_steps=True`` ile
    #: doldurulur (aksi hâlde None). DÜZELTME 2026-07-19: eski ``nsteps`` alanı
    #: ``sol.t.shape[0]`` döndürüyordu, yani t_eval verildiğinde ÇIKTI nokta sayısı,
    #: adım sayısı DEĞİL. Bu, chronoamp testindeki "implicit avantaj" iddiasını BOŞ
    #: bir assertion'a çeviriyordu (1 < 88.9 daima doğru). Tam denetimde yakalandı.


def integrate_stiff(rhs, y0, t_span, *, jac=None, t_eval=None,
                    rtol=1e-6, atol=1e-9, method="BDF",
                    count_steps=False) -> StiffSolution:
    """Stiff ODE sistemini ``t_span`` üzerinde entegre et (implicit, f64).

    ``rhs(t, y) -> dy/dt`` ve opsiyonel ``jac(t, y) -> ∂f/∂y``. Analitik Jacobian
    verilmesi stiff çözümde şiddetle önerilir (Newton adımlarını hızlandırır).

    ``count_steps=True`` çözücünün KABUL ETTİĞİ dahili adım sayısını ölçer
    (``dense_output`` açılır, iç kırılma noktaları ``sol.sol.ts``'den okunur) ve
    ``n_accepted_steps``'i doldurur. Maliyeti vardır (interpolant saklanır), bu
    yüzden opt-in'dir. "Implicit avantaj" gibi ADIM-SAYISI iddiaları bu bayrak
    olmadan ÖLÇÜLEMEZ — ``n_output_points`` adım sayısı değildir.

    Geçerlilik/dürüstlük:
    - ``method`` STIFF_METHODS içinde olmalı (explicit RK truth-path değil).
    - Çözücü başarısız olursa ``RuntimeError`` (sessiz kısmi sonuç yok).
    """
    if method not in STIFF_METHODS:
        raise ValueError(
            f"method={method!r} stiff truth-path değil; implicit bir yöntem seç "
            f"({', '.join(STIFF_METHODS)}). Explicit RK (RK45/DOP853) Robertson "
            "gibi stiff problemlerde ya ıraksar ya orantısız adım harcar."
        )
    y0 = _np.asarray(y0, dtype=float)
    sol = _solve_ivp(rhs, t_span, y0, method=method, jac=jac, t_eval=t_eval,
                     rtol=rtol, atol=atol, dense_output=bool(count_steps))
    if not sol.success:
        raise RuntimeError(f"stiff entegrasyon başarısız (method={method}): {sol.message}")
    accepted = None
    if count_steps and getattr(sol, "sol", None) is not None:
        accepted = len(sol.sol.ts) - 1        # iç kırılma noktaları → kabul edilen adım
    return StiffSolution(
        t=sol.t, y=sol.y, method=method,
        nfev=int(sol.nfev), njev=int(getattr(sol, "njev", 0) or 0),
        nlu=int(getattr(sol, "nlu", 0) or 0),
        n_output_points=int(sol.t.shape[0]),
        n_accepted_steps=accepted,
    )


def conservation_residual(y, invariant) -> float:
    """Korunum residüeli: max_t |Σ_invariant(y(·,t)) − invariant(y₀)|.

    ``y`` şekli (n_species, n_t); ``invariant`` çözümün her sütununa uygulanan,
    korunması beklenen skaler (ör. toplam derişim). Referans t=0 sütunundan alınır.
    """
    y = _np.asarray(y, dtype=float)
    vals = _np.array([invariant(y[:, i]) for i in range(y.shape[1])])
    return float(_np.max(_np.abs(vals - vals[0])))

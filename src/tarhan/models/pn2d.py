"""2D pn-diyot: kutu yöntemi + Gummel + Scharfetter-Gummel.

`pn1d`'nin 2D kardeşi ve public yüzeyi kasten aynı şekilde: bir cihaz dataclass'ı,
`solve_bias`, `iv_sweep`. Fizik de aynı — Boltzmann istatistiği, SG kenar akıları,
R = 0 (kısa-taban), ohmik kontaklar (yük-nötr + np = ni²). Değişen tek şey
geometri: tridiagonal komşuluk yerine bir `Mesh`, `solve_tridiag` yerine
`backend.solve_sparse`.

MESH ÜRETİLMEZ, OKUNUR. `numerics/mesh.py` bir üretici değildir (DESIGN-2D §3) ve
bu modül de değildir: düğümler ve üçgenler dışarıdan gelir — Gmsh'ten, DEVSIM'den,
elle. Bir mesher yazmak bu projenin altı ayını fizik yapmadan geçirme biçimidir.

Ölçekleme pn1d'nin De Mari reçetesi:
    C0 = max|N|,  δ = ni/C0,  L_D = sqrt(εs·UT/(q·C0)),  x̂ = x/L_D,  ψ̂ = ψ/UT
Bu ölçekte Poisson tam olarak ``lap(ψ̂) = n̂ − p̂ − N̂`` olur. Koordinatlar buraya
FİZİKSEL (cm) verilir ve ölçekleme içeride yapılır: kenar ağırlığı A/L ölçekten
bağımsızdır ama düğüm hacmi uzunluğun karesiyle ölçeklenir, yani mesh'i yanlış
birimde kurmak sessizce yanlış bir kaynak terimi verir.

İşaret konvansiyonları — ikisi de ölçülerek bulundu, ikisi de sessiz hata verirdi:

* Süreklilik operatörünün null uzayı ``exp(−ψ̂)``, yani DEŞİK bağıntısı.
  Elektronlar ``n = ni·exp(+ψ̂)`` uyar ve −ψ ister; bu yüzden
  ``assemble_continuity`` taşıyıcıyı zorunlu parametre alır.
* Elektron PARÇACIK akısı ile elektron AKIMI zıt işaretlidir (yük negatif).
  İkisine de +q uygulamak DEVSIM'e karşı |oran| = 1.0000 verir — büyüklük
  kusursuz, işaret ters — ve idealiteyi 1.01'den 1.03'e taşır.

Doğrulama (2D-2, DEVSIM'in kendi 495 düğümlü mesh'i üzerinde, aynı biaslarda):
    I_n oranı 1.00000 her biasta; I_p ve toplam 0.4-0.5 V'ta 5 hane
    idealite 1.0119-1.0134  (DEVSIM'in kendisi 1.0114-1.0126)
0.1 V ve altı gürültü tabanıdır: akımlar ~1e-14 ve DEVSIM'in kendi akım korunumu
orada zaten 1.4e-2'dir. Oraya bakıp sonuç çıkarmayın.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from tarhan import backend
from tarhan.models.pn1d import _contact_densities
from tarhan.numerics.assemble import assemble_continuity, assemble_poisson
from tarhan.numerics.mesh import Mesh, build_mesh


@dataclass
class PNDiode2D:
    """Vaka girdileri. Sabitler ASLA hardcode edilmez (konvansiyon-tuzağı kuralı).

    ``points`` cm cinsinden, ``net_doping`` cm^-3 ve düğüm başına, ``contacts``
    ise ad -> düğüm indeksleri. Biaslanan kontağın adı ``biased_contact``.
    """

    points: Sequence[tuple]
    triangles: Sequence[tuple]
    net_doping: Sequence[float]
    contacts: Mapping[str, Sequence[int]]
    biased_contact: str
    ni: float = 1e10                     # cm^-3
    ut: float = 0.025887193125           # V (kT/q, T=300K)
    eps_s: float = 11.1 * 8.85e-14       # F/cm
    q: float = 1.6e-19                   # C
    mu_n: float = 400.0                  # cm^2/Vs
    mu_p: float = 200.0                  # cm^2/Vs
    # türetilenler
    mesh: Mesh = field(init=False)
    C0: float = field(init=False)
    delta: float = field(init=False)
    L_D: float = field(init=False)

    def __post_init__(self):
        for name in ("ni", "ut", "eps_s", "q", "mu_n", "mu_p"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"{name}={v}: must be positive and finite")
        if not self.contacts:
            raise ValueError("at least one contact is required")
        if self.biased_contact not in self.contacts:
            raise ValueError(
                f"biased_contact {self.biased_contact!r} is not among "
                f"{sorted(self.contacts)}")
        # Contacts must be non-empty and DISJOINT. A node in two contacts is not
        # a harmless duplicate: _contact_state writes psi into a dict keyed by
        # node, so the last contact visited wins and the answer depends on
        # dictionary order. Measured on a 10-node strip, node 0 came out at
        # -13.82 or +5.50 thermal volts purely by swapping the order the
        # contacts were declared in. An empty contact is quiet in a different
        # way: it simply constrains nothing.
        seen: Dict[int, str] = {}
        for name, nodes in self.contacts.items():
            idx = np.asarray(nodes, dtype=int).ravel()
            if idx.size == 0:
                raise ValueError(f"contact {name!r} has no nodes")
            for i in idx:
                node = int(i)
                if not 0 <= node < len(self.points):
                    raise ValueError(
                        f"contact {name!r} references node {node}, outside "
                        f"0..{len(self.points) - 1}")
                if node in seen:
                    raise ValueError(
                        f"node {node} belongs to both contact {seen[node]!r} "
                        f"and {name!r}; contacts must be disjoint, or the "
                        "applied bias depends on which is visited last")
                seen[node] = name

        self.net_doping = np.asarray(self.net_doping, dtype=float)
        peak = float(np.abs(self.net_doping).max())
        if not peak > 0.0:
            raise ValueError("net_doping is identically zero; there is no junction")
        self.C0 = peak
        self.delta = self.ni / self.C0
        self.L_D = math.sqrt(self.eps_s * self.ut / (self.q * self.C0))
        # Mesh ÖLÇEKLİ kurulur: A/L ölçekten bağımsızdır ama düğüm hacmi değildir.
        self.mesh = build_mesh(
            [(float(x) / self.L_D, float(y) / self.L_D) for x, y in self.points],
            self.triangles)
        if len(self.net_doping) != self.mesh.n_nodes:
            raise ValueError(
                f"net_doping has {len(self.net_doping)} entries but the mesh has "
                f"{self.mesh.n_nodes} nodes")

    @property
    def doping_hat(self) -> np.ndarray:
        return self.net_doping / self.C0

    def _contact_state(self, v_applied: float):
        """Her kontak düğümü için (ψ̂, n̂, p̂) Dirichlet değerleri."""
        v_hat = v_applied / self.ut
        psi_bc: Dict[int, float] = {}
        n_bc: Dict[int, float] = {}
        p_bc: Dict[int, float] = {}
        for name, nodes in self.contacts.items():
            bias = v_hat if name == self.biased_contact else 0.0
            for i in np.asarray(nodes, dtype=int):
                n_c, p_c = _contact_densities(float(self.doping_hat[i]), self.delta)
                n_bc[int(i)] = n_c
                p_bc[int(i)] = p_c
                # Uygulanan bias quasi-Fermi seviyesini kaydırır; ψ onu izler.
                psi_bc[int(i)] = bias + math.log(n_c / self.delta)
        return psi_bc, n_bc, p_bc


def _poisson_newton(dev: PNDiode2D, psi, phi_n, phi_p, psi_bc,
                    tol: float = 1e-11, max_iter: int = 100):
    """Gummel-içi nonlineer Poisson: sabit quasi-Fermi'lerle damped Newton.

    Adım kelepçesi |Δψ̂| ≤ 5, pn1d ile aynı. M-matris + kelepçe ⇒ gürbüz.
    """
    delta = dev.delta
    doping = dev.doping_hat
    step_size = float("inf")
    for it in range(max_iter):
        n_h = delta * np.exp(np.clip(psi - phi_n, -700.0, 700.0))
        p_h = delta * np.exp(np.clip(phi_p - psi, -700.0, 700.0))
        system = assemble_poisson(dev.mesh, psi,
                                  charge=n_h - p_h - doping,
                                  dcharge_dpsi=n_h + p_h,
                                  dirichlet=psi_bc)
        step = backend.solve_sparse(system.rows, system.cols, system.vals,
                                    -system.residual, n=system.n_nodes)
        step_size = float(np.abs(step).max())
        if step_size > 5.0:
            step = step * (5.0 / step_size)
        psi = psi + step
        if step_size < tol:
            return psi, it + 1
    raise RuntimeError(f"Poisson-Newton yakınsamadı (son adım {step_size:.2e})")


def _continuity_solve(dev: PNDiode2D, psi, carrier: str, bc):
    """Süreklilik: R = 0 olduğundan yoğunlukta LİNEER — tek çözüm yeter."""
    coef = np.full(len(dev.mesh.edges),
                   (dev.mu_n if carrier == "electron" else dev.mu_p)
                   / max(dev.mu_n, dev.mu_p))
    system = assemble_continuity(dev.mesh, np.zeros(dev.mesh.n_nodes), psi,
                                 carrier=carrier, edge_coef=coef, dirichlet=bc)
    out = backend.solve_sparse(system.rows, system.cols, system.vals,
                               -system.residual, n=system.n_nodes)
    if float(out.min()) <= 0.0:
        raise RuntimeError(
            f"{carrier} sürekliliği pozitifliği kaybetti (min {out.min():.3e}); "
            "mesh ya da bias adımını kontrol edin")
    return out


def contact_current(dev: PNDiode2D, state, contact: str):
    """Kontak akımı [A / cm derinlik]: ``(I_n, I_p)``.

    KISITSIZ süreklilik rezidüelinin kontak düğümleri üzerindeki toplamı — yani
    o düğümlere giren net akı. Dirichlet uygulanmış hâlde bu bilgi silinmiş
    olurdu, o yüzden burada BC'siz yeniden kurulur.

    Ölçek q·μ_max·U_T·C0 bir türetmeydi; DEVSIM'e karşı doğrulandı (I_n oranı
    0.1-0.5 V'ta 1.00000). Elektron terimi negatiftir: parçacık akısı ile akım
    zıt yönlüdür.
    """
    nodes = np.asarray(dev.contacts[contact], dtype=int)
    mu_max = max(dev.mu_n, dev.mu_p)
    coef_n = np.full(len(dev.mesh.edges), dev.mu_n / mu_max)
    coef_p = np.full(len(dev.mesh.edges), dev.mu_p / mu_max)
    res_n = assemble_continuity(dev.mesh, state["n_hat"], state["psi"],
                                carrier="electron", edge_coef=coef_n).residual
    res_p = assemble_continuity(dev.mesh, state["p_hat"], state["psi"],
                                carrier="hole", edge_coef=coef_p).residual
    scale = dev.q * mu_max * dev.ut * dev.C0
    return (-float(res_n[nodes].sum()) * scale,
            float(res_p[nodes].sum()) * scale)


def solve_bias(dev: PNDiode2D, v_applied: float, state=None,
               gummel_tol: float = 1e-9, max_gummel: int = 200):
    """Tek bias noktası; ``state=None`` ise dengeden başlar."""
    psi_bc, n_bc, p_bc = dev._contact_state(v_applied)
    delta = dev.delta

    if state is None:
        psi = np.arcsinh(dev.doping_hat / (2.0 * delta))
        phi_n = np.zeros(dev.mesh.n_nodes)
        phi_p = np.zeros(dev.mesh.n_nodes)
    else:
        psi = np.array(state["psi"], copy=True)
        phi_n = np.array(state["phi_n"], copy=True)
        phi_p = np.array(state["phi_p"], copy=True)
    for i, value in psi_bc.items():
        psi[i] = value
    bias_hat = v_applied / dev.ut
    for name, nodes in dev.contacts.items():
        level = bias_hat if name == dev.biased_contact else 0.0
        for i in np.asarray(nodes, dtype=int):
            phi_n[i] = level
            phi_p[i] = level

    for g in range(max_gummel):
        psi_old = psi.copy()
        psi, _ = _poisson_newton(dev, psi, phi_n, phi_p, psi_bc)
        n_h = _continuity_solve(dev, psi, "electron", n_bc)
        p_h = _continuity_solve(dev, psi, "hole", p_bc)
        phi_n = psi - np.log(n_h / delta)
        phi_p = psi + np.log(p_h / delta)
        if float(np.abs(psi - psi_old).max()) < gummel_tol:
            break
    else:
        raise RuntimeError(f"Gummel yakınsamadı @ V={v_applied}")

    out = {"psi": psi, "phi_n": phi_n, "phi_p": phi_p,
           "n_hat": n_h, "p_hat": p_h, "gummel_iters": g + 1,
           "v_applied": v_applied}
    i_n, i_p = contact_current(dev, out, dev.biased_contact)
    out["i_n"], out["i_p"], out["i"] = i_n, i_p, i_n + i_p
    return out


def iv_sweep(dev: PNDiode2D, voltages, **kw):
    """Bias-continuation'lı I-V taraması (warm start), pn1d.iv_sweep gibi."""
    state, out = None, []
    for v in voltages:
        state = solve_bias(dev, v, state=state, **kw)
        out.append(state["i"])
    return out, state


__all__ = ["PNDiode2D", "solve_bias", "iv_sweep", "contact_current"]

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
from tarhan.numerics.assemble import (assemble_continuity, assemble_poisson,
                                     node_volumes)
from tarhan.numerics.transient import integrate_stiff
from tarhan.models.pn1d import time_scale_seconds
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
            raw = np.asarray(nodes).ravel()
            if raw.size == 0:
                raise ValueError(f"contact {name!r} has no nodes")
            # Casting to int first would swallow the mistake: 0.9 becomes node 0
            # in silence, so a contact placed by a slightly-off computation
            # lands on its neighbour and the device still solves. Check the
            # value BEFORE converting.
            if not np.issubdtype(raw.dtype, np.integer):
                as_float = raw.astype(float, copy=False)
                if not np.all(np.isfinite(as_float)):
                    raise ValueError(
                        f"contact {name!r} has a non-finite node index")
                if not np.all(as_float == np.floor(as_float)):
                    offender = as_float[as_float != np.floor(as_float)][0]
                    raise ValueError(
                        f"contact {name!r} has the non-integer node index "
                        f"{offender!r}; truncating it would silently select a "
                        "different node")
            idx = raw.astype(np.int64)
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


def coupled_residual(dev: PNDiode2D, psi, n_hat, p_hat, psi_bc):
    """How badly the CURRENT (psi, n, p) fails the coupled system.

    This is the measurement `max|dpsi| < tol` cannot make. Gummel solves
    Poisson with the OLD quasi-Fermi levels, then updates the carriers; the
    same psi with the NEW n and p no longer satisfies Poisson, and that
    mismatch is what "the solver solved its own equations" has to mean.
    Measuring the sub-solve's own residual instead would prove nothing: the
    linear continuity solve is exact by construction, as reported in review.

    Returns raw max-norms plus a dimensionless Poisson figure, normalised by
    the scale of the charge term it sits against. The NORMALISATION is a
    judgement — physics_verify is unavailable — so the raw norm is returned
    beside it and the caller can use either.
    """
    charge = n_hat - p_hat - dev.doping_hat
    system = assemble_poisson(dev.mesh, psi, charge=charge,
                              dcharge_dpsi=n_hat + p_hat, dirichlet=psi_bc)
    free = np.ones(dev.mesh.n_nodes, dtype=bool)
    for nodes in dev.contacts.values():
        free[np.asarray(nodes, dtype=int)] = False

    poisson = float(np.abs(system.residual[free]).max()) if free.any() else 0.0
    scale = float(np.abs(charge[free]).max()) if free.any() else 0.0

    mu_max = max(dev.mu_n, dev.mu_p)
    norms = {}
    for tag, values, carrier, mu in (("electron", n_hat, "electron", dev.mu_n),
                                     ("hole", p_hat, "hole", dev.mu_p)):
        coef = np.full(len(dev.mesh.edges), mu / mu_max)
        res = assemble_continuity(dev.mesh, values, psi, carrier=carrier,
                                  edge_coef=coef).residual
        norms[tag] = float(np.abs(res[free]).max()) if free.any() else 0.0

    return {"poisson": poisson,
            "poisson_relative": poisson / scale if scale > 0 else poisson,
            "electron": norms["electron"], "hole": norms["hole"]}


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
               gummel_tol: float = 1e-9, max_gummel: int = 200,
               on_iteration=None, *, min_gummel: int = 0):
    """Tek bias noktası; ``state=None`` ise dengeden başlar.

    ``on_iteration(index, total)`` her Gummel adımından ÖNCE çağrılır. Bütün
    çözüm tek bir bloklayan çağrı olduğu için, çağıranın ekrana bir şey
    çizebileceği tek an budur — 1D tarafında bu geri çağrı yokken `run solve`
    begin() ile finish() arasında sıfır bayt yazıyordu, yani uzun bir çözümde
    gösterge ancak iş bittikten sonra beliriyordu (incelemede yakalandı).
    """
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

    previous_current = None
    current_change = None
    current_history = []
    psi_step = float("inf")
    for g in range(max_gummel):
        if on_iteration is not None:
            on_iteration(g, max_gummel)
        psi_old = psi.copy()
        psi, _ = _poisson_newton(dev, psi, phi_n, phi_p, psi_bc)
        n_h = _continuity_solve(dev, psi, "electron", n_bc)
        p_h = _continuity_solve(dev, psi, "hole", p_bc)
        phi_n = psi - np.log(n_h / delta)
        phi_p = psi + np.log(p_h / delta)
        psi_step = float(np.abs(psi - psi_old).max())

        # The ANSWER, not the step. max|dpsi| under 1e-9 called 0.2 V, 0.3 V
        # and 0.5 V all converged, while the terminal current at 0.2 V was
        # still moving by ~1e-5 per pass and never settled. A criterion that
        # cannot tell a settled current from a stalled one is not a claim
        # about the result. Reported in review.
        i_n, i_p = contact_current(dev, {"psi": psi, "n_hat": n_h,
                                         "p_hat": p_h}, dev.biased_contact)
        total = i_n + i_p
        # At equilibrium the net current is zero BY CONSTRUCTION — an almost
        # exact cancellation of two large opposite components — so |dI|/|I| is
        # roundoff over roundoff and says nothing. Two normalisations were
        # tried and both were wrong: dividing by the differenced magnitude
        # made every bias look settled to 1e-16 and destroyed the
        # discrimination this criterion exists for. So the cancelling case is
        # named instead of normalised away: when the net is below 1e-10 of the
        # components that make it, there is no current to settle and the
        # potential criterion is the whole story.
        # Recorded raw. At equilibrium the net current is an exact
        # cancellation, so this ratio is roundoff over roundoff and means
        # nothing there — said plainly rather than hidden behind a cut-off
        # nobody can justify.
        # None, not a number. On the first pass there is nothing to compare
        # against; at equilibrium the net current is a cancellation and the
        # ratio is roundoff over roundoff; with a warm start a single-pass
        # solve never forms one at all. All three used to publish a float —
        # 0.8999 at 1D equilibrium, and `inf` for a one-pass solve — which a
        # machine consumer cannot tell from a measurement. Reported in review.
        if v_applied == 0.0:
            # Equilibrium: no applied bias, so there is no net current to
            # settle and the ratio is roundoff over roundoff. This is the
            # BOUNDARY CONDITION saying so, not a magnitude threshold — an
            # earlier attempt used a 1e-10 cancellation cut-off and behaved
            # differently in 1D and 2D. The artifact published 0.8999 here as
            # though it were a measurement.
            current_change = None
        elif previous_current is None or total == 0.0:
            current_change = None
        else:
            current_change = abs(total - previous_current) / abs(total)
        previous_current = total
        current_history.append(current_change)

        if psi_step < gummel_tol and g + 1 >= min_gummel:
            break
    else:
        # Falling out of the loop is only a FAILURE if the potential never
        # settled. A settled potential with an unsettled current is a real
        # state that low-bias solves genuinely reach, and it is labelled
        # rather than refused — the numbers are still good to the tolerance
        # their validation claims.
        if psi_step >= gummel_tol:
            raise RuntimeError(
                f"Gummel yakinsamadi @ V={v_applied}: "
                f"max|dpsi|={psi_step:.3e} (tol {gummel_tol:.0e})")

    out = {"psi": psi, "phi_n": phi_n, "phi_p": phi_p,
           "n_hat": n_h, "p_hat": p_h, "gummel_iters": g + 1,
           "v_applied": v_applied,
           # Recorded so an artifact can show what "converged" was worth,
           # rather than asserting it.
           "psi_step": psi_step, "current_rel_change": current_change,
           "current_history": tuple(current_history),
           "coupled_residual": coupled_residual(dev, psi, n_h, p_h, psi_bc)}
    i_n, i_p = contact_current(dev, out, dev.biased_contact)
    out["i_n"], out["i_p"], out["i"] = i_n, i_p, i_n + i_p
    return out


def _poisson_linear(dev: PNDiode2D, n_hat, p_hat, psi_bc):
    """Poisson with the carrier densities fixed — one linear solve, no Newton.

    The same reduction as the 1D transient path: with n̂ and p̂ as state
    variables the charge n̂ − p̂ − N̂ contains no ψ̂, so passing
    ``dcharge_dpsi = 0`` leaves the Jacobian equal to the bare box-method
    Laplacian and one step from any starting point lands on the exact solution.
    """
    psi = np.zeros(dev.mesh.n_nodes)
    system = assemble_poisson(dev.mesh, psi,
                              charge=n_hat - p_hat - dev.doping_hat,
                              dcharge_dpsi=np.zeros(dev.mesh.n_nodes),
                              dirichlet=psi_bc)
    step = backend.solve_sparse(system.rows, system.cols, system.vals,
                                -system.residual, n=system.n_nodes)
    return psi + step


def transient_setup(dev: PNDiode2D, v_applied: float, state=None):
    """Everything the time integrator needs, taken from the steady-state path."""
    st = solve_bias(dev, v_applied, state=state)
    psi_bc, n_bc, p_bc = dev._contact_state(v_applied)
    pinned = np.zeros(dev.mesh.n_nodes, dtype=bool)
    pinned[np.asarray(sorted(psi_bc), dtype=int)] = True
    free = np.flatnonzero(~pinned)
    mu_max = max(dev.mu_n, dev.mu_p)
    return {
        "state": st,
        "free": free,
        "psi_bc": psi_bc,
        "n_bc": n_bc,
        "p_bc": p_bc,
        "volumes": node_volumes(dev.mesh),
        "coef_n": np.full(len(dev.mesh.edges), dev.mu_n / mu_max),
        "coef_p": np.full(len(dev.mesh.edges), dev.mu_p / mu_max),
        "n_full": np.array(st["n_hat"], copy=True),
        "p_full": np.array(st["p_hat"], copy=True),
        "y_steady": np.concatenate([st["n_hat"][free], st["p_hat"][free]]),
    }


def transient_rhs(dev: PNDiode2D, setup, y):
    """d(n̂, p̂)/dt̂ at the free nodes, with ψ̂ solved from the state.

    The spatial operator is not rewritten here: ``assemble_continuity``'s
    UNCONSTRAINED residual is already the net flux at each node — it is what
    :func:`contact_current` reads to get a terminal current — so the
    accumulation is −residual/volume. Reusing it means the transient and steady
    paths cannot drift apart in their discretisation.

    The sign is settled by the relaxation test rather than by argument: the
    fixed-point check passes for EITHER sign, because the residual vanishes at
    the steady state whichever way it is fed in. Only integrating tells them
    apart — the wrong sign runs away instead of settling.
    """
    free = setup["free"]
    n_free = len(free)
    n_hat = setup["n_full"]
    p_hat = setup["p_full"]
    n_hat[free] = y[:n_free]
    p_hat[free] = y[n_free:]

    psi = _poisson_linear(dev, n_hat, p_hat, setup["psi_bc"])
    res_n = assemble_continuity(dev.mesh, n_hat, psi, carrier="electron",
                                edge_coef=setup["coef_n"]).residual
    res_p = assemble_continuity(dev.mesh, p_hat, psi, carrier="hole",
                                edge_coef=setup["coef_p"]).residual
    vol = setup["volumes"]
    return np.concatenate([-res_n[free] / vol[free],
                           -res_p[free] / vol[free]])


def transient_solve(dev: PNDiode2D, v_applied: float, *, y0=None,
                    t_span_hat=(0.0, 40.0), t_eval=None, rtol: float = 1e-8,
                    atol: float = 1e-14, method: str = "BDF", state=None):
    """Integrate (n̂, p̂) on the box mesh in scaled time.

    ``t_span_hat`` is in units of :func:`tarhan.models.pn1d.time_scale_seconds`
    — the same dielectric relaxation time from the same reference mobility, so
    the 1D and 2D transients share one clock.
    """
    setup = transient_setup(dev, v_applied, state=state)
    y_start = setup["y_steady"] if y0 is None else y0
    sol = integrate_stiff(lambda _t, y: transient_rhs(dev, setup, y),
                          y_start, t_span_hat, t_eval=t_eval, rtol=rtol,
                          atol=atol, method=method)
    setup["solution"] = sol
    setup["t_seconds"] = sol.t * time_scale_seconds(dev)
    return setup


def iv_sweep(dev: PNDiode2D, voltages, **kw):
    """Bias-continuation'lı I-V taraması (warm start), pn1d.iv_sweep gibi."""
    state, out = None, []
    for v in voltages:
        state = solve_bias(dev, v, state=state, **kw)
        out.append(state["i"])
    return out, state


__all__ = ["PNDiode2D", "solve_bias", "iv_sweep", "contact_current"]

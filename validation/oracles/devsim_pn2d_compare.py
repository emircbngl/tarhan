"""Çapraz-oracle, 2D: TARHAN'ın kutu-yöntemi denge çözümü vs DEVSIM 2.x.

Bu, `devsim_pn1d_compare.py`'nin 2D kardeşi ve aynı disiplini izler — DEVSIM
motor kodu değil, ORACLE olarak kullanılır; her parametre AÇIKÇA set edilir
(`SetSiliconParameters` varsayılanlarına güvenilmez).

Bir farkla, ve o fark bu dosyanın asıl değeri: cihaz 1D'nin kılık değiştirmiş
hâli DEĞİL. `top` kontağı sol kenarın yalnızca üst %20'sini kaplar, `bot` ise
sağ kenarın tamamını — yani çözümün gerçek bir enine yapısı vardır ve bir 1D
şema onu üretemez. DESIGN-2D §5'in 2D-1 aşaması budur.

Karşılaştırma AYNI MESH üzerinde yapılır: DEVSIM'in kendi düğümleri ve üçgenleri
çıkarılıp TARHAN'a verilir. Böylece geriye saklanacak bir ayrıklaştırma farkı
kalmaz — iki kod aynı ayrık problemi çözer ya da çözmez.

Ölçüm (2026-08-06, DEVSIM 2.10.0, 495 düğüm / 880 eleman, 14 Newton adımı):
    kendi kontak koşulumuzla çözülen 481 düğümde
    max |psi_TARHAN - psi_DEVSIM| = 2.242e-16 V,  rms 7.389e-17 V
    V_bi: TARHAN = DEVSIM = analitik = 0.953719 V (farklar TAM sıfır)

Kontak modeli bağımsız türetilir (yük-nötr + np = ni^2 => psi = V_t*asinh(N/2ni))
ve DEVSIM'in kontak potansiyeliyle 9 hanede örtüşür. Formül physicist Docker
oracle'ında doğrulandı: yük-nötrlük denklemi log(r + sqrt(r^2+1)) verir, ki bu
asinh(r)'dir; asinh(5e7) = 18.420680743952367; r -> 0 limitinde 0.
"""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np

# dio2_element_2d.py'nin vaka değerleri — TARHAN tarafında da bunlar kullanılır.
Q = 1.6e-19                       # C
EPS = 11.1 * 8.85e-14             # F/cm
NI = 1.0e10                       # cm^-3
K_B = 1.3806503e-23               # J/K
TEMP = 300.0                      # K
V_T = K_B * TEMP / Q              # V
DOPING = 1.0e18                   # cm^-3, her iki taraf
DEVICE_LEN = 1.0e-5               # cm
JUNCTION_X = 0.5e-5               # cm


def run_devsim_2d_equilibrium():
    """DEVSIM'de 2D diyotu dengede çözer; mesh + potansiyel + doping döner.

    Yalnız Poisson kurulur (sürüklenme-difüzyon yok): denge, süreklilik
    denklemleri devreye girmeden tanımlıdır ve 2D-1'in sorduğu tam olarak budur.
    """
    import devsim as ds
    from devsim.python_packages import simple_physics as sp

    ds.reset_devsim()                        # global solver kaydını da siler →
    try:                                     # UMFPACK shim'ini yeniden kaydet
        from devsim.umfpack import umfshim
        ds.set_parameter(name="direct_solver", value="custom")
        ds.set_parameter(name="solver_callback",
                         value=umfshim.local_solver_callback)
    except ImportError:
        ds.set_parameter(name="direct_solver", value="superlu")
    device, region = "MyDevice", "MyRegion"

    with contextlib.redirect_stdout(io.StringIO()):
        ds.create_2d_mesh(mesh="dio2d")
        # Referans vakanın mesh'i BİREBİR. Sondaki iki çizgi ve iki "air"
        # bölgesi süs değil: kontaklar x=0 ve x=1e-5 düzlemlerinde tanımlı ve
        # DEVSIM onları ancak orada bir bölge SINIRI varsa yaratır. Bunları
        # atıp sadeleştirmeyi denedim; `add_2d_contact` sessizce hiçbir şey
        # yaratmadı ve hata çok sonra, kontak modeli kurulurken patladı.
        for direction, pos, spacing in (
            ("x", 0.0, 1e-6), ("x", JUNCTION_X, 1e-8), ("x", DEVICE_LEN, 1e-6),
            ("y", 0.0, 1e-6), ("y", DEVICE_LEN, 1e-6),
            ("x", -1e-8, 1e-8), ("x", 1.001e-5, 1e-8),
        ):
            ds.add_2d_mesh_line(mesh="dio2d", dir=direction, pos=pos, ps=spacing)
        ds.add_2d_region(mesh="dio2d", material="Si", region=region)
        ds.add_2d_region(mesh="dio2d", material="Si", region="air1",
                         xl=-1e-8, xh=0.0)
        ds.add_2d_region(mesh="dio2d", material="Si", region="air2",
                         xl=DEVICE_LEN, xh=1.001e-5)
        # Kısmi üst kontak: enine yapıyı yaratan şey bu.
        ds.add_2d_contact(mesh="dio2d", name="top", region=region,
                          xl=0.0, xh=0.0, yl=0.8 * DEVICE_LEN, yh=DEVICE_LEN,
                          bloat=1e-10, material="metal")
        ds.add_2d_contact(mesh="dio2d", name="bot", region=region,
                          xl=DEVICE_LEN, xh=DEVICE_LEN, bloat=1e-10,
                          material="metal")
        ds.finalize_mesh(mesh="dio2d")
        ds.create_device(mesh="dio2d", device=device)

        # Parametreler AÇIKÇA — varsayılana güvenmek konvansiyon tuzağıdır.
        for name, value in (("Permittivity", EPS), ("ElectronCharge", Q),
                            ("n_i", NI), ("kT", 8.85e-14 * TEMP),
                            ("V_t", V_T), ("mu_n", 400.0), ("mu_p", 200.0)):
            ds.set_parameter(device=device, region=region, name=name, value=value)

        sp.CreateSolution(device, region, "Potential")
        for name, equation in (
            ("Acceptors", f"{DOPING}*step({JUNCTION_X}-x)"),
            ("Donors", f"{DOPING}*step(x-{JUNCTION_X})"),
            ("NetDoping", "Donors-Acceptors"),
        ):
            ds.node_model(device=device, region=region, name=name,
                          equation=equation)
        sp.CreateSiliconPotentialOnly(device, region)
        for contact in ("top", "bot"):
            # simple_physics kontak modelini "<ad>_bias" parametresi üzerinden
            # kurar; DENGE her iki kontakta sıfır bias demektir. Set etmezsen
            # hata çözüm anında ve modelin içinden gelir ("Value for bot_bias
            # not available"), yani kurulum sırasında değil.
            ds.set_parameter(device=device, name=f"{contact}_bias", value=0.0)
            sp.CreateSiliconPotentialOnlyContact(device, region, contact)
        ds.solve(type="dc", absolute_error=1.0, relative_error=1e-12,
                 maximum_iterations=30)

        def get(name):
            return np.array(ds.get_node_model_values(
                device=device, region=region, name=name))

        out = {
            "x": get("x"), "y": get("y"),
            "psi": get("Potential"), "net_doping": get("NetDoping"),
            "elements": [tuple(t) for t in
                         ds.get_element_node_list(device=device, region=region)],
        }
    return out


#: 2D-2 tarama biasları. 0.1 V ve altı KASTEN dışarıda: orada akımlar ~1e-14 ve
#: DEVSIM'in kendi akım korunumu bile 1.4e-2, yani karşılaştırılan şey gürültü.
IV_VOLTS = (0.20, 0.30, 0.40, 0.50)
#: Continuation adımları — warm start için, pn1d.iv_sweep gibi.
IV_RAMP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def run_devsim_2d_iv(volts=IV_RAMP, tau=1e10):
    """DEVSIM'de aynı cihazın I-V'si; döner: ``{V: (I_n, I_p, I_top, I_bot)}``.

    Referans vaka (dio2_element_physics) REKOMBİNASYONSUZDUR — ne USRH ne taun
    tanımlar, yani R = 0 kısa-taban, pn1d'nin varsayılan modu. `simple_physics`
    ise SRH'yi zorunlu kılar, o yüzden tau astronomik verilip kapatılır. Bu
    varsayılmadı, ÖLÇÜLDÜ: tau'yu 1e10'dan 1e12'ye çıkarmak her biastaki akımı
    tam 0.00e+00 değiştirir.
    """
    import devsim as ds
    from devsim.python_packages import simple_physics as sp

    run_devsim_2d_equilibrium()          # cihazı kurar ve dengede çözer
    device, region = "MyDevice", "MyRegion"
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for name, value in (("taun", tau), ("taup", tau), ("n1", NI), ("p1", NI)):
            ds.set_parameter(device=device, region=region, name=name, value=value)
        sp.CreateSolution(device, region, "Electrons")
        sp.CreateSolution(device, region, "Holes")
        ds.set_node_values(device=device, region=region, name="Electrons",
                           init_from="IntrinsicElectrons")
        ds.set_node_values(device=device, region=region, name="Holes",
                           init_from="IntrinsicHoles")
        sp.CreateSiliconDriftDiffusion(device, region)
        for contact in ("top", "bot"):
            sp.CreateSiliconDriftDiffusionAtContact(device, region, contact)

        def current(contact, equation):
            return ds.get_contact_current(contact=contact, equation=equation,
                                          device=device)

        for v in volts:
            ds.set_parameter(device=device, name="top_bias", value=float(v))
            ds.solve(type="dc", absolute_error=1e10, relative_error=1e-10,
                     maximum_iterations=30)
            i_n = current("top", "ElectronContinuityEquation")
            i_p = current("top", "HoleContinuityEquation")
            i_bot = (current("bot", "ElectronContinuityEquation")
                     + current("bot", "HoleContinuityEquation"))
            out[round(float(v), 4)] = (i_n, i_p, i_n + i_p, i_bot)
    return out


def compare_iv(quiet: bool = True):
    """TARHAN'ın pn2d'si ve DEVSIM, AYNI mesh üzerinde, aynı biaslarda."""
    from tarhan.models.pn2d import PNDiode2D, solve_bias

    ref = run_devsim_2d_equilibrium()
    tol = 1e-9
    on_top = np.nonzero((np.abs(ref["x"]) < tol)
                        & (ref["y"] >= 0.8 * DEVICE_LEN - tol))[0]
    on_bot = np.nonzero(np.abs(ref["x"] - DEVICE_LEN) < tol)[0]
    dev = PNDiode2D(points=list(zip(ref["x"], ref["y"])),
                    triangles=ref["elements"], net_doping=ref["net_doping"],
                    contacts={"top": on_top, "bot": on_bot},
                    biased_contact="top",
                    ni=NI, ut=V_T, eps_s=EPS, q=Q, mu_n=400.0, mu_p=200.0)

    tarhan = {}
    state = None
    for v in IV_RAMP:
        state = solve_bias(dev, v, state=state)
        tarhan[round(float(v), 4)] = (state["i_n"], state["i_p"], state["i"],
                                      state["gummel_iters"])

    devsim = run_devsim_2d_iv()
    results = {"tarhan": tarhan, "devsim": devsim, "volts": IV_VOLTS,
               "ramp": IV_RAMP, "device": dev}
    if not quiet:
        print("  V   |  I_n ratio  I_p ratio  I_tot ratio")
        for v in IV_VOLTS:
            t, d = tarhan[v], devsim[v]
            print(f"{v:5.2f} | {t[0]/d[0]:9.5f} {t[1]/d[1]:10.5f} {t[2]/d[2]:12.5f}")
    return results


def ideality(current_by_volt, volts):
    """Ardışık noktalardan idealite: ``n = ΔV / (U_T · Δ ln I)``."""
    out = []
    ordered = sorted(volts)
    for lo, hi in zip(ordered, ordered[1:]):
        out.append((hi - lo)
                   / (V_T * math.log(current_by_volt[hi] / current_by_volt[lo])))
    return out


def solve_tarhan_equilibrium(ref):
    """Aynı mesh üzerinde TARHAN'ın denge çözümü, kendi kontak koşulumuzla.

    Ölçekleme pn1d'nin De Mari reçetesi: C0 = max doping, delta = ni/C0,
    L_D = sqrt(eps*V_t/(q*C0)), psi_hat = psi/V_t. Bu ölçekte Poisson tam olarak
    ``lap(psi_hat) = n_hat - p_hat - N_hat`` olur, yani ``assemble_poisson``'ın
    sözleşmesi.
    """
    from tarhan import backend
    from tarhan.numerics.assemble import assemble_poisson
    from tarhan.numerics.mesh import build_mesh

    c0 = DOPING
    delta = NI / c0
    debye = math.sqrt(EPS * V_T / (Q * c0))

    mesh = build_mesh(list(zip(ref["x"] / debye, ref["y"] / debye)),
                      ref["elements"])
    doping_hat = ref["net_doping"] / c0

    # Ohmik kontak: yük-nötr + np = ni^2  =>  psi_hat = asinh(N_hat/(2*delta)).
    # Serbest kalan sınırlar doğal (sıfır-akı) koşulu alır — kutu yönteminde
    # bunun için hiçbir şey yapmak gerekmez, kısıtlamamak yeterlidir.
    tol = 1e-9
    on_top = (np.abs(ref["x"]) < tol) & (ref["y"] >= 0.8 * DEVICE_LEN - tol)
    on_bot = np.abs(ref["x"] - DEVICE_LEN) < tol
    contact = on_top | on_bot
    fixed = {int(i): float(np.arcsinh(doping_hat[i] / (2.0 * delta)))
             for i in np.nonzero(contact)[0]}

    psi = np.zeros(mesh.n_nodes)
    step_size = float("inf")
    for iteration in range(400):
        n_hat = delta * np.exp(np.clip(psi, -700.0, 700.0))
        p_hat = delta * np.exp(np.clip(-psi, -700.0, 700.0))
        system = assemble_poisson(mesh, psi,
                                  charge=n_hat - p_hat - doping_hat,
                                  dcharge_dpsi=n_hat + p_hat,
                                  dirichlet=fixed)
        step = backend.solve_sparse(system.rows, system.cols, system.vals,
                                    -system.residual, n=system.n_nodes)
        step_size = float(np.abs(step).max())
        if step_size > 5.0:                       # pn1d'nin adım kelepçesi
            step = step * (5.0 / step_size)
        psi = psi + step
        if step_size < 1e-11:
            break
    else:
        raise RuntimeError(
            f"2D Poisson-Newton yakınsamadı (son adım {step_size:.2e})")

    return {"psi_hat": psi, "mesh": mesh, "contact": contact,
            "iterations": iteration + 1, "delta": delta, "debye_cm": debye}


def compare(quiet: bool = True):
    """DEVSIM'i ve TARHAN'ı aynı mesh üzerinde koşup farkları döner."""
    ref = run_devsim_2d_equilibrium()
    got = solve_tarhan_equilibrium(ref)

    psi_ref_hat = ref["psi"] / V_T
    free = ~got["contact"]
    diff_volts = np.abs(got["psi_hat"] - psi_ref_hat) * V_T

    results = {
        "n_nodes": len(ref["x"]),
        "n_elements": len(ref["elements"]),
        "n_solved": int(free.sum()),
        "max_abs_volt": float(diff_volts[free].max()),
        "rms_volt": float(np.sqrt((diff_volts[free] ** 2).mean())),
        "vbi_tarhan": float(got["psi_hat"].max() - got["psi_hat"].min()) * V_T,
        "vbi_devsim": float(ref["psi"].max() - ref["psi"].min()),
        "vbi_analytic": V_T * math.log(DOPING * DOPING / (NI * NI)),
        "iterations": got["iterations"],
        "psi_hat": got["psi_hat"], "psi_ref_hat": psi_ref_hat,
        "x": ref["x"], "y": ref["y"], "contact": got["contact"],
        "mesh": got["mesh"],
    }
    if not quiet:
        print(f"nodes {results['n_nodes']}  elements {results['n_elements']}  "
              f"solved {results['n_solved']}  Newton {results['iterations']}")
        print(f"max |dpsi| {results['max_abs_volt']:.3e} V   "
              f"rms {results['rms_volt']:.3e} V")
        print(f"V_bi  TARHAN {results['vbi_tarhan']:.6f}  "
              f"DEVSIM {results['vbi_devsim']:.6f}  "
              f"analytic {results['vbi_analytic']:.6f}")
    return results

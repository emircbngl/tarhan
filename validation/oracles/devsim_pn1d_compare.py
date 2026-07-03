"""Çapraz-oracle: TARHAN pn1d vs DEVSIM 2.x (bağımsız TCAD, Apache-2.0).

DEVSIM oracle OLARAK kullanılır (Form Decision: 'oracle + canlı yedek') — motor
kodu değil, API'siyle aynı fiziksel cihaz kurulup sonuçlar karşılaştırılır.
Aynı fizik: Boltzmann, SG akıları, midgap SRH (n1=p1=ni), ohmik kontaklar
(yük-nötr + np=ni²) — DEVSIM simple_physics modeli birebir aynı formülasyon.

Parametre eşitleme (konvansiyon-tuzağı disiplini): Permittivity, n_i, mu_n,
mu_p, taun/taup, kT ve V_t AÇIKÇA TARHAN vaka değerlerine set edilir
(SetSiliconParameters default'larına güvenilmez).
"""
from __future__ import annotations

import contextlib
import io


def run_devsim_diode(na, nd, len_p, len_n, tau, *, ni=1e10, ut=0.0259,
                     eps_s=11.7 * 8.85e-14, q=1.6e-19, mu_n=1350.0, mu_p=480.0,
                     h_junction=5e-7, h_contact=2e-5, volts=(0.1, 0.2, 0.3, 0.4)):
    """DEVSIM'de aynı diyot; döner: (v_bi, {V: J[A/cm²]}). Her çağrı taze mesh kurar."""
    import devsim as ds
    from devsim.python_packages import simple_physics as sp

    ds.reset_devsim()                        # global solver kaydını da siler →
    try:                                     # UMFPACK shim'ini yeniden kaydet
        from devsim.umfpack import umfshim
        ds.set_parameter(name="direct_solver", value="custom")
        ds.set_parameter(name="solver_callback", value=umfshim.local_solver_callback)
    except ImportError:
        ds.set_parameter(name="direct_solver", value="superlu")
    device, region = "diode", "bulk"
    xj, xtot = len_p, len_p + len_n

    ds.create_1d_mesh(mesh="m")
    ds.add_1d_mesh_line(mesh="m", pos=0.0, ps=h_contact, tag="left")
    ds.add_1d_mesh_line(mesh="m", pos=xj, ps=h_junction)
    ds.add_1d_mesh_line(mesh="m", pos=xtot, ps=h_contact, tag="right")
    ds.add_1d_contact(mesh="m", name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh="m", name="right", tag="right", material="metal")
    ds.add_1d_region(mesh="m", material="Si", region=region, tag1="left", tag2="right")
    ds.finalize_mesh(mesh="m")
    ds.create_device(mesh="m", device=device)

    # parametreler — TARHAN vakasıyla BİREBİR (default'lara güven yok)
    sp.SetSiliconParameters(device, region, 300)
    for name, value in (("Permittivity", eps_s), ("ElectronCharge", q),
                        ("n_i", ni), ("V_t", ut), ("kT", q * ut),
                        ("mu_n", mu_n), ("mu_p", mu_p),
                        ("n1", ni), ("p1", ni), ("taun", tau), ("taup", tau)):
        ds.set_parameter(device=device, region=region, name=name, value=value)

    # doping: p-taraf solda (TARHAN geometrisi)
    sp.CreateNodeModel(device, region, "Acceptors", f"{na:.6e}*step({xj:.6e}-x)")
    sp.CreateNodeModel(device, region, "Donors", f"{nd:.6e}*step(x-{xj:.6e})")
    sp.CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")

    # 1) yalnız-Poisson denge
    sp.CreateSolution(device, region, "Potential")
    sp.CreateSiliconPotentialOnly(device, region)
    for c in ("left", "right"):
        sp.CreateSiliconPotentialOnlyContact(device, region, c)
        ds.set_parameter(device=device, name=sp.GetContactBiasName(c), value=0.0)
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)

    pot = ds.get_node_model_values(device=device, region=region, name="Potential")
    v_bi = pot[-1] - pot[0]

    # 2) drift-diffusion
    sp.CreateSolution(device, region, "Electrons")
    sp.CreateSolution(device, region, "Holes")
    ds.set_node_values(device=device, region=region, name="Electrons",
                       init_from="IntrinsicElectrons")
    ds.set_node_values(device=device, region=region, name="Holes",
                       init_from="IntrinsicHoles")
    sp.CreateSiliconDriftDiffusion(device, region)
    for c in ("left", "right"):
        sp.CreateSiliconDriftDiffusionAtContact(device, region, c)
    ds.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)

    # 3) bias rampası (25 mV adım — continuation) + akım okuma
    currents = {}
    targets = sorted(volts)
    v_now, i_t = 0.0, 0
    while i_t < len(targets):
        v_now = min(v_now + 0.025, targets[i_t])
        ds.set_parameter(device=device, name=sp.GetContactBiasName("left"), value=v_now)
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-8, maximum_iterations=50)
        if abs(v_now - targets[i_t]) < 1e-12:
            j_e = ds.get_contact_current(device=device, contact="left",
                                         equation=sp.ece_name)
            j_h = ds.get_contact_current(device=device, contact="left",
                                         equation=sp.hce_name)
            currents[targets[i_t]] = abs(j_e + j_h)
            i_t += 1
    return v_bi, currents


def compare(quiet: bool = True):
    """İki konfigürasyonda TARHAN vs DEVSIM; döner: sonuç sözlüğü."""
    from tarhan.models.pn1d import PNDiode1D, iv_sweep, solve_bias
    from tarhan.physics import builtin_potential

    configs = {
        "R0_kisa_taban": dict(dev=PNDiode1D(), tau=1.0,
                              len_p=3e-4, len_n=3e-4),
        "SRH_iki_rejim": dict(dev=PNDiode1D(len_p=25e-4, len_n=25e-4,
                                            tau_n=1e-8, tau_p=1e-8), tau=1e-8,
                              len_p=25e-4, len_n=25e-4),
    }
    volts = (0.1, 0.2, 0.3, 0.4)
    results = {}
    for name, cfg in configs.items():
        dev = cfg["dev"]
        js, _ = iv_sweep(dev, list(volts))
        j_tarhan = dict(zip(volts, js))
        eq = solve_bias(dev, 0.0)
        vbi_tarhan = dev.ut * float(eq["psi"][-1] - eq["psi"][0])

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            vbi_devsim, j_devsim = run_devsim_diode(
                dev.Na, dev.Nd, cfg["len_p"], cfg["len_n"], cfg["tau"],
                ni=dev.ni, ut=dev.ut, eps_s=dev.eps_s, q=dev.q,
                mu_n=dev.mu_n, mu_p=dev.mu_p, volts=volts)

        ratios = {v: j_tarhan[v] / j_devsim[v] for v in volts}
        results[name] = {"vbi_tarhan": vbi_tarhan, "vbi_devsim": vbi_devsim,
                         "vbi_analitik": builtin_potential(dev.Na, dev.Nd, dev.ni, dev.ut),
                         "j_tarhan": j_tarhan, "j_devsim": j_devsim, "ratios": ratios}
        if not quiet:
            print(f"\n== {name} ==")
            print(f"V_bi: TARHAN={vbi_tarhan:.6f}  DEVSIM={vbi_devsim:.6f}  "
                  f"fark={abs(vbi_tarhan-vbi_devsim)*1e6:.1f} uV")
            for v in volts:
                print(f"  V={v:.1f}: J_T={j_tarhan[v]:.4e}  J_D={j_devsim[v]:.4e}  "
                      f"oran={ratios[v]:.4f}")
    return results


if __name__ == "__main__":
    compare(quiet=False)

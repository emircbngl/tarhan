"""Çapraz-oracle: elektrostatik kapasitans, TARHAN vs DEVSIM `cap2d`.

DESIGN-2D aşama 2D-3′ (2D-3 DEĞİL — o blokeli; gerekçe tasarım dokümanında).
Ölçülen şey paralel-plaka kapasitörün KONTAK YÜKÜ: tek bölge, tek tip
geçirgenlik, yüksüz Laplace, iki Dirichlet kontağı.

Neden bu aşama yeni bir şey söylüyor: buradaki cevap `assemble_poisson`'ın
rezidüelinin bir AKI olduğunu kullanır. Kontak düğümleri üzerinde toplanan
kısıtsız rezidüel, o kontağın çevrelediği yüktür — Gauss yasası. 2D-2 aynı
iddiayı süreklilik denklemi için doğrulamıştı (kontak akımı); bu onun
elektrostatik ikizi ve Poisson tarafında hiç sınanmamıştı.

Ölçek çarpanı YOKTUR ve bu tesadüf değil: A/L boyutsuzdur ve yük terimi sıfır
olduğundan hacim hiç girmez, dolayısıyla rezidüel doğrudan ``ε · (A/L) · ψ`` =
C/cm verir ve mesh'in birimi sonucu değiştirmez.

Ölçüm (2026-08-07, DEVSIM 2.10.0, 8281 düğüm / 15636 eleman):
    kontak yükü oranı 1.000000000  (her ikisi 3.350171660e-12 C/cm)
    max |psi_TARHAN - psi_DEVSIM| = 6.6e-13 V   (1 V ölçeğinde)
    yük korunumu: üst + alt = -2.9e-25, yani 3.35e-12'ye karşı makine sıfırı
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import pathlib
import sys
import tempfile

import numpy as np

#: cap2d.py'nin vaka değeri (oksit geçirgenliği).
PERMITTIVITY = 3.9 * 8.85e-14        # F/cm
DEVICE = "MyDevice"
REGION = "air"                       # denklemler yalnız burada; m1/m2 sadece geometri


def run_devsim_cap2d():
    """DEVSIM'in `examples/capacitance/cap2d.py` vakasını koşar.

    Betik import edildiğinde cwd'ye .msh/.dat/vtk yazar, o yüzden geçici bir
    dizinde koşturulur — bir oracle'ın yan etkisi çalışma ağacına sızmamalı.
    """
    import devsim as ds

    # devsim_data pip tarafından sys.prefix altına konur, paketin YANINA değil.
    # Tek bir yol tahmin etmek yerine adaylar denenir ve hiçbiri tutmazsa
    # hepsi birden bildirilir — "bulunamadı" diyen ama nereye baktığını
    # söylemeyen bir hata, hata değildir.
    package_root = pathlib.Path(ds.__file__).resolve().parent
    candidates = [
        pathlib.Path(sys.prefix) / "devsim_data",
        package_root.parent / "devsim_data",
        package_root / "devsim_data",
    ]
    for data in candidates:
        case = data / "examples" / "capacitance"
        if (case / "cap2d.py").exists():
            break
    else:                                          # pragma: no cover
        raise RuntimeError(
            "cap2d.py bulunamadı; denenen yollar: "
            + ", ".join(str(c / "examples" / "capacitance") for c in candidates))

    # DEVSIM'in durumu GLOBAL'dir. Diyot oracle'ı da "MyDevice" adını kullanır,
    # yani bu fonksiyon tek başına koşarken çalışıp tam suite içinde
    # `Device "MyDevice" already exists` ile patlar — teste sıra bağımlılığı
    # olarak görünür, oysa izolasyon eksikliğidir.
    ds.reset_devsim()                              # global solver kaydını da siler →
    try:                                           # UMFPACK shim'ini yeniden kaydet
        from devsim.umfpack import umfshim
        ds.set_parameter(name="direct_solver", value="custom")
        ds.set_parameter(name="solver_callback",
                         value=umfshim.local_solver_callback)
    except ImportError:                            # pragma: no cover
        ds.set_parameter(name="direct_solver", value="superlu")

    previous_cwd = os.getcwd()
    sys.path.insert(0, str(case))
    try:
        with tempfile.TemporaryDirectory() as scratch:
            os.chdir(scratch)
            with contextlib.redirect_stdout(io.StringIO()):
                # import bir kez koşar; ikinci çağrıda cihazı YENİDEN kurmak
                # için reload gerekir, yoksa reset_devsim'den sonra ortada
                # cihaz kalmaz ve okuma adımı boşa düşer.
                if "cap2d" in sys.modules:
                    importlib.reload(sys.modules["cap2d"])
                else:
                    import cap2d                   # noqa: F401
    finally:
        os.chdir(previous_cwd)

    def nodes(name):
        return np.array(ds.get_node_model_values(device=DEVICE, region=REGION,
                                                 name=name))

    return {
        "x": nodes("x"), "y": nodes("y"), "psi": nodes("Potential"),
        "elements": [tuple(t) for t in
                     ds.get_element_node_list(device=DEVICE, region=REGION)],
        "q_top": ds.get_contact_charge(device=DEVICE, contact="top",
                                       equation="PotentialEquation"),
        "q_bot": ds.get_contact_charge(device=DEVICE, contact="bot",
                                       equation="PotentialEquation"),
    }


def solve_tarhan_cap2d(ref):
    """Aynı mesh üzerinde yüksüz Laplace + kontak yükü."""
    from tarhan import backend
    from tarhan.numerics.assemble import assemble_poisson
    from tarhan.numerics.mesh import build_mesh

    mesh = build_mesh(list(zip(ref["x"], ref["y"])), ref["elements"])
    # Kontak düğümleri: Laplace çözümünde potansiyeli TAM 1 ya da TAM 0 olan
    # düğümler yalnızca iletken plakalardır; iç düğümler kesinlikle aradadır.
    top = np.nonzero(ref["psi"] == 1.0)[0]
    bot = np.nonzero(ref["psi"] == 0.0)[0]
    if len(top) == 0 or len(bot) == 0:            # pragma: no cover
        raise RuntimeError("kontak düğümleri bulunamadı")

    fixed = {int(i): 1.0 for i in top}
    fixed.update({int(i): 0.0 for i in bot})
    coef = np.full(len(mesh.edges), PERMITTIVITY)
    zero = np.zeros(mesh.n_nodes)

    system = assemble_poisson(mesh, zero, charge=zero, dcharge_dpsi=zero,
                              edge_coef=coef, dirichlet=fixed)
    psi = backend.solve_sparse(system.rows, system.cols, system.vals,
                               -system.residual, n=system.n_nodes)

    # Yük için BC'SİZ yeniden kurulur: Dirichlet uygulanmış hâlde kontak
    # satırları kimliğe döner ve akı bilgisi silinmiş olurdu.
    flux = assemble_poisson(mesh, psi, charge=zero, dcharge_dpsi=zero,
                            edge_coef=coef)
    return {"psi": psi, "mesh": mesh, "top": top, "bot": bot,
            "q_top": float(flux.residual[top].sum()),
            "q_bot": float(flux.residual[bot].sum())}


def compare(quiet: bool = True):
    ref = run_devsim_cap2d()
    got = solve_tarhan_cap2d(ref)
    results = {
        "n_nodes": len(ref["x"]), "n_elements": len(ref["elements"]),
        "q_top_tarhan": got["q_top"], "q_top_devsim": float(ref["q_top"]),
        "q_bot_tarhan": got["q_bot"], "q_bot_devsim": float(ref["q_bot"]),
        "psi_max_abs_diff": float(np.abs(got["psi"] - ref["psi"]).max()),
        "n_top": len(got["top"]), "n_bot": len(got["bot"]),
    }
    results["q_ratio"] = results["q_top_tarhan"] / results["q_top_devsim"]
    if not quiet:
        print(f"nodes {results['n_nodes']}  elements {results['n_elements']}")
        print(f"q_top TARHAN {results['q_top_tarhan']:.9e}  "
              f"DEVSIM {results['q_top_devsim']:.9e}  ratio {results['q_ratio']:.9f}")
        print(f"max |dpsi| {results['psi_max_abs_diff']:.3e} V")
    return results

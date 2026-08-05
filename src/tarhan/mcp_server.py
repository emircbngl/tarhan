"""TARHAN MCP sunucusu — AI ajanları çekirdeği araç olarak sürer.

Desen: blender-optics MCP köprüsünün torunu (Form Decision: 'MCP bridge
lift-and-shift'). Kurallar:
  - her araç girdisi API sınırında korunur (kuruluş ilkesi #7: sıfır/NaN/absürt
    boyut dürüst hata verir, çökmez/asılmaz)
  - çıktı dizileri MAX_POINTS'e desimatlanır (ajan bağlamını şişirmez)
  - `mcp` bağımlılığı opsiyonel extra'dadır (``pip install 'tarhan[mcp]'``);
    yokluğunda DÜRÜST hata (ilke #6 — sessiz degrade yok)
  - araç fonksiyonları saf Python'dır ve mcp paketi OLMADAN test edilir
Çalıştırma: ``tarhan-mcp`` (stdio; Claude Code/Desktop MCP istemcileriyle).
"""
from __future__ import annotations

import inspect
import math

from tarhan import __version__, physics

MAX_POINTS = 800

#: Canonical upstream source. The value is returned by ``about()`` and shown
#: in the server instructions so MCP clients can identify the running project.
SOURCE_URL = "https://github.com/emircbngl/tarhan"


def _require(**kw):
    for name, val in kw.items():
        if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
            raise ValueError(f"{name}={val!r}: must be a positive, finite number")


def _decimate(xs, ys):
    n = len(xs)
    step = max(1, (n + MAX_POINTS - 1) // MAX_POINTS)
    return ([float(v) for v in xs[::step]], [float(v) for v in ys[::step]])


# ---------------------------------------------------------------- araçlar --- #

def about() -> dict:
    """READ ME FIRST: TARHAN's scope, honesty model and tool guide."""
    return {
        "name": "TARHAN", "version": __version__,
        "scope": ("Fizik-öncelikli malzeme simülatörü (pre-alpha): 1D pn-diyot "
                  "drift-diffusion (Gummel/SG, SRH), voltametri (Nernst + "
                  "Butler-Volmer CV, Nicholson), SOFC 1D hücre gerilimi, "
                  "yakıt-pili kayıp merdiveni, güneş-hücresi FF/Voc."),
        "honesty_model": ("Her formül dürüstlük katmanı taşır (first-principles/"
                          "textbook/empirical); Layer-0 kataloğu basılı sayılara "
                          "sabitli; çözücüler bağımsız TCAD (DEVSIM) ile 1e-3-altı "
                          "mutabık. formula_catalog() ile katmanları görün."),
        "tools": {
            "diode_iv": "1D pn-diyot I-V taraması (ops. SRH ömürleri)",
            "diode_band_diagram": "verilen bias'ta Ec/Ev/EFn/EFp profilleri",
            "cyclic_voltammetry": "tam CV; psi=None reversible, sayı verilirse Butler-Volmer",
            "nicholson_delta_ep": "ΔEp·n [mV] — Nicholson çalışma eğrisi noktası",
            "sofc_polarization": "1D SOFC j-V eğrisi + kayıp ayrıştırması",
            "pemfc_polarization": "0D PEMFC V(i) eğrisi + kayıp ayrıştırması (Spiegel seti)",
            "fuel_cell_losses": "PEMFC kayıp merdiveni tek-noktada (Tafel/ohmik/konsantrasyon)",
            "formula_catalog": "fizik formülleri + dürüstlük katmanları",
        },
        "units_note": "Aksi yazılmadıkça: cm-tabanlı yarıiletken birimleri; V; A/cm².",
        "license": {
            "spdx": "Apache-2.0",
            "source": SOURCE_URL,
            "note": ("Apache-2.0; lisans metnini ve NOTICE dosyasını koruyarak "
                     "kullanabilir, değiştirebilir ve yeniden dağıtabilirsiniz."),
        },
    }


def diode_iv(na_cm3: float = 1e16, nd_cm3: float = 1e16,
             v_start: float = 0.05, v_stop: float = 0.40, v_step: float = 0.05,
             tau_n_s: float | None = None, tau_p_s: float | None = None,
             len_p_um: float = 3.0, len_n_um: float = 3.0) -> dict:
    """1D pn-diode I-V (Gummel/Scharfetter-Gummel; DEVSIM-validated solver)."""
    from tarhan.models.pn1d import PNDiode1D, iv_sweep

    _require(na_cm3=na_cm3, nd_cm3=nd_cm3, v_step=v_step,
             len_p_um=len_p_um, len_n_um=len_n_um)
    if not (na_cm3 <= 1e20 and nd_cm3 <= 1e20):
        raise ValueError("doping must be <= 1e20 cm^-3 (Boltzmann validity)")
    if not (0.2 <= len_p_um <= 500 and 0.2 <= len_n_um <= 500):
        raise ValueError("side lengths must be in the 0.2-500 um range")
    if not (0.0 <= v_start <= v_stop <= 0.6):
        raise ValueError("0 <= v_start <= v_stop <= 0.6 V (yüksek-enjeksiyon sınırı)")
    # Never manufacture a point beyond the caller's validated upper bias.
    # ``round(span / step)`` did exactly that for non-dividing steps: e.g.
    # 0.50..0.60 in 0.06-V steps yielded 0.62 V, outside this API's
    # high-injection guard.  Keep regular step points below v_stop, then add
    # the requested endpoint once when it is not already represented.
    n_full_steps = int(math.floor((v_stop - v_start) / v_step))
    volts = [v_start + k * v_step for k in range(n_full_steps + 1)]
    if not math.isclose(volts[-1], v_stop, rel_tol=0.0, abs_tol=1e-12):
        volts.append(v_stop)
    if len(volts) > 60:
        raise ValueError(f"{len(volts)} bias noktası çok (<=60)")

    dev = PNDiode1D(Na=na_cm3, Nd=nd_cm3, len_p=len_p_um * 1e-4,
                    len_n=len_n_um * 1e-4, tau_n=tau_n_s, tau_p=tau_p_s)
    js, _ = iv_sweep(dev, volts)
    return {"volts": volts, "current_a_cm2": [float(j) for j in js],
            "model": "R=0 kısa-taban" if tau_n_s is None else f"SRH tau={tau_n_s:g}s"}


def diode_band_diagram(bias_v: float = 0.3, na_cm3: float = 1e16,
                       nd_cm3: float = 1e16, e_gap_ev: float = 1.12) -> dict:
    """Band diagram at a given bias (Ec/Ev/EFn/EFp [eV], x [um])."""
    from tarhan.models.pn1d import PNDiode1D, band_diagram, solve_bias

    _require(na_cm3=na_cm3, nd_cm3=nd_cm3, e_gap_ev=e_gap_ev)
    if not (0.0 <= bias_v <= 0.6):
        raise ValueError("0 <= bias_v <= 0.6 V")
    dev = PNDiode1D(Na=na_cm3, Nd=nd_cm3)
    st = solve_bias(dev, bias_v)
    bd = band_diagram(dev, st, e_gap=e_gap_ev)
    x_um = [float(v) * 1e4 for v in bd["x_cm"]]
    out = {"x_um": x_um}
    for key in ("Ec", "Ev", "EFn", "EFp"):
        out["x_um"], out[key] = _decimate(x_um, [float(v) for v in bd[key]])
    return out


def cyclic_voltammetry(psi: float | None = None, alpha: float = 0.5,
                       d_theta: float = 2e-3) -> dict:
    """Full CV (dimensionless). psi=None => reversible (Nernst); a number => Butler-Volmer.

    Returns theta = F(E-E_half)/RT; current is dimensionless (Randles-Sevcik normalised)."""
    from tarhan.numerics.voltammetry import cv_sweep

    if not (0.25 <= alpha <= 0.75):
        raise ValueError("alpha must be in the 0.25-0.75 range")
    if not (5e-4 <= d_theta <= 1e-2):
        raise ValueError("d_theta must be in the 5e-4..1e-2 range")
    K0 = None
    if psi is not None:
        _require(psi=psi)
        K0 = psi * math.sqrt(math.pi)
    th, J, n_f = cv_sweep(K0=K0, alpha=alpha, d_theta=d_theta)
    theta, current = _decimate([float(v) for v in th], [float(v) for v in J])
    return {"theta": theta, "current": current,
            "mode": "reversible (Nernst)" if psi is None else f"Butler-Volmer psi={psi:g}"}


def nicholson_delta_ep(psi: float) -> dict:
    """Nicholson working-curve point: dEp*n [mV, 25 C] (validated to +/-2 mV against 1965 Table I)."""
    from tarhan.numerics.voltammetry import nicholson_peak_separation

    _require(psi=psi)
    if not (0.05 <= psi <= 50):
        raise ValueError("psi must be in the 0.05-50 range (working-curve region)")
    return {"psi": psi, "delta_ep_n_mv": float(nicholson_peak_separation(psi))}


def sofc_polarization(j_max_a_cm2: float = 1.2, points: int = 25) -> dict:
    """1D SOFC j-V curve + loss breakdown at 0.5 A/cm^2 (O'Hayre section 6.2 model)."""
    from tarhan.models.sofc1d import Sofc1DParams, cell_voltage, polarization_curve

    _require(j_max_a_cm2=j_max_a_cm2)
    if not (2 <= int(points) <= 200):
        raise ValueError("points must be in the 2-200 range")
    p = Sofc1DParams()
    grid = [j_max_a_cm2 * (k + 1) / points for k in range(int(points))]
    curve = polarization_curve(grid, p)
    v_mid, losses = cell_voltage(min(0.5, j_max_a_cm2 / 2), p)
    return {"j_a_cm2": [c[0] for c in curve], "v_cell": [c[1] for c in curve],
            "loss_breakdown_at_mid": {"v": v_mid, **losses}}


def pemfc_polarization(j_max_a_cm2: float = 1.35, points: int = 25) -> dict:
    """0D PEMFC V(i) polarizasyon eğrisi + kayıp ayrıştırması (rank-12).

    Parametre seti: Spiegel (2008)/FuelCellStore (E_r=1.229 V, i0=10^-6.912 A/cm²,
    α=0.5, R=0.19 Ω·cm², i_L=1.4 A/cm²); merdiven standart ders-kitabı biçimi,
    oracle-doğrulamalı terimlerle. j_max < i_L olmalı."""
    from tarhan.models.pemfc0d import (
        Pemfc0DParams,
        cell_voltage,
        polarization_curve,
    )

    _require(j_max_a_cm2=j_max_a_cm2)
    if not (2 <= int(points) <= 200):
        raise ValueError("points must be in the 2-200 range")
    p = Pemfc0DParams(e_r=1.229, j0=10.0 ** -6.912, alpha=0.5, n_e=2.0,
                      r_i=0.19, j_l=1.4, T=298.15)
    if not (j_max_a_cm2 < p.j_l):
        raise ValueError(f"j_max must be < i_L={p.j_l} A/cm^2 (concentration limit)")
    grid = [j_max_a_cm2 * (k + 1) / points for k in range(int(points))]
    curve = polarization_curve(p, grid)
    v_mid, losses = cell_voltage(p, min(0.5, j_max_a_cm2 / 2))
    return {"j_a_cm2": [c[0] for c in curve], "v_cell": [c[1] for c in curve],
            "loss_breakdown_at_mid": {"v": v_mid, **losses},
            "params": "Spiegel(2008)/FuelCellStore"}


def fuel_cell_losses(j_a_cm2: float = 0.5, j0_a_cm2: float = 1e-6,
                     alpha: float = 0.5, n: float = 2.0, t_kelvin: float = 353.15,
                     asr_ohm_cm2: float = 0.15, j_l_a_cm2: float = 2.26,
                     c_conc_v: float = 0.1) -> dict:
    """PEMFC loss ladder (oracle-verified formulas; O'Hayre rank-6 set)."""
    _require(j_a_cm2=j_a_cm2, j0_a_cm2=j0_a_cm2, alpha=alpha, n=n,
             t_kelvin=t_kelvin, asr_ohm_cm2=asr_ohm_cm2, j_l_a_cm2=j_l_a_cm2,
             c_conc_v=c_conc_v)
    eta_act = physics.activation_overpotential_tafel(
        j_a_cm2, j0_a_cm2, alpha, n, 96485.0, 8.314, t_kelvin)
    eta_ohm = j_a_cm2 * asr_ohm_cm2
    eta_conc = physics.concentration_overpotential(c_conc_v, j_a_cm2, j_l_a_cm2)
    return {"eta_activation_v": eta_act, "eta_ohmic_v": eta_ohm,
            "eta_concentration_v": eta_conc,
            "total_loss_v": eta_act + eta_ohm + eta_conc}


def formula_catalog() -> list[dict]:
    """Inventory of tarhan.physics — each formula with its honesty layer (principle #4)."""
    out = []
    for name, fn in inspect.getmembers(physics, inspect.isfunction):
        doc = inspect.getdoc(fn) or ""
        lines = doc.splitlines()
        tier = next((ln.strip() for ln in lines if ln.strip().startswith("Katman:")), "")
        out.append({"name": name, "summary": lines[0] if lines else "", "tier": tier})
    return out


_TOOLS = (about, diode_iv, diode_band_diagram, cyclic_voltammetry,
          nicholson_delta_ep, sofc_polarization, pemfc_polarization,
          fuel_cell_losses, formula_catalog)


def build_server():
    """Build the FastMCP server (requires the mcp package)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:                      # dürüst hata — ilke #6
        raise SystemExit(
            "the TARHAN MCP server needs an extra dependency:\n"
            "    pip install 'tarhan[mcp]'") from exc
    server = FastMCP(
        "tarhan",
        instructions=(
            f"TARHAN {__version__} — fizik-öncelikli malzeme simülatörü (pre-alpha). "
            "Önce about() aracını çağırın: kapsam, dürüstlük modeli ve araç rehberi.\n\n"
            f"Lisans: Apache-2.0. Kaynak kod: {SOURCE_URL}\n"
            "Yeniden dağıtımda lisans metnini ve NOTICE dosyasını koruyun."
        ),
        website_url=SOURCE_URL,
    )
    for fn in _TOOLS:
        server.tool()(fn)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()

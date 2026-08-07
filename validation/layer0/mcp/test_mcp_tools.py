"""MCP araç yüzeyi testleri — araç fonksiyonları mcp paketi OLMADAN test edilir.

İlke #7: ajan-sürülebilir yüzeydeki her girdi korunur (sıfır/NaN/absürt boyut →
dürüst ValueError, çökme/asılma/OOM değil). İlke #6: mcp yokken build_server
DÜRÜST SystemExit verir (sessiz degrade yok).
"""
import math

import time

import pytest

from tarhan import mcp_server as m


def test_about_manifest():
    a = m.about()
    assert a["version"] and "tools" in a
    assert len(a["tools"]) == len(m._TOOLS) - 1   # about kendini rehberde listelemez
    assert a["license"]["spdx"] == "Apache-2.0"


def test_diode_iv_happy_path():
    r = m.diode_iv(v_start=0.1, v_stop=0.3, v_step=0.1)
    assert len(r["volts"]) == 3 == len(r["current_a_cm2"])
    assert all(j > 0 for j in r["current_a_cm2"])
    assert r["current_a_cm2"][-1] > r["current_a_cm2"][0]


def test_diode_iv_never_steps_past_requested_stop():
    """A non-dividing step still includes, but never exceeds, v_stop."""
    r = m.diode_iv(v_start=0.50, v_stop=0.60, v_step=0.06)
    assert r["volts"] == pytest.approx([0.50, 0.56, 0.60])
    assert max(r["volts"]) <= 0.60


def test_diode_iv_rejects_tiny_step_without_allocating_it():
    """The point-count guard must run BEFORE the sweep list is built.

    These tools are driven by agents, so v_step is attacker-shaped input: it is
    only checked for positive+finite. Materialising the list first means a
    request that is going to be rejected anyway allocates it — at v_step=1e-9
    that is ~6e8 points. Rejection must be cheap, so this asserts on time, not
    just on the exception.
    """
    start = time.perf_counter()
    with pytest.raises(ValueError, match="too many"):
        m.diode_iv(v_start=0.0, v_stop=0.35, v_step=1e-7)
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize("kw", [
    {"na_cm3": -1e16}, {"na_cm3": float("nan")}, {"nd_cm3": 1e22},
    {"v_start": 0.5, "v_stop": 0.4}, {"v_stop": 0.9},
    {"v_step": 1e-4},                      # >60 nokta
    {"len_p_um": 0.01}, {"len_n_um": 1e4},
])
def test_diode_iv_input_guards(kw):
    with pytest.raises(ValueError):
        m.diode_iv(**kw)


def test_band_diagram_decimated_and_consistent():
    r = m.diode_band_diagram(bias_v=0.2)
    assert len(r["x_um"]) <= m.MAX_POINTS
    gaps = [c - v for c, v in zip(r["Ec"], r["Ev"])]
    assert all(abs(g - 1.12) < 1e-9 for g in gaps)


def test_cv_modes_and_guards():
    r = m.cyclic_voltammetry()                      # reversible
    assert r["mode"].startswith("reversible") and len(r["theta"]) <= m.MAX_POINTS
    r2 = m.cyclic_voltammetry(psi=1.0)
    assert "Butler-Volmer" in r2["mode"]
    for kw in ({"alpha": 0.1}, {"d_theta": 1e-1}, {"psi": -1.0}):
        with pytest.raises(ValueError):
            m.cyclic_voltammetry(**kw)


def test_nicholson_tool_matches_table_point():
    r = m.nicholson_delta_ep(0.5)
    assert abs(r["delta_ep_n_mv"] - 105) < 2.0      # rank-14 doğrulanmış nokta
    with pytest.raises(ValueError):
        m.nicholson_delta_ep(1000.0)


def test_sofc_and_losses():
    r = m.sofc_polarization(points=10)
    assert len(r["j_a_cm2"]) == 10
    assert all(v1 > v2 for v1, v2 in zip(r["v_cell"], r["v_cell"][1:]))  # j ↑ ⇒ V ↓
    L = m.fuel_cell_losses()
    assert math.isclose(L["total_loss_v"],
                        L["eta_activation_v"] + L["eta_ohmic_v"] + L["eta_concentration_v"])
    with pytest.raises(ValueError):
        m.fuel_cell_losses(j_a_cm2=3.0)             # j > j_L dürüst hata


def test_pemfc_polarization():
    r = m.pemfc_polarization(points=12)
    assert len(r["j_a_cm2"]) == 12 == len(r["v_cell"])
    assert all(v1 > v2 for v1, v2 in zip(r["v_cell"], r["v_cell"][1:]))  # j ↑ ⇒ V ↓
    assert 0.7 < r["v_cell"][0] < 1.05                 # ilk j≈0.11 A/cm²'de makul (~0.85)
    assert r["v_cell"][-1] < r["v_cell"][0]            # i_L'ye doğru düşüş
    assert set(r["loss_breakdown_at_mid"]) >= {"v", "eta_act", "eta_ohmic", "eta_conc"}
    with pytest.raises(ValueError):
        m.pemfc_polarization(j_max_a_cm2=1.4)          # j_max >= i_L dürüst hata
    with pytest.raises(ValueError):
        m.pemfc_polarization(j_max_a_cm2=-1.0)


def test_formula_catalog_carries_honesty_tiers():
    cat = m.formula_catalog()
    assert len(cat) >= 15
    tiers = [c for c in cat if c["tier"].startswith("Katman:")]
    assert len(tiers) >= 12                          # ilke #4 yüzeyde görünür


def test_build_server_registers_all_tools():
    # Guard on what the code actually imports, not on the top-level package.
    # mcp 2.0.0 restructured and removed `mcp.server.fastmcp` — the reason
    # pyproject pins mcp>=1.1,<2 — so an environment carrying 2.x satisfies
    # importorskip("mcp"), does not skip, and then fails inside build_server
    # with a SystemExit telling the reader to install an extra that is already
    # installed. A skip guard naming a different module from the one under test
    # is not a guard. Caught by fresh-clone verification, which installs only
    # [dev] and therefore has no pin protecting it.
    pytest.importorskip("mcp.server.fastmcp",
                        reason="needs mcp 1.x: pip install 'tarhan[mcp]'")
    server = m.build_server()
    assert server is not None

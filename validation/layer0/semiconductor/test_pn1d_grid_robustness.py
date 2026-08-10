"""pn1d grid-sağlamlığı: mesh-bağımsızlık bandı + FV eklem-doping regresyon koruması.

GRID-YAKINSAMA ÇALIŞMASI BULGULARI (2026-07-04, dürüst kayıt):
1. Eklem DÜĞÜMÜNE tam-Nd doping atamak baskın O(h) hata terimiydi — FV yarı-hücre
   düzeltmesi ((Nd−Na)/2) uniform 10→5 nm J-farkını ~10× küçülttü (1.3e-8→1.7e-9).
2. J-tabanlı öz-yakınsama MERTEBE ölçümü ~1e-4-bağıl altında Gummel sabit-nokta
   gürültü tabanına çarpar (tol-yolu değişince J ~1e-4-bağıl oynar; tol'u sıkmak
   ince gridde yuvarlama platosuna takılır) — bu bir DIŞ-İTERASYON artefaktı,
   ayrıklaştırma özelliği DEĞİL. TEMİZ MERTEBE MMS ile ölçüldü (2026-07-09,
   `test_pn1d_mms_order.py`): (a) izole `_poisson_newton` ve `_continuity_solve`
   ÜRETİM operatörleri O(h²) (2.000); (b) üretim konvansiyonlarını yeniden üreten
   TEST-YEREL bir ayrıklaştırma, kuplajlı+makine-hassasiyetli çözüldüğünde de O(h²)
   (2.000) — üretimden doğrudan paylaşılan tek parça `bernoulli`; `solve_bias`'ta
   kuplajlı çözüm yolu YOK. **ÜRETİMDEKİ Gummel solve'un
   mertebesi hâlâ ÖLÇÜLMEDİ** — buradaki gürültü tabanı duruyor; kapatmak gerçek bir
   coupled-Newton motor modu ister (2026-07-15 review). Keyfî mertebe İDDİA EDİLMEDİ;
   rank-4 dersinin devamı.
3. Bugün dürüstçe sabitlenen: MESH-BAĞIMSIZLIK BANDI — pratik grid ailesi boyunca
   J'nin bağıl yayılımı küçük (ölçüm: graded-baseline vs uniform 10/5 nm ≤ ~6e-5).
"""
import numpy as np
import pytest

from tarhan.models.pn1d import PNDiode1D, _contact_densities, iv_sweep


# --------------------------------------------------------------------------
# Ohmic contact densities: catastrophic cancellation, and why it stayed hidden
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize("n_dop", [1.0, -1.0, 0.5, -2.0])
def test_contact_densities_hold_mass_action_at_every_scale(delta, n_dop):
    """n_c * p_c must equal delta^2 exactly, however small delta gets.

    The contact solves n - p = N together with n*p = delta^2. Writing both roots
    as (±N + sqrt(N^2 + 4 delta^2))/2 is algebraically right and numerically
    wrong: on the MINORITY side that is a difference of two nearly equal
    numbers, and it degrades as delta shrinks. Measured for |N| = 1 with the
    naive form:

        delta=1e-6 -> n*p/delta^2 = 0.999978   (pn1d's own default; harmless)
        delta=1e-7 -> 0.999201
        delta=1e-8 -> 1.110223                 (11% out)
        delta=1e-9 -> 0.000000                 (minority collapses to zero)

    The last line is the dangerous one, and it is silent: a zero minority
    density becomes a zero Dirichlet value in the continuity solve, mass action
    vanishes, and nothing raises. Since delta = ni/max(Na,Nd), it is reached
    simply by doping harder — 1e18 cm^-3 is an ordinary number and already
    gives delta = 1e-8.

    Computing the majority by ADDITION and the minority by division is exact at
    every scale, which is what this pins.
    """
    n_c, p_c = _contact_densities(n_dop, delta)
    assert n_c > 0.0 and p_c > 0.0
    assert n_c * p_c == pytest.approx(delta * delta, rel=1e-12)
    assert n_c - p_c == pytest.approx(n_dop, rel=1e-12, abs=1e-12)


def test_contact_densities_put_the_majority_on_the_right_side():
    """The sign of N decides which carrier is the majority, checked by hand.

    N = +1 is the n-side, so electrons dominate: n_c = 1, p_c = delta^2.
    N = -1 is the mirror image. Getting this backwards would reverse the diode's
    polarity while still satisfying mass action, so the test above could not
    catch it on its own.
    """
    delta = 1e-8
    n_c, p_c = _contact_densities(1.0, delta)
    assert n_c == pytest.approx(1.0, rel=1e-12)
    assert p_c == pytest.approx(delta * delta, rel=1e-12)

    n_c, p_c = _contact_densities(-1.0, delta)
    assert p_c == pytest.approx(1.0, rel=1e-12)
    assert n_c == pytest.approx(delta * delta, rel=1e-12)


def _j03(h0, gamma):
    seq, _ = iv_sweep(PNDiode1D(h0=h0, gamma=gamma), [0.1, 0.2, 0.3],
                      gummel_tol=1e-10, max_gummel=200)
    return seq[-1]


@pytest.fixture(scope="module")
def j_family():
    return {
        "graded_baseline": _j03(5e-7, 1.06),
        "uniform_10nm": _j03(1e-6, 1.0),
        "uniform_5nm": _j03(5e-7, 1.0),
    }


def test_mesh_independence_band(j_family):
    js = np.array(list(j_family.values()))
    spread = float((js.max() - js.min()) / js.mean())
    assert spread < 3e-4, f"mesh-bağımsızlık bandı aşıldı: {spread:.2e} ({j_family})"


def test_fv_junction_doping_regression_guard(j_family):
    """Yarı-hücre düzeltmesinin geri kaçmasına karşı koruma: uniform 10→5 nm
    J-farkı düzeltme-öncesi seviyeye (1.3e-8) dönerse FAIL."""
    d = abs(j_family["uniform_10nm"] - j_family["uniform_5nm"])
    assert d < 6e-9                                   # ölçüm: 1.7e-9; eski kod: 1.3e-8


def test_a_grid_too_large_to_build_is_refused_before_it_is_built():
    """h0=1e-12 with gamma=1 needs ~6e8 nodes and never returns.

    The earlier guard only fired when the walk stopped making progress at all,
    which this never does — it just takes forever. Estimated from the
    geometric series instead, so the failure is an error message rather than a
    process nobody can interrupt. Reported in re-review.
    """
    import pytest

    from tarhan.models.pn1d import PNDiode1D

    with pytest.raises(ValueError, match="over the"):
        PNDiode1D(h0=1e-12, gamma=1.0)
    assert len(PNDiode1D().build_grid()) == 125


@pytest.mark.parametrize("h0,gamma", [(5e-7, 1.06), (5e-7, 1.0), (1e-7, 1.02),
                                      (2e-6, 1.5), (1e-8, 1.001), (5e-7, 1.3),
                                      (1e-6, 1.1)])
def test_the_node_estimate_matches_the_grid_it_predicts(h0, gamma):
    """The guard is only as good as its arithmetic.

    `estimated_nodes` decides whether a device may be built at all, so an
    estimate that runs low lets a hang through and one that runs high refuses
    a legitimate mesh. Checked against the real builder rather than trusted:
    the first version under-counted by one or two nodes — the junction node
    and a ceiling — which is nothing at the two-million limit and was still
    wrong.
    """
    from tarhan.models.pn1d import PNDiode1D, estimated_nodes

    device = PNDiode1D(h0=h0, gamma=gamma)
    one_side = (len(device.build_grid()) + 1) // 2
    assert estimated_nodes(device.len_n, h0, gamma) == one_side


def test_the_potential_step_is_not_a_claim_about_the_answer():
    """The measurement behind the open convergence finding, run rather than
    argued.

    `max|dpsi| < 1e-9` is what both solvers stop on. It reports ~1e-13 at
    every bias below, while the terminal current's change per outer iteration
    differs by four orders of magnitude between them. A criterion that cannot
    tell those apart is not a claim about the result — which is the reviewer's
    point, and this is the evidence for it.

    No threshold is asserted, because none is established: three
    normalisations were tried and each failed (dividing by |I| breaks at
    equilibrium, dividing by the differenced components flattens everything to
    1e-16, and a cancellation cut-off behaved differently in 1D and 2D). What
    IS asserted is the gap between the two measures, so the finding cannot
    quietly disappear.
    """
    from tarhan.models.pn1d import PNDiode1D, solve_bias

    device = PNDiode1D()
    settled = solve_bias(device, 0.4)
    marginal = solve_bias(device, 0.1)

    # Both report a thoroughly converged potential...
    assert settled["psi_step"] < 1e-9
    assert marginal["psi_step"] < 1e-9

    # ...and the currents behind them are in completely different states.
    #
    # Every bound here is a RATIO. The first version asserted
    # `marginal > 1e-5`, measured 3.49e-04 on the author's machine and failed
    # CI at 4.29e-06 on ubuntu; the second still carried
    # `settled < 1e-7`, which is the same mistake left in place — both were
    # caught in review. The finding is never a magnitude, it is that one bias
    # settles and the other does not, and only a ratio survives a different
    # LAPACK.
    ratio = marginal["current_rel_change"] / settled["current_rel_change"]
    assert ratio > 100, (
        f"the two biases differ by only {ratio:.1f}x in current change and "
        "are no longer distinguishable; if the solver improved, replace this "
        "test with a real convergence gate rather than loosening it")


def test_the_current_reaches_a_noise_floor_and_which_floor_depends_on_bias():
    """The sequence, not one number off the end of it.

    The previous version of this test asked for `max_gummel=8` versus `200`
    and compared the results. The solver exits early on the potential
    tolerance, so BOTH calls ran three iterations and returned identical
    floats: it compared a number with itself and asserted `<=` on equality.
    Reported in review, and it is the second test this session that did not
    test its own claim.

    `min_gummel` exists so a diagnostic can force iterations past that exit.
    With 119 of them the behaviour is visible and it is not what I described
    twice before. Neither bias DECAYS. Both reach a noise floor and wander in
    it indefinitely:

        0.1 V   1e-5 .. 9e-4, still wandering at pass 119
        0.4 V   1e-10 .. 1e-8, from pass 3 onward

    So the finding stands — the potential step says 1e-13 for both while the
    currents behind them differ by five orders of magnitude — but the honest
    description is two different noise floors, not "plateau versus decay".
    This asserts the separation between the floors, which is the part that
    survives a different machine.
    """
    import statistics

    from tarhan.models.pn1d import PNDiode1D, solve_bias

    device = PNDiode1D()

    def floor(bias):
        state = solve_bias(device, bias, min_gummel=60, max_gummel=80)
        history = [x for x in state["current_history"] if x is not None]
        assert len(history) > 40, "min_gummel did not force the iterations"
        return statistics.median(history[10:])      # past the transient

    marginal, settled = floor(0.1), floor(0.4)
    assert marginal / settled > 100, (
        f"the two noise floors are {marginal:.1e} and {settled:.1e}, a factor "
        f"of {marginal / settled:.0f}. They used to differ by ~1e5; if the "
        "solver improved, replace this with a real convergence gate")

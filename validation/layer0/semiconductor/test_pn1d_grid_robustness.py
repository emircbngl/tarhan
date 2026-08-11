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


def test_the_potential_step_does_not_track_the_current_diagnostic():
    """Two biases, one stopping criterion, two very different diagnostics.

    Both stop well inside `gummel_tol`, and their `current_rel_change` values
    differ by orders of magnitude. That is the whole finding: the quantity the
    solver stops on does not track the quantity that describes the answer.

    This test used to say more. It called the two runs `settled` and
    `marginal`, and its docstring said "one bias settles and the other does
    not" — the exact claim withdrawn from the test below it, left standing
    here because I fixed one of the two and reported the claim gone.
    Reported in review, and the fourth time this cycle a change landed on one
    of two symmetric places.

    Without a threshold, a ratio between two diagnostics cannot say which run
    converged. It says they differ, and that is all that is asserted.
    """
    from tarhan.models.pn1d import PNDiode1D, solve_bias

    device = PNDiode1D()
    low = solve_bias(device, 0.1)
    high = solve_bias(device, 0.4)

    # Both satisfied the stopping criterion the solver actually uses.
    assert low["psi_step"] < 1e-9
    assert high["psi_step"] < 1e-9

    # And the current diagnostic behind them is nowhere near each other.
    # Measured 3.49e-4 against 2.80e-9; asserted as a ratio, because the
    # magnitudes move with the LAPACK and the separation does not.
    ratio = low["current_rel_change"] / high["current_rel_change"]
    assert ratio > 100, (
        f"the two diagnostics differ by only {ratio:.0f}x; if the solver "
        "improved, replace this with a real convergence gate")


def test_the_two_biases_stay_orders_apart_after_forced_iterations():
    """The sequence, not one number off the end of it.

    The previous version of this test asked for `max_gummel=8` versus `200`
    and compared the results. The solver exits early on the potential
    tolerance, so BOTH calls ran three iterations and returned identical
    floats: it compared a number with itself. `min_gummel` exists so a
    diagnostic can force iterations past that exit.

    WHAT THIS ASSERTS, and nothing more: after 119 forced iterations the two
    biases' tail medians stay about 1e5 apart (measured 9.7e4 whole, 1.1e5
    first half, 8.7e4 second), while both runs' max|dpsi| sits near 6e-14 —
    and both of those are now CHECKED, not quoted. That is the finding — the potential step is not a claim about
    the answer — and it is all the data supports.

    WHAT IT NO LONGER ASSERTS is that either sequence has "settled to a level"
    or "stopped decaying". Four attempts failed at that, each caught in
    review:

      1. `marginal > 1e-5` — fitted to one machine, red on ubuntu.
      2. `settled < 1e-7` — the same mistake, left one line below the comment
         describing it.
      3. `second > first / 10` — permits a tenfold fall, so a strictly
         decreasing `0.99**i` passed at 0.578.
      4. the overlap fraction below — fails in BOTH directions: a perfect
         constant plateau [1, 1, ...] scores 0.0 and would be REJECTED, while
         a decaying oscillation exp(-0.002i)(1 + 0.4 sin(2*pi*i/7)) scores
         0.40 and would be ACCEPTED.

    The real 0.1 V run has half-medians 3.93e-4 -> 2.02e-4 -> 1.85e-4 ->
    1.78e-4 -> 1.38e-4 -> 1.44e-4 across sixths. That is a slow decline, not
    a level, and no statistic I have distinguishes "converging very slowly"
    from "wandering at a floor" on 119 points. Saying so is the honest
    position; the overlap is measured below as a DESCRIPTION, with no verdict
    attached.
    """
    import statistics

    from tarhan.models.pn1d import PNDiode1D, solve_bias

    device = PNDiode1D()
    passes = 119                     # the number the docstring quotes

    def history(bias):
        state = solve_bias(device, bias, min_gummel=passes,
                           max_gummel=passes + 20)
        values = [x for x in state["current_history"] if x is not None]
        assert len(values) >= passes - 1, \
            f"asked for {passes} iterations and got {len(values)}"
        # The docstring's "while max|dpsi| reports ~1e-13" was quoted and
        # never checked — this function threw the potential step away.
        # Reported in review. Measured 6.04e-14 and 5.51e-14 on the forced
        # path; the bound is loose because the magnitude moves with the
        # LAPACK, but it has to be checked rather than asserted in prose.
        assert state["psi_step"] < 1e-11, (
            f"at {bias} V the forced run's psi_step is {state['psi_step']:.1e}, "
            "so the premise that both biases satisfy the potential criterion "
            "no longer holds")
        return values[10:]           # past the transient

    # Named by their BIAS, not by a verdict. `marginal` and `settled` were
    # the previous names and they carried the withdrawn claim into every line
    # that used them.
    low_bias, high_bias = history(0.1), history(0.4)

    # 1. The two levels are separated. This is the finding.
    separation = statistics.median(low_bias) / statistics.median(high_bias)
    assert separation > 100, (
        f"the two levels differ by only {separation:.0f}x; if the solver "
        "improved, replace this with a real convergence gate")

    # 2. Both halves of each run are ALSO orders apart from the other bias.
    #    This is the separation again, measured on halves rather than on the
    #    whole tail, so it cannot be an artefact of one window. It is not a
    #    claim about decay — see the docstring for why no such claim is made.
    for label, cut in (("first half", slice(None, len(low_bias) // 2)),
                       ("second half", slice(len(low_bias) // 2, None))):
        apart = (statistics.median(low_bias[cut])
                 / statistics.median(high_bias[cut]))
        assert apart > 100, (
            f"in the {label} the two biases are only {apart:.0f}x apart; the "
            "separation is the whole finding, so re-examine it rather than "
            "loosening this")

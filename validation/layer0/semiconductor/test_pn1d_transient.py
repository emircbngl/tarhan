"""1D transient drift-diffusion — the first device in this repo with a clock.

``numerics/transient.py`` has been validated on Robertson stiff kinetics since
before 2D existed, but it had never been coupled to a device: every
semiconductor model here solved the steady state and nothing else. This is that
coupling, and these are the assertions that decide whether it is right.

**The oracle needs no new formula, which matters this session.** physics_verify
is unavailable (the physicist MCP server is disconnected), so anything derived
from a textbook expression would have to ship marked UNVERIFIED. The strongest
check here avoids that entirely: the steady state computed by the Gummel path —
already anchored to 0.57 µV against the analytic built-in potential — must be an
exact fixed point of the transient right-hand side. A sign error, a wrong mass
term, a mismatched Scharfetter–Gummel convention or a broken Poisson reduction
all break that identity. It is the 1D counterpart of stage 2D-0.

What the fixed-point test does NOT establish is the time SCALE: a residual of
zero says nothing about how fast anything moves. That is pinned separately, by
the observation that the device relaxes on the diffusion time L²/D while the
scaled unit is the dielectric relaxation time — a ratio computed independently
of the integrator, so a wrong time unit would show up as relaxation completing
at the wrong t̂.
"""
import math

import numpy as np
import pytest

from tarhan.models.pn1d import (PNDiode1D, _contact_densities, _poisson_linear,
                                solve_bias, time_scale_seconds, transient_rhs,
                                transient_setup, transient_solve)


@pytest.fixture(scope="module")
def dev():
    return PNDiode1D()


def _pieces(dev, v):
    st = solve_bias(dev, v)
    x = st["x_hat"]
    n_dop = dev.doping_hat(x)
    nL, pL = _contact_densities(float(n_dop[0]), dev.delta)
    nR, pR = _contact_densities(float(n_dop[-1]), dev.delta)
    psi_left = v / dev.ut + math.log(nL / dev.delta)
    psi_right = math.log(nR / dev.delta)
    y = np.concatenate([st["n_hat"][1:-1], st["p_hat"][1:-1]])
    return st, x, y, psi_left, psi_right, ((nL, nR), (pL, pR))


# --- the fixed point ------------------------------------------------------

#: Every bound in this file is set from what a DEFECT would produce, not from
#: what one machine happened to measure. That distinction cost a red CI: the
#: Poisson check was first bounded at 1e-10, three digits above the 8.62e-12
#: seen on the author's macOS box — and ubuntu produced 1.624e-10 and windows
#: 1.570e-10 for the same computation, because a different LAPACK rounds the
#: tridiagonal solve differently. Nothing was wrong with the physics; the
#: tolerance had been calibrated to one platform's arithmetic.
#:
#: So these are relative to the scale of the quantity, and chosen with orders of
#: magnitude of headroom over the observed cross-platform spread while staying
#: far below anything a real error could reach. A flipped sign, a missing 1/h̄ or
#: a broken reduction gives O(1), not O(1e-9).
FIXED_POINT_BOUND = 1e-9        # observed 1.6e-13 … 3.7e-13
POISSON_REL_BOUND = 1e-8        # observed 6.2e-13 … 1.2e-11 relative


@pytest.mark.parametrize("v", [0.0, 0.30])
def test_the_steady_solution_is_a_fixed_point_of_the_transient_rhs(dev, v):
    """Measured max|dn̂/dt̂| = 1.59e-13 at equilibrium and 3.69e-13 at 0.30 V.

    A flipped sign, a missing 1/h̄, or the wrong Bernoulli argument would land
    many orders of magnitude higher, not at 1e-13. The floor is the Gummel
    tolerance (1e-9 on ψ̂) rather than machine epsilon, which is why this is not
    asserted at 1e-16.
    """
    _, x, y, psi_l, psi_r, contacts = _pieces(dev, v)
    r = transient_rhs(dev, x, y, psi_l, psi_r, contacts)
    assert np.max(np.abs(r)) < FIXED_POINT_BOUND


@pytest.mark.parametrize("v", [0.0, 0.30])
def test_the_linear_poisson_reproduces_the_newton_solution(dev, v):
    """The reduction argument, put on trial.

    The transient formulation claims that with n̂ and p̂ as state variables the
    charge carries no ψ̂, so Poisson is linear and one tridiagonal solve replaces
    the Newton loop. If that were wrong the two would disagree by something you
    could see, not by a rounding difference.

    Measured absolute differences on a ψ̂ span of ~27.6 thermal volts: 8.62e-12
    (macOS), 1.624e-10 (ubuntu), 1.570e-10 (windows) at equilibrium. Compared
    relative to max|ψ̂| so the assertion means the same thing on each.
    """
    st, x, _, psi_l, psi_r, _ = _pieces(dev, v)
    psi = _poisson_linear(dev, x, st["n_hat"], st["p_hat"], psi_l, psi_r)
    scale = float(np.max(np.abs(st["psi"])))
    assert np.max(np.abs(psi - st["psi"])) / scale < POISSON_REL_BOUND


def test_starting_at_the_steady_state_the_integrator_does_not_wander(dev):
    """Integrating a fixed point must be uneventful. Measured drift 4.50e-12."""
    r = transient_solve(dev, 0.30, t_span_hat=(0.0, 50.0))
    y0 = np.asarray(r["y_steady"])
    drift = float(np.max(np.abs(r["solution"].y[:, -1] - y0) / y0))
    assert drift < 1e-9


# --- relaxation, and the time scale ---------------------------------------

@pytest.fixture(scope="module")
def relaxation(dev):
    r = transient_solve(dev, 0.30)
    y0 = np.asarray(r["y_steady"])
    rng = np.random.default_rng(0)
    perturbed = np.maximum(y0 * (1.0 + 0.05 * rng.standard_normal(y0.shape)),
                           1e-300)
    out = transient_solve(dev, 0.30, y0=perturbed, t_span_hat=(0.0, 2.0e4))
    return y0, perturbed, out


def test_a_perturbation_relaxes_back_to_the_validated_steady_state(relaxation):
    """The claim worth making: the transient path lands where the steady path is.

    A 5% relative perturbation on every interior node decays from 1.553e-1 to
    7.61e-8 — a factor of 2e6 — over 2e4 scaled time units. The target is not an
    analytic expression but the Gummel solution itself, which is anchored to
    0.57 µV against the analytic V_bi, so agreement here inherits that anchor.
    """
    y0, perturbed, out = relaxation
    start = float(np.max(np.abs(perturbed - y0) / y0))
    end = float(np.max(np.abs(out["solution"].y[:, -1] - y0) / y0))
    assert start > 1e-2, "the perturbation was too small to be a test"
    assert end < 1e-6
    assert start / end > 1e4


def test_the_relaxation_happens_on_the_diffusion_time_not_the_scaled_unit(dev):
    """The time-scale anchor, and the reason the system is stiff.

    The scaled unit is the dielectric relaxation time, t0 = εs/(q·C0·μ_scale)
    = 4.79e-13 s here. The device relaxes on the diffusion time L²/D = 2.57e-9 s
    — a ratio of 5.37e3, computed from the device geometry and the Einstein
    relation without reference to the integrator. A wrong time unit would show
    up as relaxation finishing at the wrong t̂, so this pins it.

    UNVERIFIED by physics_verify (server unavailable this session): D = μ·U_T
    and t_diff = L²/D are used as written. The dimensional check was done by
    hand, and the ratio is corroborated by the measured decay above.
    """
    t0 = time_scale_seconds(dev)
    assert t0 == pytest.approx(dev.eps_s / (dev.q * dev.C0
                                            * max(dev.mu_n, dev.mu_p)))
    assert t0 == pytest.approx(4.794e-13, rel=1e-3)
    diffusion_time = dev.len_p ** 2 / (dev.mu_n * dev.ut)
    assert diffusion_time / t0 == pytest.approx(5.37e3, rel=1e-2)


def test_the_stiff_integrator_absorbs_that_ratio_in_a_few_hundred_steps(relaxation):
    """Stiffness is the point of delegating to BDF rather than stepping by hand.

    Measured 425 steps to cross 2e4 scaled units at a stiffness ratio of 5.4e3.
    An explicit method would need steps of order the dielectric time.
    """
    _, _, out = relaxation
    assert len(out["solution"].t) < 2000


def test_the_seconds_axis_is_the_scaled_axis_times_t0(dev):
    r = transient_solve(dev, 0.30, t_span_hat=(0.0, 10.0))
    assert np.allclose(r["t_seconds"],
                       np.asarray(r["solution"].t) * time_scale_seconds(dev))


# --- the setup is borrowed, not re-derived --------------------------------

def test_the_setup_reuses_the_steady_contact_densities(dev):
    """A second copy of the contact algebra is a second place for the
    cancellation bug to come back."""
    setup = transient_setup(dev, 0.30)
    (nL, nR), (pL, pR) = setup["contacts"]
    for n_c, p_c in ((nL, pL), (nR, pR)):
        assert n_c * p_c == pytest.approx(dev.delta ** 2, rel=1e-12)

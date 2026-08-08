"""2D transient drift-diffusion on the box mesh — the time coupling, not the mesh.

What this file tests and what it does not, stated first because the distinction
decides which failures it can catch. The SPATIAL operator is already validated
elsewhere and far more strongly: stages 2D-1 and 2D-2 run it against DEVSIM on
DEVSIM's own 495-node mesh and agree to 2.24e-16 V and an I_n ratio of 1.00000.
Repeating that here would prove nothing new. What is new is the TIME coupling —
the mass term, the linear-Poisson reduction, and the sign of the accumulation —
so the device here is deliberately small and the assertions aim at those.

Two tests, each catching a different error, and neither substitutes for the
other:

* The steady solution must be an exact fixed point of the transient right-hand
  side. This catches a wrong mass term or a broken Poisson reduction — but NOT
  a sign error, because the residual vanishes at the steady state whichever way
  it is fed in.
* A perturbed state must relax back. This is what settles the sign: with the
  accumulation backwards the integration runs away instead of settling.

physics_verify is unavailable this session (the physicist MCP server is
disconnected), so as in the 1D case the oracle is chosen to need no external
formula: the target is the steady solution the Gummel path already produces.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from tarhan.models.pn2d import (PNDiode2D, _poisson_linear,  # noqa: E402
                                transient_rhs, transient_setup,
                                transient_solve)
from test_pn2d_contacts import _strip                        # noqa: E402

#: Bounds are set from what a DEFECT would produce, not from what one machine
#: measured — the lesson a red CI taught on the 1D transient, where a bound
#: fitted to this laptop's 8.62e-12 failed against ubuntu's 1.624e-10 for the
#: identical computation. A flipped sign, a missing volume divisor or a broken
#: reduction gives O(1) here, so these sit many orders below anything real and
#: well above any plausible cross-platform spread.
FIXED_POINT_BOUND = 1e-9        # observed 2.6e-17 … 4.6e-16
POISSON_REL_BOUND = 1e-8        # observed 1.3e-16 … 3.9e-16 relative


@pytest.fixture(scope="module")
def dev():
    pts, tris, doping = _strip()
    return PNDiode2D(points=pts, triangles=tris, net_doping=doping,
                     contacts={"left": [0, 1], "right": [8, 9]},
                     biased_contact="left")


# --- the fixed point ------------------------------------------------------

@pytest.mark.parametrize("v", [0.0, 0.30])
def test_the_steady_solution_is_a_fixed_point_of_the_transient_rhs(dev, v):
    """Measured max|dŷ/dt̂| = 2.60e-17 at equilibrium and 4.60e-16 at 0.30 V.

    Sharper than the 1D counterpart because pn2d's Poisson-Newton runs to 1e-11
    rather than 1e-10, so the steady state it hands over is converged further.
    """
    setup = transient_setup(dev, v)
    r = transient_rhs(dev, setup, setup["y_steady"])
    assert np.max(np.abs(r)) < FIXED_POINT_BOUND


@pytest.mark.parametrize("v", [0.0, 0.30])
def test_the_linear_poisson_reproduces_the_newton_solution(dev, v):
    """The reduction argument on the box mesh.

    Passing ``dcharge_dpsi = 0`` leaves the Jacobian equal to the bare
    box-method Laplacian, so one step lands on the exact solution. Measured
    1.78e-15 and 5.33e-15 absolute on a ψ̂ scale of 13.816 — relative 1.3e-16
    and 3.9e-16, which is machine precision rather than agreement to a
    tolerance.
    """
    setup = transient_setup(dev, v)
    st = setup["state"]
    psi = _poisson_linear(dev, st["n_hat"], st["p_hat"], setup["psi_bc"])
    scale = float(np.max(np.abs(st["psi"])))
    assert np.max(np.abs(psi - st["psi"])) / scale < POISSON_REL_BOUND


def test_starting_at_the_steady_state_the_integrator_does_not_wander(dev):
    """Measured drift 4.05e-11 over 50 scaled time units."""
    r = transient_solve(dev, 0.30, t_span_hat=(0.0, 50.0))
    y0 = np.asarray(r["y_steady"])
    drift = float(np.max(np.abs(r["solution"].y[:, -1] - y0) / y0))
    assert drift < 1e-7


# --- relaxation, which is what fixes the sign ------------------------------

@pytest.fixture(scope="module")
def relaxation(dev):
    r = transient_solve(dev, 0.30, t_span_hat=(0.0, 1.0))
    y0 = np.asarray(r["y_steady"])
    rng = np.random.default_rng(0)
    perturbed = np.maximum(y0 * (1.0 + 0.05 * rng.standard_normal(y0.shape)),
                           1e-300)
    out = transient_solve(dev, 0.30, y0=perturbed, t_span_hat=(0.0, 1.0e3))
    return y0, perturbed, out


def test_a_perturbation_relaxes_back_to_the_steady_state(relaxation):
    """The test that settles the sign of the accumulation.

    A 5% relative perturbation decays from 1.553e-1 to 4.05e-11 by t̂ = 1e3 — a
    factor of 4e9 — and the endpoint is the steady solution's OWN numerical
    floor (4.045e-11 measured with no perturbation at all), so the perturbation
    is gone rather than merely small.

    With the accumulation sign reversed this integration diverges instead, which
    is why the fixed-point test cannot stand in for it: that one passes either
    way.
    """
    y0, perturbed, out = relaxation
    start = float(np.max(np.abs(perturbed - y0) / y0))
    end = float(np.max(np.abs(out["solution"].y[:, -1] - y0) / y0))
    assert start > 1e-2, "the perturbation was too small to be a test"
    assert end < 1e-8
    assert start / end > 1e6


def test_the_relaxation_is_monotone_enough_to_be_a_relaxation(relaxation):
    """A run that wandered and happened to land near the target is not a
    relaxation. The deviation must be no larger at the end than mid-way."""
    y0, _, out = relaxation
    ys = out["solution"].y
    mid = ys.shape[1] // 2
    dev_mid = float(np.max(np.abs(ys[:, mid] - y0) / y0))
    dev_end = float(np.max(np.abs(ys[:, -1] - y0) / y0))
    assert dev_end <= dev_mid


def test_the_1d_and_2d_transients_share_one_clock(dev):
    """Both scale time by the dielectric relaxation time from the same
    reference mobility, so a second on one axis is a second on the other."""
    from tarhan.models.pn1d import time_scale_seconds

    r = transient_solve(dev, 0.30, t_span_hat=(0.0, 10.0))
    assert np.allclose(r["t_seconds"],
                       np.asarray(r["solution"].t) * time_scale_seconds(dev))


def test_only_the_free_nodes_evolve(dev):
    """Contacts are Dirichlet. If a contact node were in the state vector its
    boundary condition would be integrated away."""
    setup = transient_setup(dev, 0.30)
    pinned = set(range(dev.mesh.n_nodes)) - {int(i) for i in setup["free"]}
    assert pinned == {0, 1, 8, 9}
    assert len(setup["y_steady"]) == 2 * len(setup["free"])

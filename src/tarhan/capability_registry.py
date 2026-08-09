"""The records themselves — every capability, and every one deliberately absent.

The schema lives in :mod:`tarhan.capabilities`; this module is data. The split is
not tidiness: a test pins these records against ``docs/DESIGN-2D.md`` §5, and
keeping the data in one importable place is what makes that check possible.

Every ``measured`` string below is copied from the docstring or the assertion of
the named test — not from memory, and not from the README. The README rounds
(0.57 µV appears there as 0.6 µV, 0.0155% as 0.016%), and a registry that
laundered a rounded number into a precise-looking claim would be doing the one
thing this project refuses.
"""
from __future__ import annotations

from typing import Tuple

from tarhan.capabilities import Capability, Evidence

_V = "validation/layer0"


class CapabilityNotFound(KeyError):
    """Asked for a capability id that is not in the registry."""


REGISTRY: Tuple[Capability, ...] = (
    Capability(
        domain="electrochemistry",
        family="diffusion",
        dimension=1,
        time="transient",
        status="validated",
        source="numerics/diffusion1d.py",
        inputs=("diffusion coefficient", "grid", "time targets"),
        produces=("concentration profile", "surface flux"),
        limits=("explicit FD: the step is bound by dt < dx^2/2D",
                "semi-infinite domain; no migration and no convection"),
        evidence=(
            Evidence("Cottrell G(T) = 1/sqrt(pi*T)",
                     "max relative error 0.0155% at n_x=200, observed spatial "
                     "order p=1.97",
                     f"{_V}/electrochem/test_rank00_cottrell_fd.py"),
            Evidence("manufactured solution is reproduced exactly",
                     "~1e-14 per node, independent of the grid",
                     f"{_V}/numerics/test_rank03_mms_exact.py"),
        ),
    ),
    Capability(
        domain="electrochemistry",
        family="chronoamperometry",
        dimension=1,
        time="transient",
        status="validated",
        source="models/chronoamp1d.py",
        inputs=("diffusion coefficient", "bulk concentration", "electrode area",
                "time span"),
        produces=("current transient", "concentration profile"),
        limits=("semi-infinite planar diffusion; no convection",
                "one species, no coupled homogeneous kinetics"),
        evidence=(
            Evidence("implicit BDF path converges to the Cottrell analytic flux",
                     "measured spatial order 2.000",
                     f"{_V}/electrochem/test_chronoamp_transient.py"),
        ),
    ),
    Capability(
        domain="fuelcell",
        family="pemfc",
        dimension=0,
        time="steady",
        status="validated",
        source="models/pemfc0d.py",
        inputs=("E_r", "exchange current density", "alpha", "R_i", "i_L"),
        produces=("polarization curve V(i)", "loss breakdown"),
        limits=("lumped 0D: no spatial gradient anywhere",
                "parameters are Spiegel's set and must be supplied, not assumed"),
        evidence=(
            Evidence("assembly of an oracle-verified loss ladder, no new formula",
                     "0.997 V at 1 mA/cm^2 and 0.582 V at 1 A/cm^2, rolling off "
                     "toward i_L = 1.4 A/cm^2",
                     f"{_V}/fuelcell/test_rank12_pemfc_polarization.py"),
        ),
    ),
    Capability(
        domain="fuelcell",
        family="sofc",
        dimension=1,
        time="steady",
        status="validated",
        source="models/sofc1d.py",
        inputs=("temperature", "current density", "layer thicknesses",
                "conductivities"),
        produces=("cell voltage", "ohmic and activation overpotentials", "ASR"),
        limits=("isothermal; no thermal transport",
                "one printed operating point is the anchor, not a swept design"),
        evidence=(
            Evidence("end-to-end printed chain from O'Hayre 3e Table 6.4",
                     "ASR 0.176 ohm*cm^2, eta_ohmic 0.088 V, eta_cathode "
                     "0.158 V, V 0.754 V — every value printed in the source",
                     f"{_V}/fuelcell/test_rank12_sofc_1d_cell.py"),
        ),
    ),
    Capability(
        domain="semiconductor",
        family="pn.drift-diffusion",
        dimension=1,
        time="steady",
        status="validated",
        source="models/pn1d.py",
        inputs=("doping profile", "grid", "bias", "SRH lifetimes (optional)"),
        produces=("psi, n, p", "terminal current", "band diagram"),
        limits=("isothermal; no impact ionisation and no tunnelling",
                "Boltzmann statistics; degenerate doping is out of range",
                "steady state only — no transient and no small-signal solve"),
        evidence=(
            Evidence("equilibrium V_bi against the analytic value",
                     "0.57 microvolt",
                     f"{_V}/semiconductor/test_flagship_pn1d_gummel.py"),
            Evidence("Shockley ideality at low injection",
                     "1.000-1.002 over 0.15-0.40 V",
                     f"{_V}/semiconductor/test_flagship_pn1d_gummel.py"),
            Evidence("discrete current conservation across the device",
                     "spread ~1e-8",
                     f"{_V}/semiconductor/test_flagship_pn1d_gummel.py"),
            Evidence("minority diffusion length emerges from the profile",
                     "L_p/sqrt(D*tau) = 0.99 with SRH active",
                     f"{_V}/semiconductor/test_pn1d_srh.py"),
        ),
    ),
    Capability(
        domain="semiconductor",
        family="pn.drift-diffusion",
        dimension=2,
        time="steady",
        status="validated",
        source="models/pn2d.py",
        inputs=("triangular mesh — supplied, or generated for the one "
                "rectangular diode shape", "net doping",
                "contact node sets", "bias"),
        produces=("psi, n, p on the mesh", "contact currents", "I-V sweep"),
        limits=("the mesh must be Delaunay; the box method's positivity "
                "guarantee rests on it",
                "one generated shape only — an axis-aligned rectangular pn "
                "diode; any other geometry must be supplied as a mesh",
                "meshes are read, never repaired",
                "steady state only",
                "not exposed on the MCP tool surface"),
        evidence=(
            Evidence("a 1D problem run through the 2D machinery reduces to the "
                     "1D scheme identically",
                     "psi agrees to 1.8e-15 over a 125-node grid spanning 27.6 "
                     "thermal volts",
                     f"{_V}/semiconductor/test_pn2d_equilibrium.py"),
            Evidence("equilibrium pn junction against DEVSIM on DEVSIM's own "
                     "mesh",
                     "max 2.242e-16 V and rms 7.389e-17 V over the 481 solved "
                     "nodes; V_bi 0.953719 V from both codes and from the "
                     "analytic expression",
                     f"{_V}/semiconductor/test_2d1_pn2d_equilibrium.py"),
            Evidence("diode I-V against DEVSIM — the first stage that exercises "
                     "transverse transport",
                     "I_n ratio 1.00000 (rel 2e-4) at every bias; ideality "
                     "1.0119-1.0134 against DEVSIM's own 1.0114-1.0126",
                     f"{_V}/semiconductor/test_2d2_pn2d_iv.py"),
            Evidence("electrostatic contact charge against DEVSIM's cap2d case",
                     "3.350171660e-12 C/cm from both codes, ratio 1.000000000, "
                     "on 8281 nodes and 15636 elements",
                     f"{_V}/semiconductor/test_2d3_capacitance.py"),
        ),
    ),
    Capability(
        domain="semiconductor",
        family="pn.drift-diffusion",
        dimension=1,
        time="transient",
        status="validated",
        source="models/pn1d.py",
        inputs=("doping profile", "grid", "bias", "initial (n, p) state",
                "scaled time span"),
        produces=("time-resolved psi, n, p", "the seconds axis"),
        limits=("no displacement current: the state evolution is solved, but "
                "the terminal current during a transient omits the eps*dE/dt "
                "term, so a switching current cannot be read off yet",
                "no SRH in the transient path — recombination is R=0 there",
                "the bias is held fixed; there is no waveform input",
                "isothermal, Boltzmann statistics, as in the steady path"),
        evidence=(
            Evidence("the validated steady state is an exact fixed point of "
                     "the transient right-hand side",
                     "max|dn/dt| 1.59e-13 at equilibrium and 3.69e-13 at "
                     "0.30 V, against a Gummel tolerance floor of 1e-9",
                     f"{_V}/semiconductor/test_pn1d_transient.py"),
            Evidence("with n and p as state variables Poisson is linear, so "
                     "one tridiagonal solve replaces the Newton loop",
                     "the linear solve reproduces the Newton potential to "
                     "1.2e-11 relative on a 27.6-thermal-volt span, worst case "
                     "across macOS, ubuntu and windows",
                     f"{_V}/semiconductor/test_pn1d_transient.py"),
            Evidence("a perturbed state relaxes back to the steady solution",
                     "a 5% perturbation decays from 1.553e-1 to 7.61e-8 over "
                     "2e4 scaled time units — a factor of 2e6",
                     f"{_V}/semiconductor/test_pn1d_transient.py"),
            Evidence("the time scale is the dielectric relaxation time while "
                     "the device relaxes on the diffusion time",
                     "t0 = 4.794e-13 s against L^2/D = 2.57e-9 s, a stiffness "
                     "ratio of 5.37e3 that BDF crosses in 425 steps",
                     f"{_V}/semiconductor/test_pn1d_transient.py"),
        ),
    ),
    Capability(
        domain="semiconductor",
        family="pn.drift-diffusion",
        dimension=2,
        time="transient",
        status="validated",
        source="models/pn2d.py",
        inputs=("triangular mesh — supplied, or generated for the one "
                "rectangular diode shape", "net doping",
                "contact node sets", "bias", "initial (n, p) state"),
        produces=("time-resolved fields on the mesh", "the seconds axis"),
        limits=("no displacement current, as in 1D: the terminal current omits "
                "the eps*dE/dt term, so a switching current cannot be read off",
                "no SRH in the transient path",
                "the bias is held fixed; there is no waveform input",
                "the spatial operator's own validation is stages 2D-1 and 2D-2 "
                "against DEVSIM; the transient tests exercise the time "
                "coupling on a small strip, not the mesh"),
        evidence=(
            Evidence("the validated steady state is an exact fixed point of "
                     "the transient right-hand side on the box mesh",
                     "max|dy/dt| 2.60e-17 at equilibrium and 4.60e-16 at "
                     "0.30 V",
                     f"{_V}/semiconductor/test_pn2d_transient.py"),
            Evidence("the linear-Poisson reduction holds on the box mesh",
                     "the linear solve reproduces the Newton potential to "
                     "1.3e-16 relative — machine precision, not a tolerance",
                     f"{_V}/semiconductor/test_pn2d_transient.py"),
            Evidence("a perturbed state relaxes back, which is what settles "
                     "the sign of the accumulation",
                     "a 5% perturbation decays from 1.553e-1 to 4.05e-11 by "
                     "t=1e3 scaled units, reaching the steady solution's own "
                     "numerical floor",
                     f"{_V}/semiconductor/test_pn2d_transient.py"),
        ),
    ),
    Capability(
        domain="semiconductor",
        family="mos.capacitance",
        dimension=2,
        time="ac",
        status="blocked",
        inputs=("MOS structure mesh", "gate bias sweep", "small-signal frequency"),
        produces=("C-V curve",),
        reason="The reference case is a small-signal AC solve wired to a lumped "
               "circuit, and this code has neither an AC layer nor a circuit "
               "layer. Stage 2D-3 of docs/DESIGN-2D.md.",
        needs="A complex-valued linear solve linearised about the DC operating "
              "point, plus circuit-node coupling to hold the terminal.",
        does_not_mean="It does NOT mean MOS electrostatics are unsupported. The "
                      "electrostatic half was split out as stage 2D-3' and is "
                      "validated against DEVSIM to a charge ratio of "
                      "1.000000000 — see the 2d steady semiconductor record.",
    ),
    Capability(
        domain="semiconductor",
        family="mosfet.drift-diffusion",
        dimension=2,
        time="steady",
        status="blocked",
        inputs=("MOSFET mesh", "terminal biases"),
        produces=("drain current", "I-V family"),
        reason="The reference mesh is not Delaunay on 22 of its interior edges, "
               "and the box method's positivity guarantee rests on exactly that "
               "property. Stage 2D-4 of docs/DESIGN-2D.md.",
        needs="Either a Delaunay oracle mesh, or an element-based discretisation "
              "that does not depend on the Delaunay condition. Clamping, "
              "re-meshing and edge-flipping each cost more than the stage is "
              "worth, and flipping would destroy the very evidence the "
              "comparison rests on.",
        does_not_mean="It does NOT mean MOSFETs are unsupported in principle. "
                      "The blocker is the reference mesh and the method's "
                      "prerequisite, not the device.",
    ),
    Capability(
        domain="semiconductor",
        family="device.drift-diffusion",
        dimension=3,
        time="steady",
        status="blocked",
        inputs=("tetrahedral mesh", "doping", "contacts", "bias"),
        produces=("psi, n, p in 3D", "terminal currents"),
        does_not_mean="It does NOT mean 3D is unreachable, and it is NOT a "
                      "solver or scaling problem — that earlier claim was "
                      "withdrawn after measurement. The obstacle is the dual "
                      "geometry on tetrahedra, and it is a solved problem in "
                      "the literature; it is simply not solved HERE yet. "
                      "Nothing about 0D, 1D or 2D is affected.",
        reason="No 3D mesh geometry exists. numerics/assemble.py is already "
               "dimension-agnostic — it consumes edge transmissibilities and "
               "node volumes and never asks what shape produced them — so what "
               "is missing is a tetrahedral build_mesh: circumcentre-based "
               "Voronoi facet AREAS instead of lengths, node volumes as "
               "(1/6)*sum(A*L) instead of (1/4), and the positivity condition "
               "restated for tetrahedra.",
        needs="That mesh builder, then validation against DEVSIM's own 3D "
              "diode: examples/diode/gmsh_diode3d.msh, 1417 nodes and 6701 "
              "tetrahedra. MEASURED 2026-08-07 in "
              "validation/layer0/numerics/test_tetrahedral_geometry.py: on a "
              "WELL-CENTRED tetrahedron the circumcentric dual is exact — the "
              "regular tetrahedron gives facet sqrt(2)/3 per edge and the "
              "(1/6)*sum(A*L) volume identity to machine precision. But the "
              "same construction OVERSHOOTS BY EXACTLY TWO on a tetrahedron "
              "whose circumcentre lies outside it, and 2555 of 6701 tetrahedra "
              "in the reference mesh (38.1%) are in that state — against the "
              "0.27% of edges that were enough to block stage 2D-4. A signed "
              "variant was attempted and is NOT trustworthy: it reproduces the "
              "mesh volume to only 66%, so its '26% of edges negative' figure "
              "is withdrawn rather than reported. TWO EARLIER CLAIMS ALSO "
              "WITHDRAWN: that a direct sparse LU 'stops being viable' here "
              "(it conflated a hypothetical 128^3 production mesh with the "
              "1417-node oracle; the validated 2D capacitance case already "
              "runs 8281 nodes through the same solver), and that no reference "
              "mesh existed.",
    ),
    Capability(
        domain="semiconductor",
        family="device.drift-diffusion",
        dimension=3,
        time="transient",
        status="planned",
        inputs=("tetrahedral mesh", "doping", "contacts", "bias waveform"),
        produces=("time-resolved psi, n, p", "terminal current transients"),
        reason="This is the roadmap's '4D': three spatial dimensions plus time, "
               "not a fourth spatial axis. It needs the 3D steady engine first, "
               "and that does not exist.",
        needs="The 3D steady capability, plus coupling the existing transient "
              "primitive (numerics/transient.py) to the device equations — that "
              "primitive is validated on stiff kinetics, never on a device.",
    ),
)


def all_capabilities() -> Tuple[Capability, ...]:
    """Every record, in registry order."""
    return REGISTRY


def get(capability_id: str) -> Capability:
    """Look one up by its derived id.

    Raises :class:`CapabilityNotFound` rather than returning ``None``, so a
    caller that forgets to check cannot silently report nothing as though it
    were something.
    """
    for cap in REGISTRY:
        if cap.id == capability_id:
            return cap
    raise CapabilityNotFound(capability_id)

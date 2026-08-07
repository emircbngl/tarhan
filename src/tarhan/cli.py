"""TARHAN CLI — the zero-config proof: ``tarhan demo``.

İki vaka (``--case``):
  cottrell  — Cottrell kronoamperometri reprodüksiyonu vs analitik G = 1/sqrt(πT)
  diode     — 1D pn-diyot (Gummel/SG amiral gemisi): I-V + band diyagramı;
              ideality 1.00±0.02 öz-denetimi
Demolar KENDİNİ DOĞRULAR: tolerans aşılırsa çıkış kodu 1 (sessiz degrade yok —
kuruluş ilkesi #6).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys

from tarhan import __version__, cliout, physics
from tarhan.capability_registry import CapabilityNotFound, all_capabilities, get
from tarhan.numerics.diffusion1d import cottrell_fd_samples

_CAP_COLUMNS = ("id", "status", "dimension", "time", "source")


def _cap_row(cap) -> dict:
    return {"id": cap.id, "status": cap.status, "dimension": cap.dimension,
            "time": cap.time, "source": cap.source}


def _capabilities_list(out: cliout.Output) -> int:
    caps = all_capabilities()
    out.emit([_cap_row(c) for c in caps], _CAP_COLUMNS)
    stuck = sum(not c.runnable for c in caps)
    out.note(f"{len(caps)} capabilities, {stuck} of them not runnable today. "
             "`tarhan capabilities show <id>` says why.")
    return cliout.EXIT_OK


def _check_import(module: str):
    """Import a dependency and report its version, or why it is not usable."""
    try:
        mod = __import__(module)
    except Exception as exc:                              # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"{module} {getattr(mod, '__version__', 'unknown version')}"


def _check_registry():
    caps = all_capabilities()
    runnable = sum(c.runnable for c in caps)
    return True, (f"{len(caps)} capabilities, {runnable} runnable, "
                  f"{len(caps) - runnable} blocked or planned")


def _check_evidence():
    """Every validated claim must still name a test that exists.

    An installed copy whose evidence points at deleted files is a copy whose
    claims cannot be re-checked. In a wheel the validation tree is not shipped,
    so absence there is expected and reported as such rather than as a fault.
    """
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[2]
    if not (repo / "validation").is_dir():
        return None, "validation tree not present (expected in a wheel install)"
    missing = [ev.test for cap in all_capabilities() for ev in cap.evidence
               if not (repo / ev.test).exists()]
    if missing:
        return False, f"{len(missing)} evidence files missing, e.g. {missing[0]}"
    total = sum(len(c.evidence) for c in all_capabilities())
    return True, f"{total} evidence files present"


@contextlib.contextmanager
def _stdout_to_stderr():
    """Hold the file descriptor, not just ``sys.stdout``.

    DEVSIM prints a BLAS/UMFPACK banner from C at import time. That lands on
    file descriptor 1 directly, so ``contextlib.redirect_stdout`` does not see
    it and it appears in the middle of a ``--format json`` stream, breaking the
    one promise this CLI makes to a machine consumer. Caught by running the new
    doctor command through a pipe — a diagnostic banner is diagnostics, so it
    goes where diagnostics go.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _check_devsim():
    try:
        with _stdout_to_stderr():
            import devsim                                 # noqa: F401
    except Exception:                                     # noqa: BLE001
        return None, "not installed — cross-oracle tests will skip"
    return True, "available for cross-oracle validation"


#: (name, what-it-is, callable). A callable returns (True | False | None,
#: detail); None means optional-and-absent, reported without failing the run.
DOCTOR_CHECKS = (
    ("numpy", "array kernel", lambda: _check_import("numpy")),
    ("scipy", "sparse solver and integrators", lambda: _check_import("scipy")),
    ("matplotlib", "plotting for the demos", lambda: _check_import("matplotlib")),
    ("registry", "loading the capability records", _check_registry),
    ("evidence", "checking every claim still names its test", _check_evidence),
    ("devsim", "optional cross-oracle simulator", _check_devsim),
)


def _capabilities_doctor(out: cliout.Output) -> int:
    """Bring the tools up, and say plainly whether they came up.

    The bar counts completed checks. It is not a timer, and no path advances it
    with elapsed time — the same rule the solver display follows, for the same
    reason: a number that looks measured has to be measured.
    """
    from tarhan.forge import Forge

    results = []
    forge = Forge([name for name, _, _ in DOCTOR_CHECKS], out, style="boot",
                  pin=False)
    with forge:
        for name, detail, check in DOCTOR_CHECKS:
            forge.begin(name, detail)
            forge.tick()
            ok, said = check()
            results.append((name, ok, said))
            forge.finish(said)
        broken = [n for n, ok, _ in results if ok is False]
        if broken:
            forge.failed(f"{len(broken)} of {len(results)} checks failed: "
                         + ", ".join(broken))
        else:
            optional = sum(1 for _, ok, _ in results if ok is None)
            forge.converged(
                f"{len(results) - optional} checks passed"
                + (f", {optional} optional not installed" if optional else ""))

    if out.fmt != "table":
        out.emit([{"check": n,
                   "status": {True: "ok", False: "FAILED", None: "absent"}[ok],
                   "detail": said}
                  for n, ok, said in results],
                 ("check", "status", "detail"))
    return (cliout.EXIT_UNAVAILABLE
            if any(ok is False for _, ok, _ in results) else cliout.EXIT_OK)


def _capabilities_show(out: cliout.Output, capability_id: str) -> int:
    """Print one record; the exit status repeats the verdict for a machine.

    A blocked capability still prints in full and still exits 3. Printing is the
    answer to the question that was asked; the status is that same answer in a
    form a script does not have to parse out of a paragraph.
    """
    try:
        cap = get(capability_id)
    except CapabilityNotFound:
        out.error(f"no such capability: {capability_id}")
        out.note("run `tarhan capabilities list` to see the ids that exist")
        return cliout.EXIT_INPUT

    if out.fmt == "table":
        lines = [f"{'id:':<16}{cap.id}",
                 f"{'status:':<16}{cap.status}",
                 f"{'dimension:':<16}{cap.dimension}",
                 f"{'time:':<16}{cap.time}"]
        if cap.source:
            lines.append(f"{'source:':<16}src/tarhan/{cap.source}")
        for label, items in (("inputs:", cap.inputs),
                             ("produces:", cap.produces),
                             ("limits:", cap.limits)):
            for i, item in enumerate(items):
                lines.append(f"{label if i == 0 else '':<16}{item}")
        for label, text in (("reason:", cap.reason), ("needs:", cap.needs),
                            ("does not mean:", cap.does_not_mean)):
            if text:
                lines.append(f"{label:<16}{text}")
        for ev in cap.evidence:
            lines.append(f"{'evidence:':<16}{ev.claim}")
            lines.append(f"{'':<16}  measured: {ev.measured}")
            lines.append(f"{'':<16}  test:     {ev.test}")
        out.detail("\n".join(lines) + "\n")
    else:
        row = dict(_cap_row(cap))
        row.update({"inputs": list(cap.inputs), "produces": list(cap.produces),
                    "limits": list(cap.limits), "reason": cap.reason,
                    "needs": cap.needs, "does_not_mean": cap.does_not_mean,
                    "evidence": [{"claim": e.claim, "measured": e.measured,
                                  "test": e.test} for e in cap.evidence]})
        out.emit([row], list(row))

    if not cap.runnable:
        out.note(f"{cap.id} is {cap.status}; exit status "
                 f"{cliout.EXIT_UNAVAILABLE} says so without parsing prose")
        return cliout.EXIT_UNAVAILABLE
    return cliout.EXIT_OK


def _demo(save: str | None, show: bool) -> int:
    targets = [0.1 * k for k in range(1, 11)]
    samples, dt, dx = cottrell_fd_samples(targets)

    print(f"TARHAN {__version__} — demo: Cottrell chronoamperometry (explicit FD)")
    print(f"grid: dX={dx:.4f}, dT={dt:.2e} (lam=0.45)\n")
    print(f"{'T':>4} | {'G_sim':>9} | {'G_analytic':>10} | {'err %':>7}")
    print("-" * 40)
    max_err = 0.0
    for t in targets:
        g_an = physics.cottrell_dimensionless(t)
        err = abs(samples[t] - g_an) / g_an * 100.0
        max_err = max(max_err, err)
        print(f"{t:4.1f} | {samples[t]:9.5f} | {g_an:10.5f} | {err:7.4f}")
    ok = max_err < 0.05
    print(f"\nmax relative error: {max_err:.4f}%  (tolerance 0.05%)  "
          f"{'PASS' if ok else 'FAIL'}")

    if save or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tt = np.linspace(0.05, 1.0, 400)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tt, 1.0 / np.sqrt(np.pi * tt), "-", label="analytic  G = 1/sqrt(pi*T)")
        ax.plot(targets, [samples[t] for t in targets], "o", label="TARHAN explicit-FD")
        ax.set_xlabel("T (dimensionless time)")
        ax.set_ylabel("G (dimensionless current)")
        ax.set_title("Cottrell reproduction - max error {:.3f}%".format(max_err))
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
            print(f"plot saved: {save}")
        if show:
            plt.show()

    return 0 if ok else 1


def _demo_diode(save: str | None, show: bool) -> int:
    from tarhan.models.pn1d import PNDiode1D, band_diagram, iv_sweep, solve_bias

    dev = PNDiode1D()
    volts = [0.05 * k for k in range(1, 9)]
    js, _ = iv_sweep(dev, volts)

    print(f"TARHAN {__version__} — demo: 1D pn-diode (Gummel + Scharfetter-Gummel)")
    print(f"Na=Nd={dev.Na:.0e} cm⁻³, ni={dev.ni:.0e}, kT/q={dev.ut} V (case inputs)\n")
    print(f"{'V [V]':>6} | {'J [A/cm²]':>12} | {'n_id':>6}")
    print("-" * 32)
    ok = True
    for i, (v, j) in enumerate(zip(volts, js)):
        nid_s = ""
        if i >= 3:                                       # −1 terimi rejimi: V ≥ 0.15
            nid = (volts[i] - volts[i - 1]) / (dev.ut * math.log(js[i] / js[i - 1]))
            nid_s = f"{nid:.4f}"
            ok &= abs(nid - 1.0) < 0.02
        print(f"{v:6.2f} | {j:12.5e} | {nid_s:>6}")
    print(f"\nideality (0.15-0.40 V) within 1.00+/-0.02: {'PASS' if ok else 'FAIL'}")

    if save or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        st = solve_bias(dev, 0.30)
        bd = band_diagram(dev, st)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.semilogy(volts, js, "o-", label="TARHAN Gummel/SG")
        vv = np.linspace(0.15, 0.40, 50)
        ax1.semilogy(vv, js[5] * np.exp((vv - volts[5]) / dev.ut), "--",
                     label="ideal slope (~60 mV/decade)")
        ax1.set_xlabel("V [V]"); ax1.set_ylabel("J [A/cm²]")
        ax1.set_title("Diode I-V"); ax1.legend()
        x_um = np.asarray(bd["x_cm"]) * 1e4
        ax2.plot(x_um, bd["Ec"], label="E_c")
        ax2.plot(x_um, bd["Ev"], label="E_v")
        ax2.plot(x_um, bd["EFn"], "--", label="E_Fn")
        ax2.plot(x_um, bd["EFp"], "--", label="E_Fp")
        ax2.set_xlabel("x [µm]"); ax2.set_ylabel("E [eV]")
        ax2.set_title("Band diagram @ 0.30 V"); ax2.legend()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
            print(f"plot saved: {save}")
        if show:
            plt.show()
    return 0 if ok else 1


def _force_utf8_stdio() -> None:
    """Windows konsolunda Türkçe çıktıyı kurtar (ilk CI koşusu yakaladı).

    Windows'ta stdout varsayılan olarak yerel kod sayfasını kullanır (ör. cp1252)
    ve çıktımızdaki 'ğ/ş/ı/ü/ö/ç' karakterlerini yazamayıp UnicodeEncodeError ile
    ÇÖKER — sayısal hiçbir sorun olmadığı hâlde demo çıkış kodu 1 döner.
    Bu CI'a özgü DEĞİL: aynı çökme gerçek Windows kullanıcısında da olur, o yüzden
    çözüm ortam değişkeni değil, akışın kendisi.

    reconfigure() Python 3.7+ TextIOWrapper'da vardır; stdout bir boruya ya da
    TextIOWrapper olmayan bir nesneye yönlendirilmişse sessizce atlanır.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _should_show(flag: bool | None) -> bool:
    """Decide whether to open an interactive plot window.

    ``plt.show()`` blocks until the window is closed. When nothing can ever
    close it — a CI job, an ssh session, a container, a piped run — the
    documented first command (`tarhan demo`) hangs forever, and because stdout
    is block-buffered when piped, it hangs having printed nothing at all. So the
    window is opt-in by context: only when stdout is a real terminal, and only
    when a display backend is actually usable. An explicit --show/--no-show
    always wins.
    """
    if flag is not None:
        return flag
    if not sys.stdout.isatty():
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="tarhan",
        description="TARHAN — physics-first materials simulator (pre-alpha)")
    parser.add_argument("--version", action="version", version=f"tarhan {__version__}")
    parser.add_argument("--format", choices=cliout.FORMATS, default="table",
                        help="result format on stdout (default: table). In json "
                             "and csv, stdout carries ONLY the result — every "
                             "note, warning and progress line goes to stderr")
    parser.add_argument("--color", choices=cliout.COLOR_MODES, default="auto",
                        help="colour on stderr (default: auto — only when it is "
                             "a terminal, never in json/csv, never with NO_COLOR)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress notes and progress; errors still print")
    sub = parser.add_subparsers(dest="command")

    p_cap = sub.add_parser(
        "capabilities",
        help="what the engine can actually do, and where it stops")
    cap_sub = p_cap.add_subparsers(dest="cap_command")
    cap_sub.add_parser("list", help="every capability and its status")
    cap_sub.add_parser(
        "doctor",
        help="bring the tools up and check them — run this after installing",
        description="Imports the dependencies, loads the capability registry "
                    "and verifies every claim still names a test that exists. "
                    "The progress bar counts completed checks, never elapsed "
                    "time. Exits 3 if a required check fails.")
    p_show = cap_sub.add_parser(
        "show",
        help="one capability in full",
        description="Exits 3 when the capability is blocked or planned. The "
                    "record still prints; the status is the same answer in a "
                    "form a script does not have to parse.")
    p_show.add_argument("capability_id", metavar="<capability-id>")

    p_demo = sub.add_parser("demo", help="zero-config reproduction demos")
    p_demo.add_argument("--case", choices=("cottrell", "diode"), default="cottrell",
                        help="demo case (default: cottrell)")
    p_demo.add_argument("--save", metavar="PATH", default=None, help="save the plot as a PNG")
    p_demo.add_argument("--show", dest="show", action="store_true", default=None,
                        help="open an interactive plot window (default: only when "
                             "stdout is a terminal)")
    p_demo.add_argument("--no-show", dest="show", action="store_false",
                        help="never open a window (headless/CI)")

    args = parser.parse_args(argv)
    out = cliout.Output(fmt=args.format, color=args.color, quiet=args.quiet)

    try:
        if args.command == "demo":
            # demo keeps its own 0/1 contract, documented in AGENTS.md. It
            # predates this output contract and people already depend on it.
            show = _should_show(args.show)
            if args.case == "diode":
                return _demo_diode(save=args.save, show=show)
            return _demo(save=args.save, show=show)

        if args.command == "capabilities":
            if args.cap_command == "list":
                return _capabilities_list(out)
            if args.cap_command == "doctor":
                return _capabilities_doctor(out)
            if args.cap_command == "show":
                return _capabilities_show(out, args.capability_id)
            p_cap.print_help()
            return cliout.EXIT_INPUT
    except Exception as exc:                       # noqa: BLE001 — the boundary
        # The last line of defence: an unexpected exception is OUR bug, and it
        # gets its own status so a caller can tell it apart from a rejected
        # input. The traceback still goes to stderr, where diagnostics belong.
        import traceback
        traceback.print_exc()
        out.error(f"internal error: {exc}")
        return cliout.EXIT_INTERNAL

    parser.print_help()
    print("\nhint: start with `tarhan capabilities list`, then `tarhan demo`.")
    return cliout.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

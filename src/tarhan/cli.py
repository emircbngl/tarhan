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
import math
import os
import sys

from tarhan import __version__, physics
from tarhan.numerics.diffusion1d import cottrell_fd_samples


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
    sub = parser.add_subparsers(dest="command")

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
    if args.command == "demo":
        show = _should_show(args.show)
        if args.case == "diode":
            return _demo_diode(save=args.save, show=show)
        return _demo(save=args.save, show=show)
    parser.print_help()
    print("\nhint: start with `tarhan demo` or `tarhan demo --case diode`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

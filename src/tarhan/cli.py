"""TARHAN CLI — sıfır-config kanıtı: ``tarhan demo``.

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
import sys

from tarhan import __version__, physics
from tarhan.numerics.diffusion1d import cottrell_fd_samples


def _demo(save: str | None, show: bool) -> int:
    targets = [0.1 * k for k in range(1, 11)]
    samples, dt, dx = cottrell_fd_samples(targets)

    print(f"TARHAN {__version__} — demo: Cottrell kronoamperometri (explicit FD)")
    print(f"grid: dX={dx:.4f}, dT={dt:.2e} (lam=0.45)\n")
    print(f"{'T':>4} | {'G_sim':>9} | {'G_analitik':>10} | {'hata %':>7}")
    print("-" * 40)
    max_err = 0.0
    for t in targets:
        g_an = physics.cottrell_dimensionless(t)
        err = abs(samples[t] - g_an) / g_an * 100.0
        max_err = max(max_err, err)
        print(f"{t:4.1f} | {samples[t]:9.5f} | {g_an:10.5f} | {err:7.4f}")
    ok = max_err < 0.05
    print(f"\nmaks. bağıl hata: {max_err:.4f}%  (tolerans %0.05)  "
          f"{'PASS' if ok else 'FAIL'}")

    if save or show:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tt = np.linspace(0.05, 1.0, 400)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tt, 1.0 / np.sqrt(np.pi * tt), "-", label="analitik  G = 1/√(πT)")
        ax.plot(targets, [samples[t] for t in targets], "o", label="TARHAN explicit-FD")
        ax.set_xlabel("T (boyutsuz zaman)")
        ax.set_ylabel("G (boyutsuz akım)")
        ax.set_title("Cottrell reprodüksiyonu — maks. hata %{:.3f}".format(max_err))
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
            print(f"grafik kaydedildi: {save}")
        if show:
            plt.show()

    return 0 if ok else 1


def _demo_diode(save: str | None, show: bool) -> int:
    from tarhan.models.pn1d import PNDiode1D, band_diagram, iv_sweep, solve_bias

    dev = PNDiode1D()
    volts = [0.05 * k for k in range(1, 9)]
    js, _ = iv_sweep(dev, volts)

    print(f"TARHAN {__version__} — demo: 1D pn-diyot (Gummel + Scharfetter-Gummel)")
    print(f"Na=Nd={dev.Na:.0e} cm⁻³, ni={dev.ni:.0e}, kT/q={dev.ut} V (vaka girdileri)\n")
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
    print(f"\nideality (0.15-0.40 V) 1.00±0.02 içinde: {'PASS' if ok else 'FAIL'}")

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
                     label="ideal eğim (60 mV/dekad-ish)")
        ax1.set_xlabel("V [V]"); ax1.set_ylabel("J [A/cm²]")
        ax1.set_title("Diyot I-V"); ax1.legend()
        x_um = np.asarray(bd["x_cm"]) * 1e4
        ax2.plot(x_um, bd["Ec"], label="E_c")
        ax2.plot(x_um, bd["Ev"], label="E_v")
        ax2.plot(x_um, bd["EFn"], "--", label="E_Fn")
        ax2.plot(x_um, bd["EFp"], "--", label="E_Fp")
        ax2.set_xlabel("x [µm]"); ax2.set_ylabel("E [eV]")
        ax2.set_title("Band diyagramı @ 0.30 V"); ax2.legend()
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
            print(f"grafik kaydedildi: {save}")
        if show:
            plt.show()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tarhan",
        description="TARHAN — physics-first materials simulator (pre-alpha)")
    parser.add_argument("--version", action="version", version=f"tarhan {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="sıfır-config reprodüksiyon demoları")
    p_demo.add_argument("--case", choices=("cottrell", "diode"), default="cottrell",
                        help="demo vakası (varsayılan: cottrell)")
    p_demo.add_argument("--save", metavar="PATH", default=None, help="grafiği PNG olarak kaydet")
    p_demo.add_argument("--no-show", action="store_true", help="pencere açma (headless/CI)")

    args = parser.parse_args(argv)
    if args.command == "demo":
        if args.case == "diode":
            return _demo_diode(save=args.save, show=not args.no_show)
        return _demo(save=args.save, show=not args.no_show)
    parser.print_help()
    print("\nipucu: `tarhan demo` veya `tarhan demo --case diode` ile başlayın.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

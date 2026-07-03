"""TARHAN CLI — sıfır-config kanıtı: ``tarhan demo``.

Demo yakınsamış bir Cottrell kronoamperometri reprodüksiyonu koşar (Britz-tarzı
explicit FD, kod bizim), simülasyonu analitik G = 1/sqrt(πT) ile karşılaştırır,
tabloyu basar ve grafiği çizer. Demo KENDİNİ DOĞRULAR: tolerans aşılırsa çıkış
kodu 1 (sessiz degrade yok — kuruluş ilkesi #6).
"""
from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tarhan",
        description="TARHAN — physics-first materials simulator (pre-alpha)")
    parser.add_argument("--version", action="version", version=f"tarhan {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="sıfır-config Cottrell reprodüksiyon demosu")
    p_demo.add_argument("--save", metavar="PATH", default=None, help="grafiği PNG olarak kaydet")
    p_demo.add_argument("--no-show", action="store_true", help="pencere açma (headless/CI)")

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _demo(save=args.save, show=not args.no_show)
    parser.print_help()
    print("\nipucu: `tarhan demo` ile başlayın.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

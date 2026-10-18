"""Design studies (spec sections 6.2, 6.3): sensitivity of the sized heat shield to V_E, gamma_E,
beta and R_n, and a comparison of the candidate heat-shield materials at the baseline entry.

Sweep ranges are fixed multiples / offsets around the baseline, listed in RANGES.
Usage: python scripts/sweeps.py [--materials ...] [--baseline ...] [--out ...]
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from common import load, parse

from tps.sizing import size, sweep
from tps.trajectory import SkipOutError

MATERIALS = ("li900", "li2200", "frci12")
RANGES = {   # parameter: (label, values as a function of the baseline value, display scale)
    "V_E": ("entry velocity $V_E$ (km/s)", lambda b: np.linspace(b - 600.0, b + 200.0, 6), 1e-3),
    "gamma_E_deg": ("entry angle $|\\gamma_E|$ (deg)", lambda b: b * np.array([0.6, 0.8, 1.0, 1.5, 2.0, 3.0]), -1.0),
    "beta": (r"ballistic coefficient $\beta$ (kg/m$^2$)", lambda b: b * np.array([0.5, 0.7, 1.0, 1.4, 2.0]), 1.0),
    "nose_radius": ("nose radius $R_n$ (m)", lambda b: b * np.array([0.5, 0.7, 1.0, 1.4, 2.0]), 1.0),
}


def safe_sweep(case, name, values):
    rows = []
    for v in values:
        print(f"  sizing {name} = {v:g}", flush=True)
        try:
            rows += sweep(case, name, [v])
        except SkipOutError:
            print(f"  {name} = {v:g}: skip-out, omitted", flush=True)
        except ValueError as err:
            print(f"  {name} = {v:g}: {err}, omitted", flush=True)
    return rows


def main():
    args = parse(__doc__)
    case = load(args)
    lines = ["| Parameter | Value | Thickness (cm) | Areal mass (kg/m²) | Peak q (W/cm²) | Heat load (MJ/m²) | Peak g |",
             "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    results = {}
    for ax, (name, (label, values_of, scale)) in zip(axes.flat, RANGES.items()):
        rows = safe_sweep(case, name, values_of(getattr(case, name)))
        results[name] = rows
        if len(rows) < 2:
            ax.set_visible(False)
            continue
        x = np.array([r["value"] for r in rows]) * scale
        ax.plot(x, [100 * r["thickness"] for r in rows], "o-", color="tab:blue")
        ax.axvline(getattr(case, name) * scale, color="0.6", ls=":", lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel("required thickness (cm)")
        # Areal mass is affine in thickness for one material (glue and skin are fixed), so a
        # second axis would draw the same curve. The table carries the mass.
        for r in rows:
            lines.append(f"| {name} | {r['value']:g} | {100 * r['thickness']:.2f} | {r['areal_mass']:.2f} | "
                         f"{r['peak_q'] / 1e4:.1f} | {r['heat_load'] / 1e6:.1f} | {r['peak_g']:.2f} |")
    fig.suptitle(f"Sensitivity of the sized {case.shield.name} heat shield (dotted: baseline)")
    fig.savefig(args.out / "figures" / "sweeps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # steep vs. shallow trade: peak heating up, heat load down (spec 6.2)
    rows = results["gamma_E_deg"]
    if len(rows) < 2:
        raise SystemExit("entry-angle sweep produced fewer than two sized cases")
    g = -np.array([r["value"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    axes[0].plot(g, [r["peak_q"] / 1e4 for r in rows], "o-")
    axes[0].set_ylabel(r"peak heat flux (W/cm$^2$)")
    axes[1].plot(g, [r["heat_load"] / 1e6 for r in rows], "o-", color="tab:red")
    axes[1].set_ylabel(r"heat load (MJ/m$^2$)")
    axes[2].plot(g, [100 * r["thickness"] for r in rows], "o-", color="tab:green")
    axes[2].set_ylabel("required thickness (cm)")
    for ax in axes:
        ax.set_xlabel(r"entry angle $|\gamma_E|$ (deg)")
    fig.suptitle("Steep vs. shallow entry: peak heating rises, heat load (and thickness) falls")
    fig.savefig(args.out / "figures" / "entry_angle_trade.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # material comparison at the baseline entry (spec 6.3)
    mat_lines = ["| Material | Density (kg/m³) | Thickness (cm) | Areal mass (kg/m²) | Peak surface T (K) | Max use T (K) |",
                 "|---|---|---|---|---|---|"]
    names, masses, thick = [], [], []
    tr = case.trajectory()
    for key in MATERIALS:
        c = load(args, shield=key)
        r = size(c, tr)
        names.append(c.shield.name.split()[0])
        masses.append(r.areal_mass)
        thick.append(100 * r.thickness)
        mat_lines.append(f"| {c.shield.name} | {c.shield.rho:.0f} | {100 * r.thickness:.2f} | {r.areal_mass:.2f} | "
                         f"{r.peak_T_surface:.0f} | {c.shield.max_use_temperature:.0f} |")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].bar(names, thick, color="tab:blue")
    axes[0].set_ylabel("required thickness (cm)")
    axes[1].bar(names, masses, color="tab:orange")
    axes[1].set_ylabel(r"areal mass, all layers (kg/m$^2$)")
    fig.suptitle("Candidate heat-shield materials at the baseline entry")
    fig.savefig(args.out / "figures" / "materials.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    text = "\n".join(lines) + "\n\n" + "\n".join(mat_lines) + "\n"
    (args.out / "results" / "sweeps.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

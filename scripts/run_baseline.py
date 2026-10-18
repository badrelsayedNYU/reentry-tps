"""Baseline case (spec sections 6.1, 6.4, 6.5): size the heat shield, then plot time histories and
temperature profiles through the wall at the sized thickness.

Usage: python scripts/run_baseline.py [--materials ...] [--baseline ...] [--out ...]
"""

import matplotlib.pyplot as plt
import numpy as np
from common import load, parse

from tps.sizing import simulate, size
from tps.trajectory import G0


def main():
    args = parse(__doc__)
    case = load(args)
    tr = case.trajectory()
    r = size(case, tr)
    res = simulate(case, r.thickness, tr)
    s = res.summary()
    t, t_q, t_b = res.t, s["t_peak_q_s"], s["t_peak_T_bondline_s"]
    t_heat = tr.t[-1]

    # ---- time histories on a shared axis (spec 6.4)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout="constrained")
    panels = [
        (axes[0, 0], tr.t, tr.h / 1e3, "altitude (km)"),
        (axes[0, 1], tr.t, tr.V / 1e3, "velocity (km/s)"),
        (axes[1, 0], tr.t, tr.g_load, "deceleration (g)"),
        (axes[1, 1], t, res.q_conv / 1e4, r"stagnation heat flux (W/cm$^2$)"),
        (axes[2, 0], t, res.T_surface, "surface temperature (K)"),
        (axes[2, 1], t, res.T_bondline, "bondline temperature (K)"),
    ]
    for ax, x, y, label in panels:
        ax.plot(x, y, lw=1.3)
        ax.set_ylabel(label)
        ax.axvline(t_q, color="tab:red", ls="--", lw=0.9)
        ax.axvline(t_b, color="tab:purple", ls=":", lw=1.1)
        ax.axvspan(t_heat, t[-1], color="0.93", zorder=0)
    axes[2, 0].axhline(case.shield.max_use_temperature, color="k", lw=0.8)
    axes[2, 0].text(t[-1], case.shield.max_use_temperature, " tile limit", va="center", fontsize=8)
    axes[2, 1].axhline(case.T_limit, color="k", lw=0.8)
    axes[2, 1].axhline(case.T_target, color="k", lw=0.8, ls="--")
    axes[2, 1].text(t[-1], case.T_limit, " limit", va="center", fontsize=8)
    axes[0, 0].plot([], [], "--", color="tab:red", label=f"peak heating, t = {t_q:.0f} s")
    axes[0, 0].plot([], [], ":", color="tab:purple", label=f"bondline peak, t = {t_b:.0f} s")
    axes[0, 0].fill_between([], [], color="0.93", label="soak-back (no convective heating)")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[2]:
        ax.set_xlabel("time from entry interface (s)")
    fig.suptitle(f"Baseline entry, {case.shield.name}, sized thickness {r.thickness * 100:.2f} cm")
    fig.savefig(args.out / "figures" / "baseline_time_histories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- temperature profiles through the wall (spec 6.5)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    xs = np.r_[0.0, res.conduction.wall.x, res.conduction.wall.thickness]
    marks = [(t_q, "peak heating"), (t_heat, "end of heating"), (0.5 * (t_heat + t_b), "soak-back"),
             (t_b, "bondline peak"), (t[-1], "end of run")]
    for tm, label in marks:
        i = int(np.argmin(np.abs(t - tm)))
        ax.plot(1e3 * xs, res.conduction.T[i], label=f"t = {t[i]:.0f} s ({label})")
    for x_if in res.conduction.wall.interfaces:
        ax.axvline(1e3 * x_if, color="0.6", ls=":", lw=1)
    ax.set_xlabel("depth from the hot surface (mm)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("Heat soaking through the wall (dotted: layer interfaces)")
    ax.legend(fontsize=8)
    fig.savefig(args.out / "figures" / "baseline_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "| Quantity | Value |", "|---|---|",
        f"| Heat-shield material | {case.shield.name} |",
        f"| Sized thickness | {r.thickness * 100:.2f} cm |",
        f"| Areal mass (all layers) | {r.areal_mass:.2f} kg/m² |",
        f"| Peak stagnation heat flux | {s['peak_q_W_m2'] / 1e4:.1f} W/cm² at t = {t_q:.0f} s |",
        f"| Heat load | {s['heat_load_J_m2'] / 1e6:.1f} MJ/m² |",
        f"| Peak deceleration | {tr.g_load.max():.2f} g |",
        f"| Peak surface temperature | {s['peak_T_surface_K']:.0f} K "
        f"(material limit {case.shield.max_use_temperature:.0f} K) |",
        f"| Peak bondline temperature | {s['peak_T_bondline_K']:.1f} K at t = {t_b:.0f} s "
        f"(target {case.T_target:.0f} K) |",
    ]
    (args.out / "results" / "baseline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

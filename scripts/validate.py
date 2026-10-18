"""Validation figures for the conduction solver against analytical solutions (spec §5.1-5.3).

The properties here are test-only round numbers, not material data.
Usage: python scripts/validate.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tps.conduction import AdiabaticBC, FluxBC, Layer, TemperatureBC, Wall, solve
from tps.materials import Material
from tps.trajectory import G0, Vehicle, allen_eggers_peaks, allen_eggers_velocity, fly

ROOT = Path(__file__).resolve().parent.parent
TEST = Material.from_values("test slab", rho=1000.0, k=10.0, cp=1000.0)  # alpha = 1e-5 m^2/s
ALPHA = 1e-5


def fig_conduction():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")

    # 1. semi-infinite solid, constant flux
    ax = axes[0]
    q, Ti = 1e5, 300.0
    t = np.linspace(0, 100, 400)
    ax.plot(t, Ti + 2 * q / 10.0 * np.sqrt(ALPHA * t / np.pi), "k-", lw=1.3, label="exact")
    for n, dt, mk in ((25, 4.0, "s"), (200, 0.5, "o")):
        res = solve(Wall([Layer(TEST, 1.0, n, 1.05 if n == 25 else 1.02)]), Ti, 100, dt, FluxBC(q), AdiabaticBC())
        ax.plot(res.t[::max(1, len(res.t) // 25)], res.T_surface[::max(1, len(res.t) // 25)], mk,
                mfc="none", ms=5, label=f"FV, {n} cells, $\\Delta t$ = {dt:g} s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("surface temperature (K)")
    ax.set_title(r"Semi-infinite solid, $q$ = 100 kW/m$^2$")
    ax.legend(fontsize=8)

    # 2. finite slab, sudden surface temperature
    ax = axes[1]
    L, Ts = 0.05, 1000.0
    wall = Wall([Layer(TEST, L, 40)])
    x = np.linspace(0, L, 300)
    for tc, color in ((10, "tab:blue"), (60, "tab:orange"), (250, "tab:green"), (1000, "tab:red")):
        n = np.arange(200)[:, None]
        lam = (2 * n + 1) * np.pi / (2 * L)
        ex = Ts + (Ti - Ts) * np.sum(4 / ((2 * n + 1) * np.pi) * np.sin(lam * x) * np.exp(-lam**2 * ALPHA * tc), axis=0)
        ax.plot(1e3 * x, ex, "-", color=color, lw=1.2)
        res = solve(wall, Ti, tc, 0.5, TemperatureBC(Ts), AdiabaticBC())
        ax.plot(1e3 * wall.x, res.T_cells[-1], "o", color=color, mfc="none", ms=4, label=f"t = {tc} s")
    ax.plot([], [], "k-", label="exact (Fourier series)")
    ax.set_xlabel("depth $x$ (mm)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("Slab, sudden surface temperature, adiabatic back")
    ax.legend(fontsize=8)

    # 3. two-layer steady state
    ax = axes[2]
    A = Material.from_values("A", 1500.0, 0.5, 1200.0)
    B = Material.from_values("B", 2700.0, 150.0, 900.0)
    LA, LB, Th, Tc = 0.02, 0.005, 1200.0, 300.0
    wall = Wall([Layer(A, LA, 20, 1.08), Layer(B, LB, 8)])
    res = solve(wall, Tc, 1e12, 1e12, TemperatureBC(Th), TemperatureBC(Tc))
    qs = (Th - Tc) / (LA / 0.5 + LB / 150.0)
    xs = np.array([0, LA, LA + LB])
    ax.plot(1e3 * xs, [Th, Th - qs * LA / 0.5, Tc], "k-", lw=1.2, label="exact (series resistance)")
    ax.plot(1e3 * wall.x, res.T_cells[-1], "o", mfc="none", ms=4, label="FV (stretched in layer A)")
    ax.axvline(1e3 * LA, color="0.6", ls=":", lw=1)
    ax.text(1e3 * LA * 0.55, 1150, "layer A\n$k$ = 0.5", fontsize=8)
    ax.text(1e3 * LA * 1.03, 1150, "layer B\n$k$ = 150", fontsize=8)
    ax.set_xlabel("depth $x$ (mm)")
    ax.set_ylabel("temperature (K)")
    ax.set_title("Two layers, steady state")
    ax.legend(fontsize=8, loc="center left")

    fig.suptitle("Conduction solver vs. analytical solutions (test-only properties)")
    out = ROOT / "figures" / "validation_conduction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_allen_eggers():
    """Validation 5: trajectory vs. Allen–Eggers (V_E = 7.5 km/s, gamma_E = -20 deg, beta = 500 kg/m^2)."""
    VE, gE, beta = 7500.0, -20.0, 500.0
    veh = Vehicle(beta)
    ae = allen_eggers_peaks(VE, gE, beta)
    runs = (("simulation, no gravity / flat Earth", fly(veh, VE, gE, gravity=False, curvature=False), "tab:blue", "--"),
            ("simulation, full physics", fly(veh, VE, gE), "tab:red", "-"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")

    h = np.linspace(15e3, 120e3, 800)
    axes[0].plot(allen_eggers_velocity(h, VE, gE, beta) / VE, h / 1e3, "k-", lw=2.5, alpha=0.35, label="Allen–Eggers")
    rho = 1.225 * np.exp(-h / 7200.0)
    a_ae = rho * allen_eggers_velocity(h, VE, gE, beta) ** 2 / (2 * beta) / G0
    axes[1].plot(a_ae, h / 1e3, "k-", lw=2.5, alpha=0.35, label="Allen–Eggers")
    for label, tr, color, ls in runs:
        axes[0].plot(tr.V / VE, tr.h / 1e3, ls, color=color, lw=1.2, label=label)
        axes[1].plot(tr.g_load, tr.h / 1e3, ls, color=color, lw=1.2, label=label)
    for key, mk, name in (("decel", "o", "peak deceleration"), ("heating", "s", "peak heating")):
        axes[0].plot(ae[key]["V"] / VE, ae[key]["h"] / 1e3, mk, color="k", mfc="none", ms=8, label=f"A–E {name}")
    axes[0].set_xlabel("$V/V_E$")
    axes[1].set_xlabel("aerodynamic load (g)")
    for ax in axes[:2]:
        ax.set_ylabel("altitude (km)")
        ax.set_ylim(15, 80)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_title(r"Velocity, $V_E$ = 7.5 km/s, $\gamma_E$ = -20°, $\beta$ = 500 kg/m$^2$", fontsize=10)
    axes[1].set_title(f"Deceleration: peak {ae['decel']['a'] / G0:.1f} g (A–E)", fontsize=10)

    ax = axes[2]
    gammas = np.array([-10.0, -15.0, -20.0, -30.0, -45.0, -60.0, -85.0])
    for v, color in ((7500.0, "tab:red"), (15000.0, "tab:purple")):
        resid = []
        for g in gammas:
            tr = fly(veh, v, g)
            ta, _ = tr.peak(lambda V, gg, hh, r: r * V**2 / (2 * beta))
            resid.append(100 * (tr.state(ta)[0] / (v * np.exp(-0.5)) - 1))
        ax.plot(-gammas, resid, "o-", color=color, label=f"$V_E$ = {v / 1e3:g} km/s")
        ax.axhline(100 * 2 * np.e * 7200 * G0 / v**2, color=color, ls=":", lw=1)
    ax.set_xlabel(r"entry angle $|\gamma_E|$ (deg)")
    ax.set_ylabel(r"$V$ at peak deceleration: simulation vs. A–E (%)")
    ax.set_title("Residual: gravity along the path (steep), path curvature (shallow)\n"
                 "dotted: gravity/drag ratio at peak deceleration, $2eHg/V_E^2$", fontsize=9)
    ax.legend(fontsize=8)
    out = ROOT / "figures" / "validation_allen_eggers.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    fig_conduction()
    fig_allen_eggers()


if __name__ == "__main__":
    main()

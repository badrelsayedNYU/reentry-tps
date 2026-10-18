"""Validation 6 (spec section 5.6): bondline peak temperature vs. grid spacing and time step,
at the baseline sized thickness. Each series refines one parameter with the other held fine; the
error is measured against the finest run and the observed order is the log2 of successive ratios.

Usage: python scripts/convergence.py [--materials ...] [--baseline ...] [--out ...]
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from common import load, parse

from tps.sizing import peak_bondline, size


def series(case, tr, L, field, values):
    return np.array([peak_bondline(replace(case, **{field: v}), L, tr) for v in values])


def main():
    args = parse(__doc__)
    case = load(args)
    tr = case.trajectory()
    L = size(case, tr).thickness

    cells = np.array([10, 20, 40, 80, 160, 320])
    dts = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25])
    T_cells = series(replace(case, dt=0.25, shield_stretch=1.0), tr, L, "shield_cells", cells)
    T_dt = series(replace(case, shield_cells=160), tr, L, "dt", dts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    lines = ["| Refinement | Values | Peak bondline T (K) | Error vs finest (K) | Observed order |",
             "|---|---|---|---|---|"]
    for ax, x, T, label, name in ((axes[0], L / cells, T_cells, r"heat-shield cell size $\Delta x$ (mm)", "cells"),
                                  (axes[1], dts, T_dt, r"time step $\Delta t$ (s)", "dt")):
        err = np.abs(T[:-1] - T[-1])
        order = np.log2(err[:-1] / err[1:])
        xx = 1e3 * x[:-1] if name == "cells" else x[:-1]
        ax.loglog(xx, err, "o-", label="|T_peak - T_peak(finest)|")
        ref = 2 if name == "cells" else 1
        ax.loglog(xx, err[-1] * (xx / xx[-1]) ** ref, "k--", lw=0.8, label=f"slope {ref}")
        ax.set_xlabel(label)
        ax.set_ylabel("bondline peak error (K)")
        ax.legend(fontsize=8)
        for i, v in enumerate(cells if name == "cells" else dts):
            lines.append(f"| {name} | {v:g} | {T[i]:.3f} | {abs(T[i] - T[-1]) if i < len(T) - 1 else 0:.3f} | "
                         f"{order[i - 1]:.2f} |" if 0 < i < len(T) - 1 else
                         f"| {name} | {v:g} | {T[i]:.3f} | {abs(T[i] - T[-1]):.3f} | |")
    axes[0].set_title("Space (uniform cells, $\\Delta t$ = 0.25 s)")
    axes[1].set_title("Time (backward Euler, 160 cells)")
    fig.suptitle(f"Convergence of the bondline peak temperature, {case.shield.name}, L = {L * 100:.2f} cm")
    fig.savefig(args.out / "figures" / "convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (args.out / "results" / "convergence.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

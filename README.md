# Reentry Heating & TPS Thermal Model

How thick does a capsule's heat shield have to be? This project answers that by coupling three
models — an entry trajectory, a stagnation-point heating correlation, and a transient multi-layer
conduction model of the wall — and searching for the thinnest heat shield that keeps the bondline
(the adhesive between the tiles and the aluminium structure) below its temperature limit, including
the heat that keeps soaking inward long after the heating has stopped.

> **Status:** models implemented and validated (below). Baseline sizing, sensitivity studies and
> material comparison are pending the baseline mission definition and wall layup.

## Why this matters

Aerothermodynamic heating is the defining constraint of entry-vehicle design. A capsule returning
from low Earth orbit converts roughly 30 MJ of kinetic energy per kilogram of vehicle into heat in
the shock layer; most of it is carried away by the flow, but the fraction that reaches the surface
still drives the vehicle's mass. Thermal-protection mass is paid for on every flight, so sizing it
accurately — not just conservatively — matters.

## The three coupled models

**1. Trajectory** — planar point mass over a non-rotating spherical Earth (`src/tps/trajectory.py`):

$$\dot V = -\frac{D}{m} - g\sin\gamma,\qquad
\dot\gamma = \frac{1}{V}\Big[\frac{L}{m} - \Big(g - \frac{V^2}{R_E+h}\Big)\cos\gamma\Big],\qquad
\dot h = V\sin\gamma$$

with $D/m = \rho V^2/2\beta$, $L = (L/D)\,D$, $g = g_0 (R_E/(R_E+h))^2$ and an exponential
atmosphere $\rho = \rho_0 e^{-h/H}$ ($H$ = 7.2 km). Integrated with RK45 (rtol 1e-8) from the
120 km entry interface to $V$ = 500 m/s; skip-out is detected.

**2. Stagnation-point heating** — Sutton–Graves (`src/tps/heating.py`):

$$q_{conv} = k\,\sqrt{\rho/R_n}\,V^3,\qquad k = 1.7415\times10^{-4}\ \mathrm{kg^{1/2}/m}$$

with an optional hot-wall correction $q(1 - h_w/h_0)$, and re-radiation from the surface:
$-k\,\partial T/\partial x|_0 = q_{conv} - \varepsilon\sigma(T_s^4 - T_\infty^4)$, where $T_\infty$ comes from
the 1976 U.S. Standard Atmosphere (checked against the printed tables at six altitudes).

**3. Wall conduction** — $\rho c_p(T)\,\partial T/\partial t = \partial_x(k(T)\,\partial_x T)$ through
[heat shield | bondline | structure] (`src/tps/conduction.py`):
- implicit finite volume (backward Euler, Crank–Nicolson option), tridiagonal solve per step;
- cells never straddle a layer boundary; series (harmonic-mean) conductance at interfaces;
- geometric stretching clusters cells at the hot surface;
- energy-conserving enthalpy form with temperature-dependent $k$ and $c_p$; the $T^4$ boundary
  condition is Newton-linearized and iterated to 1e-6 K each step;
- the run continues after heating ends (soak-back) until the bondline peak has passed.

**Sizing** (`src/tps/sizing.py`): Brent's method on the heat-shield thickness, such that the peak
bondline temperature equals the limit minus a margin. The result is never thinner than needed and
at most 0.1 mm thicker.

## Assumptions

- Non-ablating wall: a reusable conduction/re-radiation system (ceramic tiles), no pyrolysis or
  surface recession.
- No shock-layer radiative heating (valid below about 10 km/s, i.e. for low-Earth-orbit returns).
- Stagnation point only, one-dimensional conduction through the wall.
- The tile coating is represented only by its emissivity; the strain-isolation pad used on the
  Space Shuttle between tile and adhesive is not modelled.
- Tile conductivity at 1 atm (the highest value in the tables, conservative for the bondline; the
  bondline peaks during soak-back, low in the atmosphere).

## Materials

All properties are from NASA RP-1289, *Thermal Protection Materials: Thermophysical Property Data*
(Williams & Curry, 1992), transcribed in their printed units with page citations in
[`data/materials.yaml`](data/materials.yaml) and converted to SI on loading.

| Layer | Material | Density (kg/m³) | Max use temperature (K) |
|---|---|---|---|
| Heat shield | LI-900 silica tile | 144 | 1533 |
| Heat shield | LI-2200 silica tile | 352 | 1756 |
| Heat shield | FRCI-12 fibrous composite | 192 | 1756 |
| Coating | RCG, emissivity 0.85 | — | — |
| Bondline | RTV-560 silicone adhesive | 1410 | 561 |
| Structure | Aluminium 2219 | 2813 | 450 |

## Validation

### Conduction solver vs. analytical solutions

![Conduction validation](figures/validation_conduction.png)

| Case | Reference | Result |
|---|---|---|
| Semi-infinite solid, constant flux | $T_s - T_i = (2q/k)\sqrt{\alpha t/\pi}$ | within 0.5%; first order in time, second order in space, as designed |
| Finite slab, sudden surface temperature | Fourier series | within 0.2% of the temperature step |
| Two-layer steady state | series thermal resistance | exact to round-off; slope ratio = inverse conductivity ratio |
| $k(T) = a + bT$ steady state | Kirchhoff transform | second-order convergence |
| Constant heating with re-radiation | radiative equilibrium $\varepsilon\sigma(T^4 - T_\infty^4) = q$ | to 1e-6 |
| **Energy conservation** with radiation and tabulated $k(T)$, $c_p(T)$ | $\int(q_{in} - q_{rad} - q_{back})\,dt = \Delta E_{stored}$ | **closes to 1e-13** (spec: 0.1%) |

### Trajectory vs. Allen–Eggers

![Allen–Eggers validation](figures/validation_allen_eggers.png)

Allen & Eggers' ballistic solution assumes constant flight-path angle and no gravity. With gravity
and curvature switched off, the integrated trajectory reproduces it to 1e-6. With full physics, on
the steep ballistic case ($V_E$ = 7.5 km/s, $\gamma_E$ = −20°, $\beta$ = 500 kg/m²):

| | Allen–Eggers | Simulation | Difference |
|---|---|---|---|
| $V/V_E$ at peak deceleration | $e^{-1/2}$ = 0.6065 | 0.6127 | +1.0% |
| $V/V_E$ at peak heating | $e^{-1/6}$ = 0.8465 | 0.8571 | +1.25% |
| Peak deceleration | 50.1 g | 52.6 g | +4.9% |

The residual is the gravity component along the path, which Allen–Eggers neglects: it keeps the
vehicle slightly faster at every altitude. Its size relative to drag at peak deceleration is
$2eHg/V_E^2$, so doubling the entry speed should cut it about fourfold — the simulation shows
exactly that (right panel). For shallow entries, path curvature takes over as the dominant effect.

### Grid and time-step convergence

*Pending (computed at the baseline sized thickness).*

## Results

*Pending: baseline time histories, sized thickness and areal mass, sensitivity to entry speed,
angle, ballistic coefficient and nose radius, the steep-vs-shallow trade, and the material
comparison.*

## Limitations and next steps

- **Stagnation point only.** Next: Lees' laminar heating distribution around the blunt body, to size
  the shield away from the nose.
- **No ablation.** Next: pyrolysis and surface recession for ablative heat shields (Avcoat, PICA),
  needed for faster lunar and Mars returns.
- **No shock-layer radiation**, which becomes significant above roughly 10 km/s.
- **Simplified atmosphere** for the trajectory (exponential).
- **A correlation instead of a flow solution.** In blunt-body CFD, a resolved shock layer gives the
  wall heat flux directly; replacing Sutton–Graves with CFD heating at a few trajectory points is
  the natural link between this model and that work.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # validation suite
python scripts/make_figures.py  # every figure and results table
```

Design inputs live in [`data/baseline.yaml`](data/baseline.yaml) and
[`data/materials.yaml`](data/materials.yaml); the scripts refuse to run while any value is still
marked TODO.

## References

- Allen, H. J.; Eggers, A. J.: *A Study of the Motion and Aerodynamic Heating of Ballistic Missiles
  Entering the Earth's Atmosphere at High Supersonic Speeds*, NACA Report 1381, 1958.
- Sutton, K.; Graves, R. A.: *A General Stagnation-Point Convective-Heating Equation for Arbitrary
  Gas Mixtures*, NASA TR R-376, 1971.
- Williams, S. D.; Curry, D. M.: *Thermal Protection Materials: Thermophysical Property Data*,
  NASA RP-1289, 1992.
- *U.S. Standard Atmosphere, 1976*, NOAA-S/T 76-1562.
- Anderson, J. D.: *Hypersonic and High-Temperature Gas Dynamics*, AIAA.
- Incropera, F. P. et al.: *Fundamentals of Heat and Mass Transfer*, Wiley.
- Patankar, S. V.: *Numerical Heat Transfer and Fluid Flow*, Hemisphere, 1980.

"""Material properties: constants or temperature tables, with an exact enthalpy integral.

A property is either a constant or a table of (T, value) pairs, linearly interpolated and held
constant beyond the table ends. Specific enthalpy e(T) = integral of c_p dT is evaluated exactly
for that piecewise-linear c_p, so the conduction solver can use the energy-conserving form.
"""

from dataclasses import dataclass

import numpy as np


class Constant:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, T):
        return np.full_like(np.asarray(T, dtype=float), self.value)

    def integral(self, T):
        """integral_0^T value dT' = value * T."""
        return self.value * np.asarray(T, dtype=float)


class Table:
    """Piecewise-linear property in T, constant extrapolation outside [T_0, T_last]."""

    def __init__(self, T, values):
        self.T = np.asarray(T, dtype=float)
        self.v = np.asarray(values, dtype=float)
        if self.T.ndim != 1 or self.T.size < 2 or np.any(np.diff(self.T) <= 0):
            raise ValueError("table temperatures must be strictly increasing, at least 2 points")
        # cumulative integral at the nodes, measured from T = 0 with constant value below T_0
        seg = 0.5 * (self.v[1:] + self.v[:-1]) * np.diff(self.T)
        self.cum = self.v[0] * self.T[0] + np.r_[0.0, np.cumsum(seg)]

    def __call__(self, T):
        return np.interp(T, self.T, self.v)

    def integral(self, T):
        """integral_0^T f(T') dT' for the piecewise-linear f (exact)."""
        T = np.asarray(T, dtype=float)
        out = np.empty_like(T)
        lo, hi = T <= self.T[0], T >= self.T[-1]
        out[lo] = self.v[0] * T[lo]
        out[hi] = self.cum[-1] + self.v[-1] * (T[hi] - self.T[-1])
        mid = ~(lo | hi)
        if np.any(mid):
            j = np.searchsorted(self.T, T[mid]) - 1
            dT = T[mid] - self.T[j]
            slope = (self.v[j + 1] - self.v[j]) / (self.T[j + 1] - self.T[j])
            out[mid] = self.cum[j] + self.v[j] * dT + 0.5 * slope * dT**2
        return out


def make_property(spec):
    """A float gives a Constant; a pair of sequences (T, values) gives a Table."""
    if np.isscalar(spec):
        return Constant(spec)
    T, values = spec
    return Table(T, values)


@dataclass
class Material:
    name: str
    rho: float          # density [kg/m^3]
    k: object           # conductivity [W/(m K)], Constant or Table
    cp: object          # specific heat [J/(kg K)], Constant or Table
    emissivity: float = None
    source: str = ""
    max_use_temperature: float = None   # K

    @classmethod
    def from_values(cls, name, rho, k, cp, emissivity=None, source="", max_use_temperature=None):
        return cls(name, float(rho), make_property(k), make_property(cp), emissivity, source,
                   max_use_temperature)

    def enthalpy(self, T):
        """Specific enthalpy e(T) = integral_0^T c_p dT' [J/kg] (exact for the tabulated c_p)."""
        return self.cp.integral(T)


class MissingDataError(ValueError):
    """Raised when data/materials.yaml still contains TODO placeholders."""


def _todos(node, path=""):
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _todos(v, f"{path}.{k}" if path else str(k))]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _todos(v, f"{path}[{i}]")]
    return [path] if isinstance(node, str) and node.strip().upper().startswith("TODO") else []


# Unit conversions for entries transcribed in the source's own units (exact, IT Btu):
#   T: R -> K, k: Btu/(ft h R) -> W/(m K), c_p: Btu/(lbm R) -> J/(kg K), rho: lbm/ft^3 -> kg/m^3
UNITS = {
    "si": dict(T=1.0, k=1.0, cp=1.0, rho=1.0),
    "english": dict(T=1.0 / 1.8, k=1.730734666, cp=4186.8, rho=16.01846337),
}


def _prop(spec, fT, fv):
    if isinstance(spec, dict):
        return make_property((np.asarray(spec["T"], float) * fT, np.asarray(spec["value"], float) * fv))
    return make_property(float(spec) * fv)


def load_materials(path, require=None):
    """Read the materials file. Returns (materials: dict key -> Material (SI), raw data dict).

    Entries may be written in their source's units (`units: english`) so every number can be checked
    against the printed table; conversion to SI happens here. Refuses to return anything if a
    required entry (or, by default, anything in the file) still holds a TODO: material property
    values are never guessed (spec ground rules).
    """
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh)
    keys = require if require is not None else list(data.get("materials", {}))
    scope = {k: data["materials"][k] for k in keys}
    missing = _todos(scope, "materials")
    if require is None:
        for section in ("limits",):
            missing += _todos(data.get(section, {}), section)
    if missing:
        raise MissingDataError("data/materials.yaml has values still to be supplied (with a source): "
                               + ", ".join(missing))
    materials = {}
    for key, e in scope.items():
        f = UNITS[e.get("units", "si")]
        t_max = e.get("max_use_temperature")
        materials[key] = Material(e["name"], float(e["rho"]) * f["rho"], _prop(e["k"], f["T"], f["k"]),
                                  _prop(e["cp"], f["T"], f["cp"]), e.get("emissivity"), e["source"],
                                  None if t_max is None else float(t_max) * f["T"])
    return materials, data

"""Atmosphere models (spec §2.1).

1976 reference values are read from the printed tables of U.S. Standard Atmosphere, 1976
(NOAA-S/T 76-1562, NASA NTRS 19770009539), Table I, geometric altitude; pressures converted
from mb to Pa. At 86 km the table's kinetic temperature (186.87 K) differs from the molecular-scale
temperature the layer formulas give (186.946 K) by 0.04%.
"""

import numpy as np
import pytest

from tps.atmosphere import RHO0, exponential_density, temperature, us1976

TABLE = {  # h [m]: (T [K], p [Pa], rho [kg/m^3])
    0: (288.150, 101325.0, 1.2250),
    11000: (216.774, 22699.0, 0.36480),
    27000: (223.536, 1879.9, 2.9298e-2),
    38000: (244.818, 377.13, 5.3666e-3),
    60000: (247.021, 21.958, 3.0968e-4),
    86000: (186.87, 0.37338, 6.958e-6),
}


@pytest.mark.parametrize("h", TABLE)
def test_us1976_matches_printed_table(h):
    T, p, rho = TABLE[h]
    a = us1976(h)
    if h == 86000:  # kinetic vs. molecular-scale temperature differ by 0.04% here
        assert a["T"][0] == pytest.approx(T, rel=5e-4)
    else:           # printed to 0.001 K
        assert a["T"][0] == pytest.approx(T, abs=6e-4)
    assert a["p"][0] == pytest.approx(p, rel=1e-4)
    assert a["rho"][0] == pytest.approx(rho, rel=1e-4)


def test_temperature_held_above_86_km():
    np.testing.assert_allclose(temperature([90e3, 120e3]), temperature(86e3)[0], rtol=1e-12)


def test_exponential_density():
    assert exponential_density(0.0) == RHO0
    assert exponential_density(7200.0) == pytest.approx(RHO0 / np.e)
    assert exponential_density(14400.0, H=7200.0) == pytest.approx(RHO0 / np.e**2)

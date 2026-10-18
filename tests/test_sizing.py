"""Sizing by root-finding and the configuration loader (spec section 6.1). Test-only properties."""

from dataclasses import replace
from pathlib import Path

import pytest

from tps.materials import Material, MissingDataError
from tps.sizing import Case, load_case, peak_bondline, size

ROOT = Path(__file__).resolve().parent.parent
T_TABLE = [200.0, 600.0, 1200.0, 2000.0]
SHIELD = Material.from_values("test shield", 200.0, (T_TABLE, [0.04, 0.06, 0.10, 0.20]),
                              (T_TABLE, [700.0, 1100.0, 1250.0, 1300.0]), emissivity=0.85,
                              source="test-only")
GLUE = Material.from_values("test glue", 1400.0, 0.3, 1200.0, source="test-only")
METAL = Material.from_values("test metal", 2800.0, 130.0, 880.0, source="test-only")


@pytest.fixture(scope="module")
def case():
    return Case(SHIELD, GLUE, METAL, 0.0002, 0.002, 295.0, V_E=7800.0, gamma_E_deg=-2.0, beta=350.0,
                lift_to_drag=0.3, nose_radius=2.0, shield_cells=20, dt=2.0, soak_time=1800.0,
                T_limit=450.0, margin=20.0)


@pytest.fixture(scope="module")
def sized(case):
    tr = case.trajectory()
    return tr, size(case, tr, tol=2e-4)


def test_sized_thickness_meets_target(case, sized):
    tr, r = sized
    assert r.peak_T_bondline <= case.T_target + 0.5
    assert peak_bondline(case, r.thickness - 2e-4, tr) > case.T_target   # thinner is not enough
    assert r.areal_mass == pytest.approx(SHIELD.rho * r.thickness + 1400 * 0.0002 + 2800 * 0.002)


def test_peak_decreases_with_thickness(case, sized):
    tr, r = sized
    peaks = [peak_bondline(case, L, tr) for L in (0.5 * r.thickness, r.thickness, 2 * r.thickness)]
    assert peaks[0] > peaks[1] > peaks[2]


def test_harder_entry_needs_more_shield(case, sized):
    _, r = sized
    harder = replace(case, beta=600.0)   # heavier vehicle: decelerates lower, in denser air
    assert size(harder, tol=2e-4).thickness > r.thickness


def test_config_refuses_todos(tmp_path):
    import yaml

    base = yaml.safe_load((ROOT / "data" / "baseline.yaml").read_text())
    base["mission"]["V_E"] = "TODO"
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(base))
    with pytest.raises(MissingDataError, match="baseline.mission.V_E"):
        load_case(ROOT / "data" / "materials.yaml", path)

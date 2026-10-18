"""Sizing by root-finding and the configuration loader (spec section 6.1). Test-only properties."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tps.materials import Material, MissingDataError
from tps.sizing import Case, UncapturedPeakError, load_case, peak_bondline, simulate, size

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


@pytest.mark.parametrize("captured_at_limit", [False, True])
def test_soak_limit_requires_a_confirmed_peak(case, monkeypatch, captured_at_limit):
    calls = []
    def fake_run(*args, soak_time, **kwargs):
        calls.append(soak_time)
        return SimpleNamespace(summary=lambda: {
            "bondline_peak_captured": captured_at_limit and soak_time == 7.0,
        })
    monkeypatch.setattr("tps.sizing.run", fake_run)
    short = replace(case, soak_time=2.0, max_soak_time=7.0)
    if captured_at_limit:
        assert simulate(short, 0.07, trajectory=object()).summary()["bondline_peak_captured"]
    else:
        with pytest.raises(UncapturedPeakError, match="increase max_soak_time"):
            simulate(short, 0.07, trajectory=object())
    assert calls == [2.0, 4.0, 7.0]


def test_initial_soak_is_capped(case, monkeypatch):
    calls = []
    def fake_run(*args, soak_time, **kwargs):
        calls.append(soak_time)
        return SimpleNamespace(summary=lambda: {"bondline_peak_captured": False})
    monkeypatch.setattr("tps.sizing.run", fake_run)
    with pytest.raises(UncapturedPeakError):
        simulate(replace(case, soak_time=20.0, max_soak_time=7.0), 0.07, trajectory=object())
    assert calls == [7.0]


@pytest.mark.parametrize("field,value", [("soak_time", 0), ("soak_time", -1),
                                        ("max_soak_time", 0), ("max_soak_time", float("inf"))])
def test_invalid_soak_windows_are_refused(case, field, value):
    with pytest.raises(ValueError, match="finite and positive"):
        simulate(replace(case, **{field: value}), 0.07, trajectory=object())


def test_sizing_rejects_a_physically_truncated_run(case):
    with pytest.raises(UncapturedPeakError, match="bondline peak not captured"):
        size(replace(case, soak_time=1.0, max_soak_time=1.0), L_lo=0.07, L_hi=0.1)


def test_sizing_does_not_require_an_unnecessarily_thick_trial(case, monkeypatch):
    def peak(case, thickness, trajectory):
        if thickness > 0.128:
            raise UncapturedPeakError("oversized trial has not reached its peak")
        return case.T_target + 1000 * (0.05 - thickness)
    monkeypatch.setattr("tps.sizing.peak_bondline", peak)
    monkeypatch.setattr("tps.sizing.simulate", lambda case, L, tr: SimpleNamespace(summary=lambda: {
        "peak_T_bondline_K": peak(case, L, tr), "peak_T_surface_K": 1000.0,
    }))
    result = size(case, trajectory=object())
    assert 0.05 <= result.thickness <= 0.0501


def test_sizing_reports_when_the_maximum_thickness_is_insufficient(case, monkeypatch):
    monkeypatch.setattr("tps.sizing.peak_bondline", lambda *args: case.T_target + 1)
    with pytest.raises(ValueError, match="exceeds the bondline target"):
        size(case, trajectory=object(), L_hi=0.1)


def test_config_loads_maximum_soak_time(tmp_path):
    import yaml

    base = yaml.safe_load((ROOT / "data" / "baseline.yaml").read_text())
    base["numerics"]["max_soak_time"] = 36000.0
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(base))
    assert load_case(ROOT / "data" / "materials.yaml", path).max_soak_time == 36000.0

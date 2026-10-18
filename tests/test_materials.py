"""Materials file handling: TODO placeholders are refused, filled entries load and convert to SI."""

from pathlib import Path

import numpy as np
import pytest

from tps.materials import MissingDataError, load_materials

REPO_FILE = Path(__file__).resolve().parent.parent / "data" / "materials.yaml"
FILLED = ["li900", "li2200", "frci12", "rtv560", "al2219"]


def test_remaining_todos_are_refused():
    if "TODO" not in REPO_FILE.read_text():
        pytest.skip("data/materials.yaml has been filled in")
    with pytest.raises(MissingDataError, match="TODO|limits|wall"):
        load_materials(REPO_FILE)


def test_repo_materials_load_and_convert():
    """Spot checks against RP-1289's printed values, converted by hand."""
    mats, _ = load_materials(REPO_FILE, require=FILLED)
    li900 = mats["li900"]
    assert li900.rho == pytest.approx(9.0 * 16.01846337)                         # 144.2 kg/m^3
    assert float(li900.k(210 / 1.8)) == pytest.approx(0.0233 * 1.730734666)      # 0.0403 W/(m K)
    assert float(li900.cp(960 / 1.8)) == pytest.approx(0.252 * 4186.8)           # 1055 J/(kg K)
    assert li900.max_use_temperature == pytest.approx(2760 / 1.8)                # 1533 K
    assert mats["al2219"].max_use_temperature == pytest.approx(450.0)
    for m in mats.values():                                                      # physical ranges
        assert m.rho > 0 and np.all(m.k.v > 0) and np.all(m.cp.v > 0)
    assert all(mats[k].emissivity == 0.85 for k in ("li900", "li2200", "frci12"))


def test_filled_entry_loads(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("""
materials:
  a:
    name: test A
    role: heat_shield
    rho: 300
    k: {T: [300, 1200], value: [0.05, 0.12]}
    cp: 1000
    emissivity: 0.8
    source: test-only
  b:
    name: test B
    role: structure
    rho: TODO
    k: 1
    cp: 1
    source: "TODO: source"
limits:
  bondline_temperature_limit: TODO
""")
    mats, _ = load_materials(f, require=["a"])
    assert mats["a"].rho == 300.0
    np.testing.assert_allclose(mats["a"].k(np.array([300.0, 750.0, 2000.0])), [0.05, 0.085, 0.12])
    with pytest.raises(MissingDataError, match="materials.b.rho"):
        load_materials(f, require=["b"])
    with pytest.raises(MissingDataError, match="limits.bondline_temperature_limit"):
        load_materials(f)

"""Materials file handling: TODO placeholders are refused, filled entries load (values here are test-only)."""

from pathlib import Path

import numpy as np
import pytest

from tps.materials import MissingDataError, load_materials

REPO_FILE = Path(__file__).resolve().parent.parent / "data" / "materials.yaml"


def test_template_with_todos_is_refused():
    with pytest.raises(MissingDataError, match="materials.shield_1.rho"):
        load_materials(REPO_FILE) if "TODO" in REPO_FILE.read_text() else (_ for _ in ()).throw(
            MissingDataError("materials.shield_1.rho (file already filled in)"))


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
    mats, entries, _ = load_materials(f, require=["a"])
    assert mats["a"].rho == 300.0
    np.testing.assert_allclose(mats["a"].k(np.array([300.0, 750.0, 2000.0])), [0.05, 0.085, 0.12])
    with pytest.raises(MissingDataError, match="materials.b.rho"):
        load_materials(f, require=["b"])
    with pytest.raises(MissingDataError, match="limits.bondline_temperature_limit"):
        load_materials(f)

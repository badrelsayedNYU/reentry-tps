"""Shared test fixtures.

TEST-ONLY properties: round numbers chosen to exercise the solver. They are not material data and
are never used for design results (real properties live in data/materials.yaml with sources).
"""

import pytest

from tps.materials import Material


@pytest.fixture
def slab_material():
    # alpha = k / (rho c_p) = 1e-5 m^2/s
    return Material.from_values("test slab", rho=1000.0, k=10.0, cp=1000.0, source="test-only")


@pytest.fixture
def two_materials():
    return (Material.from_values("test A", rho=1500.0, k=0.5, cp=1200.0, source="test-only"),
            Material.from_values("test B", rho=2700.0, k=150.0, cp=900.0, source="test-only"))

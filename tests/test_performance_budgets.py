from random import Random

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import MeshSnapshotValidator, build_mesh_fingerprint
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.performance_budget import PerformanceSample, RelativePerformanceBudget, measure_median
from tests.test_seeded_geometry_fuzz import _ngon


def test_relative_budget_rejects_superlinear_regression():
    budget = RelativePerformanceBudget(maximum_time_ratio_per_size_ratio=2.0)
    budget.assert_within((PerformanceSample(100, 0.01), PerformanceSample(1_000, 0.08)))
    with pytest.raises(AssertionError, match="exceeded budget"):
        budget.assert_within((PerformanceSample(100, 0.01), PerformanceSample(1_000, 0.5)))


def test_mesh_validation_and_fingerprint_scale_with_generous_relative_budget():
    rng = Random(0xB16B00B5); samples = []
    for size in (100, 500, 2_000):
        snapshot = _ngon(rng, size, size); validator = MeshSnapshotValidator()
        elapsed = measure_median(lambda: (validator.validate_or_raise(snapshot), build_mesh_fingerprint(snapshot)), repeats=3, warmups=1)
        samples.append(PerformanceSample(size=size, seconds=elapsed))
    RelativePerformanceBudget(maximum_time_ratio_per_size_ratio=3.0).assert_within(samples)

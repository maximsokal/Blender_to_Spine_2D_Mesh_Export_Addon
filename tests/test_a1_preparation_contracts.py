import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectStage,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_preparation_contracts import (
    A1ObjectPreparationError,
    freeze_statistics,
    warning_issue,
)


def test_statistics_are_merged_and_frozen():
    statistics = freeze_statistics({"vertices": 4}, {"mode": "OBJECT_BAKE"})

    assert dict(statistics) == {"vertices": 4, "mode": "OBJECT_BAKE"}
    with pytest.raises(TypeError):
        statistics["vertices"] = 8


def test_statistics_reject_bool_values_that_hide_type_errors():
    with pytest.raises(TypeError, match="statistics values"):
        freeze_statistics({"camera": True})


def test_warning_factory_preserves_stage_object_and_context():
    issue = warning_issue(
        stage=A1SingleObjectStage.ANALYZE_MATERIALS,
        code="MATERIAL_NOTE",
        message="Unsupported fallback",
        object_id="Hero",
        context={"slot_index": 2},
    )

    assert issue.severity is IssueSeverity.WARNING
    assert issue.stage == A1SingleObjectStage.ANALYZE_MATERIALS.value
    assert issue.object_id == "Hero"
    assert issue.context == {"slot_index": 2}


def test_preparation_error_preserves_partial_diagnostics_immutably():
    cause = RuntimeError("failed")
    error = A1ObjectPreparationError(
        stage=A1SingleObjectStage.UNWRAP_TEXTURE_UV,
        object_id="Hero",
        cause=cause,
        statistics={"uv_loop_count": 10},
        warnings=(),
    )

    assert error.cause is cause
    assert error.stage is A1SingleObjectStage.UNWRAP_TEXTURE_UV
    assert dict(error.statistics) == {"uv_loop_count": 10}
    with pytest.raises(TypeError):
        error.statistics["uv_loop_count"] = 0

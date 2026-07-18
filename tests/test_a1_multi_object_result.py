import logging

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import A1MultiObjectStage
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_result import (
    build_multi_object_failure_result,
)


def test_shared_failure_result_preserves_stage_component_and_operation():
    result = build_multi_object_failure_result(
        logger=logging.getLogger("test.a1.multi.result"),
        operation="A1 mixed-object output",
        stage=A1MultiObjectStage.COMPOSE_DOCUMENT,
        exc=ValueError("composition failed"),
        statistics={"object_count": 3},
        warnings=(),
        component_id="object_2:Armor",
        object_id="Armor",
        object_stage="BUILD_RIG",
    )

    assert result.success is False
    assert result.statistics == {"object_count": 3}
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == A1MultiObjectStage.COMPOSE_DOCUMENT.error_code
    assert issue.object_id == "Armor"
    assert issue.context == {
        "exception_type": "ValueError",
        "operation": "A1 mixed-object output",
        "component_id": "object_2:Armor",
        "object_stage": "BUILD_RIG",
    }


def test_shared_failure_result_rejects_untyped_warning_container():
    with pytest.raises(TypeError, match="warnings must be a tuple"):
        build_multi_object_failure_result(
            logger=logging.getLogger("test.a1.multi.result.invalid"),
            operation="A1 multi-object output",
            stage=A1MultiObjectStage.VALIDATE_REQUEST,
            exc=ValueError("invalid"),
            statistics={},
            warnings=[],
        )

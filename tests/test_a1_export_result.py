import logging

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectStage,
    ExportIssue,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_export_result import (
    build_a1_failure_result,
)


def test_shared_a1_failure_result_preserves_single_object_context():
    warning = ExportIssue(
        severity=IssueSeverity.WARNING,
        stage="READ_GEOMETRY",
        code="TEST_WARNING",
        message="warning",
        object_id="Hero",
    )
    result = build_a1_failure_result(
        logger=logging.getLogger("test.a1.single.result"),
        operation="A1 single-object output",
        stage=A1SingleObjectStage.ASSEMBLE_DOCUMENT,
        exc=ValueError("assembly failed"),
        statistics={"region_count": 3},
        warnings=(warning,),
        object_id="Hero",
    )

    assert result.success is False
    assert result.statistics == {"region_count": 3}
    assert result.issues[0] is warning
    issue = result.issues[-1]
    assert issue.code == A1SingleObjectStage.ASSEMBLE_DOCUMENT.error_code
    assert issue.object_id == "Hero"
    assert issue.context == {
        "exception_type": "ValueError",
        "operation": "A1 single-object output",
    }


def test_shared_a1_failure_result_rejects_invalid_stage():
    with pytest.raises(TypeError, match="stage must be"):
        build_a1_failure_result(
            logger=logging.getLogger("test.a1.result.invalid"),
            operation="invalid",
            stage="ASSEMBLE_DOCUMENT",
            exc=ValueError("invalid"),
            statistics={},
        )

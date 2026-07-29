from dataclasses import replace
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    SpineSerializer,
    SpineValidationError,
    build_connected_group_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_serialization_validator import (
    ConnectedGroupSerializationValidator,
)

from test_connected_group_document import connected_objects, settings


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_OWNER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_multi_object_output.py"
)


def test_connected_serializer_allows_only_historical_order_ties():
    result = build_connected_group_document(connected_objects(), settings())

    with pytest.raises(SpineValidationError) as strict_error:
        SpineSerializer().to_dict(result.document)
    assert {
        issue.code for issue in strict_error.value.issues
    } == {"DUPLICATE_CONSTRAINT_ORDER"}

    serialized = SpineSerializer(
        validator=ConnectedGroupSerializationValidator()
    ).to_dict(result.document)
    assert serialized["skeleton"]["spine"] == "4.2.43"
    assert len(serialized["transform"]) == len(result.document.transform)


def test_connected_serializer_does_not_hide_any_other_validation_error():
    result = build_connected_group_document(connected_objects(), settings())
    broken_transform = replace(result.document.transform[0], target="missing_target")
    broken = replace(
        result.document,
        transform=(broken_transform, *result.document.transform[1:]),
    )

    with pytest.raises(SpineValidationError) as caught:
        SpineSerializer(
            validator=ConnectedGroupSerializationValidator()
        ).to_dict(broken)

    codes = {issue.code for issue in caught.value.issues}
    assert "MISSING_CONSTRAINT_TARGET" in codes
    assert "DUPLICATE_CONSTRAINT_ORDER" not in codes


def test_multi_output_selects_relaxed_validator_only_for_connected_result():
    source = OUTPUT_OWNER.read_text(encoding="utf-8")

    assert "def _serializer_for_composition(" in source
    assert "isinstance(composition, ConnectedGroupBuildResult)" in source
    assert "ConnectedGroupSerializationValidator()" in source
    assert "return SpineSerializer()" in source

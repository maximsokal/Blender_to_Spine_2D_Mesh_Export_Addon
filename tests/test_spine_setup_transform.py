from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADLESS = ROOT / "tests" / "blender_headless"
if str(HEADLESS) not in sys.path:
    sys.path.insert(0, str(HEADLESS))

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (  # noqa: E402
    Bone,
    SpineDocument,
)
from spine_setup_transform import (  # noqa: E402
    SpineSetupTransformError,
    evaluate_spine_setup_bone,
    evaluate_spine_setup_bone_position,
)


def _document(*bones: Bone) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=bones,
        slots=(),
        skins=(),
        animations={},
    )


def test_two_axis_connected_wrapper_evaluates_to_expected_object_origin() -> None:
    document = _document(
        Bone(name="root"),
        Bone(name="all_objects_main", parent="root", x=100.0, y=200.0),
        Bone(name="all_objects_base", parent="all_objects_main"),
        Bone(
            name="all_objects_flatten",
            parent="all_objects_base",
            scale_x=0.0,
        ),
        Bone(name="all_objects_rotation", parent="all_objects_flatten"),
        Bone(
            name="all_objects_layer_0_scale",
            parent="all_objects_rotation",
            rotation=90.0,
            y=30.0,
            extras={"inherit": "onlyTranslation"},
        ),
        Bone(
            name="all_objects_layer_0",
            parent="all_objects_layer_0_scale",
            rotation=-90.0,
        ),
        Bone(
            name="Object_main",
            parent="all_objects_layer_0",
            x=25.0,
            y=-30.0,
        ),
    )

    result = evaluate_spine_setup_bone(document, "Object_main")

    assert result.position == pytest.approx((125.0, 200.0), abs=1.0e-12)
    assert result.transform.a == pytest.approx(1.0, abs=1.0e-12)
    assert result.transform.b == pytest.approx(0.0, abs=1.0e-12)
    assert result.transform.c == pytest.approx(0.0, abs=1.0e-12)
    assert result.transform.d == pytest.approx(1.0, abs=1.0e-12)


def test_normal_inheritance_rotates_child_translation_and_basis() -> None:
    document = _document(
        Bone(name="root"),
        Bone(name="parent", parent="root", x=10.0, y=20.0, rotation=90.0),
        Bone(name="child", parent="parent", x=5.0, rotation=-90.0),
    )

    result = evaluate_spine_setup_bone(document, "child")

    assert result.position == pytest.approx((10.0, 25.0), abs=1.0e-12)
    assert result.transform.a == pytest.approx(1.0, abs=1.0e-12)
    assert result.transform.b == pytest.approx(0.0, abs=1.0e-12)
    assert result.transform.c == pytest.approx(0.0, abs=1.0e-12)
    assert result.transform.d == pytest.approx(1.0, abs=1.0e-12)


def test_only_translation_keeps_parent_space_origin_but_resets_basis_inheritance() -> None:
    document = _document(
        Bone(name="root"),
        Bone(name="parent", parent="root", x=10.0, y=20.0, rotation=90.0),
        Bone(
            name="child",
            parent="parent",
            x=5.0,
            extras={"inherit": "onlyTranslation"},
        ),
    )

    result = evaluate_spine_setup_bone(document, "child")

    assert result.position == pytest.approx((10.0, 25.0), abs=1.0e-12)
    assert result.transform.a == pytest.approx(1.0, abs=1.0e-12)
    assert result.transform.b == pytest.approx(0.0, abs=1.0e-12)
    assert result.transform.c == pytest.approx(0.0, abs=1.0e-12)
    assert result.transform.d == pytest.approx(1.0, abs=1.0e-12)


def test_transform_alias_is_supported_when_not_conflicting() -> None:
    document = _document(
        Bone(name="root"),
        Bone(
            name="child",
            parent="root",
            x=3.0,
            y=4.0,
            extras={"transform": "onlyTranslation"},
        ),
    )

    assert evaluate_spine_setup_bone_position(document, "child") == (3.0, 4.0)


def test_conflicting_inherit_aliases_fail_closed() -> None:
    document = _document(
        Bone(name="root"),
        Bone(
            name="child",
            parent="root",
            extras={
                "inherit": "normal",
                "transform": "onlyTranslation",
            },
        ),
    )

    with pytest.raises(SpineSetupTransformError, match="conflicting"):
        evaluate_spine_setup_bone(document, "child")


def test_unsupported_inherit_mode_fails_closed() -> None:
    document = _document(
        Bone(name="root"),
        Bone(
            name="child",
            parent="root",
            extras={"inherit": "noScale"},
        ),
    )

    with pytest.raises(SpineSetupTransformError, match="unsupported inherit mode"):
        evaluate_spine_setup_bone(document, "child")


def test_non_zero_shear_fails_closed() -> None:
    document = _document(
        Bone(name="root"),
        Bone(
            name="child",
            parent="root",
            extras={"shearY": 15.0},
        ),
    )

    with pytest.raises(SpineSetupTransformError, match="unsupported non-zero shearY"):
        evaluate_spine_setup_bone(document, "child")


def test_parent_cycle_fails_closed() -> None:
    document = _document(
        Bone(name="a", parent="b"),
        Bone(name="b", parent="a"),
    )

    with pytest.raises(SpineSetupTransformError, match="parent cycle"):
        evaluate_spine_setup_bone(document, "a")


def test_missing_parent_fails_closed() -> None:
    document = _document(Bone(name="child", parent="missing"))

    with pytest.raises(SpineSetupTransformError, match="Missing parent bone"):
        evaluate_spine_setup_bone(document, "child")

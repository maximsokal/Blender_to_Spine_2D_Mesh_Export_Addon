from __future__ import annotations

import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_serialization_validator import (
    ConnectedGroupSerializationValidator,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    IKConstraint,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs.runtime_constraint_order import (
    normalize_runtime_constraint_orders,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _legacy_three_axis_document() -> SpineDocument:
    bones = (
        Bone("root"),
        Bone("constraint", parent="root"),
        Bone("target", parent="root"),
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=bones,
        slots=(),
        skins=(),
        ik=(
            IKConstraint(
                name="scale_ik",
                order=3,
                bones=("constraint",),
                target="target",
            ),
        ),
        transform=(
            TransformConstraint(
                name="rotation_x",
                order=1,
                bones=("constraint",),
                target="target",
            ),
            TransformConstraint(
                name="rotation_y",
                order=2,
                bones=("constraint",),
                target="target",
            ),
            TransformConstraint(
                name="rotation_z",
                order=5,
                bones=("constraint",),
                target="target",
            ),
            TransformConstraint(
                name="scale",
                order=4,
                bones=("constraint",),
                target="target",
            ),
            TransformConstraint(
                name="scale_compensator",
                order=6,
                bones=("constraint",),
                target="target",
            ),
        ),
    )


def test_spine38_codec_normalizes_legacy_three_axis_orders_without_mutation() -> None:
    document = _legacy_three_axis_document()

    payload = json.loads(
        serialize_spine_document(document, SpineJsonTarget.SPINE_3_8)
    )

    assert payload["skeleton"]["spine"] == "3.8.99"
    assert [item["order"] for item in payload["ik"]] == [2]
    assert [item["order"] for item in payload["transform"]] == [0, 1, 4, 3, 5]
    assert sorted(
        item["order"]
        for collection in (payload["ik"], payload["transform"])
        for item in collection
    ) == list(range(6))

    assert document.ik[0].order == 3
    assert [item.order for item in document.transform] == [1, 2, 5, 4, 6]


def test_spine42_codec_resolves_connected_order_ties_for_runtime_cache() -> None:
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(
            Bone("root"),
            Bone("constraint", parent="root"),
            Bone("target", parent="root"),
        ),
        slots=(),
        skins=(),
        ik=(
            IKConstraint(
                name="ik_first",
                order=0,
                bones=("constraint",),
                target="target",
            ),
        ),
        transform=(
            TransformConstraint(
                name="transform_tied",
                order=0,
                bones=("constraint",),
                target="target",
            ),
            TransformConstraint(
                name="transform_late",
                order=6,
                bones=("constraint",),
                target="target",
            ),
        ),
    )

    payload = json.loads(
        serialize_spine_document(
            document,
            SpineJsonTarget.SPINE_4_2,
            validator=ConnectedGroupSerializationValidator(),
        )
    )

    assert [item["order"] for item in payload["ik"]] == [0]
    assert [item["order"] for item in payload["transform"]] == [1, 2]
    assert document.ik[0].order == 0
    assert [item.order for item in document.transform] == [0, 6]


def test_runtime_order_normalizer_uses_legacy_collection_precedence() -> None:
    output = {
        "ik": [{"name": "ik", "order": 4}],
        "transform": [{"name": "transform", "order": 4}],
        "path": [{"name": "path", "order": 4}],
        "physics": [{"name": "physics", "order": 4}],
    }

    assignments = normalize_runtime_constraint_orders(
        output,
        collections=("ik", "transform", "path", "physics"),
    )

    assert [(item.collection, item.runtime_order) for item in assignments] == [
        ("ik", 0),
        ("transform", 1),
        ("path", 2),
        ("physics", 3),
    ]
    assert output["ik"][0]["order"] == 0
    assert output["transform"][0]["order"] == 1
    assert output["path"][0]["order"] == 2
    assert output["physics"][0]["order"] == 3


def test_runtime_order_normalizer_finishes_ancestor_wrapper_before_object_rig() -> None:
    output = {
        "bones": [
            {"name": "root"},
            {"name": "global_rotate", "parent": "root"},
            {"name": "global_layer", "parent": "global_rotate"},
            {"name": "global_helper", "parent": "global_rotate"},
            {"name": "global_target", "parent": "root"},
            {"name": "object_main", "parent": "global_layer"},
            {"name": "object_scale_parent", "parent": "object_main"},
            {"name": "object_rotate", "parent": "object_scale_parent"},
            {"name": "object_depth", "parent": "object_rotate"},
            {"name": "object_helper", "parent": "object_main"},
            {"name": "object_target", "parent": "object_main"},
        ],
        "ik": [
            {
                "name": "global_ik",
                "order": 2,
                "bones": ["global_helper"],
                "target": "global_target",
            },
            {
                "name": "object_ik",
                "order": 3,
                "bones": ["object_helper"],
                "target": "object_target",
            },
        ],
        "transform": [
            {
                "name": "global_rotation_x",
                "order": 0,
                "bones": ["global_rotate"],
                "target": "global_target",
            },
            {
                "name": "object_rotation_x",
                "order": 1,
                "bones": ["object_rotate"],
                "target": "object_target",
            },
            {
                "name": "global_scale",
                "order": 4,
                "bones": ["global_rotate", "global_layer"],
                "target": "global_target",
            },
            {
                "name": "object_scale",
                "order": 5,
                "bones": ["object_rotate"],
                "target": "object_target",
            },
            {
                "name": "global_scale_depth",
                "order": 6,
                "bones": ["global_layer"],
                "target": "global_helper",
            },
            {
                "name": "object_scale_depth",
                "order": 7,
                "bones": ["object_depth"],
                "target": "object_helper",
            },
            {
                "name": "global_rotation_y",
                "order": 8,
                "bones": ["global_layer"],
                "target": "global_target",
            },
            {
                "name": "object_rotation_y",
                "order": 9,
                "bones": ["object_depth"],
                "target": "object_target",
            },
        ],
    }

    assignments = normalize_runtime_constraint_orders(
        output,
        collections=("ik", "transform", "path", "physics"),
    )

    assert [item.name for item in assignments] == [
        "global_rotation_x",
        "global_ik",
        "global_scale",
        "global_scale_depth",
        "global_rotation_y",
        "object_rotation_x",
        "object_ik",
        "object_scale",
        "object_scale_depth",
        "object_rotation_y",
    ]
    assert [item["order"] for item in output["ik"]] == [1, 6]
    assert [item["order"] for item in output["transform"]] == [
        0,
        5,
        2,
        7,
        3,
        8,
        4,
        9,
    ]


def test_runtime_order_normalizer_rejects_cyclic_hierarchy_before_mutation() -> None:
    output = {
        "bones": [
            {"name": "root"},
            {"name": "left", "parent": "root"},
            {"name": "left_child", "parent": "left"},
            {"name": "right", "parent": "root"},
            {"name": "right_child", "parent": "right"},
        ],
        "transform": [
            {
                "name": "left_reads_right",
                "order": 4,
                "bones": ["left"],
                "target": "right_child",
            },
            {
                "name": "right_reads_left",
                "order": 7,
                "bones": ["right"],
                "target": "left_child",
            },
        ],
    }

    with pytest.raises(ValueError, match="cyclic runtime dependency"):
        normalize_runtime_constraint_orders(
            output,
            collections=("ik", "transform", "path", "physics"),
        )

    assert [item["order"] for item in output["transform"]] == [4, 7]

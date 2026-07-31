"""Spine 4.3.23 unified-constraint codec regressions."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.validator import (
    SpineValidationError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    Spine43JsonCodec,
    SpineJsonCodecContext,
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


PROPERTIES = ("rotate", "x", "y", "scaleX", "scaleY", "shearY")


def _document() -> SpineDocument:
    attachment = MeshAttachment(
        name="mesh",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
        width=64.0,
        height=64.0,
        sequence={"count": 2, "start": 0, "digits": 4, "setup": 0},
    )
    return SpineDocument(
        skeleton={
            "spine": "4.2.43",
            "images": "images/",
            "referenceScale": 100,
        },
        bones=(
            Bone("root"),
            Bone(
                "control",
                parent="root",
                extras={"inherit": "onlyTranslation"},
            ),
        ),
        slots=(Slot("slot", bone="control", attachment="mesh"),),
        skins=(
            Skin(
                "default",
                {"slot": {"mesh": attachment}},
                constraints=("ik", "transform"),
            ),
        ),
        ik=(
            IKConstraint(
                "ik",
                1,
                ("control",),
                "root",
                extras={"compress": True, "stretch": True, "mix": 0.75},
            ),
        ),
        transform=(
            TransformConstraint(
                "transform",
                0,
                ("control",),
                "root",
                extras={
                    "local": True,
                    "relative": True,
                    "rotation": 12.5,
                    "x": -4.0,
                    "y": 8.0,
                    "scaleX": -1.0,
                    "scaleY": 0.25,
                    "shearY": 3.0,
                    "mixRotate": 0.25,
                    "mixX": 0.5,
                    "mixY": 0.75,
                    "mixScaleX": 0.6,
                    "mixScaleY": 0.7,
                    "mixShearY": 0.0,
                },
            ),
        ),
        animations={"animation": {}},
    )


def _legacy_three_axis_sparse_order_document() -> SpineDocument:
    """Reproduce the canonical Legacy 3-Axis order schedule used by production."""

    base = _document()
    return replace(
        base,
        skins=(replace(base.skins[0], constraints=()),),
        ik=(
            IKConstraint(
                "legacy_scale_ik",
                3,
                ("control",),
                "root",
                extras={"compress": True, "stretch": True},
            ),
        ),
        transform=(
            TransformConstraint(
                "legacy_rotation_x",
                1,
                ("control",),
                "root",
            ),
            TransformConstraint(
                "legacy_rotation_y",
                2,
                ("control",),
                "root",
            ),
            TransformConstraint(
                "legacy_rotation_z",
                5,
                ("control",),
                "root",
            ),
            TransformConstraint(
                "legacy_scale",
                4,
                ("control",),
                "root",
            ),
            TransformConstraint(
                "legacy_scale_compensator",
                6,
                ("control",),
                "root",
            ),
        ),
    )


def test_spine43_codec_builds_unified_constraints_in_authored_order() -> None:
    payload = json.loads(
        serialize_spine_document(
            _document(),
            SpineJsonTarget.SPINE_4_3,
        )
    )

    assert payload["skeleton"]["spine"] == "4.3.23"
    assert payload["skeleton"]["referenceScale"] == 100
    assert payload["bones"][1]["inherit"] == "onlyTranslation"
    assert "transform" not in payload["bones"][1]
    assert "ik" not in payload
    assert "transform" not in payload

    constraints = payload["constraints"]
    assert [item["name"] for item in constraints] == ["transform", "ik"]
    assert all("order" not in item for item in constraints)

    transform = constraints[0]
    assert transform["type"] == "transform"
    assert transform["source"] == "root"
    assert "target" not in transform
    assert transform["localSource"] is True
    assert transform["localTarget"] is True
    assert transform["additive"] is True
    assert "local" not in transform
    assert "relative" not in transform
    assert tuple(transform["properties"]) == PROPERTIES
    for property_name in PROPERTIES:
        assert transform["properties"][property_name] == {
            "to": {property_name: {}}
        }

    for field_name, expected in (
        ("rotation", 12.5),
        ("x", -4.0),
        ("y", 8.0),
        ("scaleX", -1.0),
        ("scaleY", 0.25),
        ("shearY", 3.0),
        ("mixRotate", 0.25),
        ("mixX", 0.5),
        ("mixY", 0.75),
        ("mixScaleX", 0.6),
        ("mixScaleY", 0.7),
        ("mixShearY", 0.0),
    ):
        assert transform[field_name] == expected

    ik = constraints[1]
    assert ik["type"] == "ik"
    assert ik["target"] == "root"
    assert ik["compress"] is True
    assert ik["stretch"] is True
    assert ik["mix"] == 0.75

    skin = payload["skins"][0]
    assert skin["ik"] == ["ik"]
    assert skin["transform"] == ["transform"]
    assert "constraints" not in skin
    assert skin["attachments"]["slot"]["mesh"]["sequence"] == {
        "count": 2,
        "start": 0,
        "digits": 4,
        "setup": 0,
    }


def test_spine43_codec_does_not_mutate_the_canonical_document() -> None:
    document = _document()
    before = document

    serialize_spine_document(document, "4.3.23")

    assert document is before
    assert document.skeleton["spine"] == "4.2.43"
    assert document.bones[1].extras == {"inherit": "onlyTranslation"}
    assert document.skins[0].constraints == ("ik", "transform")
    assert document.transform[0].target == "root"
    assert document.transform[0].extras["local"] is True
    assert document.transform[0].extras["relative"] is True


def test_spine43_codec_compacts_legacy_sparse_orders_by_dependency_order() -> None:
    document = _legacy_three_axis_sparse_order_document()

    payload = json.loads(
        serialize_spine_document(document, SpineJsonTarget.SPINE_4_3)
    )

    assert [item["name"] for item in payload["constraints"]] == [
        "legacy_rotation_x",
        "legacy_rotation_y",
        "legacy_scale_ik",
        "legacy_scale",
        "legacy_rotation_z",
        "legacy_scale_compensator",
    ]
    assert all("order" not in item for item in payload["constraints"])
    assert tuple(item.order for item in document.ik) == (3,)
    assert tuple(item.order for item in document.transform) == (1, 2, 5, 4, 6)


def test_spine43_codec_rejects_duplicate_global_orders() -> None:
    document = _document()
    broken = replace(
        document,
        ik=(replace(document.ik[0], order=0),),
    )

    with pytest.raises(SpineValidationError, match="DUPLICATE_CONSTRAINT_ORDER"):
        serialize_spine_document(broken, SpineJsonTarget.SPINE_4_3)


def test_spine43_codec_rejects_untyped_root_constraint_families() -> None:
    document = _document()
    broken = replace(
        document,
        extras={"path": [{"name": "unsupported"}]},
    )

    with pytest.raises(ValueError, match="untyped root constraint families"):
        serialize_spine_document(broken, SpineJsonTarget.SPINE_4_3)


def test_spine43_codec_rejects_reserved_properties_payload() -> None:
    document = _document()
    broken_transform = replace(
        document.transform[0],
        extras={**document.transform[0].extras, "properties": {}},
    )
    broken = replace(document, transform=(broken_transform,))

    with pytest.raises(ValueError, match="properties is reserved"):
        serialize_spine_document(broken, SpineJsonTarget.SPINE_4_3)


def test_spine43_codec_rejects_wrong_context_target() -> None:
    with pytest.raises(ValueError, match="requires SPINE_4_3"):
        Spine43JsonCodec().to_json(
            _document(),
            context=SpineJsonCodecContext(target=SpineJsonTarget.SPINE_4_2),
        )

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineCompositionError,
    SpineDocument,
    SpineDocumentComponent,
    compose_spine_documents,
)


def _raw_mesh(*, weighted):
    vertices = (
        [1, 0, 0.0, 0.0, 1.0] * 3
        if weighted
        else [-50.0, 50.0, 50.0, 50.0, 0.0, -50.0]
    )
    return {
        "type": "mesh",
        "uvs": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
        "triangles": [0, 1, 2],
        "vertices": vertices,
        "hull": 3,
        "edges": [0, 2, 2, 4, 4, 0],
        "width": 100.0,
        "height": 100.0,
    }


def _component(component_id, attachment):
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot(f"{component_id}_slot", "root", attachment="mesh"),),
        skins=(
            Skin(
                name="default",
                attachments={
                    f"{component_id}_slot": {"mesh": attachment},
                },
            ),
        ),
        animations={},
    )
    return SpineDocumentComponent(
        component_id=component_id,
        document=document,
    )


def test_public_composition_allows_raw_unweighted_mesh_without_bone_indices():
    result = compose_spine_documents(
        (_component("plain", _raw_mesh(weighted=False)),),
    )

    attachment = result.document.skins[0].attachments["plain_slot"]["mesh"]
    assert attachment["vertices"] == [
        -50.0,
        50.0,
        50.0,
        50.0,
        0.0,
        -50.0,
    ]


def test_public_composition_rejects_raw_weighted_mesh_with_ambiguous_bone_indices():
    with pytest.raises(SpineCompositionError, match="typed MeshAttachment"):
        compose_spine_documents(
            (_component("weighted", _raw_mesh(weighted=True)),),
        )


def test_public_domain_exports_guarded_composer_not_low_level_function():
    import Blender_to_Spine2D_Mesh_Exporter.domain.spine as spine

    assert spine.compose_spine_documents.__module__.endswith("composition_service")

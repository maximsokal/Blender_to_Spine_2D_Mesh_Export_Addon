from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    MeshAttachment,
    Skin,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.deform_timeline_contract import (
    validate_animation_deform_timelines,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import (
    AttachmentReference,
    LinkedMeshResolver,
    validate_setup_linked_meshes,
)


RAW_MESH = {
    "type": "mesh",
    "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "triangles": [0, 1, 2],
    "vertices": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "hull": 3,
}


def linked_skins():
    return (
        Skin(
            "default",
            {
                "slot": {
                    "parent": deepcopy(RAW_MESH),
                    "child": {
                        "type": "linkedmesh",
                        "parent": "parent",
                    },
                }
            },
        ),
    )


def deform_animation():
    return {
        "animation": {
            "attachments": {
                "default": {
                    "slot": {
                        "child": {
                            "deform": [
                                {"vertices": [0.1, -0.1]},
                                {"time": 1.0},
                            ]
                        }
                    }
                }
            }
        }
    }


def validate_deform(animations, skins, *, resolver=None):
    validate_animation_deform_timelines(
        animations,
        skins=skins,
        slot_names=("slot",),
        path="document.animations",
        linked_mesh_resolver=resolver,
    )


def test_setup_validation_returns_the_validated_reusable_resolver():
    skins = linked_skins()

    resolver = validate_setup_linked_meshes(
        skins,
        path="document.skins",
    )

    assert isinstance(resolver, LinkedMeshResolver)
    assert resolver.skins is skins
    resolved = resolver.resolve(
        AttachmentReference("default", "slot", "child")
    )
    assert resolved.terminal == AttachmentReference(
        "default",
        "slot",
        "parent",
    )


def test_deform_boundary_accepts_the_exact_validated_resolver():
    skins = linked_skins()
    resolver = validate_setup_linked_meshes(skins)

    validate_deform(deform_animation(), skins, resolver=resolver)


def test_deform_boundary_keeps_direct_call_compatibility_without_resolver():
    skins = linked_skins()

    validate_deform(deform_animation(), skins)


def test_deform_boundary_rejects_non_resolver_values():
    skins = linked_skins()

    with pytest.raises(
        TypeError,
        match="linked_mesh_resolver must be LinkedMeshResolver or None",
    ):
        validate_deform(deform_animation(), skins, resolver=object())


def test_deform_boundary_rejects_resolver_from_another_skin_tuple():
    skins = linked_skins()
    equivalent_but_distinct_skins = linked_skins()
    stale_resolver = validate_setup_linked_meshes(equivalent_but_distinct_skins)

    with pytest.raises(
        ValueError,
        match="built from the exact skins tuple",
    ):
        validate_deform(deform_animation(), skins, resolver=stale_resolver)


def test_terminal_geometry_capacity_is_shared_by_multiple_linked_sources():
    parent = MeshAttachment(
        name="parent",
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
    )
    skins = (
        Skin(
            "default",
            {
                "slot": {
                    "parent": parent,
                    "first": {
                        "type": "linkedmesh",
                        "parent": "parent",
                    },
                    "second": {
                        "type": "linkedmesh",
                        "parent": "parent",
                    },
                }
            },
        ),
    )
    animations = {
        "animation": {
            "attachments": {
                "default": {
                    "slot": {
                        "first": {"deform": [{"vertices": [0.1, 0.2]}]},
                        "second": {"deform": [{"offset": 4, "vertices": [0.3, 0.4]}]},
                    }
                }
            }
        }
    }
    resolver = validate_setup_linked_meshes(skins)

    validate_deform(animations, skins, resolver=resolver)

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import Skin
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import (
    LinkedMeshResolver,
    validate_setup_linked_meshes,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.sequence_timeline_contract import (
    validate_animation_sequence_timelines,
)


def sequence_skins():
    return (
        Skin(
            "default",
            {
                "slot": {
                    "item": {
                        "type": "region",
                        "sequence": {"count": 4, "start": 0},
                    }
                }
            },
        ),
    )


def sequence_animation():
    return {
        "animation": {
            "attachments": {
                "default": {
                    "slot": {
                        "item": {
                            "sequence": [
                                {"mode": "hold", "index": 0},
                                {"time": 1.0, "mode": "loop", "delay": 0.1},
                            ]
                        }
                    }
                }
            }
        }
    }


def validate_sequence(animations, skins, *, resolver=None):
    validate_animation_sequence_timelines(
        animations,
        skins=skins,
        slot_names=("slot",),
        path="document.animations",
        linked_mesh_resolver=resolver,
    )


def test_sequence_boundary_accepts_the_exact_validated_resolver():
    skins = sequence_skins()
    resolver = validate_setup_linked_meshes(skins)

    validate_sequence(sequence_animation(), skins, resolver=resolver)

    assert isinstance(resolver, LinkedMeshResolver)
    assert resolver.skins is skins


def test_sequence_boundary_keeps_direct_call_compatibility_without_resolver():
    skins = sequence_skins()

    validate_sequence(sequence_animation(), skins)


def test_sequence_boundary_rejects_non_resolver_values():
    skins = sequence_skins()

    with pytest.raises(
        TypeError,
        match="linked_mesh_resolver must be LinkedMeshResolver or None",
    ):
        validate_sequence(sequence_animation(), skins, resolver=object())


def test_sequence_boundary_rejects_resolver_from_another_skin_tuple():
    skins = sequence_skins()
    equivalent_but_distinct_skins = sequence_skins()
    stale_resolver = validate_setup_linked_meshes(equivalent_but_distinct_skins)

    with pytest.raises(
        ValueError,
        match="built from the exact skins tuple",
    ):
        validate_sequence(sequence_animation(), skins, resolver=stale_resolver)


def test_sequence_lookup_uses_the_indexed_setup_attachment():
    skins = sequence_skins()
    resolver = validate_setup_linked_meshes(skins)
    indexed_sequence = skins[0].attachments["slot"]["item"]["sequence"]

    indexed_sequence["count"] = 0

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        validate_sequence(sequence_animation(), skins, resolver=resolver)

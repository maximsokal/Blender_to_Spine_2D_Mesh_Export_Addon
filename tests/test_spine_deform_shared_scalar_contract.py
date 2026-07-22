from copy import deepcopy
from math import inf, nan

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.deform_timeline_contract as deform_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import (
    LinkedMeshResolver,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import MeshAttachment, Skin
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    require_finite_number,
    require_name,
)


def mesh(name="mesh"):
    return MeshAttachment(
        name=name,
        uvs=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        triangles=(0, 1, 2),
        vertices=(0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        hull=3,
    )


def validate(frames, *, skin_name="default", attachment_name="mesh"):
    skins = (Skin("default", {"slot": {"mesh": mesh()}}),)
    animations = {
        "idle": {
            "attachments": {
                skin_name: {
                    "slot": {
                        attachment_name: {"deform": frames},
                    }
                }
            }
        }
    }
    deform_contract.validate_animation_deform_timelines(
        animations,
        skins=skins,
        slot_names=("slot",),
        path="document.animations",
    )
    return animations


def test_deform_contract_holds_exact_shared_scalar_functions():
    assert deform_contract._require_name is require_name
    assert deform_contract._require_finite_number is require_finite_number


def test_number_sequence_is_validated_without_copy_or_normalization():
    values = [0, -1.5, 2]
    source = deepcopy(values)

    result = deform_contract._require_number_sequence(values, "vertices")

    assert result is values
    assert values == source


@pytest.mark.parametrize("value", (True, False, None, "0", (), {}))
def test_vertex_component_preserves_strict_numeric_type_diagnostic(value):
    with pytest.raises(TypeError, match=r"vertices\[1\] must be a finite number"):
        deform_contract._require_number_sequence([0.0, value], "vertices")


@pytest.mark.parametrize("value", (inf, -inf, nan))
def test_vertex_component_preserves_non_finite_diagnostic(value):
    with pytest.raises(ValueError, match=r"vertices\[1\] must be finite"):
        deform_contract._require_number_sequence([0.0, value], "vertices")


def test_non_finite_stream_fails_before_weighted_decoder(monkeypatch):
    calls = []

    def recording_decoder(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("decoder must not be called")

    monkeypatch.setattr(
        deform_contract,
        "decode_weighted_vertices",
        recording_decoder,
    )

    with pytest.raises(ValueError, match=r"vertices\[1\] must be finite"):
        deform_contract._deform_capacity_from_vertices(
            [0.0, inf],
            expected_coordinate_count=4,
            path="vertices",
        )

    assert calls == []


def test_skin_name_fails_before_resolver_lookup(monkeypatch):
    skins = (Skin("default", {"slot": {"mesh": mesh()}}),)
    resolver = LinkedMeshResolver(skins, path="document.skins")
    calls = []
    original_require_skin = resolver.require_skin

    def recording_require_skin(skin_name, *, path):
        calls.append((skin_name, path))
        return original_require_skin(skin_name, path=path)

    monkeypatch.setattr(resolver, "require_skin", recording_require_skin)

    with pytest.raises(ValueError, match="skin name cannot be empty"):
        deform_contract.validate_animation_deform_timelines(
            {"idle": {"attachments": {" ": {}}}},
            skins=skins,
            slot_names=("slot",),
            path="document.animations",
            linked_mesh_resolver=resolver,
        )

    assert calls == []


def test_attachment_name_preserves_shared_name_diagnostic():
    with pytest.raises(ValueError, match="attachment name cannot be empty"):
        validate([{}], attachment_name=" ")


def test_valid_deform_payload_is_not_mutated():
    frames = [
        {"vertices": [0.25, -0.5]},
        {"time": 1.0, "offset": 2, "vertices": [0.0, 0.0]},
    ]
    source = deepcopy(frames)

    validate(frames)

    assert frames == source


def test_xy_pair_alignment_remains_deform_specific():
    with pytest.raises(ValueError, match="preserve X/Y pair alignment"):
        deform_contract._require_non_negative_even_int(1, "offset")

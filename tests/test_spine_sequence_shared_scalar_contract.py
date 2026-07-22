from copy import deepcopy
from math import inf, nan

import pytest

import Blender_to_Spine2D_Mesh_Exporter.domain.spine.sequence_timeline_contract as sequence_contract
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.linked_mesh_contract import (
    LinkedMeshResolver,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import Skin
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_scalar_contract import (
    require_finite_number,
    require_name,
)


def setup_skin():
    return Skin(
        "default",
        {
            "slot": {
                "item": {
                    "type": "region",
                    "sequence": {"count": 4},
                }
            }
        },
    )


def test_sequence_contract_holds_exact_shared_scalar_functions():
    assert sequence_contract._require_name is require_name
    assert sequence_contract._require_finite_number is require_finite_number


@pytest.mark.parametrize("value", (True, False, None, "0", (), {}))
def test_sequence_time_preserves_strict_numeric_type_diagnostic(value):
    with pytest.raises(
        TypeError,
        match=r"sequence\[0\]\.time must be a finite number",
    ):
        sequence_contract._validate_sequence_timeline(
            [{"time": value}],
            path="sequence",
        )


@pytest.mark.parametrize("value", (inf, -inf, nan))
def test_sequence_delay_preserves_non_finite_diagnostic(value):
    with pytest.raises(
        ValueError,
        match=r"sequence\[0\]\.delay must be finite",
    ):
        sequence_contract._validate_sequence_timeline(
            [{"delay": value}],
            path="sequence",
        )


def test_inherited_delay_is_evaluated_without_payload_mutation():
    frames = [
        {"mode": "hold", "delay": 0.25},
        {"time": 1, "mode": "once"},
    ]
    source = deepcopy(frames)

    sequence_contract._validate_sequence_timeline(
        frames,
        path="sequence",
    )

    assert frames == source


def test_missing_effective_delay_still_fails_for_non_hold_mode():
    with pytest.raises(
        ValueError,
        match="delay must resolve to a value greater than 0",
    ):
        sequence_contract._validate_sequence_timeline(
            [{"mode": "once"}],
            path="sequence",
        )


def test_exact_runtime_frame_packing_bound_is_unchanged():
    sequence_contract._validate_sequence_timeline(
        [{"index": 1_048_575}],
        path="sequence",
    )

    with pytest.raises(ValueError, match="exact runtime frame packing"):
        sequence_contract._validate_sequence_timeline(
            [{"index": 1_048_576}],
            path="sequence",
        )


def test_skin_name_fails_before_resolver_lookup(monkeypatch):
    skins = (setup_skin(),)
    resolver = LinkedMeshResolver(skins, path="document.skins")
    calls = []
    original_require_skin = LinkedMeshResolver.require_skin

    def recording_require_skin(self, skin_name, *, path):
        calls.append((skin_name, path))
        return original_require_skin(self, skin_name, path=path)

    monkeypatch.setattr(
        LinkedMeshResolver,
        "require_skin",
        recording_require_skin,
    )

    with pytest.raises(ValueError, match="skin name cannot be empty"):
        sequence_contract.validate_animation_sequence_timelines(
            {"idle": {"attachments": {" ": {}}}},
            skins=skins,
            slot_names=("slot",),
            path="document.animations",
            linked_mesh_resolver=resolver,
        )

    assert calls == []


def test_attachment_name_fails_before_attachment_lookup(monkeypatch):
    skins = (setup_skin(),)
    resolver = LinkedMeshResolver(skins, path="document.skins")
    calls = []
    original_get_attachment = LinkedMeshResolver.get_attachment

    def recording_get_attachment(self, reference, *, path):
        calls.append((reference, path))
        return original_get_attachment(self, reference, path=path)

    monkeypatch.setattr(
        LinkedMeshResolver,
        "get_attachment",
        recording_get_attachment,
    )

    with pytest.raises(ValueError, match="attachment name cannot be empty"):
        sequence_contract.validate_animation_sequence_timelines(
            {
                "idle": {
                    "attachments": {
                        "default": {
                            "slot": {
                                " ": {"sequence": [{}]},
                            }
                        }
                    }
                }
            },
            skins=skins,
            slot_names=("slot",),
            path="document.animations",
            linked_mesh_resolver=resolver,
        )

    assert calls == []

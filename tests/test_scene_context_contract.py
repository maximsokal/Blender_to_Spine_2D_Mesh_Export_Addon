from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_context_contract import (
    BlenderSceneContextError,
    require_context_scene_consistency,
    rna_identity,
)


class _RnaValue:
    def __init__(self, pointer, name):
        self._pointer = pointer
        self.name = name

    def as_pointer(self):
        return self._pointer


def test_matching_context_and_explicit_scene_are_accepted():
    scene_from_context = _RnaValue(101, "Scene")
    explicit_proxy = _RnaValue(101, "Scene")

    require_context_scene_consistency(
        SimpleNamespace(scene=scene_from_context),
        explicit_proxy,
    )


def test_mismatched_context_and_explicit_scene_are_rejected():
    context_scene = _RnaValue(101, "Scene_A")
    explicit_scene = _RnaValue(202, "Scene_B")

    with pytest.raises(BlenderSceneContextError, match="same Scene"):
        require_context_scene_consistency(
            SimpleNamespace(scene=context_scene),
            explicit_scene,
        )


def test_context_without_scene_is_rejected_when_scene_is_explicit():
    with pytest.raises(BlenderSceneContextError, match="has no scene"):
        require_context_scene_consistency(SimpleNamespace(), object())


def test_omitted_owner_does_not_create_a_false_mismatch():
    require_context_scene_consistency(None, object())
    require_context_scene_consistency(SimpleNamespace(scene=object()), None)


def test_rna_identity_rejects_invalid_or_failing_pointers():
    with pytest.raises(BlenderSceneContextError, match="invalid pointer"):
        rna_identity(_RnaValue(0, "Invalid"))

    class Broken:
        def as_pointer(self):
            raise RuntimeError("gone")

    with pytest.raises(BlenderSceneContextError, match="resolve RNA identity"):
        rna_identity(Broken())

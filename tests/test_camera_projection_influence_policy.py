from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_state import (
    configure_camera_visibility,
    preserve_camera_projection_state,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    CameraProjectionInfluencePolicy,
)


class _FakeObject:
    def __init__(self, name: str, *, object_type: str = "MESH") -> None:
        self.name = name
        self.type = object_type
        self.hide_render = False
        self.visible_camera = True
        self.visible_shadow = True
        self.visible_glossy = True
        self.visible_transmission = True


class _FakeScene:
    def __init__(self, objects: tuple[_FakeObject, ...]) -> None:
        self.objects = objects
        self.world = object()
        self.frame_current = 7

    def frame_set(self, frame: int) -> None:
        self.frame_current = frame


def test_default_policy_hides_other_geometry_from_camera_only() -> None:
    source = _FakeObject("Source")
    dependency = _FakeObject("Dependency")
    light = _FakeObject("Light", object_type="LIGHT")
    scene = _FakeScene((source, dependency, light))

    configure_camera_visibility(
        source,
        scene,
        isolate=True,
        influence_policy=CameraProjectionInfluencePolicy(),
    )

    assert source.hide_render is False
    assert source.visible_camera is True
    assert dependency.visible_camera is False
    assert dependency.visible_shadow is True
    assert dependency.visible_glossy is True
    assert dependency.visible_transmission is True
    assert light.visible_camera is True


def test_policy_can_disable_shadow_reflection_and_transmission_rays_independently() -> None:
    source = _FakeObject("Source")
    dependency = _FakeObject("Dependency")
    scene = _FakeScene((source, dependency))

    configure_camera_visibility(
        source,
        scene,
        isolate=True,
        influence_policy=CameraProjectionInfluencePolicy(
            include_scene_shadows=False,
            include_scene_reflection_transmission=False,
            world_affects_lighting_reflections=True,
        ),
    )

    assert dependency.visible_camera is False
    assert dependency.visible_shadow is False
    assert dependency.visible_glossy is False
    assert dependency.visible_transmission is False
    assert source.visible_shadow is True
    assert source.visible_glossy is True
    assert source.visible_transmission is True


def test_projection_state_restores_world_frame_and_all_ray_visibility() -> None:
    source = _FakeObject("Source")
    dependency = _FakeObject("Dependency")
    source.hide_render = True
    dependency.visible_camera = False
    dependency.visible_shadow = False
    dependency.visible_glossy = False
    dependency.visible_transmission = False
    scene = _FakeScene((source, dependency))
    original_world = scene.world

    with preserve_camera_projection_state(scene):
        source.hide_render = False
        source.visible_camera = False
        dependency.visible_camera = True
        dependency.visible_shadow = True
        dependency.visible_glossy = True
        dependency.visible_transmission = True
        scene.world = None
        scene.frame_current = 99

    assert source.hide_render is True
    assert source.visible_camera is True
    assert dependency.visible_camera is False
    assert dependency.visible_shadow is False
    assert dependency.visible_glossy is False
    assert dependency.visible_transmission is False
    assert scene.world is original_world
    assert scene.frame_current == 7


def test_policy_contract_rejects_non_boolean_fields() -> None:
    try:
        CameraProjectionInfluencePolicy(
            include_scene_shadows=1,  # type: ignore[arg-type]
        )
    except TypeError:
        pass
    else:
        raise AssertionError("non-boolean policy field must be rejected")

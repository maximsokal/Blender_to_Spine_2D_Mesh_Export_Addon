from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    BakeSceneState,
    configure_scene_for_bake,
    preserve_bake_scene_state,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeEvaluationScope,
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    TextureFormat,
    build_bake_plan,
)


class FakeScene:
    def __init__(self):
        self.render = SimpleNamespace(
            engine="BLENDER_EEVEE",
            image_settings=SimpleNamespace(file_format="JPEG", color_mode="RGB"),
            bake=SimpleNamespace(
                margin=16,
                use_clear=False,
                use_selected_to_active=False,
                use_cage=False,
                cage_extrusion=0.0,
                view_from="ABOVE_SURFACE",
                use_pass_direct=True,
                use_pass_indirect=True,
                use_pass_color=False,
                use_pass_diffuse=False,
                use_pass_glossy=False,
                use_pass_transmission=False,
                use_pass_emit=False,
            ),
        )
        self.cycles = SimpleNamespace(bake_type="NORMAL", samples=32)
        self.camera = None
        self.frame_current = 12

    def frame_set(self, value):
        self.frame_current = value


def make_plan(
    tmp_path,
    *,
    selected_to_active=False,
    procedural=False,
    texture_format=TextureFormat.PNG,
):
    kind = MaterialKind.PROCEDURAL if procedural else MaterialKind.IMAGE
    analysis = ObjectMaterialAnalysis(
        "Cube",
        (MaterialAnalysis(0, "Material", kind),),
    )
    return build_bake_plan(
        analysis,
        BakeSettings(
            width=256,
            height=128,
            output_directory=tmp_path,
            output_stem="Cube",
            texture_format=texture_format,
            selected_to_active=selected_to_active,
        ),
    )


def test_configure_scene_maps_explicit_diffuse_and_selected_to_active(tmp_path):
    scene = FakeScene()
    plan = make_plan(tmp_path, selected_to_active=True)
    execution = BakeExecutionSettings(samples=128, color_mode="RGBA")

    configure_scene_for_bake(
        scene,
        plan,
        execution,
        bake_mode=BakeMode.DIFFUSE,
        evaluation_scope=BakeEvaluationScope.LOCAL,
    )

    assert scene.render.engine == "CYCLES"
    assert scene.render.image_settings.file_format == "PNG"
    assert scene.render.image_settings.color_mode == "RGBA"
    assert scene.render.bake.margin == 4
    assert scene.render.bake.use_clear
    assert scene.render.bake.use_selected_to_active
    assert scene.render.bake.use_cage
    assert scene.render.bake.cage_extrusion == 0.1
    assert scene.render.bake.view_from == "ABOVE_SURFACE"
    assert not scene.render.bake.use_pass_direct
    assert not scene.render.bake.use_pass_indirect
    assert scene.render.bake.use_pass_color
    assert scene.cycles.bake_type == "DIFFUSE"
    assert scene.cycles.samples == 128


def test_configure_scene_enables_blender_52_combined_contributions(tmp_path):
    scene = FakeScene()
    plan = make_plan(tmp_path, procedural=True)
    assert plan.bake_mode is BakeMode.COMBINED

    configure_scene_for_bake(
        scene,
        plan,
        BakeExecutionSettings(),
        bake_mode=BakeMode.COMBINED,
        evaluation_scope=BakeEvaluationScope.LOCAL,
    )

    assert scene.render.bake.view_from == "ABOVE_SURFACE"
    assert scene.render.bake.use_pass_direct
    assert scene.render.bake.use_pass_indirect
    assert scene.render.bake.use_pass_color
    assert scene.render.bake.use_pass_diffuse
    assert scene.render.bake.use_pass_glossy
    assert scene.render.bake.use_pass_transmission
    assert scene.render.bake.use_pass_emit
    assert scene.cycles.bake_type == "COMBINED"


def test_camera_scope_uses_active_camera_view_and_transmission(tmp_path):
    scene = FakeScene()
    scene.camera = object()
    plan = make_plan(tmp_path, procedural=True)

    configure_scene_for_bake(
        scene,
        plan,
        BakeExecutionSettings(),
        bake_mode=BakeMode.COMBINED,
        evaluation_scope=BakeEvaluationScope.CAMERA,
    )

    assert scene.render.bake.view_from == "ACTIVE_CAMERA"
    assert scene.render.bake.use_pass_transmission
    assert scene.render.bake.use_pass_glossy
    assert scene.render.bake.use_pass_direct
    assert scene.render.bake.use_pass_indirect


def test_camera_scope_requires_scene_camera(tmp_path):
    scene = FakeScene()

    with pytest.raises(Exception, match="requires scene.camera"):
        configure_scene_for_bake(
            scene,
            make_plan(tmp_path, procedural=True),
            BakeExecutionSettings(),
            bake_mode=BakeMode.COMBINED,
            evaluation_scope=BakeEvaluationScope.CAMERA,
        )


def test_configure_scene_requires_explicit_bake_mode_and_scope(tmp_path):
    scene = FakeScene()
    plan = make_plan(tmp_path)

    with pytest.raises(TypeError):
        configure_scene_for_bake(scene, plan, BakeExecutionSettings())

    with pytest.raises(TypeError, match="bake_mode must be BakeMode"):
        configure_scene_for_bake(
            scene,
            plan,
            BakeExecutionSettings(),
            bake_mode=None,
            evaluation_scope=BakeEvaluationScope.LOCAL,
        )

    with pytest.raises(TypeError, match="evaluation_scope must be BakeEvaluationScope"):
        configure_scene_for_bake(
            scene,
            plan,
            BakeExecutionSettings(),
            bake_mode=BakeMode.DIFFUSE,
            evaluation_scope=None,
        )


def test_preserve_bake_scene_state_restores_after_exception(tmp_path):
    scene = FakeScene()
    state_before = BakeSceneState.capture(scene)

    with pytest.raises(RuntimeError):
        with preserve_bake_scene_state(scene):
            configure_scene_for_bake(
                scene,
                make_plan(tmp_path, selected_to_active=True),
                BakeExecutionSettings(samples=64),
                bake_mode=BakeMode.DIFFUSE,
                evaluation_scope=BakeEvaluationScope.LOCAL,
            )
            scene.frame_set(99)
            raise RuntimeError("simulated bake failure")

    state_after = BakeSceneState.capture(scene)
    assert state_after == state_before


def test_preserve_restores_camera_view_and_combined_contributions(tmp_path):
    scene = FakeScene()
    scene.camera = object()
    state_before = BakeSceneState.capture(scene)

    with preserve_bake_scene_state(scene):
        configure_scene_for_bake(
            scene,
            make_plan(tmp_path, procedural=True),
            BakeExecutionSettings(),
            bake_mode=BakeMode.COMBINED,
            evaluation_scope=BakeEvaluationScope.CAMERA,
        )
        assert scene.render.bake.view_from == "ACTIVE_CAMERA"
        assert scene.render.bake.use_pass_transmission

    assert BakeSceneState.capture(scene) == state_before


def test_capture_rejects_missing_required_blender_52_bake_property():
    scene = FakeScene()
    del scene.render.bake.use_pass_transmission

    with pytest.raises(Exception, match="use_pass_transmission"):
        BakeSceneState.capture(scene)


class FormatAwareImageSettings:
    """Small RNA-like fake whose color-mode enum depends on file format."""

    def __init__(self, *, file_format: str, color_mode: str):
        self._file_format = file_format
        self._color_mode = color_mode

    @property
    def file_format(self):
        return self._file_format

    @file_format.setter
    def file_format(self, value):
        self._file_format = value
        if value == "JPEG" and self._color_mode == "RGBA":
            self._color_mode = "RGB"

    @property
    def color_mode(self):
        return self._color_mode

    @color_mode.setter
    def color_mode(self, value):
        if self._file_format == "JPEG" and value == "RGBA":
            raise TypeError("JPEG does not expose RGBA")
        self._color_mode = value


def test_configure_scene_resolves_default_rgba_to_rgb_for_jpeg(tmp_path):
    scene = FakeScene()
    scene.render.image_settings = FormatAwareImageSettings(
        file_format="PNG",
        color_mode="RGBA",
    )

    configure_scene_for_bake(
        scene,
        make_plan(tmp_path, texture_format=TextureFormat.JPEG),
        BakeExecutionSettings(),
        bake_mode=BakeMode.DIFFUSE,
        evaluation_scope=BakeEvaluationScope.LOCAL,
    )

    assert scene.render.image_settings.file_format == "JPEG"
    assert scene.render.image_settings.color_mode == "RGB"


def test_scene_restore_returns_format_before_dependent_color_mode():
    scene = FakeScene()
    scene.render.image_settings = FormatAwareImageSettings(
        file_format="PNG",
        color_mode="RGBA",
    )
    state_before = BakeSceneState.capture(scene)

    with pytest.raises(RuntimeError, match="forced after format switch"):
        with preserve_bake_scene_state(scene):
            scene.render.image_settings.file_format = "JPEG"
            raise RuntimeError("forced after format switch")

    assert BakeSceneState.capture(scene) == state_before


def test_execution_settings_validation():
    with pytest.raises(ValueError):
        BakeExecutionSettings(samples=0)
    with pytest.raises(ValueError):
        BakeExecutionSettings(color_mode="INVALID")
    with pytest.raises(ValueError):
        BakeExecutionSettings(generated_color=(0.0, 0.0, 0.0))

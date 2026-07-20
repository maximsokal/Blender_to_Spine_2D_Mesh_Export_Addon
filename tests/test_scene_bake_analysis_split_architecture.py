import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import scene_bake_analyzer
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import scene_bake_rna
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_capture import (
    analyse_bake_contexts,
    analyse_scene_bake_context,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_error import (
    SceneBakeAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_resources import (
    analyse_camera,
    analyse_light,
    analyse_object_bake_context,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_runtime import (
    validate_runtime_scene_context,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_world import (
    analyse_world,
)


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _source(name: str) -> str:
    return (ADAPTER / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=name)


def _top_level_definitions(name: str):
    return tuple(
        node
        for node in _tree(name).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _matrix(scale: float = 1.0):
    return tuple(
        tuple(scale if row == column else 0.0 for column in range(4))
        for row in range(4)
    )


class _Collection:
    def __init__(self, name: str):
        self.name = name


class _Object:
    def __init__(self, name: str, object_type: str, data=None):
        self.name = name
        self.type = object_type
        self.data = data
        self.matrix_world = _matrix()
        self.hide_render = False
        self.visible_camera = True
        self.visible_shadow = True
        self.users_collection = (_Collection("Collection"),)
        self.animation_data = None


class _LightData:
    def __init__(self):
        self.type = "AREA"
        self.energy = 1000.0
        self.color = (1.0, 1.0, 1.0)
        self.use_shadow = True
        self.animation_data = None


class _CameraData:
    def __init__(self):
        self.type = "PERSP"
        self.lens = 50.0
        self.ortho_scale = 6.0
        self.clip_start = 0.1
        self.clip_end = 1000.0
        self.animation_data = None


class _World:
    def __init__(self):
        self.name = "World"
        self.color = (0.05, 0.05, 0.05)
        self.use_nodes = False
        self.node_tree = None
        self.animation_data = None


class _ViewSettings:
    def __init__(self):
        self.view_transform = "Standard"
        self.look = ""
        self.exposure = 0.0
        self.gamma = 1.0


class _Scene:
    def __init__(self, objects, camera):
        self.name = "Scene"
        self.objects = tuple(objects)
        self.camera = camera
        self.world = _World()
        self.view_settings = _ViewSettings()
        self.render = SimpleNamespace(engine="CYCLES")
        self.frame_current = 1


def _fixture():
    source = _Object("Source", "MESH", SimpleNamespace(animation_data=None))
    occluder = _Object("Occluder", "MESH", SimpleNamespace(animation_data=None))
    light = _Object("Key", "LIGHT", _LightData())
    camera = _Object("Camera", "CAMERA", _CameraData())
    scene = _Scene((camera, source, light, occluder), camera)
    return source, occluder, light, camera, scene


def test_scene_bake_analyzer_is_compatibility_only():
    assert _top_level_definitions("scene_bake_analyzer.py") == ()
    source = _source("scene_bake_analyzer.py")
    for owner in (
        "scene_bake_capture",
        "scene_bake_error",
        "scene_bake_resources",
        "scene_bake_rna",
        "scene_bake_runtime",
        "scene_bake_world",
    ):
        assert owner in source


def test_physical_scene_bake_ownership_boundaries():
    rna = _source("scene_bake_rna.py")
    world = _source("scene_bake_world.py")
    resources = _source("scene_bake_resources.py")
    capture = _source("scene_bake_capture.py")
    runtime = _source("scene_bake_runtime.py")

    for forbidden in (
        "ObjectBakeContext",
        "SceneBakeContext",
        "WorldBakeSnapshot",
        "LightBakeSnapshot",
        "CameraBakeSnapshot",
    ):
        assert forbidden not in rna
    assert "resolved_scene.objects" not in world
    assert "validate_runtime_scene_context" not in resources
    assert "validate_runtime_scene_context" not in capture
    for forbidden in ("OUTPUT_WORLD", "BACKGROUND", "input_socket", "node_tree"):
        assert forbidden not in runtime


def test_production_callers_and_package_use_physical_owners():
    planning = _source("a1_texture_planning.py")
    semantic = _source("semantic_bake_validation.py")
    camera = _source("camera_projection_validation.py")
    package = _source("__init__.py")

    assert "from .scene_bake_capture import analyse_bake_contexts" in planning
    assert "from .scene_bake_runtime import validate_runtime_scene_context" in semantic
    assert "from .scene_bake_runtime import validate_runtime_scene_context" in camera
    assert "from .scene_bake_analyzer import" not in planning
    assert "from .scene_bake_analyzer import" not in semantic
    assert "from .scene_bake_analyzer import" not in camera
    assert "from .scene_bake_capture import" in package
    assert "from .scene_bake_error import SceneBakeAnalysisError" in package
    assert "from .scene_bake_resources import analyse_object_bake_context" in package
    assert "from .scene_bake_runtime import validate_runtime_scene_context" in package


def test_explicit_scene_and_context_do_not_load_bpy(monkeypatch):
    source, _occluder, _light, _camera, scene = _fixture()

    def fail_load():
        raise AssertionError("bpy was loaded")

    monkeypatch.setattr(scene_bake_rna, "load_bpy", fail_load)
    direct = analyse_scene_bake_context(scene=scene)
    through_context = analyse_scene_bake_context(context=SimpleNamespace(scene=scene))
    assert direct.scene_name == "Scene"
    assert through_context == direct
    assert analyse_object_bake_context(source).source_object_id == "Source"


def test_missing_scene_uses_lazy_bpy_error(monkeypatch):
    monkeypatch.setattr(
        scene_bake_rna,
        "load_bpy",
        lambda: SimpleNamespace(context=None),
    )
    with pytest.raises(SceneBakeAnalysisError, match="Blender Scene"):
        analyse_scene_bake_context()


def test_capture_is_deterministic_and_negative_light_energy_is_clamped():
    source, _occluder, light, _camera, scene = _fixture()
    light.data.energy = -20.0
    object_context, scene_context = analyse_bake_contexts(source, scene=scene)

    assert object_context.collection_names == ("Collection",)
    assert tuple(item.object_id for item in scene_context.lights) == ("Key",)
    assert scene_context.lights[0].energy == 0.0
    assert scene_context.visible_object_ids == ("Camera", "Key", "Occluder", "Source")
    assert scene_context.shadow_caster_ids == ("Occluder", "Source")
    assert analyse_world(scene).background_strength == 1.0


def test_non_finite_resources_and_invalid_camera_clip_fail_consistently():
    _source, _occluder, light, camera, scene = _fixture()
    light.data.energy = float("nan")
    with pytest.raises(SceneBakeAnalysisError, match="Light 'Key' energy"):
        analyse_light(light)

    light.data.energy = 1.0
    camera.data.clip_start = 10.0
    camera.data.clip_end = 1.0
    with pytest.raises(SceneBakeAnalysisError, match="clip_end"):
        analyse_camera(scene)

    camera.data.clip_start = 0.1
    camera.data.clip_end = 100.0
    scene.view_settings.gamma = 0.0
    with pytest.raises(SceneBakeAnalysisError, match="color-management snapshot"):
        analyse_scene_bake_context(scene=scene)


def test_runtime_allows_frame_and_numeric_animation_values_to_change():
    source, _occluder, light, camera, scene = _fixture()
    expected_object, expected_scene = analyse_bake_contexts(source, scene=scene)

    scene.frame_current = 19
    source.matrix_world = _matrix(2.0)
    light.data.energy = 25.0
    light.data.color = (0.2, 0.4, 0.8)
    light.matrix_world = _matrix(3.0)
    camera.data.lens = 85.0
    camera.data.clip_end = 500.0
    camera.matrix_world = _matrix(4.0)
    scene.world.color = (0.8, 0.2, 0.1)

    validate_runtime_scene_context(
        source,
        expected_object,
        expected_scene,
        scene=scene,
    )


def test_runtime_rejects_source_and_scene_set_changes_in_fixed_order():
    source, occluder, _light, _camera, scene = _fixture()
    expected_object, expected_scene = analyse_bake_contexts(source, scene=scene)

    source.users_collection = (_Collection("Moved"),)
    source.hide_render = True
    occluder.hide_render = True

    with pytest.raises(SceneBakeAnalysisError) as captured:
        validate_runtime_scene_context(
            source,
            expected_object,
            expected_scene,
            scene=scene,
        )
    message = str(captured.value)
    assert message.index("source collection membership changed") < message.index(
        "source render visibility changed"
    )
    assert message.index("source render visibility changed") < message.index(
        "render-visible object set changed"
    )
    assert "shadow-caster set changed" in message


def test_runtime_rejects_world_camera_light_and_color_structure_changes():
    source, _occluder, light, camera, scene = _fixture()
    expected_object, expected_scene = analyse_bake_contexts(source, scene=scene)

    scene.world.use_nodes = True
    camera.data.type = "ORTHO"
    light.data.type = "POINT"
    scene.view_settings.view_transform = "AgX"

    with pytest.raises(SceneBakeAnalysisError) as captured:
        validate_runtime_scene_context(
            source,
            expected_object,
            expected_scene,
            scene=scene,
        )
    message = str(captured.value)
    assert "World structure changed" in message
    assert "active camera structure changed" in message
    assert "visible light structure changed" in message
    assert "color management changed" in message


def test_runtime_rejects_animation_presence_changes():
    source, _occluder, light, camera, scene = _fixture()
    expected_object, expected_scene = analyse_bake_contexts(source, scene=scene)

    source.animation_data = SimpleNamespace(action=object(), drivers=())
    light.data.animation_data = SimpleNamespace(action=object(), drivers=())
    camera.data.animation_data = SimpleNamespace(action=object(), drivers=())
    scene.world.animation_data = SimpleNamespace(action=object(), drivers=())

    with pytest.raises(SceneBakeAnalysisError) as captured:
        validate_runtime_scene_context(
            source,
            expected_object,
            expected_scene,
            scene=scene,
        )
    message = str(captured.value)
    assert "source animation status changed" in message
    assert "World structure changed" in message
    assert "active camera structure changed" in message
    assert "visible light structure changed" in message


def test_facade_retains_historical_alias_identity():
    assert scene_bake_analyzer.SceneBakeAnalysisError is SceneBakeAnalysisError
    assert scene_bake_analyzer.analyse_bake_contexts is analyse_bake_contexts
    assert scene_bake_analyzer.analyse_scene_bake_context is analyse_scene_bake_context
    assert scene_bake_analyzer.analyse_object_bake_context is analyse_object_bake_context
    assert scene_bake_analyzer.validate_runtime_scene_context is validate_runtime_scene_context
    assert scene_bake_analyzer._analyse_light is analyse_light
    assert scene_bake_analyzer._analyse_camera is analyse_camera
    assert scene_bake_analyzer._analyse_world is analyse_world

"""Real Blender 5.2 pixel tests for Camera Projection scene influence."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback

import bpy
from mathutils import Vector

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeExecutionSettings,
    CameraProjectionInfluencePolicy,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _create_mesh_object,
)
from run_camera_projection_integration import (  # noqa: E402
    _aim_at,
    _configure_scene,
    _create_camera,
    _create_cube,
    _read_pixels,
)


def _create_quad(name: str):
    return _create_mesh_object(
        name,
        (
            (-2.0, -2.0, 0.0),
            (2.0, -2.0, 0.0),
            (2.0, 2.0, 0.0),
            (-2.0, 2.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )


def _create_principled_material(
    name: str,
    *,
    base_color: tuple[float, float, float, float],
    metallic: float,
    roughness: float,
):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    material.node_tree.links.new(
        principled.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material


def _create_area_light(name: str, *, location: tuple[float, float, float]):
    data = bpy.data.lights.new(name=f"{name}_Data", type="AREA")
    data.energy = 900.0
    data.shape = "DISK"
    data.size = 1.2
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    _aim_at(obj, Vector((0.0, 0.0, 0.0)))
    return obj


def _settings(
    output_directory: Path,
    stem: str,
    policy: CameraProjectionInfluencePolicy,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version="4.2.43",
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
            bake_margin=1,
        ),
        prefix=stem,
        output_stem=stem,
        json_output_stem=stem,
        source_geometry_mode=A1SourceGeometryMode.EVALUATED,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        bake_execution=BakeExecutionSettings(
            samples=16,
            texture_export_mode=A1TextureExportMode.CAMERA_PROJECTION,
            camera_influence_policy=policy,
        ),
    )


def _export_pixels(
    source,
    output_directory: Path,
    stem: str,
    policy: CameraProjectionInfluencePolicy,
) -> tuple[float, ...]:
    _activate_only(source)
    result = export_a1_single_object(
        source,
        _settings(output_directory, stem, policy),
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(result.success, f"Camera influence export failed: {result.issues}")
    _assert(len(result.output_files) == 2, "static export must write JSON plus PNG")
    texture_path = result.output_files[1]
    _assert(texture_path.is_file(), f"missing rendered texture: {texture_path}")
    return _read_pixels(texture_path)


def _mean_visible_luminance(pixels: tuple[float, ...]) -> float:
    total = weight = 0.0
    for offset in range(0, len(pixels), 4):
        alpha = float(pixels[offset + 3])
        if alpha <= 0.05:
            continue
        red = float(pixels[offset])
        green = float(pixels[offset + 1])
        blue = float(pixels[offset + 2])
        total += (0.2126 * red + 0.7152 * green + 0.0722 * blue) * alpha
        weight += alpha
    _assert(weight > 0.0, "render contains no visible source pixels")
    return total / weight


def _mean_visible_rgb(pixels: tuple[float, ...]) -> tuple[float, float, float]:
    red = green = blue = weight = 0.0
    for offset in range(0, len(pixels), 4):
        alpha = float(pixels[offset + 3])
        if alpha <= 0.05:
            continue
        red += float(pixels[offset]) * alpha
        green += float(pixels[offset + 1]) * alpha
        blue += float(pixels[offset + 2]) * alpha
        weight += alpha
    _assert(weight > 0.0, "render contains no visible source pixels")
    return red / weight, green / weight, blue / weight


def test_camera_hidden_object_shadow_toggle_changes_render() -> None:
    _clear_scene()
    _configure_scene()
    _create_camera()
    scene = bpy.context.scene
    scene.world.color = (0.0, 0.0, 0.0)
    if scene.world and scene.world.use_nodes:
        background = next(
            node
            for node in scene.world.node_tree.nodes
            if node.bl_idname == "ShaderNodeBackground"
        )
        background.inputs["Strength"].default_value = 0.0

    source = _create_quad("ShadowReceiver")
    source.data.materials.append(
        _create_principled_material(
            "ShadowReceiverMaterial",
            base_color=(0.8, 0.8, 0.8, 1.0),
            metallic=0.0,
            roughness=0.85,
        )
    )
    caster = _create_cube("CameraHiddenShadowCaster")
    caster.scale = (0.65, 0.65, 0.65)
    caster.location = (0.35, 0.15, 1.4)
    _create_area_light("ShadowKey", location=(1.8, -1.6, 4.0))

    with tempfile.TemporaryDirectory(prefix="spine2d-shadow-policy-") as directory:
        root = Path(directory)
        with_shadow = _export_pixels(
            source,
            root / "with-shadow",
            "WithShadow",
            CameraProjectionInfluencePolicy(
                include_scene_shadows=True,
                include_scene_reflection_transmission=False,
                world_affects_lighting_reflections=False,
            ),
        )
        without_shadow = _export_pixels(
            source,
            root / "without-shadow",
            "WithoutShadow",
            CameraProjectionInfluencePolicy(
                include_scene_shadows=False,
                include_scene_reflection_transmission=False,
                world_affects_lighting_reflections=False,
            ),
        )

    with_value = _mean_visible_luminance(with_shadow)
    without_value = _mean_visible_luminance(without_shadow)
    _assert(
        without_value > with_value + 0.025,
        "disabling scene-object shadows did not brighten the rendered source; "
        f"with={with_value}, without={without_value}",
    )


def test_world_toggle_changes_metallic_reflection_but_keeps_transparent_background() -> None:
    _clear_scene()
    _configure_scene()
    _create_camera()
    scene = bpy.context.scene
    world = scene.world
    _assert(world is not None and world.use_nodes, "test requires a node World")
    background = next(
        node
        for node in world.node_tree.nodes
        if node.bl_idname == "ShaderNodeBackground"
    )
    background.inputs["Color"].default_value = (0.02, 0.2, 1.0, 1.0)
    background.inputs["Strength"].default_value = 2.5

    source = _create_cube("WorldReflectionSource")
    source.data.materials.append(
        _create_principled_material(
            "WorldReflectionMaterial",
            base_color=(0.15, 0.15, 0.15, 1.0),
            metallic=1.0,
            roughness=0.08,
        )
    )

    with tempfile.TemporaryDirectory(prefix="spine2d-world-policy-") as directory:
        root = Path(directory)
        with_world = _export_pixels(
            source,
            root / "with-world",
            "WithWorld",
            CameraProjectionInfluencePolicy(
                include_scene_shadows=False,
                include_scene_reflection_transmission=False,
                world_affects_lighting_reflections=True,
            ),
        )
        without_world = _export_pixels(
            source,
            root / "without-world",
            "WithoutWorld",
            CameraProjectionInfluencePolicy(
                include_scene_shadows=False,
                include_scene_reflection_transmission=False,
                world_affects_lighting_reflections=False,
            ),
        )

    with_rgb = _mean_visible_rgb(with_world)
    without_rgb = _mean_visible_rgb(without_world)
    delta = sum(abs(first - second) for first, second in zip(with_rgb, without_rgb, strict=True))
    _assert(
        delta > 0.12,
        "World toggle did not change metallic reflection enough; "
        f"with={with_rgb}, without={without_rgb}, delta={delta}",
    )


def main() -> None:
    tests = (
        test_camera_hidden_object_shadow_toggle_changes_render,
        test_world_toggle_changes_metallic_reflection_but_keeps_transparent_background,
    )
    failures: list[tuple[str, str]] = []
    print(f"Blender version: {bpy.app.version_string}")
    for test in tests:
        print(f"[CAMERA-INFLUENCE] RUN {test.__name__}")
        try:
            test()
        except Exception:
            failures.append((test.__name__, traceback.format_exc()))
            print(f"[CAMERA-INFLUENCE] FAIL {test.__name__}")
        else:
            print(f"[CAMERA-INFLUENCE] PASS {test.__name__}")
        finally:
            _clear_scene()

    if failures:
        for name, details in failures:
            print(f"\n--- {name} ---\n{details}")
        raise SystemExit(1)
    print(f"[CAMERA-INFLUENCE] PASS {len(tests)} integration tests")


if __name__ == "__main__":
    main()

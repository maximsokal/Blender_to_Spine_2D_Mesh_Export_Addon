"""Reusable real-Blender fixtures and assertions for semantic texture baking.

The helpers deliberately build tiny scenes at runtime instead of committing binary
``.blend`` fixtures. This keeps the fixtures reviewable while still exercising the
real Blender 5.2 RNA, Cycles, image codecs, operators, depsgraph, and cleanup paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable
import warnings

import bpy

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    analyse_object_materials,
    read_source_mesh_snapshot,
    unwrap_snapshot_uv,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
    write_uv_coordinate,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeMode,
    BakeSettings,
    TextureFormat,
    build_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TEMPORARY_PREFIX = "__Spine2D"


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """User-visible Blender context that one bake must restore exactly."""

    active_object_name: str | None
    selected_object_names: tuple[str, ...]
    mode: str


@dataclass(frozen=True, slots=True)
class SceneBakeSnapshot:
    """Independent snapshot of every Scene value changed by semantic baking."""

    frame_current: int
    render_engine: str
    file_format: str
    color_mode: str
    bake_margin: int
    bake_use_clear: bool
    bake_selected_to_active: bool
    bake_use_cage: bool
    cage_extrusion: float
    use_pass_direct: bool
    use_pass_indirect: bool
    use_pass_color: bool
    cycles_bake_type: str
    cycles_samples: int


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Decoded image dimensions and immutable straight-RGBA pixels."""

    width: int
    height: int
    channels: int
    pixels: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("decoded image dimensions must be positive")
        if self.channels != 4:
            raise ValueError("semantic bake integration tests require RGBA images")
        expected = self.width * self.height * self.channels
        if len(self.pixels) != expected:
            raise ValueError(
                f"decoded image has {len(self.pixels)} values, expected {expected}"
            )
        if not all(isfinite(value) for value in self.pixels):
            raise ValueError("decoded image contains NaN or infinite values")

    @property
    def rgba_pixels(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(
            tuple(float(value) for value in self.pixels[offset : offset + 4])
            for offset in range(0, len(self.pixels), 4)
        )


def create_mesh_object(
    name: str,
    vertices: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, ...], ...],
):
    """Create one linked Mesh object with a deterministic planar source UV map."""

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.validate(clean_customdata=False)
    mesh.update(calc_edges=True, calc_edges_loose=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    uv_layer = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True

    minimum_x = min(vertex[0] for vertex in vertices)
    maximum_x = max(vertex[0] for vertex in vertices)
    minimum_y = min(vertex[1] for vertex in vertices)
    maximum_y = max(vertex[1] for vertex in vertices)
    size_x = max(maximum_x - minimum_x, 1.0)
    size_y = max(maximum_y - minimum_y, 1.0)
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = int(mesh.loops[loop_index].vertex_index)
            x_value, y_value, _z_value = vertices[vertex_index]
            write_uv_coordinate(
                uv_layer,
                int(loop_index),
                (
                    (float(x_value) - minimum_x) / size_x,
                    (float(y_value) - minimum_y) / size_y,
                ),
                expected_length=len(mesh.loops),
            )
    return obj


def create_quad(name: str = "BakeSource"):
    return create_mesh_object(
        name,
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )


def create_two_quad_object(name: str = "TwoMaterialSource"):
    return create_mesh_object(
        name,
        (
            (-2.0, -1.0, 0.0),
            (-0.2, -1.0, 0.0),
            (-0.2, 1.0, 0.0),
            (-2.0, 1.0, 0.0),
            (0.2, -1.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.2, 1.0, 0.0),
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )


def create_sentinel(name: str = "Sentinel"):
    return create_mesh_object(
        name,
        ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.5, 1.0, 0.0)),
        ((0, 1, 2),),
    )


def _enable_material_nodes(material) -> None:
    """Enable nodes while silencing Blender 5.2's Blender-6.0 deprecation notice."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Material.use_nodes.*",
            category=DeprecationWarning,
        )
        material.use_nodes = True


def _material_uses_nodes(material) -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Material.use_nodes.*",
            category=DeprecationWarning,
        )
        return bool(material.use_nodes)


def new_emission_material(
    name: str,
    color: tuple[float, float, float],
):
    """Create a minimal deterministic emission graph and return its color socket."""

    material = bpy.data.materials.new(name=name)
    _enable_material_nodes(material)
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    color_socket = emission.inputs["Color"]
    color_socket.default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material, color_socket


def new_principled_material(
    name: str,
    color: tuple[float, float, float],
    *,
    alpha: float = 1.0,
):
    material = bpy.data.materials.new(name=name)
    _enable_material_nodes(material)
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Alpha"].default_value = alpha
    material.node_tree.links.new(
        principled.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material


def new_transparent_material(name: str):
    material = bpy.data.materials.new(name=name)
    _enable_material_nodes(material)
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    material.node_tree.links.new(
        transparent.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material


def new_image_alpha_material(
    name: str,
    *,
    alpha: float,
):
    """Create one generated source image feeding Principled color and alpha."""

    material = bpy.data.materials.new(name=name)
    _enable_material_nodes(material)
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 1.0
    image_node = nodes.new(type="ShaderNodeTexImage")
    image = bpy.data.images.new(
        name=f"{name}_Image",
        width=2,
        height=2,
        alpha=True,
        float_buffer=True,
    )
    pixel = (0.05, 0.75, 0.15, alpha)
    image.generated_color = pixel
    image.pixels[:] = list(pixel) * 4
    image.update()
    image_node.image = image
    material.node_tree.links.new(
        image_node.outputs["Color"],
        principled.inputs["Base Color"],
    )
    material.node_tree.links.new(
        image_node.outputs["Alpha"],
        principled.inputs["Alpha"],
    )
    material.node_tree.links.new(
        principled.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material, image


def new_transparent_mix_material(
    name: str,
    *,
    factor: float,
    transparent_first: bool,
):
    """Create a Mix Shader whose socket order determines final opacity."""

    material = bpy.data.materials.new(name=name)
    _enable_material_nodes(material)
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mix = nodes.new(type="ShaderNodeMixShader")
    mix.inputs[0].default_value = factor
    transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (0.75, 0.08, 0.04, 1.0)
    principled.inputs["Roughness"].default_value = 1.0
    first = (
        transparent.outputs["BSDF"]
        if transparent_first
        else principled.outputs["BSDF"]
    )
    second = (
        principled.outputs["BSDF"]
        if transparent_first
        else transparent.outputs["BSDF"]
    )
    material.node_tree.links.new(first, mix.inputs[1])
    material.node_tree.links.new(second, mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def activate_only(obj, *, mode: str = "OBJECT") -> None:
    """Make one object active/selected and optionally leave it in Edit Mode."""

    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        result = bpy.ops.object.mode_set(mode="OBJECT")
        if "FINISHED" not in result:
            raise RuntimeError(f"unable to leave current mode: {result!r}")
    for candidate in tuple(bpy.context.scene.objects):
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    resolved_mode = mode.strip().upper()
    if resolved_mode != "OBJECT":
        result = bpy.ops.object.mode_set(mode=resolved_mode)
        if "FINISHED" not in result:
            raise RuntimeError(f"unable to enter {resolved_mode}: {result!r}")


def capture_context() -> ContextSnapshot:
    active = bpy.context.view_layer.objects.active
    return ContextSnapshot(
        active_object_name=None if active is None else active.name_full,
        selected_object_names=tuple(
            sorted(candidate.name_full for candidate in bpy.context.selected_objects)
        ),
        mode=str(bpy.context.mode),
    )


def capture_scene_bake_state() -> SceneBakeSnapshot:
    scene = bpy.context.scene
    return SceneBakeSnapshot(
        frame_current=int(scene.frame_current),
        render_engine=str(scene.render.engine),
        file_format=str(scene.render.image_settings.file_format),
        color_mode=str(scene.render.image_settings.color_mode),
        bake_margin=int(scene.render.bake.margin),
        bake_use_clear=bool(scene.render.bake.use_clear),
        bake_selected_to_active=bool(scene.render.bake.use_selected_to_active),
        bake_use_cage=bool(scene.render.bake.use_cage),
        cage_extrusion=float(scene.render.bake.cage_extrusion),
        use_pass_direct=bool(scene.render.bake.use_pass_direct),
        use_pass_indirect=bool(scene.render.bake.use_pass_indirect),
        use_pass_color=bool(scene.render.bake.use_pass_color),
        cycles_bake_type=str(scene.cycles.bake_type),
        cycles_samples=int(scene.cycles.samples),
    )


def _plain_socket_value(value: object) -> object:
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        resolved = tuple(float(component) for component in value)  # type: ignore[arg-type]
    except Exception:
        return type(value).__name__
    return resolved


def material_fingerprint(material) -> tuple[object, ...]:
    """Capture graph topology, active node, selection, and simple socket defaults."""

    nodes = tuple(
        sorted(
            (
                node.name,
                node.bl_idname,
                bool(node.select),
                tuple(
                    (socket.name, _plain_socket_value(socket.default_value))
                    for socket in node.inputs
                    if hasattr(socket, "default_value")
                ),
            )
            for node in material.node_tree.nodes
        )
    )
    links = tuple(
        sorted(
            (
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            for link in material.node_tree.links
        )
    )
    active = material.node_tree.nodes.active
    return (
        material.name_full,
        _material_uses_nodes(material),
        None if active is None else active.name,
        nodes,
        links,
    )


def datablock_signature() -> tuple[frozenset[str], ...]:
    """Capture all datablock classes semantic baking may allocate temporarily."""

    return tuple(
        frozenset(item.name_full for item in collection)
        for collection in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.collections,
            bpy.data.materials,
            bpy.data.images,
            bpy.data.node_groups,
        )
    )


def temporary_datablock_names() -> tuple[str, ...]:
    names: list[str] = []
    for collection in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.collections,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.node_groups,
    ):
        names.extend(
            item.name_full
            for item in collection
            if item.name_full.startswith(TEMPORARY_PREFIX)
        )
    return tuple(sorted(names))


def prepare_bake_plan(
    obj,
    output_directory: Path,
    output_stem: str,
    *,
    width: int = 32,
    height: int = 32,
    diffuse_mode: BakeMode = BakeMode.EMIT,
    procedural_mode: BakeMode = BakeMode.EMIT,
    selected_to_active: bool = False,
    texture_format: TextureFormat = TextureFormat.PNG,
    sequence_start_frame: int = 0,
    sequence_frame_count: int = 0,
):
    """Run production source capture, UV unwrap, material analysis, and planning."""

    source_snapshot = read_source_mesh_snapshot(obj)
    target_snapshot = unwrap_snapshot_uv(
        source_snapshot,
        UvUnwrapSettings(layer_name="SpineBakeUV"),
    ).snapshot
    analysis = analyse_object_materials(
        obj,
        render_target="CYCLES",
        source_object_id=source_snapshot.source_object_id,
    )
    plan = build_bake_plan(
        analysis,
        BakeSettings(
            width=width,
            height=height,
            output_directory=output_directory,
            output_stem=output_stem,
            uv_layer_name="SpineBakeUV",
            texture_format=texture_format,
            margin_pixels=1,
            selected_to_active=selected_to_active,
            cage_extrusion=0.05,
            diffuse_mode=diffuse_mode,
            procedural_mode=procedural_mode,
            sequence_start_frame=sequence_start_frame,
            sequence_frame_count=sequence_frame_count,
        ),
    )
    return target_snapshot, analysis, plan


def load_image(path: Path) -> LoadedImage:
    """Decode a committed texture through Blender and release the loaded datablock."""

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.alpha_mode = "STRAIGHT"
        return LoadedImage(
            width=int(image.size[0]),
            height=int(image.size[1]),
            channels=int(image.channels),
            pixels=tuple(float(value) for value in image.pixels[:]),
        )
    finally:
        bpy.data.images.remove(image)


def mean_rgba(
    pixels: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    resolved = tuple(pixels)
    if not resolved:
        raise ValueError("at least one RGBA pixel is required")
    count = float(len(resolved))
    return tuple(
        sum(pixel[channel] for pixel in resolved) / count
        for channel in range(4)
    )


def dominant_pixel_count(image: LoadedImage, channel: int) -> int:
    """Count strongly red/green/blue opaque pixels without exact color matching."""

    if channel not in {0, 1, 2}:
        raise ValueError("channel must be 0, 1, or 2")
    count = 0
    for rgba in image.rgba_pixels:
        red, green, blue, alpha = rgba
        if alpha <= 0.05:
            continue
        values = (red, green, blue)
        other_values = tuple(
            values[index] for index in range(3) if index != channel
        )
        if values[channel] > 0.2 and values[channel] > max(other_values) * 1.35:
            count += 1
    return count


__all__ = [
    "PNG_SIGNATURE",
    "ContextSnapshot",
    "LoadedImage",
    "SceneBakeSnapshot",
    "activate_only",
    "capture_context",
    "capture_scene_bake_state",
    "create_quad",
    "create_sentinel",
    "create_two_quad_object",
    "datablock_signature",
    "dominant_pixel_count",
    "load_image",
    "material_fingerprint",
    "mean_rgba",
    "new_emission_material",
    "new_image_alpha_material",
    "new_principled_material",
    "new_transparent_material",
    "new_transparent_mix_material",
    "prepare_bake_plan",
    "temporary_datablock_names",
]

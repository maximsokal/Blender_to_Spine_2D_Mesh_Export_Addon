"""Copy Blender 5.2+ materials and prepare isolated active bake image nodes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from math import isfinite
from typing import Any, Iterable, Iterator, Tuple
from uuid import uuid4

from ..domain.baking import BakePassPlan
from ..domain.baking.generated_materials import GeneratedMaterialPlan
from ..domain.geometry import MeshSnapshot
from .mesh_writer import build_mesh_topology_correspondence
from .render_engine_contract import render_engine_contract


logger = logging.getLogger(__name__)


class BakeMaterialError(RuntimeError):
    """Raised when temporary bake materials cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class PreparedBakeMaterials:
    materials: Tuple[Any, ...]
    image_nodes: Tuple[Any, ...]
    placeholder_slot_indices: Tuple[int, ...]
    used_material_indices: Tuple[int, ...]
    render_target: str = "CYCLES"

    def __post_init__(self) -> None:
        for field_name in (
            "materials",
            "image_nodes",
            "placeholder_slot_indices",
            "used_material_indices",
        ):
            if not isinstance(getattr(self, field_name), tuple):
                raise TypeError(f"{field_name} must be tuple")
        normalized_target = render_engine_contract(self.render_target).shader_target
        object.__setattr__(self, "render_target", normalized_target)

    def assign_image(self, image: Any) -> None:
        if image is None:
            raise BakeMaterialError("image cannot be None")
        for node in self.image_nodes:
            node.image = image

    @contextmanager
    def prepare_pass(self, pass_plan: BakePassPlan) -> Iterator[None]:
        """Apply one pass preparation only to owned temporary material copies."""

        if not isinstance(pass_plan, BakePassPlan):
            raise TypeError("pass_plan must be BakePassPlan")
        from .scene_material_preparation import temporary_prepare_scene_material_pass

        with temporary_prepare_scene_material_pass(
            self.materials,
            pass_plan,
            used_material_indices=self.used_material_indices,
            render_target=self.render_target,
        ):
            yield


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise BakeMaterialError("Blender bpy module is unavailable") from exc
    return bpy


def _clear_material_slots(mesh: Any) -> None:
    materials = getattr(mesh, "materials", None)
    if materials is None:
        raise BakeMaterialError("Target mesh has no material collection")
    try:
        materials.clear()
    except Exception as exc:
        raise BakeMaterialError("Unable to clear target material slots") from exc


def _material_node_tree(material: Any, *, label: str) -> Any:
    """Return the mandatory Blender 5.2+ material node tree."""

    if material is None:
        raise BakeMaterialError(f"{label} material is missing")
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise BakeMaterialError(
            f"{label} material has no node tree; Blender 5.2+ materials must expose one"
        )
    nodes = getattr(node_tree, "nodes", None)
    links = getattr(node_tree, "links", None)
    if nodes is None or links is None:
        raise BakeMaterialError(f"{label} material node tree is incomplete")
    return node_tree


def _create_placeholder_material(bpy_module: Any, slot_index: int, token: str) -> Any:
    material = bpy_module.data.materials.new(
        name=f"__Spine2D_EmptySlot_{slot_index}_{token}"
    )
    try:
        _material_node_tree(material, label=f"Placeholder slot {slot_index}")
        try:
            material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
        except Exception:
            logger.debug("Placeholder diffuse_color is not writable", exc_info=True)
        return material
    except Exception:
        try:
            if getattr(material, "users", 0) == 0:
                bpy_module.data.materials.remove(material)
        except Exception:
            logger.exception("Failed to remove invalid placeholder material")
        raise


def _copy_material(
    bpy_module: Any,
    source_material: Any,
    slot_index: int,
    token: str,
) -> Any:
    if source_material is None:
        return _create_placeholder_material(bpy_module, slot_index, token)
    _material_node_tree(source_material, label=f"Source slot {slot_index}")
    try:
        copied = source_material.copy()
    except Exception as exc:
        raise BakeMaterialError(
            f"Unable to copy source material in slot {slot_index}"
        ) from exc
    copied.name = f"__Spine2D_Bake_{slot_index}_{token}"
    try:
        _material_node_tree(copied, label=f"Copied slot {slot_index}")
        return copied
    except Exception:
        try:
            if getattr(copied, "users", 0) == 0:
                bpy_module.data.materials.remove(copied)
        except Exception:
            logger.exception("Failed to remove invalid copied material")
        raise


def _create_active_bake_node(material: Any, slot_index: int, token: str) -> Any:
    node_tree = _material_node_tree(material, label=f"Bake slot {slot_index}")
    nodes = node_tree.nodes
    try:
        for node in nodes:
            node.select = False
        bake_node = nodes.new(type="ShaderNodeTexImage")
        bake_node.name = f"__Spine2D_BakeTarget_{slot_index}_{token}"
        bake_node.label = "Spine2D temporary bake target"
        bake_node.select = True
        nodes.active = bake_node
        return bake_node
    except Exception as exc:
        raise BakeMaterialError(
            f"Unable to create active bake target node in material slot {slot_index}"
        ) from exc


def _create_generated_color_material(
    bpy_module: Any,
    plan: GeneratedMaterialPlan,
    token: str,
) -> Any:
    """Create one owned Emission material reading the generated CORNER color."""

    material = bpy_module.data.materials.new(
        name=f"{plan.material_name}_{token}"
    )
    try:
        node_tree = _material_node_tree(material, label="Generated color")
        node_tree.nodes.clear()
        output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        output.name = "Material Output"
        emission = node_tree.nodes.new(type="ShaderNodeEmission")
        emission.name = "Spine2D Generated Emission"
        try:
            color_node = node_tree.nodes.new(type="ShaderNodeVertexColor")
            color_node.layer_name = plan.color_attribute_name
        except Exception:
            color_node = node_tree.nodes.new(type="ShaderNodeAttribute")
            color_node.attribute_name = plan.color_attribute_name
        color_output = color_node.outputs.get("Color")
        emission_input = emission.inputs.get("Color")
        emission_output = emission.outputs.get("Emission")
        surface_input = output.inputs.get("Surface")
        if (
            color_output is None
            or emission_input is None
            or emission_output is None
            or surface_input is None
        ):
            raise RuntimeError("generated material nodes expose unexpected sockets")
        node_tree.links.new(color_output, emission_input)
        node_tree.links.new(emission_output, surface_input)
        try:
            material.diffuse_color = plan.face_colors[0]
        except Exception:
            logger.debug("Generated diffuse_color is not writable", exc_info=True)
        return material
    except Exception as exc:
        try:
            if getattr(material, "users", 0) == 0:
                bpy_module.data.materials.remove(material)
        except Exception:
            logger.exception("Failed to remove invalid generated material")
        raise BakeMaterialError(
            f"Unable to create generated color material: {exc}"
        ) from exc


def _assign_generated_display_color(
    attribute_value: Any,
    color: tuple[float, float, float, float],
) -> None:
    """Store one display-referred color through Blender 5.2 color management.

    ``FloatColorAttributeValue.color_srgb`` converts the stable generated palette
    into the active scene-linear working space. Writing ``color`` directly would
    reinterpret the same numeric tuple in ACEScg, Linear Rec.2020, or another OCIO
    working space and would change the visible diagnostic palette.
    """

    if attribute_value is None:
        raise BakeMaterialError("Generated color attribute value is missing")
    if not isinstance(color, tuple) or len(color) != 4:
        raise BakeMaterialError("Generated display color must contain four values")
    resolved = tuple(float(component) for component in color)
    if not all(isfinite(component) for component in resolved):
        raise BakeMaterialError("Generated display color contains a non-finite value")
    if any(component < 0.0 or component > 1.0 for component in resolved):
        raise BakeMaterialError("Generated display color must be within [0, 1]")
    if not hasattr(attribute_value, "color_srgb"):
        raise BakeMaterialError(
            "Blender 5.2 FloatColorAttributeValue.color_srgb is unavailable"
        )
    try:
        attribute_value.color_srgb = resolved
    except Exception as exc:
        raise BakeMaterialError(
            f"Unable to write generated display color {resolved!r}"
        ) from exc


def _write_generated_corner_colors(
    target_mesh: Any,
    plan: GeneratedMaterialPlan,
) -> Any:
    """Write one color-managed face color to every mapped Blender loop."""

    attributes = getattr(target_mesh, "color_attributes", None)
    if attributes is None:
        raise BakeMaterialError("Target mesh has no color_attributes collection")
    existing = attributes.get(plan.color_attribute_name)
    if existing is not None:
        try:
            attributes.remove(existing)
        except Exception as exc:
            raise BakeMaterialError(
                f"Unable to replace color attribute '{plan.color_attribute_name}'"
            ) from exc
    try:
        attribute = attributes.new(
            name=plan.color_attribute_name,
            type="FLOAT_COLOR",
            domain="CORNER",
        )
        correspondence = build_mesh_topology_correspondence(
            plan.target_snapshot,
            target_mesh,
            stage="generated-color-write",
        )
        for face, color in zip(
            plan.target_snapshot.faces,
            plan.face_colors,
            strict=True,
        ):
            for loop_id in face.loop_ids:
                mesh_loop_index = correspondence.mesh_loop_index_for(loop_id)
                _assign_generated_display_color(
                    attribute.data[mesh_loop_index],
                    color,
                )
        return attribute
    except Exception as exc:
        try:
            created = attributes.get(plan.color_attribute_name)
            if created is not None:
                attributes.remove(created)
        except Exception:
            logger.exception("Failed to remove partial generated color attribute")
        if isinstance(exc, BakeMaterialError):
            raise
        raise BakeMaterialError(
            f"Unable to write generated CORNER colors: {exc}"
        ) from exc


def _remove_color_attribute(mesh: Any, attribute: Any | None) -> None:
    if attribute is None:
        return
    attributes = getattr(mesh, "color_attributes", None)
    if attributes is None:
        return
    try:
        attributes.remove(attribute)
    except Exception:
        logger.exception("Failed to remove generated color attribute")


def _apply_face_material_indices(
    target_snapshot: MeshSnapshot,
    target_mesh: Any,
    face_material_indices: Iterable[int],
    *,
    material_slot_count: int,
) -> None:
    """Restore face-slot bindings through exact snapshot-to-polygon correspondence."""

    if not isinstance(target_snapshot, MeshSnapshot):
        raise TypeError("target_snapshot must be MeshSnapshot")
    if not isinstance(material_slot_count, int) or isinstance(material_slot_count, bool):
        raise TypeError("material_slot_count must be int")
    if material_slot_count < 0:
        raise ValueError("material_slot_count must be a non-negative integer")

    try:
        resolved = tuple(face_material_indices)
    except TypeError as exc:
        raise TypeError("face_material_indices must be iterable") from exc
    polygons = tuple(getattr(target_mesh, "polygons", ()))
    expected_face_count = len(target_snapshot.faces)
    if len(resolved) != expected_face_count:
        raise BakeMaterialError(
            f"Received {len(resolved)} face material indices for "
            f"{expected_face_count} snapshot faces"
        )
    if len(polygons) != expected_face_count:
        raise BakeMaterialError(
            f"Target mesh contains {len(polygons)} polygons for "
            f"{expected_face_count} snapshot faces"
        )

    for face_index, material_index in enumerate(resolved):
        if not isinstance(material_index, int) or isinstance(material_index, bool):
            raise BakeMaterialError(
                f"face_material_indices[{face_index}] must be an integer"
            )
        if material_index < 0 or material_index >= material_slot_count:
            raise BakeMaterialError(
                f"Face {face_index} references material slot {material_index}, but "
                f"only {material_slot_count} slots exist"
            )

    try:
        correspondence = build_mesh_topology_correspondence(
            target_snapshot,
            target_mesh,
            stage="bake-material-index-assignment",
        )
        polygon_index_by_face_id = dict(correspondence.face_to_polygon_index)
        polygon_index_by_face_position = tuple(
            polygon_index_by_face_id[face.id] for face in target_snapshot.faces
        )
    except Exception as exc:
        raise BakeMaterialError(
            "Unable to map snapshot faces to temporary bake polygons"
        ) from exc

    try:
        for face_position, material_index in enumerate(resolved):
            polygon_index = polygon_index_by_face_position[face_position]
            polygons[polygon_index].material_index = material_index
    except Exception as exc:
        raise BakeMaterialError(
            "Unable to restore target polygon material indices"
        ) from exc

    actual = tuple(
        int(polygons[polygon_index].material_index)
        for polygon_index in polygon_index_by_face_position
    )
    if actual != resolved:
        raise BakeMaterialError(
            "Blender changed target polygon material indices after assignment: "
            f"expected_by_snapshot_face={resolved}, actual_by_snapshot_face={actual}"
        )


def _remove_materials(bpy_module: Any, materials: Iterable[Any]) -> None:
    for material in reversed(tuple(materials)):
        try:
            if getattr(material, "users", 0) == 0:
                bpy_module.data.materials.remove(material)
        except Exception:
            logger.exception("Failed to remove temporary bake material")


@contextmanager
def temporary_bake_materials(
    source_obj: Any,
    target_obj: Any,
    *,
    target_snapshot: MeshSnapshot,
    used_material_indices: Iterable[int],
    face_material_indices: Iterable[int],
    render_target: str = "CYCLES",
    generated_material: GeneratedMaterialPlan | None = None,
) -> Iterator[PreparedBakeMaterials]:
    """Prepare source copies or one generated material on an isolated target."""

    if source_obj is None or target_obj is None:
        raise BakeMaterialError("source_obj and target_obj are required")
    if not isinstance(target_snapshot, MeshSnapshot):
        raise TypeError("target_snapshot must be MeshSnapshot")
    if generated_material is not None and not isinstance(
        generated_material,
        GeneratedMaterialPlan,
    ):
        raise TypeError("generated_material must be GeneratedMaterialPlan or None")

    normalized_target = render_engine_contract(render_target).shader_target
    source_slots = tuple(getattr(source_obj, "material_slots", ()))
    used = tuple(sorted(set(used_material_indices)))
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in used
    ):
        raise ValueError("used_material_indices must contain non-negative integers")
    if generated_material is None and used and max(used) >= len(source_slots):
        raise BakeMaterialError(
            f"Target faces reference material slot {max(used)}, but source object has "
            f"only {len(source_slots)} slots"
        )

    bpy_module = _load_bpy()
    target_mesh = getattr(target_obj, "data", None)
    if target_mesh is None:
        raise BakeMaterialError("target_obj.data is missing")
    token = uuid4().hex
    copied_materials: list[Any] = []
    image_nodes: list[Any] = []
    placeholder_indices: list[int] = []
    generated_attribute = None

    try:
        _clear_material_slots(target_mesh)
        if generated_material is not None:
            if used != (0,):
                raise BakeMaterialError(
                    "Generated material execution requires only synthetic slot zero"
                )
            generated_copy = _create_generated_color_material(
                bpy_module,
                generated_material,
                token,
            )
            copied_materials.append(generated_copy)
            target_mesh.materials.append(generated_copy)
            _apply_face_material_indices(
                target_snapshot,
                target_mesh,
                face_material_indices,
                material_slot_count=1,
            )
            generated_attribute = _write_generated_corner_colors(
                target_mesh,
                generated_material,
            )
            image_nodes.append(
                _create_active_bake_node(generated_copy, 0, token)
            )
        else:
            for slot_index, slot in enumerate(source_slots):
                source_material = getattr(slot, "material", None)
                copied_material = _copy_material(
                    bpy_module,
                    source_material,
                    slot_index,
                    token,
                )
                copied_materials.append(copied_material)
                if source_material is None:
                    placeholder_indices.append(slot_index)
                target_mesh.materials.append(copied_material)
                if slot_index in used:
                    if source_material is None:
                        raise BakeMaterialError(
                            f"Target geometry uses empty source material slot {slot_index}"
                        )
                    image_nodes.append(
                        _create_active_bake_node(
                            copied_material,
                            slot_index,
                            token,
                        )
                    )

            _apply_face_material_indices(
                target_snapshot,
                target_mesh,
                face_material_indices,
                material_slot_count=len(copied_materials),
            )

        if not image_nodes:
            raise BakeMaterialError(
                "No used material slots received active bake nodes"
            )
        prepared = PreparedBakeMaterials(
            materials=tuple(copied_materials),
            image_nodes=tuple(image_nodes),
            placeholder_slot_indices=tuple(placeholder_indices),
            used_material_indices=used,
            render_target=normalized_target,
        )
        yield prepared
    finally:
        _remove_color_attribute(target_mesh, generated_attribute)
        try:
            _clear_material_slots(target_mesh)
        except Exception:
            logger.exception("Failed to clear temporary target material slots")
        _remove_materials(bpy_module, copied_materials)

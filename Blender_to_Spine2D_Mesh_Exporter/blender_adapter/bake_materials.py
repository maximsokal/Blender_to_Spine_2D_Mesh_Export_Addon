"""Copy source materials and prepare isolated active image nodes for baking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterable, Iterator, Tuple
from uuid import uuid4

from ..domain.baking import BakePassPlan

logger = logging.getLogger(__name__)


class BakeMaterialError(RuntimeError):
    """Raised when temporary bake materials cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class PreparedBakeMaterials:
    materials: Tuple[Any, ...]
    image_nodes: Tuple[Any, ...]
    placeholder_slot_indices: Tuple[int, ...]
    used_material_indices: Tuple[int, ...]

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


def _create_placeholder_material(bpy_module: Any, slot_index: int, token: str) -> Any:
    material = bpy_module.data.materials.new(
        name=f"__Spine2D_EmptySlot_{slot_index}_{token}"
    )
    material.use_nodes = True
    try:
        material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    except Exception:
        logger.debug("Placeholder diffuse_color is not writable", exc_info=True)
    return material


def _copy_material(
    bpy_module: Any,
    source_material: Any,
    slot_index: int,
    token: str,
) -> Any:
    if source_material is None:
        return _create_placeholder_material(bpy_module, slot_index, token)
    try:
        copied = source_material.copy()
    except Exception as exc:
        raise BakeMaterialError(
            f"Unable to copy source material in slot {slot_index}"
        ) from exc
    copied.name = f"__Spine2D_Bake_{slot_index}_{token}"
    return copied


def _ensure_node_tree(material: Any, slot_index: int) -> Any:
    try:
        material.use_nodes = True
    except Exception as exc:
        raise BakeMaterialError(
            f"Unable to enable nodes for copied material in slot {slot_index}"
        ) from exc
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        raise BakeMaterialError(
            f"Copied material in slot {slot_index} has no node tree"
        )
    return node_tree


def _create_active_bake_node(material: Any, slot_index: int, token: str) -> Any:
    node_tree = _ensure_node_tree(material, slot_index)
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


def _apply_face_material_indices(
    target_mesh: Any,
    face_material_indices: Iterable[int],
    *,
    material_slot_count: int,
) -> None:
    """Restore polygon-slot bindings only after Blender material slots exist."""

    if not isinstance(material_slot_count, int) or material_slot_count < 0:
        raise ValueError("material_slot_count must be a non-negative integer")
    resolved = tuple(face_material_indices)
    polygons = tuple(getattr(target_mesh, "polygons", ()))
    if len(resolved) != len(polygons):
        raise BakeMaterialError(
            f"Received {len(resolved)} face material indices for "
            f"{len(polygons)} target polygons"
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
        for polygon, material_index in zip(polygons, resolved):
            polygon.material_index = material_index
    except Exception as exc:
        raise BakeMaterialError(
            "Unable to restore target polygon material indices"
        ) from exc

    actual = tuple(int(polygon.material_index) for polygon in polygons)
    if actual != resolved:
        raise BakeMaterialError(
            "Blender changed target polygon material indices after assignment: "
            f"expected={resolved}, actual={actual}"
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
    used_material_indices: Iterable[int],
    face_material_indices: Iterable[int],
) -> Iterator[PreparedBakeMaterials]:
    """Copy source slots, restore polygon bindings, and create active bake nodes."""

    if source_obj is None or target_obj is None:
        raise BakeMaterialError("source_obj and target_obj are required")
    source_slots = tuple(getattr(source_obj, "material_slots", ()))
    used = tuple(sorted(set(used_material_indices)))
    if any(not isinstance(index, int) or index < 0 for index in used):
        raise ValueError("used_material_indices must contain non-negative integers")
    if used and max(used) >= len(source_slots):
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

    try:
        _clear_material_slots(target_mesh)
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
                    _create_active_bake_node(copied_material, slot_index, token)
                )

        _apply_face_material_indices(
            target_mesh,
            face_material_indices,
            material_slot_count=len(copied_materials),
        )
        if not image_nodes:
            raise BakeMaterialError("No used material slots received active bake nodes")

        prepared = PreparedBakeMaterials(
            materials=tuple(copied_materials),
            image_nodes=tuple(image_nodes),
            placeholder_slot_indices=tuple(placeholder_indices),
            used_material_indices=used,
        )
        yield prepared
    finally:
        try:
            _clear_material_slots(target_mesh)
        except Exception:
            logger.exception("Failed to clear temporary target material slots")
        _remove_materials(bpy_module, copied_materials)

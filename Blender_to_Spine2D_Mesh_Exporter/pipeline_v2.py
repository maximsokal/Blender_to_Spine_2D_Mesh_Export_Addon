# pylint: disable=import-error,too-many-locals,too-many-branches,too-many-statements
"""Resource-safe export pipeline used as the v0.24 migration bridge.

The legacy ``main.py`` still contains geometry and Spine conversion routines.
This module extracts orchestration and ownership from that file while keeping
its public API stable.  ``install()`` replaces only the unsafe entry points;
callers can continue importing them from ``main`` during the staged rewrite.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import os
from math import sqrt
from typing import Iterable, Sequence

import bpy
import bmesh

from .blender_context import (
    BlenderContextSnapshot,
    activate_object,
    managed_bmesh,
    remove_object_if_alive,
    scene_bool,
)

logger = logging.getLogger(__name__)
_LEGACY_MAIN = None


class ExportPipelineError(RuntimeError):
    """Expected pipeline failure with a message suitable for the Blender log."""


def _legacy():
    if _LEGACY_MAIN is None:
        raise RuntimeError("pipeline_v2.install(main_module) was not called")
    return _LEGACY_MAIN


def _mesh_object(obj: object, argument_name: str = "obj") -> object:
    if obj is None:
        raise TypeError(f"{argument_name} must not be None")
    if getattr(obj, "type", None) != "MESH":
        raise TypeError(f"{argument_name} must be a Blender MESH object")
    if getattr(obj, "data", None) is None:
        raise TypeError(f"{argument_name}.data must be a Mesh datablock")
    return obj


def _normalise_edges(segmentation_data: Iterable[Sequence[int]] | None) -> set[frozenset[int]]:
    normalised: set[frozenset[int]] = set()
    if not segmentation_data:
        return normalised

    for index, pair in enumerate(segmentation_data):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            logger.warning("Ignoring malformed segmentation edge #%d: %r", index, pair)
            continue
        try:
            first, second = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            logger.warning("Ignoring non-integer segmentation edge #%d: %r", index, pair)
            continue
        if first == second or first < 0 or second < 0:
            logger.warning("Ignoring degenerate segmentation edge #%d: %r", index, pair)
            continue
        normalised.add(frozenset((first, second)))
    return normalised


def apply_segmentation_seams(obj, segmentation_data) -> None:
    """Apply segment boundaries and always release the temporary BMesh."""
    obj = _mesh_object(obj)
    target_edges = _normalise_edges(segmentation_data)
    if not target_edges:
        logger.debug("No segmentation seams to apply to %s", obj.name)
        return

    marked = 0
    with managed_bmesh(obj.data, write_back=True) as bm:
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        for edge in bm.edges:
            key = frozenset((edge.verts[0].index, edge.verts[1].index))
            if key in target_edges:
                edge.seam = True
                marked += 1

    missing = len(target_edges) - marked
    if missing > 0:
        logger.warning(
            "Applied %d of %d segmentation seams to %s; %d edges were not found",
            marked,
            len(target_edges),
            obj.name,
            missing,
        )
    else:
        logger.debug("Applied %d segmentation seams to %s", marked, obj.name)


def copy_orig_face_id_layer(source_obj, target_obj) -> bool:
    """Copy ``orig_face_id`` by face index with deterministic BMesh cleanup."""
    source_obj = _mesh_object(source_obj, "source_obj")
    target_obj = _mesh_object(target_obj, "target_obj")

    with managed_bmesh(source_obj.data) as source_bm:
        source_bm.faces.ensure_lookup_table()
        source_layer = source_bm.faces.layers.int.get("orig_face_id")
        if source_layer is None:
            logger.warning("%s has no orig_face_id face layer", source_obj.name)
            return False

        source_values = [face[source_layer] for face in source_bm.faces]

    with managed_bmesh(target_obj.data, write_back=True) as target_bm:
        target_bm.faces.ensure_lookup_table()
        target_layer = target_bm.faces.layers.int.get("orig_face_id")
        if target_layer is None:
            target_layer = target_bm.faces.layers.int.new("orig_face_id")

        copy_count = min(len(source_values), len(target_bm.faces))
        for face_index in range(copy_count):
            target_bm.faces[face_index][target_layer] = source_values[face_index]

        if len(source_values) != len(target_bm.faces):
            logger.warning(
                "orig_face_id face-count mismatch: %s=%d, %s=%d; copied %d",
                source_obj.name,
                len(source_values),
                target_obj.name,
                len(target_bm.faces),
                copy_count,
            )

    return True


def get_texture_dimensions(obj, default_texture_width, default_texture_height):
    """Read the first image texture size without suppressing unexpected errors."""
    width = int(default_texture_width)
    height = int(default_texture_height)

    try:
        material = getattr(obj, "active_material", None)
        node_tree = getattr(material, "node_tree", None) if material else None
        nodes = getattr(node_tree, "nodes", ()) if node_tree else ()
        for node in nodes:
            image = getattr(node, "image", None)
            if getattr(node, "type", None) == "TEX_IMAGE" and image is not None:
                image_width, image_height = image.size[:2]
                width, height = int(image_width), int(image_height)
                break
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        logger.warning("Failed to read texture dimensions; defaults are used", exc_info=True)

    return width, height


def triangulate_mesh(obj):
    """Return a triangulated BMesh; ownership is transferred to the caller."""
    obj = _mesh_object(obj)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        faces = list(bm.faces)
        if faces:
            bmesh.ops.triangulate(
                bm,
                faces=faces,
                quad_method="BEAUTY",
                ngon_method="BEAUTY",
            )
        return bm
    except Exception:
        bm.free()
        logger.exception("Failed to triangulate BMesh for %s", obj.name)
        raise


def collect_vertices(bm, obj_name):
    """Collect UV/position data without taking ownership of *bm*.

    The caller that received the BMesh from :func:`triangulate_mesh` is solely
    responsible for ``bm.free()``.  This removes the legacy double-free path.
    """
    if bm is None:
        raise TypeError("bm must not be None")

    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        raise ExportPipelineError(f"UV layer not found after triangulation: {obj_name}")

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    triangles: list[int] = []
    vertex_map: dict[tuple[float, float, int], int] = {}
    vertex_list: list[tuple[float, float]] = []
    vertex_positions_list: list[object] = []

    for face in bm.faces:
        if len(face.loops) != 3:
            raise ExportPipelineError(
                f"Triangulated BMesh contains a {len(face.loops)}-sided face: {obj_name}"
            )
        for loop in face.loops:
            uv = loop[uv_layer].uv
            key = (round(float(uv.x), 6), round(float(uv.y), 6), int(loop.vert.index))
            vertex_index = vertex_map.get(key)
            if vertex_index is None:
                vertex_index = len(vertex_list)
                vertex_map[key] = vertex_index
                vertex_list.append((float(uv.x), float(uv.y)))
                vertex_positions_list.append(loop.vert.co.copy())
            triangles.append(vertex_index)

    if len(triangles) % 3:
        raise ExportPipelineError(
            f"Triangle index list for {obj_name} has invalid length {len(triangles)}"
        )

    logger.debug(
        "Collected %d unique vertices and %d triangles from %s",
        len(vertex_list),
        len(triangles) // 3,
        obj_name,
    )
    return vertex_list, vertex_positions_list, triangles, obj_name


def calculate_stretch_values(obj):
    """Calculate stretch values while freeing the BMesh on every exit path."""
    obj = _mesh_object(obj)
    legacy = _legacy()

    with managed_bmesh(obj.data) as bm:
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            logger.debug("No active UV layer found for stretch calculation on %s", obj.name)
            return None

        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        vertex_groups = defaultdict(list)
        precision = 4
        for vert in bm.verts:
            vertex_groups[round(float(vert.co.z), precision)].append(vert)

        processed_faces: set[int] = set()
        group_areas = defaultdict(lambda: {"area_3d": 0.0, "area_uv": 0.0})
        minimum_area = 1e-4

        for vertices in vertex_groups.values():
            for vert in vertices:
                for face in vert.link_faces:
                    if face.index in processed_faces:
                        continue
                    processed_faces.add(face.index)

                    z_values = [round(float(v.co.z), precision) for v in face.verts]
                    z_key = z_values[0] if all(z == z_values[0] for z in z_values) else round(
                        sum(z_values) / len(z_values), precision
                    )
                    area_3d = float(face.calc_area())
                    area_uv = 0.5 * abs(
                        sum(
                            face.loops[index][uv_layer].uv.x
                            * face.loops[index - 1][uv_layer].uv.y
                            - face.loops[index - 1][uv_layer].uv.x
                            * face.loops[index][uv_layer].uv.y
                            for index in range(len(face.loops))
                        )
                    )
                    if area_3d <= minimum_area or area_uv <= minimum_area:
                        logger.warning(
                            "Face %d on %s has too small areas: 3D=%s UV=%s",
                            face.index,
                            obj.name,
                            area_3d,
                            area_uv,
                        )
                        continue
                    group_areas[z_key]["area_3d"] += area_3d
                    group_areas[z_key]["area_uv"] += area_uv

    raw_stretch = {}
    for z_key, areas in group_areas.items():
        if areas["area_3d"] > minimum_area and areas["area_uv"] > minimum_area:
            raw_stretch[z_key] = sqrt(areas["area_3d"] / areas["area_uv"])
        else:
            raw_stretch[z_key] = 1.0

    smoothed = legacy.smooth_stretch_values(raw_stretch, window_size=5)
    limited = legacy.limit_stretch_changes(smoothed, max_change=0.1)
    return {z_key: max(0.5, min(2.0, value)) for z_key, value in limited.items()}


def apply_transformations(obj) -> None:
    """Apply rotation/scale to exactly *obj* and restore prior context."""
    obj = _mesh_object(obj)
    snapshot = BlenderContextSnapshot.capture()
    try:
        activate_object(obj)
        result = bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        if isinstance(result, set) and "CANCELLED" in result:
            raise ExportPipelineError(f"Transform apply was cancelled for {obj.name}")
    finally:
        snapshot.restore()


@dataclass(slots=True)
class ExportSession:
    """Own temporary datablocks and the user's Blender context for one export."""

    source_obj: object
    context: BlenderContextSnapshot = field(default_factory=BlenderContextSnapshot.capture)
    temporary_objects: list[object] = field(default_factory=list)
    segment_objects: list[object] = field(default_factory=list)
    original_copy: object | None = None
    textured_obj: object | None = None
    created_source_uv_name: str | None = None
    previous_face_id_map: object | None = None
    had_face_id_map: bool = False

    def track(self, obj: object | None) -> object | None:
        if obj is not None and all(existing is not obj for existing in self.temporary_objects):
            self.temporary_objects.append(obj)
        return obj

    def restore_source_metadata(self) -> None:
        data = getattr(self.source_obj, "data", None)
        if data is None:
            return

        if self.created_source_uv_name:
            try:
                layer = data.uv_layers.get(self.created_source_uv_name)
                if layer is not None:
                    data.uv_layers.remove(layer)
            except (AttributeError, ReferenceError, RuntimeError):
                logger.warning("Failed to remove temporary source UV layer", exc_info=True)

        try:
            if self.had_face_id_map:
                data["face_id_map"] = self.previous_face_id_map
            elif "face_id_map" in data:
                del data["face_id_map"]
        except (AttributeError, KeyError, ReferenceError, RuntimeError, TypeError):
            logger.warning("Failed to restore face_id_map metadata", exc_info=True)

    def cleanup(self, preserve_debug_artifacts: bool) -> None:
        legacy = _legacy()
        if preserve_debug_artifacts:
            logger.info(
                "Debug artifacts preserved explicitly by scene property "
                "spine2d_preserve_debug_artifacts"
            )
            return

        if self.segment_objects and self.original_copy is not None:
            try:
                legacy.delete_segment_artifacts(self.segment_objects, self.original_copy)
            except Exception:
                logger.exception("Failed to delete segment artifacts")

        removed_ids: set[int] = set()
        removed_count = 0
        for obj in reversed(self.temporary_objects):
            identity = id(obj)
            if identity in removed_ids:
                continue
            removed_ids.add(identity)
            if remove_object_if_alive(obj):
                removed_count += 1
        logger.info("Removed %d temporary Blender objects", removed_count)


def _resolve_output_directory(output_dir: str | None) -> str:
    legacy = _legacy()
    candidate = None
    if output_dir:
        try:
            candidate = bpy.path.abspath(output_dir)
        except (AttributeError, RuntimeError, TypeError):
            candidate = output_dir
        if isinstance(candidate, bytes):
            candidate = candidate.decode("utf-8")
        if candidate and os.path.isdir(candidate):
            return candidate

    candidate = legacy.get_default_output_dir()
    if isinstance(candidate, bytes):
        candidate = candidate.decode("utf-8")
    if not isinstance(candidate, str) or not candidate:
        raise ExportPipelineError("Unable to resolve the JSON output directory")
    os.makedirs(candidate, exist_ok=True)
    return candidate


def _ensure_source_uv(obj, session: ExportSession) -> str:
    uv_layers = obj.data.uv_layers
    if not uv_layers:
        before_names = {layer.name for layer in uv_layers}
        _legacy().smart_uv_project(obj, obj.name)
        after_names = {layer.name for layer in obj.data.uv_layers}
        created_names = sorted(after_names - before_names)
        if created_names:
            session.created_source_uv_name = created_names[0]

    active = obj.data.uv_layers.active
    if active is None and obj.data.uv_layers:
        active = obj.data.uv_layers[0]
    if active is None:
        raise ExportPipelineError(f"Object {obj.name} has no usable UV map")
    return active.name


def _copy_seams(source_obj, target_obj) -> None:
    seam_keys: set[tuple[int, int]] = set()
    with managed_bmesh(source_obj.data) as source_bm:
        source_bm.verts.ensure_lookup_table()
        source_bm.edges.ensure_lookup_table()
        for edge in source_bm.edges:
            if edge.seam:
                seam_keys.add(tuple(sorted((edge.verts[0].index, edge.verts[1].index))))

    if not seam_keys:
        return

    with managed_bmesh(target_obj.data, write_back=True) as target_bm:
        target_bm.verts.ensure_lookup_table()
        target_bm.edges.ensure_lookup_table()
        for edge in target_bm.edges:
            key = tuple(sorted((edge.verts[0].index, edge.verts[1].index)))
            edge.seam = key in seam_keys


def _new_segment_objects(before_ids: set[int], base_name: str) -> list[object]:
    result = []
    for obj in bpy.context.scene.objects:
        try:
            pointer = int(obj.as_pointer())
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            pointer = id(obj)
        if pointer in before_ids:
            continue
        if obj.name.startswith(f"{base_name}_Segment_"):
            result.append(obj)
    return sorted(result, key=lambda item: item.name)


def _object_ids() -> set[int]:
    result: set[int] = set()
    for obj in bpy.context.scene.objects:
        try:
            result.add(int(obj.as_pointer()))
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            result.add(id(obj))
    return result


def _prepare_textured_uv(textured_obj, segmentation_data) -> str:
    legacy = _legacy()
    if segmentation_data:
        uv_name = legacy.unwrap_respecting_seams(
            textured_obj,
            method="ANGLE_BASED",
            margin=0.001,
        )
        if not uv_name:
            uv_name = legacy.smart_uv_project(textured_obj, textured_obj.name)
    else:
        uv_name = legacy.smart_uv_project(textured_obj, textured_obj.name)

    if textured_obj.data.uv_layers:
        active = textured_obj.data.uv_layers.active or textured_obj.data.uv_layers[0]
        active.name = "UVMap_for_texturing"
        return active.name
    if isinstance(uv_name, str) and uv_name:
        return uv_name
    raise ExportPipelineError(f"Textured object {textured_obj.name} has no UV layer")


def _export_segments(
    segment_objects,
    original_copy,
    textured_obj,
    texture_width,
    texture_height,
    z_groups_info,
    original_z_groups,
    save_directory,
    original_world_location,
):
    legacy = _legacy()
    results = []
    original_uv_pairs = None
    textured_uv_pairs = None

    for segment in segment_objects:
        is_original = segment is original_copy or segment.name == original_copy.name
        if not is_original and original_uv_pairs is None:
            raise ExportPipelineError("Original UV/3D pairs were not generated before segments")
        if not is_original and textured_uv_pairs is None:
            raise ExportPipelineError("Textured UV/3D pairs were not generated before segments")

        result = legacy._export_segment(
            segment,
            original_copy.name,
            textured_obj=textured_obj,
            TEXTURE_WIDTH=texture_width,
            TEXTURE_HEIGHT=texture_height,
            original_uv3d_pairs=original_uv_pairs,
            textured_uv3d_pairs_global=textured_uv_pairs,
            z_groups_info=z_groups_info if is_original else {},
            original_z_groups=original_z_groups,
            output_dir=save_directory,
            local_z_groups=original_z_groups if is_original else None,
            world_location=original_world_location,
        )
        if not result:
            raise ExportPipelineError(f"Segment export failed: {segment.name}")
        results.append(result)

        if is_original:
            original_uv_pairs = result.get("_uv3d_pairs")
            textured_uv_pairs = result.get("textured_uv3d_pairs")
            if not original_uv_pairs or not textured_uv_pairs:
                raise ExportPipelineError("Original export did not produce UV/3D correspondence")

    return results


def save_uv_as_json(obj, TEXTURE_WIDTH: int, TEXTURE_HEIGHT: int, output_dir: str | None = None):
    """Run one export with deterministic cleanup and context restoration."""
    obj = _mesh_object(obj)
    texture_width = int(TEXTURE_WIDTH)
    texture_height = int(TEXTURE_HEIGHT)
    if texture_width <= 0 or texture_height <= 0:
        raise ValueError("Texture dimensions must be positive integers")

    legacy = _legacy()
    session = ExportSession(source_obj=obj)
    preserve_debug = scene_bool("spine2d_preserve_debug_artifacts", False)

    try:
        original_world_location = obj.matrix_world.translation.copy()
        save_directory = _resolve_output_directory(output_dir)

        # UV operators are context-sensitive.  Activate the explicit source
        # object instead of assuming that the caller prepared Blender context.
        activate_object(obj)
        source_uv_name = _ensure_source_uv(obj, session)

        try:
            session.had_face_id_map = "face_id_map" in obj.data
            if session.had_face_id_map:
                session.previous_face_id_map = deepcopy(obj.data["face_id_map"])
        except (KeyError, TypeError, AttributeError):
            session.had_face_id_map = False
        legacy.assign_face_ids(obj)

        original_copy = obj.copy()
        original_copy.data = obj.data.copy()
        original_copy.name = f"{obj.name}_copy_for_uv"
        bpy.context.collection.objects.link(original_copy)
        session.original_copy = session.track(original_copy)

        _copy_seams(obj, original_copy)
        activate_object(original_copy)

        post_unwrap_info = legacy.main_preprocessing(original_copy)
        if not post_unwrap_info:
            raise ExportPipelineError(f"Preprocessing failed for {original_copy.name}")
        z_groups_info = post_unwrap_info.get("z_groups_info", {})
        original_z_groups = sorted(z_groups_info.keys())

        scene = bpy.context.scene
        seam_mode = scene.spine2d_seam_maker_mode
        angle_limit = float(scene.spine2d_angle_limit)
        objects_before_cut = _object_ids()
        segmentation_data = legacy.plane_cut.execute_smart_cut(
            original_copy,
            angle_limit=angle_limit,
            seam_mode=seam_mode,
            capture_uv_data=True,
        )
        if segmentation_data is None:
            segmentation_data = []
        apply_segmentation_seams(original_copy, segmentation_data)

        base_name = original_copy.name.removesuffix("_copy_for_uv")
        generated_segments = _new_segment_objects(objects_before_cut, base_name)
        for segment in generated_segments:
            session.track(segment)
        session.segment_objects = [original_copy, *generated_segments]

        textured_obj, _seams_info = legacy.mark_seams_on_copy(obj, segmentation_data)
        if textured_obj is None:
            raise ExportPipelineError("Failed to create the texturing copy")
        session.textured_obj = session.track(textured_obj)

        if "uv_island_segments" in original_copy:
            legacy.transfer_uv_islands_between_objects(original_copy, textured_obj)

        copy_orig_face_id_layer(original_copy, textured_obj)
        target_uv_name = _prepare_textured_uv(textured_obj, segmentation_data)

        if not legacy.bake_textures_for_object(
            textured_obj,
            target_uv_name,
            obj,
            source_uv_name,
        ):
            raise ExportPipelineError("Texture baking failed")

        legacy.transfer_baked_uvs_to_segments(
            textured_obj,
            session.segment_objects,
            target_uv_name,
        )

        results = _export_segments(
            session.segment_objects,
            original_copy,
            textured_obj,
            texture_width,
            texture_height,
            z_groups_info,
            original_z_groups,
            save_directory,
            original_world_location,
        )
        merged = legacy.merge_spine_json_dicts(results[0], results[1:])
        final_path = os.path.join(save_directory, f"{obj.name}_merged.json")
        legacy.write_json(merged, final_path)
        logger.info("Export completed: %s", final_path)
        return final_path
    except Exception:
        logger.exception("Fatal error in resource-safe save_uv_as_json")
        return None
    finally:
        try:
            session.cleanup(preserve_debug)
        finally:
            session.restore_source_metadata()
            session.context.restore()


def install(main_module) -> None:
    """Install v2 implementations behind the legacy ``main`` API."""
    global _LEGACY_MAIN
    _LEGACY_MAIN = main_module

    replacements = {
        "apply_segmentation_seams": apply_segmentation_seams,
        "copy_orig_face_id_layer": copy_orig_face_id_layer,
        "get_texture_dimensions": get_texture_dimensions,
        "triangulate_mesh": triangulate_mesh,
        "collect_vertices": collect_vertices,
        "calculate_stretch_values": calculate_stretch_values,
        "apply_transformations": apply_transformations,
        "save_uv_as_json": save_uv_as_json,
    }
    for name, implementation in replacements.items():
        setattr(main_module, name, implementation)
    logger.info("Installed resource-safe export pipeline v2")

# uv_operations.py
"""
Blender to Spine2D Mesh Exporter
Copyright (c) 2025 Maxim Sokolenko

This file is part of Blender to Spine2D Mesh Exporter.

Blender to Spine2D Mesh Exporter is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

Blender to Spine2D Mesh Exporter is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with Blender to Spine2D Mesh Exporter. If not, see <https://www.gnu.org/licenses/>.
This module contains a suite of specialized functions for creating, manipulating, and analyzing UV maps on Blender objects. It is a critical component of the export pipeline, responsible for preparing the UV layouts that are essential for texture baking and the final Spine mesh.

The key functionalities are:
1.  UV Unwrapping and Projection: It provides several methods for UV unwrapping, including a robust `smart_uv_project` function that can automatically estimate the best angle limit and fall back to different methods if the initial unwrap quality is poor. It also includes functions for simple top-down projection and unwrapping that respects pre-defined seams.
2.  Texel Density Equalization: The `rescale_uv_islands` function is a core feature that intelligently scales individual UV islands to equalize their texel density. It calculates the ratio of 3D mesh area to UV area for each island and rescales them to a median reference, ensuring textures are not stretched or compressed unevenly across the model.
3.  UV Data Transfer: It includes crucial functions for transferring UV data between different objects. `transfer_baked_uvs_to_segments` is used to copy the final, baked UV layout from the texturing object to all the individual mesh segments, using original face and vertex IDs to ensure a perfect match.
4.  UV Island Management: It can group faces into UV islands, analyze their properties (like area and vertex count), and even create a new UV map from stored island data, allowing for complex UV workflows.
5.  Utility Functions: It contains various helper functions for tasks like resetting UVs, centering a UV layout, flipping UVs or images vertically, and calculating the real-world dimensions of a mesh based on its UV layout.

ATTENTION: - The functions in this module directly manipulate an object's mesh data, specifically its UV layers. The `rescale_uv_islands` function is a complex operation that can significantly alter the UV layout; its success depends on a good initial unwrap. The UV transfer functions are critical for the final output and rely on the `orig_face_id` layer being correctly preserved during the mesh cutting process. Any errors in these operations can lead to incorrect texture mapping on the final Spine mesh.
Author: Maxim Sokolenko
"""
import bpy
import logging

logger = logging.getLogger(__name__)
import bmesh
import random
import math
from statistics import median
from mathutils import Vector
from math import fabs

# Note: avoid duplicated import of Vector
from collections import defaultdict, deque, Counter


def create_uv_from_stored_islands(
    obj: bpy.types.Object, stored_islands_json: str
) -> bool:
    """
    Create UV mapping from stored UV island data.

    Args:
        obj: Target object to apply UV mapping
        stored_islands_json: JSON string containing UV island data

    Returns:
        bool: Success status
    """
    import json

    try:
        # Parse stored UV data
        uv_islands = json.loads(stored_islands_json)
        logger.info(f"Creating UV from {len(uv_islands)} stored islands")

        # Ensure object has UV layer
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap_from_segments")

        uv_layer = obj.data.uv_layers.active
        if not uv_layer:
            logger.error("No active UV layer after creation")
            return False

        # Create BMesh for processing
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        bm_uv_layer = bm.loops.layers.uv.active
        if not bm_uv_layer:
            bm_uv_layer = bm.loops.layers.uv.new("UVMap_from_segments")

        # Apply UV coordinates from stored data
        processed_faces = set()

        for island in uv_islands:
            for face_data in island["face_loops"]:
                face_idx = face_data["face_index"]
                if face_idx >= len(bm.faces):
                    continue

                face = bm.faces[face_idx]
                if face in processed_faces:
                    continue

                # Apply UV coordinates to face loops
                for loop_idx, loop_data in enumerate(face_data["loops"]):
                    if loop_idx < len(face.loops):
                        loop = face.loops[loop_idx]
                        uv_coords = loop_data["uv"]
                        loop[bm_uv_layer].uv = (uv_coords[0], uv_coords[1])

                processed_faces.add(face)

        # Update mesh
        bm.to_mesh(obj.data)
        bm.free()

        logger.info(f"Successfully applied UV data to {len(processed_faces)} faces")
        return True

    except Exception as e:
        logger.exception(f"Error creating UV from stored islands: {e}")
        return False


def transfer_uv_islands_between_objects(
    source_obj: bpy.types.Object, target_obj: bpy.types.Object
) -> bool:
    """
    Transfer stored UV island data from source object to target object.

    Args:
        source_obj: Object containing stored UV island data
        target_obj: Object to apply UV mapping to

    Returns:
        bool: Success status
    """
    if "uv_island_segments" not in source_obj:
        logger.error(f"No UV island data found on {source_obj.name}")
        return False

    stored_data = source_obj["uv_island_segments"]
    return create_uv_from_stored_islands(target_obj, stored_data)


def transfer_baked_uvs_to_segments(textured_obj, segment_objs, bake_uv_map_name):
    """
    Transfers UV coordinates from a baked object to individual segments.

    Uses 'orig_face_id' to match polygons and 'vert.index' to
    match vertices within a polygon.

    :param textured_obj: The object onto which the texture was baked (has the correct UVs).
    :param segment_objs: A list of segment objects to which the UVs should be transferred.
    :param bake_uv_map_name: The name of the UV map on textured_obj from which to take the data.
    """
    logger.info("=== Start transferring baked UVs to segments ===")
    if not segment_objs:
        logger.warning("The segment list is empty, no UV transfer is required.")
        return

    # --- 1. Create a lookup dictionary from textured_obj ---
    uv_lookup = {}
    bm_tex = bmesh.new()
    try:
        bm_tex.from_mesh(textured_obj.data)
        orig_face_layer = bm_tex.faces.layers.int.get("orig_face_id")
        uv_layer_tex = bm_tex.loops.layers.uv.get(bake_uv_map_name)

        if not (orig_face_layer and uv_layer_tex):
            raise RuntimeError(
                f"On '{textured_obj.name}' required layers are missing ('orig_face_id' or '{bake_uv_map_name}')."
            )

        for face in bm_tex.faces:
            orig_face_id = face[orig_face_layer]
            for loop in face.loops:
                # Key: (ID of the original polygon, index of the vertex in that polygon)
                # Value: UV coordinate
                key = (orig_face_id, loop.vert.index)
                uv_lookup[key] = loop[uv_layer_tex].uv.copy()

        logger.debug(f"A UV map (lookup) of {len(uv_lookup)} entries is created.")

    finally:
        bm_tex.free()

    # --- 2. Apply UV coordinates to each segment ---
    for seg_obj in segment_objs:
        bm_seg = bmesh.new()
        try:
            bm_seg.from_mesh(seg_obj.data)
            # Segments should also have this layer, inherited during slicing
            seg_orig_face_layer = bm_seg.faces.layers.int.get("orig_face_id")
            uv_layer_seg = bm_seg.loops.layers.uv.active

            if not (seg_orig_face_layer and uv_layer_seg):
                logger.warning(
                    f"Skipping segment '{seg_obj.name}': required layers missing."
                )
                continue

            updated_count = 0
            for face in bm_seg.faces:
                orig_face_id = face[seg_orig_face_layer]
                for loop in face.loops:
                    key = (orig_face_id, loop.vert.index)
                    if key in uv_lookup:
                        loop[uv_layer_seg].uv = uv_lookup[key]
                        updated_count += 1
                    else:
                        logger.warning(
                            f"No UV found for key for segment '{seg_obj.name}' {key}."
                        )

            bm_seg.to_mesh(seg_obj.data)
            seg_obj.data.update()
            logger.info(
                f"Segment '{seg_obj.name}' has {updated_count} UV coordinates updated."
            )

        finally:
            bm_seg.free()

    logger.info("=== UV transfer to segments completed ===")


def rescale_uv_islands(obj, pre_unwrap_info=None):
    """
    Improved version: Adjusts the scale of UV islands to equalize texel density,
    using the ratio of areas and a weighted centroid to preserve proportions.
    """
    try:
        mesh = obj.data
        if obj.mode == "EDIT":
            bm = bmesh.from_edit_mesh(mesh)
        else:
            bm = bmesh.new()
            bm.from_mesh(mesh)

        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            logger.error("[rescale_uv_islands_v2] No active UV layer found. Skipping.")
            if obj.mode != "EDIT":
                bm.free()
            return

        # 1. Group faces into UV islands (by seams)
        islands = []
        visited_faces = set()
        for face in bm.faces:
            if face.hide or face in visited_faces:
                continue

            current_island = []
            q = [face]
            visited_faces.add(face)
            while q:
                f = q.pop(0)
                current_island.append(f)
                for edge in f.edges:
                    if edge.seam:
                        continue
                    for linked_face in edge.link_faces:
                        if linked_face not in visited_faces:
                            visited_faces.add(linked_face)
                            q.append(linked_face)
            islands.append(current_island)
        logger.debug(f"[rescale_uv_islands_v2] {len(islands)} UV islands found.")

        # 2. For each island, calculate the scale based on the AREA RATIO
        island_scales = []
        for idx, island_faces in enumerate(islands):
            # Calculate the sum of areas in 3D
            total_area_3d = sum(f.calc_area() for f in island_faces)
            # Calculate the sum of areas in UV space
            total_area_uv = sum(
                _compute_face_uv_area(f, uv_layer) for f in island_faces
            )

            if total_area_uv > 1e-9:
                # Scale = square root of the area ratio. The root is needed because area is a squared quantity.
                scale = math.sqrt(total_area_3d / total_area_uv)
                island_scales.append(scale)
                logger.debug(
                    f"[rescale_uv_islands_v2] Island {idx}: 3D area={total_area_3d:.4f}, UV area={total_area_uv:.4f}, Scale={scale:.4f}"
                )
            else:
                # If the UV area is zero, the island is degenerate, skip it
                island_scales.append(0.0)
                logger.warning(
                    f"[rescale_uv_islands_v2] Island {idx} has zero UV area."
                )

        # 3. Determine the REFERENCE SCALE based on the MEDIAN of all calculated scales.
        #    This is much more reliable than using data from before the unwrap (pre_unwrap_info).
        valid_scales = [s for s in island_scales if s > 0]
        if not valid_scales:
            logger.error(
                "[rescale_uv_islands_v2] No valid islands were found to calculate the reference scale."
            )
            if obj.mode != "EDIT":
                bm.free()
            return

        ref_scale = median(valid_scales)
        logger.info(
            f"[rescale_uv_islands_v2] Reference scale (median) = {ref_scale:.4f}"
        )

        # 4. Scale each island to the reference scale
        for idx, island_faces in enumerate(islands):
            current_scale = island_scales[idx]
            if current_scale <= 1e-9:
                continue

            # Calculate the scaling factor for the current island
            scale_factor = ref_scale / current_scale

            # LIMIT the factor to avoid extreme distortions
            scale_factor = max(0.2, min(scale_factor, 5.0))

            if abs(scale_factor - 1.0) < 1e-4:
                continue  # The scale is already almost perfect, skip

            # Find the AREA-WEIGHTED CENTROID to scale around the correct point
            total_uv_area_island = 0.0
            accum_center = Vector((0.0, 0.0))
            for f in island_faces:
                face_uv = [loop[uv_layer].uv.copy() for loop in f.loops]

                # Shoelace formula for area and centroid
                area_acc = 0.0
                centroid_acc = Vector((0.0, 0.0))
                n = len(face_uv)
                for i in range(n):
                    p1 = face_uv[i]
                    p2 = face_uv[(i + 1) % n]
                    cross = p1.x * p2.y - p2.x * p1.y
                    area_acc += cross
                    centroid_acc += (p1 + p2) * cross

                face_area = abs(0.5 * area_acc)
                if face_area > 1e-9:
                    face_centroid = centroid_acc / (6.0 * area_acc)
                    total_uv_area_island += face_area
                    accum_center += face_centroid * face_area

            if total_uv_area_island > 1e-9:
                centroid = accum_center / total_uv_area_island
            else:
                # Fallback to a simple geometric center if the area is zero
                all_uvs = [l[uv_layer].uv for f in island_faces for l in f.loops]
                centroid = (
                    sum(all_uvs, Vector((0.0, 0.0))) / len(all_uvs)
                    if all_uvs
                    else Vector((0.5, 0.5))
                )

            logger.debug(
                f"[rescale_uv_islands_v2] Island {idx}: scale_factor={scale_factor:.4f}, centroid=({centroid.x:.3f}, {centroid.y:.3f})"
            )

            # Apply UV scaling relative to the weighted centroid
            for f in island_faces:
                for loop in f.loops:
                    uv = loop[uv_layer].uv
                    uv.x = centroid.x + (uv.x - centroid.x) * scale_factor
                    uv.y = centroid.y + (uv.y - centroid.y) * scale_factor

        # 5. Update the mesh data
        if obj.mode == "EDIT":
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        else:
            bm.to_mesh(mesh)
            mesh.update()

        logger.info("[rescale_uv_islands_v2] Texel density alignment is complete.")

    except Exception as e:
        logger.exception(f"[rescale_uv_islands_v2] Exception: {e}")
    finally:
        # [RECOMMENDATION] Guaranteed memory release
        if bm and obj.mode != "EDIT":
            bm.free()


def reset_uv(obj, obj_name):
    # logger.debug(f"starting reset_uv function for object {obj.name}")
    """
    Resets the existing UV unwrap by deleting all UV layers and creating a new one.
    Prints a message to the console.
    """
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Select only the copy we are working with
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")  # Switch to Edit Mode

    # Create BMesh from the current mesh
    bm = bmesh.from_edit_mesh(obj.data)

    # Delete all existing UV layers
    uv_layers = bm.loops.layers.uv[:]
    for layer in uv_layers:
        bm.loops.layers.uv.remove(layer)

    # Create a new UV layer
    uv_map_name = f"UVMap_{obj.name}"
    new_uv_layer_bmesh = bm.loops.layers.uv.new(uv_map_name)

    # Update the UV unwrap
    bmesh.update_edit_mesh(obj.data)

    bpy.ops.object.mode_set(mode="OBJECT")  # Return to Object Mode

    # Return the new UV layer
    new_uv_layer = obj.data.uv_layers[-1]
    logger.debug(
        f"[INFO] Created new UV map '{new_uv_layer.name}' for object {obj.name}"
    )
    return new_uv_layer


def project_uv_top(obj, obj_name, TEXTURE_WIDTH, TEXTURE_HEIGHT):
    """
    Top-down projection UV unwrap without distorting proportions.
    The larger side of the object will occupy exactly 50% of the UV square,
    the smaller side will maintain the real aspect ratio.
    """
    # logger.debug("[project_uv_top] %s", obj.name)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")

    bm = bmesh.from_edit_mesh(obj.data)

    # create (or get) the UV layer
    uv_layer = bm.loops.layers.uv.get(f"UVMap_{obj.name}") or bm.loops.layers.uv.new(
        f"UVMap_{obj.name}"
    )

    # bounds along the axes
    mins = Vector(
        (
            min(v.co.x for v in bm.verts),
            min(v.co.y for v in bm.verts),
            min(v.co.z for v in bm.verts),
        )
    )
    maxs = Vector(
        (
            max(v.co.x for v in bm.verts),
            max(v.co.y for v in bm.verts),
            max(v.co.z for v in bm.verts),
        )
    )

    ranges = {"x": maxs.x - mins.x, "y": maxs.y - mins.y, "z": maxs.z - mins.z}

    # take the two longest axes
    axis_u, axis_v = sorted(ranges, key=ranges.get, reverse=True)[:2]
    range_u, range_v = ranges[axis_u], ranges[axis_v]
    max_range = max(range_u, range_v)
    scale = 0.5 / max_range  # 0.5 = half of the UV square

    center_u = (mins["xyz".index(axis_u)] + maxs["xyz".index(axis_u)]) / 2
    center_v = (mins["xyz".index(axis_v)] + maxs["xyz".index(axis_v)]) / 2

    for face in bm.faces:
        for loop in face.loops:
            coord = loop.vert.co
            u_raw = coord["xyz".index(axis_u)]
            v_raw = coord["xyz".index(axis_v)]

            u = (u_raw - center_u) * scale + 0.5
            v = (v_raw - center_v) * scale + 0.5
            loop[uv_layer].uv = Vector((u, v))

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    logger.debug("[project_uv_top] done for %s", obj.name)


def calculate_uv_dimensions(
    obj, obj_name, TEXTURE_WIDTH, TEXTURE_HEIGHT, pre_unwrap=True
):
    """
    FIXED VERSION v7:
    Added object centering - the center of the Z-range becomes the zero point.
    """
    logger.debug(f"[calculate_uv_dimensions_v7] start for «{obj.name}»")

    try:
        from .config import calc_uniform_scale

        mesh = obj.data
        if not mesh.vertices:
            raise Exception("The object contains no vertices.")

        # --- 1. Use the same uniform_scale ---
        uniform_scale = calc_uniform_scale(TEXTURE_WIDTH, TEXTURE_HEIGHT)
        logger.debug(
            f"[calculate_uv_dimensions_v7] Uniform scale: {uniform_scale:.4f} px/BU"
        )

        # --- 2. Get 3D dimensions ---
        all_x = [v.co.x for v in mesh.vertices]
        all_y = [v.co.y for v in mesh.vertices]
        all_z = [v.co.z for v in mesh.vertices]

        model_width_3d = max(all_x) - min(all_x) if all_x else 0
        model_length_3d = max(all_y) - min(all_y) if all_y else 0
        model_height_3d = max(all_z) - min(all_z) if all_z else 0

        # --- 3. Group vertices by Z-coordinate ---
        clustered_verts = defaultdict(list)
        for v in mesh.vertices:
            clustered_verts[round(v.co.z, 4)].append(v)

        # --- 4. FIX: Calculate the CENTER of the Z-range for centering ---
        all_z_keys = list(clustered_verts.keys())
        min_z_coord = min(all_z_keys) if all_z_keys else 0.0
        max_z_coord = max(all_z_keys) if all_z_keys else 0.0
        center_z_coord = (
            min_z_coord + max_z_coord
        ) / 2.0  # <- CENTER of the object along Z

        logger.debug(
            f"[calculate_uv_dimensions_v7] Z range: [{min_z_coord:.4f}, {max_z_coord:.4f}]"
        )
        logger.debug(
            f"[calculate_uv_dimensions_v7] Z center: {center_z_coord:.4f} (centering point)"
        )

        z_groups_info = {}

        for z_val, verts in clustered_verts.items():
            group_xs = [v.co.x for v in verts]
            group_ys = [v.co.y for v in verts]

            group_width_3d = max(group_xs) - min(group_xs) if group_xs else 0
            group_length_3d = max(group_ys) - min(group_ys) if group_ys else 0

            # --- 5. CENTERING: Displacement relative to the CENTER of the Z-range ---
            z_displacement = (
                z_val - center_z_coord
            )  # <- Relative to the center, not the minimum!
            height_real_pixels = z_displacement * uniform_scale

            z_groups_info[z_val] = {
                "width_real_pixels": group_width_3d * uniform_scale,
                "length_real_pixels": group_length_3d * uniform_scale,
                "height_real_pixels": height_real_pixels,  # <- Now centered coordinates
            }

            logger.debug(
                f"[calculate_uv_dimensions_v7] Z={z_val:.4f}: "
                f"z_displacement={z_displacement:.4f} (from center), "
                f"height_pixels={height_real_pixels:.2f}"
            )

        # --- 6. Return with centered coordinates ---
        return {
            "z_groups_info": z_groups_info,
            "_meta": {
                "pixels_per_bu": uniform_scale,
                "min_z_coord": min_z_coord,
                "max_z_coord": max_z_coord,
                "center_z_coord": center_z_coord,  # Save for debugging
            },
            "model_height_real": model_height_3d,
        }

    except Exception as e:
        logger.exception(f"[calculate_uv_dimensions_v7] {e}")
        return {}


def recenter_uv(obj):
    # logger.debug(f"starting recenter_uv function for object {obj.name}")
    """
    Centers the UV unwrap so that the geometric centroid (weighted by area) of all faces becomes (0.5, 0.5).
    If the UVs go outside the [0..1] range after shifting, scaling is performed.
    """
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Select only the copy we are working with
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.get(f"UVMap_{obj.name}")
    if not uv_layer:
        raise Exception(
            f"UV layer 'UVMap_{obj.name}' not found, cannot perform recenter_uv."
        )

    total_area = 0.0
    accum_center = Vector((0.0, 0.0))
    for face in bm.faces:
        face_uv = [loop[uv_layer].uv.copy() for loop in face.loops]
        area_acc = 0.0
        centroid_acc = Vector((0.0, 0.0))
        n = len(face_uv)
        for i in range(n):
            x_i, y_i = face_uv[i].x, face_uv[i].y
            x_next, y_next = face_uv[(i + 1) % n].x, face_uv[(i + 1) % n].y
            cross = x_i * y_next - x_next * y_i
            area_acc += cross
            centroid_acc += Vector((x_i + x_next, y_i + y_next)) * cross
        face_area = 0.5 * area_acc
        if fabs(face_area) > 1e-10:
            face_centroid = centroid_acc / (6.0 * face_area)
        else:
            face_centroid = sum(face_uv, Vector((0.0, 0.0))) / n
        total_area += fabs(face_area)
        accum_center += face_centroid * fabs(face_area)

    if total_area < 1e-10:
        overall_centroid = Vector((0.5, 0.5))
    else:
        overall_centroid = accum_center / total_area

    shift = Vector((0.5, 0.5)) - overall_centroid

    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv += shift

    # Check bounding box after shifting and scale if necessary
    all_uv = [loop[uv_layer].uv for face in bm.faces for loop in face.loops]
    min_u = min(uv.x for uv in all_uv)
    max_u = max(uv.x for uv in all_uv)
    min_v = min(uv.y for uv in all_uv)
    max_v = max(uv.y for uv in all_uv)
    range_u = max_u - min_u
    range_v = max_v - min_v

    if (
        range_u > 1.0
        or range_v > 1.0
        or min_u < 0.0
        or min_v < 0.0
        or max_u > 1.0
        or max_v > 1.0
    ):
        scale_factor = min(
            (1.0 / range_u) if range_u > 0 else 1.0,
            (1.0 / range_v) if range_v > 0 else 1.0,
        )
        for face in bm.faces:
            for loop in face.loops:
                uv = loop[uv_layer].uv
                uv.x = 0.5 + (uv.x - 0.5) * scale_factor
                uv.y = 0.5 + (uv.y - 0.5) * scale_factor
        for face in bm.faces:
            for loop in face.loops:
                uv = loop[uv_layer].uv
                uv.x = max(0.0, min(1.0, uv.x))
                uv.y = max(0.0, min(1.0, uv.y))

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode="OBJECT")
    logger.debug(
        f"The UV map is centered using area weighted centroid for the object {obj.name} and its uv_layer{uv_layer}."
    )


def unwrap_after_calculations(obj):
    # logger.debug("starting unwrap_after_calculations function")
    """
    Performs a standard UV unwrap after calculations.
    Prints a message to the console.
    """
    # Deselect all objects
    bpy.ops.object.select_all(action="DESELECT")

    # Select only the copy we are working with
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")  # Switch to Edit Mode
    bpy.ops.mesh.select_all(action="SELECT")  # Select all vertices
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)  # Perform UV unwrap
    bpy.ops.object.mode_set(mode="OBJECT")  # Return to Object Mode
    logger.debug("Standard UV mapping was performed after calculations.")


def cleanup_uv_data(obj, bm=None):
    logger.debug(f"Run cleanup_uv_data function for object {obj.name}")
    """
    Cleans up UV data after processing an object.
    """
    try:
        # If BMesh is passed, free it
        if bm:
            bm.free()

        # Switch to Object Mode
        if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        logger.debug(f"[INFO] UV data cleared for object {obj.name}")

    except Exception as e:
        logger.error(f"[ERROR] Error clearing UV data: {e}")


def unwrap_respecting_seams(obj, method="ANGLE_BASED", margin=0.001):
    """
    Performs a regular unwrap so that Blender respects custom seams (edge.seam).
    :param obj: MESH type object
    :param method: 'ANGLE_BASED' or 'CONFORMAL'
    :param margin: margin between islands
    :return: name of the active UV map (str) or None
    """

    # logger.debug(f"[unwrap_respecting_seams] Starting unwrap for object {obj.name} with method={method}, margin={margin}")

    try:
        # Deselect all objects
        bpy.ops.object.select_all(action="DESELECT")

        # Select only the copy we are working with
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # Switch to EDIT mode
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        # Select all faces
        bpy.ops.mesh.select_all(action="SELECT")
        # Call unwrap
        bpy.ops.uv.unwrap(method=method, margin=margin)
        # Return to OBJECT mode
        bpy.ops.object.mode_set(mode="OBJECT")

        # Check the active UV map
        uv_map = obj.data.uv_layers.active
        if uv_map:
            logger.debug(
                f"[unwrap_respecting_seams] unwrap complete, active UV map '{uv_map.name}'"
            )
            return uv_map.name
        else:
            logger.warning(
                "[unwrap_respecting_seams] Active UV layer not found after unwrap"
            )
            return None

    except Exception as e:
        logger.error(f"[unwrap_respecting_seams] Error: {e}")
        # Reset mode if necessary
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except:
            pass
        return None


def _round_vec(vec: Vector, ndigits: int):
    return (round(vec.x, ndigits), round(vec.y, ndigits), round(vec.z, ndigits))


def _compute_face_uv_area(face: bmesh.types.BMFace, uv_layer) -> float:
    """Area of a face in UV space (shoelace formula)."""
    loops = face.loops
    if len(loops) < 3:
        return 0.0
    area = 0.0
    for i, lp in enumerate(loops):
        p1 = loops[i][uv_layer].uv
        p2 = loops[(i + 1) % len(loops)][uv_layer].uv
        area += p1.x * p2.y - p2.x * p1.y
    return abs(area) * 0.5


def _build_islands(bm, uv_layer, tol: float) -> tuple[dict, dict]:
    """
    Cluster faces of a BMesh into UV islands.

    Islands are groups of faces whose UV coordinates lie within a tolerance for each coordinate.
    This implementation performs a depth-first search over the faces, grouping any neighbouring
    face whose corresponding UV loop coordinate differences are within ``tol`` on both the U and V axes.

    Args:
        bm: The BMesh to operate on.
        uv_layer: The UV loop layer to reference when comparing coordinates.
        tol: Tolerance for grouping faces.  Faces are considered part of the same island when
            any loop on the neighbouring face differs by at most ``tol`` in both the U and V
            components.

    Returns:
        A tuple ``(f2island, island2faces)`` where ``f2island`` maps each face to its island
        index and ``island2faces`` maps each island index to the set of faces it contains.
    """
    f2island: dict[bmesh.types.BMFace, int] = {}
    island2faces: dict[int, set] = defaultdict(set)
    unvisited = set(bm.faces)
    isl = 0
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        f2island[seed] = isl
        island2faces[isl].add(seed)
        while stack:
            f = stack.pop()
            for lp in f.loops:
                base_uv = lp[uv_layer].uv
                for lf in lp.vert.link_faces:
                    # skip already assigned faces
                    if lf in f2island:
                        continue
                    # Determine if any UV on the linked face is within tolerance on both axes
                    match = False
                    for l in lf.loops:
                        uv = l[uv_layer].uv
                        if (
                            abs(base_uv.x - uv.x) <= tol
                            and abs(base_uv.y - uv.y) <= tol
                        ):
                            match = True
                            break
                    if match:
                        f2island[lf] = isl
                        island2faces[isl].add(lf)
                        if lf in unvisited:
                            unvisited.remove(lf)
                            stack.append(lf)
        isl += 1
    return f2island, island2faces


# ──────────────────────────────────────────────────────────
#   ANGLE_LIMIT ESTIMATION
# ──────────────────────────────────────────────────────────


def estimate_angle_limit(obj: bpy.types.Object, sample_edges: int = 2000) -> float:
    """Estimates the angle_limit using the angles between adjacent Face Normals."""
    try:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        edges = [e for e in bm.edges if len(e.link_faces) == 2]
        if not edges:
            bm.free()
            return 66.0
        random.shuffle(edges)
        angles = [
            e.link_faces[0].normal.angle(e.link_faces[1].normal)
            for e in edges[:sample_edges]
        ]
        bm.free()
        if not angles:
            return 66.0
        deg = median(angles) * 180.0 / math.pi
        return max(15.0, min(88.0, deg))
    except Exception as ex:
        logger.exception(f"[estimate_angle_limit] {ex}")
        return 66.0


# ──────────────────────────────────────────────────────────
#   UNWRAP QUALITY ANALYSIS
# ──────────────────────────────────────────────────────────


def analyse_islands(bm, uv_layer, tol: float = 1e-4):
    """Returns island statistics: areas, vertex counts, etc."""
    f2island, island2faces = _build_islands(bm, uv_layer, tol)
    island_stats = []
    for isl, faces in island2faces.items():
        verts = set()
        area = 0.0
        for f in faces:
            verts.update(v.index for v in f.verts)
            area += _compute_face_uv_area(f, uv_layer)
        island_stats.append(
            {
                "faces": len(faces),
                "verts": len(verts),
                "area": area,
            }
        )
    if not island_stats:
        return {}
    areas = [s["area"] for s in island_stats]
    verts_per_island = [s["verts"] for s in island_stats]
    stats = {
        "count": len(island_stats),
        "min_area": min(areas),
        "max_area": max(areas),
        "area_ratio": (max(areas) / max(min(areas), 1e-8)),
        "avg_verts": sum(verts_per_island) / len(verts_per_island),
        "small_islands": sum(1 for a in areas if a < 1e-5),
    }
    return stats


# ──────────────────────────────────────────────────────────
#   SMART UV PROJECT
# ──────────────────────────────────────────────────────────


def smart_uv_project(
    obj: bpy.types.Object,
    obj_name: str,
    angle_limit: float = 66.0,
    island_margin: float = 0.03,
    *,
    auto_adjust: bool = True,
    tol: float = 1e-4,
    island_limits: tuple[int, int] = (1, 2000),
    quality_thresholds: dict | None = None,
) -> str:
    """Extended version of Smart UV Project.

    quality_thresholds = {
        "area_ratio": 50.0,      # max/min island area
        "avg_verts_min": 3.0,    # min. average vertex count/island
        "small_islands_max": 25  # maximum allowed number of very small islands
    }
    """

    if quality_thresholds is None:
        quality_thresholds = {
            "area_ratio": 50.0,
            "avg_verts_min": 3.0,
            "small_islands_max": 25,
        }

    # 1) ─── Type and mode check ──────────────────────
    if obj.type != "MESH":
        raise TypeError("smart_uv_project only supports MESH objects.")
    start_mode = obj.mode
    if start_mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    # 2) ─── Automatic parameter selection ────────────────────────
    if auto_adjust:
        angle_limit = estimate_angle_limit(obj)
        logger.debug(f"[smart_uv_project] auto angle_limit → {angle_limit:.1f}")

    # 3) ─── First unwrap attempt (Smart Project) ──────────
    result = _do_smart_project(obj, obj_name, angle_limit, island_margin)
    stats = _log_and_check_quality(obj, result, tol, island_limits, quality_thresholds)

    # 4) ─── If quality is poor → dynamic margin ───────
    if not stats["ok"]:
        # let's try to increase the margin, but not more than 0.15
        new_margin = min(island_margin * 2.5, 0.15)
        if abs(new_margin - island_margin) > 1e-6:
            logger.info(f"[smart_uv_project] Retry with larger margin {new_margin:.3f}")
            result = _do_smart_project(obj, obj_name, angle_limit, new_margin)
            stats = _log_and_check_quality(
                obj, result, tol, island_limits, quality_thresholds
            )

    # 5) ─── Fallback → unwrap_respecting_seams ────────────
    if not stats["ok"]:
        logger.warning("[smart_uv_project] Fallback → unwrap_respecting_seams")
        result = unwrap_respecting_seams(
            obj, method="ANGLE_BASED", margin=island_margin
        )
        stats = _log_and_check_quality(
            obj, result, tol, island_limits, quality_thresholds
        )

    # 6) ─── Second fallback → CONFORMAL and angle_limit 88 ─
    if not stats["ok"]:
        logger.warning(
            "[smart_uv_project] Second fallback → CONFORMAL + angle_limit 88"
        )
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="CONFORMAL", margin=0.001)
        bpy.ops.object.mode_set(mode="OBJECT")
        result = obj.data.uv_layers.active.name
        stats = _log_and_check_quality(
            obj, result, tol, island_limits, quality_thresholds
        )

    # 7) ─── If it's still bad — report, but return the UV layer
    if not stats["ok"]:
        logger.error(
            "[smart_uv_project] Failed to achieve acceptable UV quality after all fallbacks."
        )

    # 8) ─── Return to the original mode ───────────────────
    if start_mode != obj.mode:
        try:
            bpy.ops.object.mode_set(mode=start_mode)
        except Exception:
            pass

    return result


# ──────────────────────────────────────────────────────────
#   INTERNAL HELPERS
# ──────────────────────────────────────────────────────────


def _do_smart_project(obj, obj_name, angle_limit, island_margin):
    """Executes Smart UV Project, creates/selects a UV layer."""
    target_uv = f"UVMap_{obj_name}"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    if target_uv not in [uv.name for uv in obj.data.uv_layers]:
        bpy.ops.uv.smart_project(
            angle_limit=angle_limit, island_margin=island_margin, correct_aspect=True
        )
        if obj.data.uv_layers:
            obj.data.uv_layers.active.name = target_uv
    else:
        obj.data.uv_layers.active = obj.data.uv_layers[target_uv]
        bpy.ops.uv.smart_project(
            angle_limit=angle_limit, island_margin=island_margin, correct_aspect=True
        )
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.uv_layers.active = obj.data.uv_layers[target_uv]
    return target_uv


def _log_and_check_quality(obj, uv_name, tol, island_limits, qt):
    """Logs statistics and checks if the result meets the standards."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.get(uv_name)
    if uv_layer is None:
        bm.free()
        return {"ok": False}
    stats = analyse_islands(bm, uv_layer, tol)
    bm.free()
    if not stats:
        return {"ok": False}
    logger.debug(
        f"[UV-stats] islands={stats['count']}  minA={stats['min_area']:.4e}  "
        f"maxA={stats['max_area']:.4e}  ratio={stats['area_ratio']:.1f}  "
        f"avgVerts={stats['avg_verts']:.1f}  tiny={stats['small_islands']}"
    )

    min_is, max_is = island_limits
    ok = (
        min_is <= stats["count"] <= max_is
        and stats["area_ratio"] <= qt["area_ratio"]
        and stats["avg_verts"] >= qt["avg_verts_min"]
        and stats["small_islands"] <= qt["small_islands_max"]
    )
    return {**stats, "ok": ok}


# ----------------------------------------------------------


def cleanup_after_object_processing(obj, bm=None):
    # logger.debug(f"starting cleanup_after_object_processing function for object {obj.name}")
    """Complete cleanup of all data after processing an object"""
    try:
        # Cleanup BMesh
        if bm:
            bm.free()

        # Cleanup object's UV data
        if obj.data.uv_layers:
            for uv_layer in obj.data.uv_layers:
                obj.data.uv_layers.remove(uv_layer)

        # Reset transformations
        obj.matrix_world.identity()

        # Cleanup temporary object properties
        if hasattr(obj, "processed_uv"):
            del obj.processed_uv

        # Reset object mode
        if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        logger.debug(f"[INFO] Complete data cleanup for an object {obj.name}")

    except Exception as e:
        logger.error(f"[ERROR] Error clearing object data {obj.name}: {e}")


def faces_match(face_a, face_b, tol=1e-4):
    """
    Compares two faces based on their vertices in local coordinates.
    For each face, vertices (Vector) are collected and sorted by their rounded coordinates.
    Faces are considered matching if their vertex lists have the same length and
    the distance for each corresponding pair of vertices is less than tol.
    """
    # If the number of vertices is different, the faces do not match
    if len(face_a.verts) != len(face_b.verts):
        return False

    # Collect lists of local coordinates
    coords_a = [v.co.copy() for v in face_a.verts]
    coords_b = [v.co.copy() for v in face_b.verts]

    # Sort the lists by rounded values (so the order does not depend on the original topology)
    coords_a_sorted = sorted(
        coords_a, key=lambda v: (round(v.x, 4), round(v.y, 4), round(v.z, 4))
    )
    coords_b_sorted = sorted(
        coords_b, key=lambda v: (round(v.x, 4), round(v.y, 4), round(v.z, 4))
    )

    # Comparison of vertex pairs with tolerance
    for va, vb in zip(coords_a_sorted, coords_b_sorted):
        if (va - vb).length > tol:
            return False
    return True


VERBOSE_LOOKUP_KEYS = False  # enable if a stream of all keys is needed
VERBOSE_DUPLICATES = True  # log duplicates
MAX_LIST_PREVIEW = 200  # trimming for long lists

# ---------------- Helper utilities ----------------

# NOTE: the following duplicate helpers are retained for backward compatibility only.
# The primary implementations of ``_round_vec`` and ``_build_islands`` are defined above.
# Renaming these duplicates prevents them from clobbering the main definitions.


def _round_vec_duplicate(vec: Vector, ndigits: int):
    """
    Duplicate definition kept for reference; delegates to the primary ``_round_vec``.
    """
    # Call the original helper defined earlier in this module
    return (round(vec.x, ndigits), round(vec.y, ndigits), round(vec.z, ndigits))


def _clip(obj, max_len: int = MAX_LIST_PREVIEW):
    s = str(obj)
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"… ({len(s)} chars)"


# --- DFS-clustering ---


def _build_islands_duplicate(bm, uv_layer, tol):
    """
    Duplicate DFS-based clustering implementation. Retained for backward compatibility.
    Refer to the primary ``_build_islands`` defined earlier for the actual logic.
    """
    f2island: dict[bmesh.types.BMFace, int] = {}
    island2faces: dict[int, set] = defaultdict(set)
    unvisited = set(bm.faces)
    isl = 0
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        f2island[seed] = isl
        island2faces[isl].add(seed)
        while stack:
            f = stack.pop()
            for lp in f.loops:
                base_uv = lp[uv_layer].uv
                for lf in lp.vert.link_faces:
                    if lf in f2island:
                        continue
                    # Use coordinate-wise tolerance check as in the primary implementation
                    match = False
                    for l in lf.loops:
                        uv = l[uv_layer].uv
                        if (
                            abs(base_uv.x - uv.x) <= tol
                            and abs(base_uv.y - uv.y) <= tol
                        ):
                            match = True
                            break
                    if match:
                        f2island[lf] = isl
                        island2faces[isl].add(lf)
                        if lf in unvisited:
                            unvisited.remove(lf)
                            stack.append(lf)
        isl += 1
    return f2island, island2faces


# --- Search for deque[loop] with relaxed precision ---


def _get_loop_for_pos(pos_key, pos2loops):
    if pos_key in pos2loops and pos2loops[pos_key]:
        return pos2loops[pos_key]
    key3 = tuple(round(c, 3) for c in pos_key)
    if key3 in pos2loops and pos2loops[key3]:
        return pos2loops[key3]
    key2 = tuple(round(c, 2) for c in pos_key)
    return pos2loops.get(key2, deque())


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------


def update_uv_from_texture_copy(
    obj_A, obj_B, segment_uv3d_pairs, textured_uv3d_pairs, tol=1e-4
):
    """Transfers UV coordinates from *obj_B* (texture copy) → *obj_A* (segment).
    Logging is maximally detailed for quick diagnosis of complex cases.
    """
    bm_A = bm_B = None
    try:
        # ---- sanity
        if obj_A.type != "MESH" or obj_B.type != "MESH":
            raise TypeError("Both objects must be MESH.")
        if not (segment_uv3d_pairs and textured_uv3d_pairs):
            raise ValueError("segment_uv3d_pairs or textured_uv3d_pairs are empty.")

        # logger.debug(f"[update_uv] segment_uv3d_pairs: {_clip(segment_uv3d_pairs)}")
        # logger.debug(f"[update_uv] textured_uv3d_pairs: {_clip(textured_uv3d_pairs)}")

        # ---- BMesh
        bm_A = bmesh.new()
        bm_A.from_mesh(obj_A.data)
        bm_A.faces.ensure_lookup_table()
        bm_B = bmesh.new()
        bm_B.from_mesh(obj_B.data)
        bm_B.faces.ensure_lookup_table()

        uv_A = bm_A.loops.layers.uv.get(f"UVMap_{obj_A.name}")
        uv_B = bm_B.loops.layers.uv.get("UVMap_for_texturing")
        if not uv_A or not uv_B:
            raise RuntimeError("UV layers not found.")

        face_id_layer = bm_A.faces.layers.int.get("face_id")
        orig_face_layer = bm_B.faces.layers.int.get("orig_face_id")
        if face_id_layer is None or orig_face_layer is None:
            raise RuntimeError("face_id / orig_face_id layers are missing.")

        face_ids_A = [f[face_id_layer] for f in bm_A.faces]
        logger.debug(f"[update_uv] face_id list obj_A: {_clip(face_ids_A)}")

        # ---- islands
        f2island_A, _ = _build_islands(bm_A, uv_A, tol)
        f2island_B, _ = _build_islands(bm_B, uv_B, tol)
        logger.debug(
            f"[update_uv] islands: seg={len(set(f2island_A.values()))}  tex={len(set(f2island_B.values()))}"
        )

        # ---- build lookup
        uv_lookup = defaultdict(deque)
        duplicate_counter = Counter()
        for fB in bm_B.faces:
            isl_B = f2island_B[fB]
            oid = fB[orig_face_layer]
            for lp in fB.loops:
                for nd in (4, 3, 2):
                    pos_key = _round_vec(lp.vert.co, nd)
                    for key in ((oid, isl_B, pos_key), (oid, None, pos_key)):
                        uv_lookup[key].append(lp[uv_B].uv.copy())
                        duplicate_counter[(key[0], key[2])] += 1
                        if VERBOSE_LOOKUP_KEYS:
                            logger.debug(f"[lookup_add] {key} → {lp[uv_B].uv}")
        logger.debug(f"[update_uv] lookup keys total: {len(uv_lookup)}")
        if VERBOSE_DUPLICATES:
            dup_sample = [(k, c) for k, c in duplicate_counter.items() if c > 1][:20]
            if dup_sample:
                logger.debug(f"[lookup] multi-UV entries (sample): {dup_sample}")

        # ---- pos → loops map
        pos2loops = defaultdict(deque)
        loop_island_cache = {}
        for fA in bm_A.faces:
            isl_A = f2island_A[fA]
            for lp in fA.loops:
                for nd in (4, 3, 2):
                    pos2loops[_round_vec(lp.vert.co, nd)].append(lp)
                loop_island_cache[(lp.vert.index, _round_vec(lp.vert.co, 4))] = isl_A

        # ---- stats
        match_precision = {4: 0, 3: 0, 2: 0}

        # ---- main loop
        updated_uvs = []
        missed = []
        for idx, (_, _, pos) in enumerate(segment_uv3d_pairs):
            pos_vec = Vector(pos)
            key4 = _round_vec(pos_vec, 4)
            loops_deque = _get_loop_for_pos(key4, pos2loops)
            if not loops_deque:
                missed.append(
                    {
                        "idx": idx,
                        "face_id": None,
                        "vertex_pos": tuple(pos),
                        "island_A": None,
                        "attempted_keys": [],
                        "reason": "no_loop",
                    }
                )
                updated_uvs.extend([0.0, 0.0])
                continue

            lp_A = loops_deque.popleft()
            fid = lp_A.face[face_id_layer]
            if not isinstance(fid, int):
                missed.append(
                    {
                        "idx": idx,
                        "face_id": None,
                        "vertex_pos": tuple(pos),
                        "island_A": None,
                        "attempted_keys": [],
                        "reason": "fid_none",
                    }
                )
                uv_cur = lp_A[uv_A].uv
                updated_uvs.extend([uv_cur.x, uv_cur.y])
                continue

            isl_A = loop_island_cache.get(
                (lp_A.vert.index, _round_vec(lp_A.vert.co, 4)), None
            )
            uv_vec = None
            attempted_keys = []
            for nd in (4, 3, 2):
                pos_key = _round_vec(lp_A.vert.co, nd)
                key_island = (fid, isl_A, pos_key)
                key_naked = (fid, None, pos_key)
                attempted_keys.extend([key_island, key_naked])
                if uv_lookup.get(key_island):
                    uv_vec = uv_lookup[key_island].popleft()
                    match_precision[nd] += 1
                    break
                if uv_lookup.get(key_naked):
                    logger.debug(
                        f"[conflict] face_id={fid} island {isl_A} empty, but found without island"
                    )
                    uv_vec = uv_lookup[key_naked].popleft()
                    match_precision[nd] += 1
                    break
            if uv_vec is None:
                missed.append(
                    {
                        "idx": idx,
                        "face_id": fid,
                        "vertex_pos": tuple(pos),
                        "island_A": isl_A,
                        "attempted_keys": attempted_keys,
                        "reason": "no_uv_found",
                    }
                )
                uv_vec = lp_A[uv_A].uv.copy()
            else:
                lp_A[uv_A].uv = uv_vec
            updated_uvs.extend([uv_vec.x, uv_vec.y])

        bm_A.to_mesh(obj_A.data)
        obj_A.data.update()

        # ---- final logs
        if missed:
            logger.warning(
                f"[update_uv] missed {len(missed)}/{len(segment_uv3d_pairs)} – sample: {_clip(missed[:5])}"
            )
        # logger.info(f"[update_uv] precision stats: {match_precision}")
        # logger.info(f"[update_uv] finished: {len(updated_uvs)//2}/{len(segment_uv3d_pairs)} UV written")
        logger.debug(f"[update_uv] updated_uvs: {_clip(updated_uvs)}")
        return updated_uvs

    except Exception as exc:
        logger.exception(f"update_uv_from_texture_copy EXCEPTION: {exc}")
        return []
    finally:
        if bm_A:
            bm_A.free()
        if bm_B:
            bm_B.free()


def flip_uv_map_vertically(obj):
    """
    Flips the active UV map of the object vertically.
    For each element in the UV layer, it replaces V with (1.0 - V).

    Args:
        obj (bpy.types.Object): the object for which to flip the UV map.

    Returns:
        the object's UV layer (after updating).
    """

    # Switch to Object Mode if necessary
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    uv_layer = obj.data.uv_layers.active
    if not uv_layer:
        raise Exception(f"Object {obj.name} does not have an active UV layer.")

    # Iterate over each data element of the UV layer
    for uv_data in uv_layer.data:
        # Save the current UV value
        current_uv = uv_data.uv.copy()
        # Flip the vertical coordinate: new V = 1.0 - old V
        uv_data.uv[1] = 1.0 - current_uv[1]

    return uv_layer


def flip_image_vertically(image):
    # logger.debug("[flip_image_vertically] starting flip_image_vertically function")
    try:
        # Get image dimensions (width, height)
        width, height = image.size
        # Copy the current pixel data (each pixel is defined by 4 values: R, G, B, A)
        pixels = list(image.pixels)
        # Calculate the length of one row in the list (width * 4)
        row_length = width * 4
        new_pixels = [0.0] * len(pixels)
        # Iterate through all rows and write them in reverse order
        for row in range(height):
            # The index of the row that should replace the current one
            new_row = height - row - 1
            start_old = new_row * row_length
            end_old = start_old + row_length
            start_new = row * row_length
            new_pixels[start_new : start_new + row_length] = pixels[start_old:end_old]
        # Write the flipped data back to the image.
        image.pixels[:] = new_pixels
        image.update()
    except Exception as e:
        logger.exception(f"[flip_image_vertically] Error: {e}")


# ──────────────────────────────────────────────────────────
def log_uv_bounds(
    obj: bpy.types.Object,
    uv_name: str,
    log: logging.Logger,
    prefix: str = "[UV-bounds]",
) -> None:
    """
    Logs the min/max U-V of the active `uv_name` layer.
    Useful for immediately seeing from the log whether the UV islands
    have gone beyond the 0…1 range and whether their area is non-zero.
    """
    try:
        if uv_name not in obj.data.uv_layers:
            log.warning(f"{prefix} {obj.name}: UV layer '{uv_name}' missing.")
            return
        uv_layer = obj.data.uv_layers[uv_name].data
        min_u = min(lp.uv.x for lp in uv_layer)
        max_u = max(lp.uv.x for lp in uv_layer)
        min_v = min(lp.uv.y for lp in uv_layer)
        max_v = max(lp.uv.y for lp in uv_layer)
        log.debug(
            f"{prefix} {obj.name} U[{min_u:.3f}, {max_u:.3f}] "
            f"V[{min_v:.3f}, {max_v:.3f}]"
        )
    except Exception as e:
        log.exception(f"{prefix} {obj.name}: UV analysis error— {e}")


def register():
    logger.debug("[LOG] Registration uv_operations.py")
    pass


def unregister():
    logger.debug("[LOG] Unregistration uv_operations.py")
    pass

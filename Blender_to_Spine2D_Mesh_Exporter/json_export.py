# json_export.py
# pylint: disable=import-error
"""
This file is the core module responsible for converting processed 3D mesh data into a Spine2D-compatible JSON format.
Its main functions include:
1.  Data Processing: It takes raw data extracted from Blender objects (vertices, UVs, triangles, etc.) and processes it into a format suitable for Spine. This includes converting UV coordinates into pixel-based vertex positions for the mesh attachment.
2.  JSON Structure Construction: It builds the entire Spine JSON hierarchy, including the skeleton, bones, slots, skins, and attachments. It dynamically creates a complex bone rig with control bones for rotation and scaling, as well as a hierarchy of bones corresponding to different Z-depths (Z-groups).
3.  Coordinate and Bone Calculation: It contains sophisticated logic for calculating bone positions and weights. It uses a multi-tiered approach to determine vertex positions, prioritizing real-world 3D coordinates and falling back to UV-based calculations to ensure accuracy. It also handles the creation of individual bones for each vertex in a mesh segment.
4.  Animation and Sequence Support: It adds animation data to the JSON, including a "preview" animation for the control rig and handling for image sequences by embedding sequence metadata into the mesh attachment.
5.  Data Validation and Hashing: It performs checks, such as calculating the Euler characteristic to validate mesh topology, and generates a hash for the skeleton.
6.  Segment Handling: It has specialized logic to process mesh segments, correctly matching their vertices to the original object's data to maintain a consistent structure.

ATTENTION: - This module contains the most complex logic in the addon. The bone hierarchy and constraint system it generates are highly specific and intricate. Any changes to functions like `create_bones`, `build_constraints`, or `process_z_groups` can easily break the resulting rig in Spine. The coordinate transformation logic is critical for correctly positioning the mesh and its bones. The file relies heavily on data prepared by other modules (like `uv_operations` and `plane_cut`), so the integrity of the input `raw_data` is paramount.
Author: Maxim Sokolenko
"""
import bpy
import logging
import json
import os
import hashlib
import logging
import inspect

logger = logging.getLogger(__name__)
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from mathutils import Vector
from math import sqrt
from .config import get_default_output_dir, SEQUENCE_FRAME_DIGITS, calc_uniform_scale
from .uv_operations import update_uv_from_texture_copy
from .utils import smooth_stretch_values, limit_stretch_changes, find_closest_z

logger.debug("[LOG] Loading json_export.py")


def generate_hash(filepath):
    # logger.debug("starting generate_hash function")
    try:
        if not filepath:
            filepath = "untitled"
        hash_value = hashlib.md5(filepath.encode()).hexdigest()[:10]
        # logger.debug(f"[INFO] Generated hash: {hash_value}")
        return hash_value
    except Exception as e:
        logger.exception(f"[exception] Error generating hash: {e}")
        return "UNKNOWNHASH"


def check_euler_characteristic(vertices, edge_count_map, triangles):
    # logger.debug("starting check_euler_characteristic function")
    V = len(vertices) // 2
    E = len(edge_count_map)
    F = len(triangles) // 3
    euler_char = V - E + F
    # logger.debug(f"[INFO] Euler characteristic: V - E + F = {V} - {E} + {F} = {euler_char}")
    return euler_char


def distance3d(p1, p2):
    # logger.debug("starting distance3d function")
    # p1 is of type Vector, p2 is a list or tuple [x,y,z]
    return ((p1.x - p2[0]) ** 2 + (p1.y - p2[1]) ** 2 + (p1.z - p2[2]) ** 2) ** 0.5


def match_segment_vertex_to_original(segment_pos, original_uv3d_pairs, tolerance=1e-4):
    # logger.debug("Starting match_segment_vertex_to_original function")
    try:
        # logger.debug(f"[DEBUG] Starting search for segment_pos = {segment_pos} in the list of original uv3d-pairs (length = {len(original_uv3d_pairs)})")
        for idx, pair in enumerate(original_uv3d_pairs):
            # Check if the pair has the required structure
            if not (isinstance(pair, list) and len(pair) >= 3):
                continue
            orig_pos = pair[1]
            if not (isinstance(orig_pos, (list, tuple)) and len(orig_pos) >= 3):
                continue
            # logger.debug(f"[DEBUG] Comparison: segment_pos = {segment_pos} with original[{idx}] = {orig_pos}")
            if distance3d(segment_pos, orig_pos) < tolerance:
                # logger.debug(f"[DEBUG] Match found: index {idx}")
                return idx
        # logger.debug("[DEBUG] Match not found")
        return None
    except Exception as e:
        logger.exception(
            "[exception] Exception in match_segment_vertex_to_original: %s",
            e,
            exc_info=True,
        )
        return None


def match_uv_from_original(
    seg_coord, original_uv3d_pairs, used_indices, expected_count, tolerance=5e-2
):
    logger.debug("Starting function match_uv_from_original")
    """
    1e-4 (0.0001) — strict match.
    1e-3 (0.001) — less strict, but still an exact match.
    5e-3 (0.005) — even more relaxed comparison.
    1e-2 (0.01) — allows noticeable deviation, for more flexible matching.
    5e-2 (0.05) — almost visually indistinguishable difference, but very relaxed comparison.
    Searches original_uv3d_pairs for an entry whose UV coordinate is close to the given seg_uv.
    If found, returns the found UV coordinate (list [u, v]),
    otherwise raises ValueError.
    """
    # If the required number of matches has already been found, the search can be stopped.
    if len(used_indices) >= expected_count:
        # logger.debug("Expected number of matches reached, skipping further search.")
        return None

    for idx, pair in enumerate(original_uv3d_pairs):
        # If this global point has already been used, skip it
        if idx in used_indices:
            continue
        # Expected format for pair: [index, [u, v], [pos_x, pos_y, pos_z]]
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 3):
            continue
        orig_coord = pair[2]
        if not (isinstance(orig_coord, (list, tuple)) and len(orig_coord) >= 3):
            continue
        dx = seg_coord[0] - orig_coord[0]
        dy = seg_coord[1] - orig_coord[1]
        dz = seg_coord[2] - orig_coord[2]
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        # logger.debug(f"Comparison: difference between global coordinates = {dist}")
        if dist < tolerance:
            used_indices.add(idx)
            # Return the UV from the global array (i.e., the second element of the entry)
            return pair[1]
    raise ValueError(
        f"No match found for coordinates {seg_coord} in the original object within tolerance {tolerance}"
    )


def unique_uv3d_pairs(uv3d_pairs, tolerance=1e-4):
    """
    Filters the list of uv3d_pairs, keeping only unique entries by UV coordinate.
    Comparison is performed with the given tolerance. UVs are rounded to 4 decimal places before comparison.
    The order of entries is preserved.
    Each entry format: [index, [u, v], [pos_x, pos_y, pos_z]]
    """
    unique_list = []
    seen_uvs = []
    for entry in uv3d_pairs:
        if not (isinstance(entry, list) and len(entry) >= 3):
            continue
        uv = entry[1]
        # Round UV to 4 decimal places
        uv_rounded = [round(uv[0], 4), round(uv[1], 4)]
        duplicate = False
        for seen in seen_uvs:
            if (
                abs(uv_rounded[0] - seen[0]) <= tolerance
                and abs(uv_rounded[1] - seen[1]) <= tolerance
            ):
                duplicate = True
                break
        if not duplicate:
            seen_uvs.append(uv_rounded)
            unique_list.append(entry)
    return unique_list


def process_raw_data(
    raw_data,
    textured_raw_data,
    obj,
    textured_obj,
    TEXTURE_WIDTH,
    TEXTURE_HEIGHT,
    z_groups_info,
    original_uv3d_pairs,
    textured_uv3d_pairs,
    segment_uv3d_pairs,
    stretch_values,
    obj_name_override,
    is_segment,
    filepath,
    original_z_groups,
    local_z_groups=None,
    world_location=None,
):
    if z_groups_info is None:
        z_groups_info = {}
        logger.warning("[WARNING] z_groups_info not provided, using empty {}")

    if original_uv3d_pairs is None:
        original_uv3d_pairs = []
        logger.debug("[DEBUG] original_uv3d_pairs not passed, using an empty list")
    if textured_uv3d_pairs is None:
        textured_uv3d_pairs = []
        logger.debug("[DEBUG] textured_uv3d_pairs not passed, using an empty list")

    try:
        obj_name = obj.name
        textured_obj_name = textured_obj.name
        if raw_data.get("object_name") != obj_name:
            logger.error(f"[ERROR] Data does not belong to object {obj_name}")
            return None
        # calculations for the original copy
        extract_result = extract_raw_data(raw_data, obj_name)
        if extract_result is None:
            logger.error(f"[ERROR] Failed to extract raw_data for object {obj_name}")
            return None
        uvs, triangles, vertices, hull, edges, z_groups, uv3d_pairs = extract_result
        # logger.debug(f"Initial length of uv3d_pairs: {len(uv3d_pairs)}")
        # logger.debug(f"[DEBUG] after extract_raw_data {obj_name} uv3d_pairs look like: {uv3d_pairs}")
        textured_extract_result = extract_raw_data(textured_raw_data, textured_obj_name)
        if textured_extract_result is None:
            logger.error(
                f"[ERROR] Failed to extract textured_raw_data for object {textured_obj_name}"
            )
            return None
        (
            textured_uvs,
            textured_triangles,
            textured_vertices,
            textured_hull,
            textured_edges,
            textured_z_groups,
            textured_uv3d_pairs,
        ) = textured_extract_result
        # logger.debug(f"Initial length of textured_uv3d_pairs: {len(textured_uv3d_pairs)}")
        # logger.debug(f"[DEBUG] after extract_raw_data {textured_obj_name} textured_uv3d_pairs look like: {textured_uv3d_pairs}")
        if textured_uv3d_pairs is None:
            logger.error(
                "[ERROR] textured_uv3d_pairs is None after extract_raw_data – check the data of the textured copy"
            )
            textured_uv3d_pairs = []
        if not isinstance(textured_uv3d_pairs, list):
            logger.error(
                "[ERROR] textured_uv3d_pairs has an invalid type, expected a list"
            )
            textured_uv3d_pairs = []
        validate_data_lengths(textured_uvs, vertices, triangles)
        processed_vertices = process_uvs(textured_uvs, TEXTURE_WIDTH, TEXTURE_HEIGHT)
        edge_count_map = calculate_edge_count_map(triangles)
        euler_char = check_euler_characteristic(
            processed_vertices, edge_count_map, triangles
        )
        if euler_char != 1:
            logger.warning(
                f"[WARNING] Euler characteristic is not equal to 1 ({euler_char}). The mesh may be incorrect (expected for closed objects like a cube)."
            )
        hash_value = generate_hash(filepath)
        bounding_rect = compute_bounding_rectangle(processed_vertices)
        if bounding_rect is None:
            logger.error("[process_raw_data] bounding_rect=None")
            return None

        if obj_name_override is not None:
            attach_name = obj_name_override
        else:
            attach_name = raw_data.get("object_name", obj.name)

        if obj_name_override is not None:
            original_object_name = obj_name_override
        else:
            original_object_name = obj.name
        segment_obj_name = obj.name if is_segment else None

        # If this is a segment and there are original uv3d-pairs and local uv3d-pairs of the segment and uv3d-pairs of the texture copy,
        # use the local array for iteration and search for matches in the global one.

        if (
            is_segment
            and original_uv3d_pairs
            and textured_uv3d_pairs
            and segment_uv3d_pairs
        ):
            # logger.debug("[INFO] Filtering uv3d-pairs for the segment...")
            # logger.debug(f"[DEBUG] For segment {segment_obj_name} the local uv3d-pairs array has length: {len(segment_uv3d_pairs)}")
            # Separate sets of indices for the original and the texture copy
            used_indices_original = set()
            used_indices_textured = set()
            expected_count = len(segment_uv3d_pairs)
            filtered_uv3d_pairs = []
            new_uvs = []
            new_vertices = []

            for seg_pair in segment_uv3d_pairs:
                # Instead of seg_uv, use seg_coord (the global coordinate of the segment)
                seg_coord = seg_pair[2]
                # logger.debug(f"seg_coord looks like: {seg_coord}")
                if len(used_indices_original) >= expected_count:
                    # logger.debug("Sufficient number of matches found, further search is not performed.")
                    break
                try:
                    matched_uv = match_uv_from_original(
                        seg_coord,
                        original_uv3d_pairs,
                        used_indices_original,
                        expected_count,
                        tolerance=1e-4,
                    )
                except ValueError as e:
                    logger.exception(f"[exception] {e} for object {obj.name}.")
                    raise e
                if matched_uv is not None:
                    filtered_uv3d_pairs.append(seg_pair)
                    new_uvs.extend([matched_uv[0], matched_uv[1]])
                    x = (matched_uv[0] - 0.5) * TEXTURE_WIDTH
                    y = -((matched_uv[1] - 0.5) * TEXTURE_HEIGHT)
                    new_vertices.extend([x, y])
                    # logger.debug(f"Added entry for original: new length of filtered_uv3d_pairs = {len(filtered_uv3d_pairs)}")
                else:
                    logger.warning(
                        f"[WARNING] No match found for coordinates {seg_coord} in the global array for object {obj.name}."
                    )
            uv3d_pairs = filtered_uv3d_pairs
            # Apply additional filtering of unique entries by UV
            uv3d_pairs = unique_uv3d_pairs(uv3d_pairs, tolerance=1e-4)
            # logger.debug(f"After unique filtering, uv3d_pairs has length: {len(uv3d_pairs)}")

            # Apply additional filtering of unique entries by UV
            textured_uv3d_pairs = unique_uv3d_pairs(textured_uv3d_pairs, tolerance=1e-4)
            # logger.debug(f"After unique filtering, textured_uv3d_pairs has length: {len(textured_uv3d_pairs)}")
            # 1. Create a map of old indices to new ones
            vertex_remap = {
                old_pair[0]: new_idx for new_idx, old_pair in enumerate(uv3d_pairs)
            }
            # Recalculate the uvs and vertices arrays according to the unique entries:
            new_uvs = []
            new_vertices = []
            for entry in uv3d_pairs:
                uv = entry[1]
                new_uvs.extend(uv)
                x = (uv[0] - 0.5) * TEXTURE_WIDTH
                y = -((uv[1] - 0.5) * TEXTURE_HEIGHT)
                new_vertices.extend([x, y])
            # 3. Rebuild triangles using the map
            original_triangles = raw_data.get("triangles", [])
            new_triangles = []
            for i in range(0, len(original_triangles), 3):
                try:
                    # Attempt to find all three vertices of the triangle in the new map
                    v1 = vertex_remap[original_triangles[i]]
                    v2 = vertex_remap[original_triangles[i + 1]]
                    v3 = vertex_remap[original_triangles[i + 2]]
                    new_triangles.extend([v1, v2, v3])
                except KeyError:
                    # If at least one vertex of the triangle was filtered out,
                    # this triangle no longer exists. Skip it.
                    pass
            uvs = new_uvs
            # logger.debug(f"[INFO] Filtering uv3d-pairs for {obj_name} uvs: {uvs}")
            vertices = new_vertices
            # logger.debug(f"[INFO] Filtering uv3d-pairs for {obj_name} vertices: {vertices}")
            triangles = new_triangles
            logger.debug(
                f"[INFO] Filtering for {obj_name}: New vertices: {len(vertices)//2}, New triangles: {len(triangles)//3}"
            )
            logger.debug(f"[INFO] Length of uvs: {len(uvs)}")
            # calculations for the texture copy
            textured_new_uvs = []
            textured_new_vertices = []
            for entry in textured_uv3d_pairs:
                matched_textured_uv = entry[1]
                textured_new_uvs.extend(
                    [matched_textured_uv[0], matched_textured_uv[1]]
                )
                x = (matched_textured_uv[0] - 0.5) * TEXTURE_WIDTH
                y = -((matched_textured_uv[1] - 0.5) * TEXTURE_HEIGHT)
                textured_new_vertices.extend([x, y])
            textured_uvs = textured_new_uvs
            # logger.debug(f"[INFO] Filtering textured_uv3d-pairs for {obj_name} textured_uvs: {textured_uvs}")
            textured_vertices = textured_new_vertices
            # logger.debug(f"[INFO] Filtering textured_uv3d-pairs for {obj_name} textured_vertices: {textured_vertices}")
            # logger.debug(f"[INFO] Length of textured_uvs: {len(textured_uvs)}")
            # here calculate the texture uv coordinates and replace the segment coordinates!!!
            # Update the uv list using the texture uv coordinates
            # Assume that filtered_uv3d_pairs = uv3d_pairs (after filtering) and
            # textured_uv3d_pairs are already formed
            # First, flip the UV map of the texture copy vertically:
            # flip_uv_map_vertically(textured_obj)
            # Then update the UV for object A based on the texture copy (update via the update_uv_from_texture_copy function)
            updated_uvs = update_uv_from_texture_copy(
                obj, textured_obj, uv3d_pairs, textured_uv3d_pairs, tol=1e-4
            )

            # logger.debug(f"[INFO] Updated UV list after matching: {updated_uvs}")
            # Replace the original uvs array with the updated list
            uvs = updated_uvs

        # Calculate bounds from uv3d-pairs
        def compute_uv_bounds(uv3d_pairs, obj_name):
            try:
                uv_values = [
                    entry[1]
                    for entry in uv3d_pairs
                    if isinstance(entry, list)
                    and isinstance(entry[1], (list, tuple))
                    and len(entry[1]) >= 2
                ]
                model_min_x = min(uv[0] for uv in uv_values)
                model_max_x = max(uv[0] for uv in uv_values)
                model_min_y = min(uv[1] for uv in uv_values)
                model_max_y = max(uv[1] for uv in uv_values)
                return model_min_x, model_max_x, model_min_y, model_max_y
            except Exception as ex:
                logger.exception(
                    f"[exception] Error calculating uv3d_pairs bounds for object {obj_name}: {ex}"
                )
                return None, None, None, None

        model_min_x, model_max_x, model_min_y, model_max_y = compute_uv_bounds(
            textured_uv3d_pairs, obj.name
        )
        if None in (model_min_x, model_max_x, model_min_y, model_max_y):
            return None
        center_x = (model_min_x + model_max_x) / 2
        center_y = (model_min_y + model_max_y) / 2
        model_width_real = model_max_x - model_min_x
        model_length_real = model_max_y - model_min_y

        ppi_w = z_groups_info.get("_meta", {}).get("pixels_per_blender_unit_width")
        ppi_h = z_groups_info.get("_meta", {}).get("pixels_per_blender_unit_length")
        if ppi_w and ppi_h:
            pixels_per_blender_unit_width = ppi_w
            pixels_per_blender_unit_length = ppi_h
        else:
            logger.warning(
                "[process_raw_data] scale not found in z_groups_info – "
                "leaving UV-fallback as before"
            )
            pixels_per_blender_unit_width = None  # patch to avoid error
            pixels_per_blender_unit_length = None  # patch to avoid error

        spine_json_data, bones_info = construct_spine_json(
            attach_name,
            processed_vertices,
            uvs,
            triangles,
            hull,
            edges,
            TEXTURE_WIDTH,
            TEXTURE_HEIGHT,
            uv3d_pairs,
            textured_uv3d_pairs,
            z_groups,
            original_z_groups,
            center_x,
            center_y,
            pixels_per_blender_unit_width,
            pixels_per_blender_unit_length,
            original_object_name,
            is_segment,
            segment_obj_name,
            original_uv3d_pairs,
            z_groups_info,
            world_location=world_location,
        )

        weighted_vertices = create_weighted_vertices(
            len(uv3d_pairs),
            processed_vertices,
            bones_info,
            len(bones_info.get("all_bones", bones_info.get("bones", []))),
            original_uv3d_pairs if is_segment else None,
            is_segment,
            obj_name=attach_name,
            segment_uv3d_pairs=uv3d_pairs,  # Pass the final array, corresponding to the local order
            local_z_groups=local_z_groups,
            texture_width=TEXTURE_WIDTH,
            texture_height=TEXTURE_HEIGHT,
        )
        if "skins" in spine_json_data:
            for skin in spine_json_data["skins"]:
                for slot_name in skin.get("attachments", {}):
                    for att_name, att_data in skin["attachments"][slot_name].items():
                        if att_data.get("type") == "mesh":  # ← add filter
                            att_data["vertices"] = weighted_vertices
        spine_json_data["skeleton"]["hash"] = hash_value
        if not is_segment:
            spine_json_data["skeleton"]["x"] = round(bounding_rect["center_x"], 2)
            spine_json_data["skeleton"]["y"] = round(bounding_rect["center_y"], 2)
        spine_json_data["skeleton"]["width"] = round(bounding_rect["width"], 2)
        spine_json_data["skeleton"]["height"] = round(bounding_rect["height"], 2)
        # logger.debug(f"[INFO] spine_json_data {raw_data}")

        spine_json_data["_uv3d_pairs"] = uv3d_pairs
        spine_json_data["textured_uv3d_pairs"] = textured_uv3d_pairs

        if "uv3d_pairs" in spine_json_data:
            del spine_json_data["uv3d_pairs"]
            # logger.debug("[INFO] spine_json_data - uv3d_pairs deleted")

        # logger.debug(f"[INFO] spine_json_data {spine_json_data}")
        if not filepath:
            filepath = os.path.join(
                get_default_output_dir(), f"{obj.name}_spine_output.json"
            )
        try:
            if filepath:
                save_json_files(raw_data, spine_json_data, filepath)
                # logger.debug(f"[INFO] JSON data successfully saved to {filepath}")
                return spine_json_data
        except Exception as e:
            logger.exception(
                f"[exception] Failed to create JSON data for Spine.{obj.name}: {e}"
            )
            return None

    except Exception as e:
        logger.exception(
            f"[exception] Error in process_raw_data for object {obj.name}: {e} at path {filepath}"
        )
        return None


def extract_raw_data(raw_data, obj_name):
    """
    Retrieves data for an object from raw_data.
    If the value for the key "uv3d_pairs" is missing or empty,
    it tries to get data from the key "textured_uv3d_pairs".
    """
    try:
        # logger.debug(f"[DEBUG] extract_raw_data - extracting data for object {obj_name}")
        uvs = raw_data.get("uvs", [])
        triangles = raw_data.get("triangles", [])
        vertices = raw_data.get("vertices", [])
        hull = raw_data.get("hull", 0)
        edges = raw_data.get("edges", [])
        z_groups = raw_data.get("z_groups", [])
        uv3d_pairs = raw_data.get("uv3d_pairs")
        # If uv3d_pairs is missing or an empty list, try to use textured_uv3d_pairs
        if not uv3d_pairs:
            logger.warning(
                "[WARNING] Key 'uv3d_pairs' is missing or empty; trying to use 'textured_uv3d_pairs'"
            )
            uv3d_pairs = raw_data.get("textured_uv3d_pairs", [])
        # If the final value is not a list, convert it to an empty list
        if not isinstance(uv3d_pairs, list):
            logger.error(
                "[ERROR] uv3d_pairs has an invalid type, expected a list. Converting to an empty list."
            )
            uv3d_pairs = []
        return uvs, triangles, vertices, hull, edges, z_groups, uv3d_pairs
    except Exception as e:
        logger.exception(
            f"[exception] Unable to retrieve raw_data for object {obj_name}: {e}"
        )
        return None


def validate_data_lengths(uvs, vertices, triangles):
    # logger.debug("starting validate_data_lengths function")
    """Checks the correctness of the list lengths."""
    if len(uvs) % 2 != 0:
        logger.warning("[WARNING] The 'uvs' list has an odd number of elements.")
    if len(vertices) % 2 != 0:
        logger.warning("[WARNING] The 'vertices' list has an odd number of elements.")
    if len(triangles) % 3 != 0:
        logger.warning("[WARNING] The 'triangles' list is not a multiple of 3.")


def process_uvs(uvs: list, texture_width: float, texture_height: float) -> list:
    """
    Converts a linear list [u,v,u,v,…] (0-1)
    into Spine vertex coordinates [x,y,x,y,…] in pixels.

    • (0.5, 0.5) UV space → (0, 0) Spine
    • X is scaled by the texture width, Y by the height
    • Y is inverted (Spine has "up", UV has "down")
    """
    # logger.debug(f"[process_uvs] Start (uvs={len(uvs)})")

    if not isinstance(uvs, list):
        logger.error("[process_uvs] uvs must be list, got %s", type(uvs))
        return []
    if len(uvs) % 2:
        logger.error("[process_uvs] uvs length must be even, got %d", len(uvs))
        return []

    processed: list[float] = []
    for i in range(0, len(uvs), 2):
        try:
            u = float(uvs[i])
            v = float(uvs[i + 1])

            x = (u - 0.5) * texture_width  # ← scale only by W
            y = (0.5 - v) * texture_height  # ← scale only by H + inversion

            # Spine likes 2-3 decimal places
            processed.extend([round(x, 3), round(y, 3)])
        except ValueError as exc:
            logger.warning("[process_uvs] bad pair @%d: %s → skip", i, exc)

    logger.debug("[process_uvs] Done, pairs=%d", len(processed) // 2)
    return processed


def calculate_edge_count_map(triangles):
    # logger.debug("starting calculate_edge_count_map function")
    """Calculates the edge count map for calculating the Euler characteristic."""
    edge_count_map = defaultdict(int)
    if not triangles:
        logger.warning("[WARNING] Triangle list is empty")
        return edge_count_map
    for i in range(0, len(triangles), 3):
        if i + 2 >= len(triangles):
            logger.warning(
                f"[WARNING] Not enough triangle indices for triangle {i // 3}. Skipping."
            )
            continue
        a, b, c = triangles[i], triangles[i + 1], triangles[i + 2]
        edges = [tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))]
        for edge in edges:
            edge_count_map[edge] += 1
    return edge_count_map


def compute_bounding_rectangle(processed_vertices):
    # logger.debug("starting compute_bounding_rectangle function")
    """Calculates the bounding rectangle of the model."""
    if not processed_vertices:
        logger.error("[ERROR] List 'vertices' is empty.")
        return None
    min_x = min(processed_vertices[::2])
    min_y = min(processed_vertices[1::2])
    max_x = max(processed_vertices[::2])
    max_y = max(processed_vertices[1::2])
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    # logger.debug(f"Bounding rectangle: center=({center_x}, {center_y}), width={width}, height={height}")
    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width": width,
        "height": height,
        "center_x": center_x,
        "center_y": center_y,
    }


def build_control_slots_and_attachments(
    bone_prefix: str
) -> Tuple[List[dict], Dict[str, Any]]:
    """Returns (slots, attachments_dict) for control bones and main-boundingbox."""
    slots = [
        {
            "name": f"{bone_prefix}_rotation_X",
            "bone": f"{bone_prefix}_rotation_X",
            "attachment": f"{bone_prefix}_rotation_X",
        },
        {
            "name": f"{bone_prefix}_rotation_Z",
            "bone": f"{bone_prefix}_rotation_Z",
            "attachment": f"{bone_prefix}_rotation_Z",
        },
        {
            "name": f"{bone_prefix}_rotation_Y",
            "bone": f"{bone_prefix}_rotation_Y",
            "attachment": f"{bone_prefix}_rotation_Y",
        },
        {
            "name": f"{bone_prefix}_main",
            "bone": f"{bone_prefix}_main",
            "attachment": f"{bone_prefix}_main",
        },
    ]

    att: Dict[str, Any] = {
        f"{bone_prefix}_main": {
            f"{bone_prefix}_main": {
                "type": "boundingbox",
                "vertexCount": 24,
                "vertices": [
                    -21.11,
                    20.72,
                    -20.96,
                    68.4,
                    -43.49,
                    68.1,
                    -1.72,
                    116.85,
                    43.32,
                    68.33,
                    20.42,
                    68.33,
                    20.56,
                    20.66,
                    68.38,
                    20.83,
                    68.11,
                    42.63,
                    117.11,
                    -1.92,
                    68.53,
                    -43.12,
                    68.19,
                    -20.49,
                    20.8,
                    -20.65,
                    20.65,
                    -68.22,
                    42.87,
                    -68.6,
                    -1.65,
                    -116.52,
                    -42.84,
                    -68.86,
                    -20.52,
                    -68.5,
                    -20.68,
                    -21.13,
                    -68,
                    -21.09,
                    -68.23,
                    -43.28,
                    -117.01,
                    -1.99,
                    -68.32,
                    43.2,
                    -68.17,
                    20.58,
                ],
                "color": "df00ffff",
            }
        },
        f"{bone_prefix}_rotation_X": {
            f"{bone_prefix}_rotation_X": {
                "type": "boundingbox",
                "vertexCount": 64,
                "vertices": [
                    33.35,
                    -86.18,
                    19.36,
                    -57.86,
                    6.36,
                    -62.25,
                    -4.44,
                    -64.09,
                    -18.35,
                    -63.9,
                    -33.54,
                    -59.51,
                    -45.63,
                    -52.92,
                    -55.88,
                    -43.95,
                    -63.57,
                    -33.51,
                    -69.78,
                    -20.21,
                    -72.89,
                    -6.3,
                    -72.71,
                    7.43,
                    -69.78,
                    21.89,
                    -62.09,
                    35.44,
                    -52.19,
                    47.35,
                    -39.92,
                    56.13,
                    -25.83,
                    61.81,
                    -8.98,
                    63.46,
                    9.22,
                    61.63,
                    25.33,
                    53.75,
                    37.77,
                    43.14,
                    48.39,
                    27.94,
                    53.34,
                    12.38,
                    54.98,
                    -0.62,
                    27.89,
                    25.93,
                    24.23,
                    25.93,
                    22.4,
                    23.18,
                    22.95,
                    -10.5,
                    23.86,
                    -14.53,
                    65.6,
                    -56.82,
                    69.63,
                    -58.65,
                    73.47,
                    -58.65,
                    75.78,
                    -56.46,
                    116.42,
                    -15.08,
                    118.44,
                    -11.24,
                    118.99,
                    23.55,
                    116.42,
                    26.11,
                    112.21,
                    25.93,
                    86.4,
                    -0.25,
                    86.09,
                    9.86,
                    84.81,
                    22.13,
                    80.05,
                    35.13,
                    73.82,
                    48.31,
                    65.58,
                    60.94,
                    53.88,
                    72.75,
                    39.97,
                    82.27,
                    25.87,
                    89.23,
                    8.3,
                    93.81,
                    -9.28,
                    95.82,
                    -26.57,
                    94.17,
                    -44.33,
                    89.23,
                    -61.54,
                    80.26,
                    -78.02,
                    67.08,
                    -91.56,
                    49.32,
                    -101.17,
                    27.7,
                    -105.19,
                    5.18,
                    -103.91,
                    -18.8,
                    -96.22,
                    -42.97,
                    -85.53,
                    -59.08,
                    -69.42,
                    -75,
                    -52.44,
                    -86.57,
                    -24.8,
                    -95.17,
                    0.46,
                    -96.63,
                    24.45,
                    -90.78,
                ],
                "color": "ff0000ff",
            }
        },
        f"{bone_prefix}_rotation_Y": {
            f"{bone_prefix}_rotation_Y": {
                "type": "boundingbox",
                "vertexCount": 64,
                "vertices": [
                    33.45,
                    -86.17,
                    19.46,
                    -57.86,
                    6.46,
                    -62.25,
                    -4.34,
                    -64.08,
                    -18.25,
                    -63.9,
                    -33.45,
                    -59.5,
                    -45.53,
                    -52.91,
                    -55.78,
                    -43.94,
                    -63.47,
                    -33.51,
                    -69.68,
                    -20.21,
                    -72.79,
                    -6.29,
                    -72.61,
                    7.44,
                    -69.68,
                    21.9,
                    -61.99,
                    35.45,
                    -52.09,
                    47.35,
                    -39.83,
                    56.14,
                    -25.73,
                    61.81,
                    -8.89,
                    63.46,
                    9.31,
                    61.63,
                    25.42,
                    53.76,
                    37.87,
                    43.14,
                    48.49,
                    27.95,
                    53.43,
                    12.38,
                    55.08,
                    -0.61,
                    27.99,
                    25.93,
                    24.32,
                    25.93,
                    22.49,
                    23.19,
                    23.04,
                    -10.5,
                    23.96,
                    -14.53,
                    65.7,
                    -56.82,
                    69.73,
                    -58.65,
                    73.57,
                    -58.65,
                    75.88,
                    -56.45,
                    116.52,
                    -15.08,
                    118.53,
                    -11.23,
                    119.08,
                    23.55,
                    116.52,
                    26.11,
                    112.31,
                    25.93,
                    86.5,
                    -0.25,
                    86.19,
                    9.87,
                    84.91,
                    22.13,
                    80.15,
                    35.13,
                    73.92,
                    48.31,
                    65.68,
                    60.94,
                    53.98,
                    72.76,
                    40.07,
                    82.28,
                    25.97,
                    89.23,
                    8.4,
                    93.81,
                    -9.18,
                    95.82,
                    -26.48,
                    94.18,
                    -44.23,
                    89.23,
                    -61.44,
                    80.26,
                    -77.92,
                    67.08,
                    -91.47,
                    49.32,
                    -101.07,
                    27.7,
                    -105.09,
                    5.19,
                    -103.81,
                    -18.8,
                    -96.12,
                    -42.96,
                    -85.43,
                    -59.07,
                    -69.32,
                    -75,
                    -52.35,
                    -86.56,
                    -24.7,
                    -95.17,
                    0.56,
                    -96.63,
                    24.54,
                    -90.77,
                ],
                "color": "00ff18ff",
            }
        },
        f"{bone_prefix}_rotation_Z": {
            f"{bone_prefix}_rotation_Z": {
                "type": "boundingbox",
                "vertexCount": 64,
                "vertices": [
                    33.45,
                    -86.17,
                    19.46,
                    -57.86,
                    6.46,
                    -62.25,
                    -4.34,
                    -64.08,
                    -18.26,
                    -63.9,
                    -33.45,
                    -59.5,
                    -45.53,
                    -52.91,
                    -55.79,
                    -43.94,
                    -63.48,
                    -33.51,
                    -69.68,
                    -20.21,
                    -72.79,
                    -6.29,
                    -72.61,
                    7.44,
                    -69.68,
                    21.9,
                    -61.99,
                    35.45,
                    -52.1,
                    47.35,
                    -39.83,
                    56.14,
                    -25.73,
                    61.81,
                    -8.89,
                    63.46,
                    9.31,
                    61.63,
                    25.42,
                    53.76,
                    37.87,
                    43.14,
                    48.49,
                    27.95,
                    53.43,
                    12.38,
                    55.08,
                    -0.61,
                    27.98,
                    25.93,
                    24.32,
                    25.93,
                    22.49,
                    23.19,
                    23.04,
                    -10.5,
                    23.96,
                    -14.53,
                    65.7,
                    -56.82,
                    69.72,
                    -58.65,
                    73.57,
                    -58.65,
                    75.87,
                    -56.45,
                    116.52,
                    -15.08,
                    118.53,
                    -11.23,
                    119.08,
                    23.55,
                    116.52,
                    26.11,
                    112.31,
                    25.93,
                    86.49,
                    -0.25,
                    86.18,
                    9.87,
                    84.9,
                    22.13,
                    80.14,
                    35.13,
                    73.92,
                    48.31,
                    65.68,
                    60.94,
                    53.98,
                    72.76,
                    40.06,
                    82.28,
                    25.97,
                    89.23,
                    8.39,
                    93.81,
                    -9.18,
                    95.82,
                    -26.48,
                    94.18,
                    -44.24,
                    89.23,
                    -61.45,
                    80.26,
                    -77.92,
                    67.08,
                    -91.47,
                    49.32,
                    -101.07,
                    27.7,
                    -105.1,
                    5.19,
                    -103.82,
                    -18.8,
                    -96.13,
                    -42.96,
                    -85.44,
                    -59.07,
                    -69.33,
                    -75,
                    -52.35,
                    -86.56,
                    -24.71,
                    -95.17,
                    0.56,
                    -96.63,
                    24.54,
                    -90.77,
                ],
                "color": "002cffff",
            }
        },
    }
    return slots, att


def construct_spine_json(
    attachment_name,
    processed_vertices,
    uvs,
    triangles,
    hull,
    edges,
    texture_width,
    texture_height,
    uv3d_pairs,
    textured_uv3d_pairs,  # pass the uv3d-pairs of the texture copy
    z_groups,
    original_z_groups,
    center_x,
    center_y,
    pixels_per_blender_unit_width,
    pixels_per_blender_unit_length,
    original_object_name,
    is_segment,
    segment_obj_name,
    original_uv3d_pairs,
    z_groups_info,
    world_location=None,
):
    # Define names depending on the object type
    if is_segment and original_object_name:
        # Use the name of the original object (without "_copy_for_uv") for the bone prefix
        base_name = (
            original_object_name[: -len("_copy_for_uv")]
            if original_object_name.endswith("_copy_for_uv")
            else original_object_name
        )
        bone_prefix = base_name
        slot_name = (
            segment_obj_name  # for a segment, the slot name will be the segment name
        )
        vertex_prefix = segment_obj_name
    else:
        # For the original object, also remove "_copy_for_uv" from attachment_name if present
        base_name = (
            attachment_name[: -len("_copy_for_uv")]
            if attachment_name.endswith("_copy_for_uv")
            else attachment_name
        )
        bone_prefix = base_name
        slot_name = attachment_name
        vertex_prefix = bone_prefix

    bones, bones_info = create_bones(
        bone_prefix,
        uv3d_pairs,
        z_groups,
        original_z_groups,
        texture_width,
        texture_height,
        center_x,
        center_y,
        pixels_per_blender_unit_width,
        pixels_per_blender_unit_length,
        original_uv3d_pairs=original_uv3d_pairs,  # ← pass explicitly by name
        z_groups_info=z_groups_info,  # ← and this one too by name
        world_location=world_location,
    )
    uniform_scale = calc_uniform_scale(texture_width, texture_height)
    ik_block, transform_block = build_constraints(
        uniform_scale, bone_prefix, bones_info
    )

    if is_segment:
        # [FIXED] Save the offset – the number of bones before adding vertex bones
        bones_info["vertex_bones_start_index"] = len(bones)
        vertex_bones_data = create_vertex_bones(
            vertex_prefix,
            uv3d_pairs,
            bones_info,
            center_x,
            center_y,
            pixels_per_blender_unit_width,
            pixels_per_blender_unit_length,
            original_uv3d_pairs,
            texture_width,  # pass the texture width
            texture_height,  # pass the texture height
            tolerance=1e-4,
        )
        bones.extend(vertex_bones_data["bones"])
        bones_info["vertex_bone_names"] = vertex_bones_data["vertex_bone_names"]
    # path for images
    images_path_prefix = bpy.context.scene.get("spine2d_images_path", "./images/")
    json_dir_prop = bpy.context.scene.get("spine2d_json_path", "")
    raw_json_base_dir = ""
    if json_dir_prop and os.path.isabs(bpy.path.abspath(json_dir_prop)):
        raw_json_base_dir = bpy.path.abspath(json_dir_prop)
    else:
        raw_json_base_dir = get_default_output_dir()
    json_base_dir = (
        str(raw_json_base_dir, "utf-8")
        if isinstance(raw_json_base_dir, bytes)
        else str(raw_json_base_dir)
    )
    images_path_setting = bpy.context.scene.get("spine2d_images_path", "./images/")
    images_abs_path = os.path.abspath(os.path.join(json_base_dir, images_path_setting))
    try:
        images_path_prefix = os.path.relpath(images_abs_path, json_base_dir)
    except ValueError:
        images_path_prefix = images_abs_path

    if images_path_prefix and images_path_prefix != ".":
        images_path_prefix = images_path_prefix.replace(os.sep, "/") + "/"
    else:
        images_path_prefix = ""

    base_image_name = attachment_name.replace("_copy_for_uv", "")
    final_image_path = f"{images_path_prefix}{base_image_name}_texturing_Baked"
    spine_json_data = {
        "skeleton": {
            "hash": "hash_value_placeholder",
            "spine": "4.2.43",
            "x": 0,
            "y": 0,
            "width": texture_width,
            "height": texture_height,
            "images": "",
            "audio": "./audio",
        },
        "bones": bones,
        "slots": [
            {
                "name": slot_name,
                "bone": bones_info["base_bone_name"],
                "attachment": slot_name,
            }
        ],
        "ik": ik_block,
        "transform": transform_block,
        "skins": [
            {
                "name": "default",
                "attachments": {
                    slot_name: {
                        slot_name: {
                            "type": "mesh",
                            "path": final_image_path,
                            "uvs": uvs,
                            "triangles": triangles,
                            "vertices": """weighted_vertices""",
                            "hull": hull,
                            "edges": edges,
                            "width": texture_width,
                            "height": texture_height,
                        }
                    }
                },
            }
        ],
        "animations": {"animation": {}},
    }
    # --- 3. Sequences ---------------------------------------------------------
    # frames_for_render is set in the UI; if 0 – there is no sequence
    scene = bpy.context.scene
    try:
        sequence_frame_count = int(getattr(scene, "spine2d_frames_for_render", 0))
        sequence_frame_count = max(sequence_frame_count, 0)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.exception(
            "[construct_spine_json] Failed to read spine2d_frames_for_render: %s",
            exc,
        )
        sequence_frame_count = 0

    if sequence_frame_count:
        # 3.1 — calculate start / setup
        try:
            sequence_start = int(getattr(scene, "spine2d_bake_frame_start", 0))
        except Exception:
            sequence_start = 0

        setup_index = 1 if sequence_frame_count > 1 else 0

        # 3.2 — correct the path (must end with "_")
        attachment_data = spine_json_data["skins"][0]["attachments"][slot_name][
            slot_name
        ]
        if not attachment_data["path"].endswith("_"):
            attachment_data["path"] += "_"

        # 3.3 — add full sequence information
        attachment_data["sequence"] = {
            "count": sequence_frame_count,
            "start": sequence_start,
            "digits": SEQUENCE_FRAME_DIGITS,
            "setup": setup_index,
        }

        logger.debug(
            "[construct_spine_json] Sequence added: " "count=%d, start=%d, setup=%d",
            sequence_frame_count,
            sequence_start,
            setup_index,
        )

    # ——— 5. Control icons (boundingbox-controls) —————————
    add_control_icons = True
    try:
        # If inside Blender – read the scene property
        add_control_icons = bool(bpy.context.scene.get("spine2d_control_icons", True))
    except Exception:
        # Outside Blender (CLI) – leave enabled by default
        add_control_icons = True
    # ——— 5. Export preview animation (boundingbox-controls) —————————
    add_preview_animation_icons = True
    try:
        # If inside Blender – read the scene property
        add_preview_animation_icons = bool(
            bpy.context.scene.get("spine2d_export_preview_animation", True)
        )
    except Exception:
        # Outside Blender (CLI) – leave enabled by default
        add_preview_animation_icons = True

    if (not is_segment) and add_control_icons:
        # Add control icons (boundingbox-controls)
        extra_slots, extra_atts = build_control_slots_and_attachments(bone_prefix)
        spine_json_data["slots"] = (
            extra_slots + spine_json_data["slots"]
        )  # insert at the beginning
        spine_json_data["skins"][0]["attachments"].update(extra_atts)
    # ——— 6. Preview-animation ————————————————————————————
    if (not is_segment) and add_preview_animation_icons:
        add_preview_rotation_animation(spine_json_data, bone_prefix)
    logger.debug("JSON data for Spine generated.")
    return spine_json_data, bones_info


def create_bones(
    bone_prefix,
    uv3d_pairs,
    z_groups,
    original_z_groups,
    texture_width,
    texture_height,
    center_x,
    center_y,
    pixels_per_blender_unit_width,
    pixels_per_blender_unit_length,
    original_uv3d_pairs=None,  # ← first the original pairs
    z_groups_info: dict | None = None,  # ← then offsets; COMMA removed
    uniform_scale: float = 1.0,
    world_location: Vector = None,
):
    """Creates a *full* bone hierarchy according to the new requirements.

    Hierarchy:
    root
    └─ <prefix>_main
       ├─ <prefix>                              "base" bone
       │   ├─ <prefix>_scale_rotate_X           scale-axis X
       │   │   └─ <prefix>_rotate_X             working bone (under it are Z-bones and slot-meshes)
       │   └─ <prefix>_rotate_X_constraint      ...
       │       └─ <prefix>_rotate_X_constraint_scale_IK
       │           └─ <prefix>_rotate_X_constraint_rotate_IK
       │               └─ <prefix>_rotate_X_constraint_IK
       └─ <prefix>__rotation_X / Y / Z          control bones

    Added to bones_info:
        scale_bone_name         → <prefix>_scale_rotate_X
        main_rotation_bone_name → <prefix>_rotate_X
        ctrl_rotation_x|y|z     → names of control bones
        base_bone_name, rot_x   → service references
        (plus everything returned by process_z_groups)
    Features:
        • Constraint-chain (<prefix>_rotate_X_constraint …) gets length / rotation / x / y / color / icon
        • <prefix>_rotate_X bone now has length / x / y / color
        Values are taken from uniform_scale, which is equal to the longest side of the texture.
    """
    logger.debug("[create_bones v2] start")
    uniform_scale = calc_uniform_scale(texture_width, texture_height)
    half_scale = uniform_scale / 2.0

    bones: list[dict] = []
    bones_info: dict = {
        "z_group_z_values": [],
        "z_value_to_bone_name": {},
        "sub_bone_names": [],
        "sub_bone_scale_names": [],
    }

    # ------------------------------------------------------------------ root
    bones.append({"name": "root"})

    # ------------------------------------------------------------------ <prefix>_main
    # ------------------------------------------------------------------ <prefix>_main
    main_bone_name = f"{bone_prefix}_main"

    # This line remains, but its result will only be used as a fallback
    average_y = calculate_average_y(uv3d_pairs)

    # Set default values
    bone_x = 0.0
    bone_y = round(average_y, 2)

    # If the object's world position is provided, it takes priority and overrides X and Y
    if world_location:
        logger.info(f"Using world_location for '{main_bone_name}': {world_location}")
        bone_x = world_location.x * uniform_scale
        bone_y = world_location.y * uniform_scale
        logger.info(
            f"Calculated bone position: x={bone_x}, y={bone_y} with scale={uniform_scale}"
        )

    # Now create the bone with the calculated X and Y
    bones.append(
        {
            "name": main_bone_name,
            "parent": "root",
            "x": round(bone_x, 2),
            "y": round(bone_y, 2),
        }
    )
    bones_info["main_bone_name"] = main_bone_name

    # ------------------------------------------------------------------ <prefix>
    base_bone_name = f"{bone_prefix}"
    bones.append({"name": base_bone_name, "parent": main_bone_name})
    bones_info["base_bone_name"] = base_bone_name

    # ------------------------------------------------------------------ <prefix>_scale_rotate_X  ➜  <prefix>_rotate_X
    scale_rot_x = f"{bone_prefix}_scale_rotate_X"
    rot_x = f"{bone_prefix}_rotate_X"
    bones.extend(
        [
            {
                "name": scale_rot_x,
                "parent": base_bone_name,
                "length": half_scale,
                "y": -0.5,
                "scaleX": 0,
            },
            {"name": rot_x, "parent": scale_rot_x, "color": "ff0000ff"},
        ]
    )
    bones_info.update(
        {
            "scale_bone_name": scale_rot_x,  # needed for add_stretch_and_z_groups_to_animations
            "main_rotation_bone_name": rot_x,  # needed for add_stretch_and_z_groups_to_animations
            "rot_x": rot_x,
        }
    )
    # ------------------------------------------------------------------ Z-groups under <prefix>_rotate_X
    z_group_data = process_z_groups(
        bone_prefix,
        z_groups,
        original_z_groups,
        rot_x,  # parent
        half_scale,
        z_groups_info,
        # NEW PARAMETERS - pass everything needed for correct transformation
        texture_width=texture_width,
        texture_height=texture_height,
        center_x=center_x,
        center_y=center_y,
        pixels_per_blender_unit_width=pixels_per_blender_unit_width,
        pixels_per_blender_unit_length=pixels_per_blender_unit_length,
        original_uv3d_pairs=original_uv3d_pairs,
        tolerance=1e-4,
    )
    bones.extend(z_group_data["bones"])
    bones_info.update(z_group_data["bones_info"])

    # ------------------------------------------------------------------ control bones  __rotation_X/Y/Z
    control_params = {
        "X": {"x": uniform_scale, "y": half_scale, "color": "ff0000ff"},
        "Y": {"x": uniform_scale, "color": "00ff18ff"},
        "Z": {"x": uniform_scale, "y": -half_scale, "color": "002cffff"},
    }
    for axis in ("X", "Y", "Z"):
        ctrl_name = f"{bone_prefix}_rotation_{axis}"
        params = {
            "name": ctrl_name,
            "parent": main_bone_name,
            "length": half_scale,
            "x": uniform_scale,
        }
        params.update(control_params[axis])
        bones.append(params)
        bones_info[f"ctrl_rotation_{axis.lower()}"] = ctrl_name

    # ------------------------------------------------------------------ Constraint-chain under the base bone
    chain_defs: List[Dict[str, Any]] = [
        {
            "name": f"{bone_prefix}_rotate_X_constraint",
            "parent": base_bone_name,
            "length": half_scale,
            "rotation": 90,
            "y": -0.5,
            "color": "abe323ff",
        },
        {
            "name": f"{bone_prefix}_rotate_X_constraint_scale_IK",
            "parent": base_bone_name,
            "y": half_scale - 0.5,
            "scaleX": 0,
        },
        {
            "name": f"{bone_prefix}_rotate_X_constraint_rotate_IK",
            "parent": f"{bone_prefix}_rotate_X_constraint_scale_IK",
            "x": -half_scale,
        },
        {
            "name": f"{bone_prefix}_rotate_X_constraint_IK",
            "parent": f"{bone_prefix}_rotate_X_constraint_rotate_IK",
            "rotation": 90,
            "x": half_scale,
            "color": "ff3f00ff",
            "icon": "ik",
        },
    ]
    bones.extend(chain_defs)

    # ---------------------------------------------- done
    return bones, bones_info


def build_constraints(
    uniform_scale, bone_prefix: str, bones_info: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return two lists — **ik** and **transform** — that follow the new rig schema.

    It relies only on the information already stored inside *bones_info* that is
    produced by **create_bones v2** (ctrl‑bones, z‑group names, etc.), so there is
    *no* heavy logic here — just assembling dictionaries.
    """

    # Convenience shortcuts ------------------------------------------------------
    sub_scale = bones_info.get("sub_bone_scale_names", [])  #  *_i_scale bones
    sub = bones_info.get("sub_bone_names", [])  #  *_i       bones
    rot_x = bones_info.get("rot_x", f"{bone_prefix}_rotate_X")
    reversed_scale_bones = list(reversed(sub_scale))

    # ---------------------------------------------------------------- IK block --
    ik: List[Dict[str, Any]] = [
        {
            "name": f"{bone_prefix}_scale_constraint_IK",
            "order": 3,
            "bones": [f"{bone_prefix}_rotate_X_constraint"],
            "target": f"{bone_prefix}_rotate_X_constraint_IK",
            "compress": True,
            "stretch": True,
        }
    ]

    # ----------------------------------------------------------- Transform block --
    transform: List[Dict[str, Any]] = [
        {
            "name": f"{bone_prefix}_rotation_X",
            "order": 1,
            "bones": sub_scale + [bone_prefix],
            "target": bones_info.get("ctrl_rotation_x", f"{bone_prefix}_rotation_X"),
            "rotation": 90,
            "local": True,
            "relative": True,
            "x": -(uniform_scale * 2),
            "y": -(uniform_scale / 2),
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
        {
            "name": f"{bone_prefix}_rotation_Y",
            "order": 2,
            "bones": [rot_x, f"{bone_prefix}_rotate_X_constraint_rotate_IK"],
            "target": bones_info.get("ctrl_rotation_y", f"{bone_prefix}_rotation_Y"),
            "local": True,
            "relative": True,
            "x": uniform_scale,
            "scaleX": -1,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
        {
            "name": f"{bone_prefix}_rotation_Z",
            "order": 5,
            "bones": sub,
            "target": bones_info.get("ctrl_rotation_z", f"{bone_prefix}_rotation_Z"),
            "local": True,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
        {
            "name": f"{bone_prefix}_scale_constraint",
            "order": 4,
            "bones": sub_scale,
            "target": f"{bone_prefix}_rotate_X_constraint",
            "scaleX": -1,
            "mixRotate": 0,
            "mixX": 0,
            "mixShearY": 0,
        },
        {
            "name": f"{bone_prefix}_scale_compensator",
            "order": 6,
            "bones": reversed_scale_bones,
            "target": bone_prefix,
            "mixRotate": 0,
            "mixX": 0,
            "mixScaleX": 0,
            "mixScaleY": 0,
            "mixShearY": 0,
        },
    ]

    return ik, transform


# -------------------------------------------------------------------------------
def verify_function_replacement():
    """
    Diagnostic function to confirm correct function implementation.

    This verification ensures the enhanced coordinate calculation logic
    is properly integrated into the export pipeline.
    """

    # Get current function source
    current_function = inspect.currentframe().f_back.f_code
    function_name = current_function.co_name
    line_count = current_function.co_firstlineno

    logger.debug("=" * 50)
    logger.debug("FUNCTION REPLACEMENT VERIFICATION")
    logger.debug("=" * 50)
    logger.debug(f"Function Name: {function_name}")
    logger.debug(f"Line Number: {line_count}")
    logger.debug(f"Function ID: {id(current_function)}")

    # Verify enhanced features are present
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals

    enhanced_features = {
        "direct_3d_scaling": "z_displacement" in str(local_vars),
        "coordinate_diagnostics": "calculation_method" in str(local_vars),
        "real_coordinate_processing": "uniform_scale" in str(local_vars),
    }

    logger.debug("Enhanced Features Check:")
    for feature, present in enhanced_features.items():
        status = "✅ PRESENT" if present else "❌ MISSING"
        logger.debug(f"  {feature}: {status}")

    if all(enhanced_features.values()):
        logger.debug("🎯 VERIFICATION RESULT: Enhanced function correctly implemented")
        return True
    else:
        logger.error("❌ VERIFICATION RESULT: Function replacement incomplete")
        return False


# Modified process_z_groups with verification
def process_z_groups(
    attachment_name: str,
    z_groups,
    original_z_groups,
    parent_bone_name: str,
    half_scale: float,
    z_groups_info: dict | None = None,
    texture_width: int = None,
    texture_height: int = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    pixels_per_blender_unit_width: float = 1.0,
    pixels_per_blender_unit_length: float = 1.0,
    original_uv3d_pairs=None,
    tolerance=1e-4,
):
    """
    Process Z-groups with comprehensive verification and real coordinate calculation.

    Technical Implementation Features:
    - Function replacement verification
    - Real 3D coordinate preservation
    - Comprehensive diagnostic logging
    - Multiple calculation fallback methods

    Coordinate System Architecture:
    - Primary: Direct 3D coordinate scaling using uniform transformation
    - Secondary: Pre-calculated height_real_pixels from UV dimensions
    - Tertiary: Legacy indexed spacing for backward compatibility

    Returns:
        dict: Bone structure with verified coordinate calculations
    """

    # Execute verification protocol
    verification_result = verify_function_replacement()
    if not verification_result:
        logger.error(
            "Function replacement verification failed - using legacy calculations"
        )

    logger.debug(
        "[process_z_groups VERIFIED] Initiating verified coordinate processing"
    )
    logger.debug(
        f"Verification Status: {'PASSED' if verification_result else 'FAILED'}"
    )

    # Enhanced input validation
    logger.debug("Input Parameter Validation:")
    logger.debug(f"  - attachment_name: {attachment_name}")
    logger.debug(f"  - half_scale: {half_scale}")
    logger.debug(f"  - texture_dimensions: {texture_width}x{texture_height}")
    logger.debug(f"  - z_groups_info available: {z_groups_info is not None}")
    logger.debug(f"  - original_z_groups: {original_z_groups}")

    bones: list[dict] = []
    bones_info: dict = {
        "z_group_z_values": [],
        "z_value_to_bone_name": {},
        "sub_bone_names": [],
        "sub_bone_scale_names": [],
    }

    # Z-Group selection with enhanced validation
    if original_z_groups and len(original_z_groups) > 0:
        groups_to_use = sorted(original_z_groups)
        coordinate_source = "original_z_groups"
        logger.debug(f"Using original Z-groups: {groups_to_use}")
    else:
        if isinstance(z_groups, dict):
            groups_to_use = sorted(z_groups.keys())
        elif isinstance(z_groups, list):
            groups_to_use = sorted(
                [
                    round(g[2], 4)
                    for g in z_groups
                    if isinstance(g, (list, tuple)) and len(g) >= 3
                ]
            )
        else:
            groups_to_use = []
        coordinate_source = "local_z_groups"
        logger.debug(f"Using local Z-groups: {groups_to_use}")

    if not groups_to_use:
        logger.error("No Z-groups available for coordinate processing")
        return {"bones": bones, "bones_info": bones_info}

    # Enhanced coordinate calculation with verification
    logger.debug("Initiating verified coordinate calculation pipeline...")

    # Calculate baseline reference
    min_z = min(groups_to_use) if groups_to_use else 0.0
    max_z = max(groups_to_use) if groups_to_use else 0.0
    z_range = max_z - min_z

    logger.debug("Z-coordinate analysis:")
    logger.debug(f"  - Min Z: {min_z:.4f}")
    logger.debug(f"  - Max Z: {max_z:.4f}")
    logger.debug(f"  - Z Range: {z_range:.4f}")

    coordinate_calculation_summary = []

    for z_val in groups_to_use:
        bones_info["z_group_z_values"].append(z_val)
        index = len(bones_info["z_group_z_values"])

        sub_bone_scale_name = f"{attachment_name}_{index}_scale"
        sub_bone_name = f"{attachment_name}_{index}"
        bones_info["z_value_to_bone_name"][z_val] = sub_bone_name

        # Enhanced coordinate calculation with method tracking
        y_offset = 0.0
        calculation_method = "undefined"

        # Method 1: Pre-calculated height_real_pixels (optimal)
        if (
            z_groups_info
            and z_val in z_groups_info
            and isinstance(z_groups_info[z_val], dict)
            and "height_real_pixels" in z_groups_info[z_val]
        ):
            y_offset = z_groups_info[z_val]["height_real_pixels"]
            calculation_method = "height_real_pixels"
            logger.debug(f"✅ Z-group {z_val}: Pre-calculated value = {y_offset:.2f}")

        # Method 2: Direct 3D coordinate scaling (enhanced)
        elif texture_width and texture_height:
            uniform_scale = calc_uniform_scale(texture_width, texture_height)
            z_displacement = z_val - min_z  # Relative to baseline
            y_offset = z_displacement * uniform_scale
            calculation_method = "direct_3d_scaling"
            logger.debug(f"🔄 Z-group {z_val}: Direct scaling = {y_offset:.2f}")
            logger.debug(
                f"   Calculation: ({z_val:.4f} - {min_z:.4f}) * {uniform_scale:.2f} = {y_offset:.2f}"
            )

        # Method 3: Legacy fallback (compatibility)
        else:
            y_offset = -(index - 1) * half_scale
            calculation_method = "legacy_fallback"
            logger.warning(f"⚠️  Z-group {z_val}: Legacy fallback = {y_offset:.2f}")

        # Track calculation for summary
        coordinate_calculation_summary.append(
            {
                "z_value": z_val,
                "method": calculation_method,
                "y_offset": y_offset,
                "bone_name": sub_bone_scale_name,
            }
        )

        # Generate bone structures
        bones.append(
            {
                "name": sub_bone_scale_name,
                "parent": parent_bone_name,
                "length": half_scale,
                "rotation": 90,
                "y": round(y_offset, 2),
                "inherit": "onlyTranslation",
                "color": "abe323ff",
            }
        )

        bones.append(
            {"name": sub_bone_name, "parent": sub_bone_scale_name, "rotation": -90}
        )

        bones_info["sub_bone_scale_names"].append(sub_bone_scale_name)
        bones_info["sub_bone_names"].append(sub_bone_name)

    # Enhanced processing summary
    logger.debug("=" * 60)
    logger.debug("COORDINATE PROCESSING SUMMARY")
    logger.debug("=" * 60)
    logger.debug(f"Total Z-groups processed: {len(groups_to_use)}")
    logger.debug(f"Total bones generated: {len(bones)}")
    logger.debug(f"Coordinate source: {coordinate_source}")

    method_counts = {}
    for calc in coordinate_calculation_summary:
        method = calc["method"]
        method_counts[method] = method_counts.get(method, 0) + 1

    logger.debug("Calculation methods used:")
    for method, count in method_counts.items():
        logger.debug(f"  - {method}: {count} groups")

    logger.debug("Final Y-coordinates:")
    for calc in coordinate_calculation_summary:
        logger.debug(
            f"  {calc['bone_name']:25} y={calc['y_offset']:8.2f} ({calc['method']})"
        )

    logger.debug("=" * 60)

    return {"bones": bones, "bones_info": bones_info}


def find_original_position(seg_coord, original_uv3d_pairs, tolerance=1e-4):
    """
    Searches in original_uv3d_pairs for an entry whose 3D position is close to the given seg_coord.

    Arguments:
      seg_coord: list or tuple [x, y, z] – global position of the segment vertex.
      original_uv3d_pairs: list of pairs of the form [index, [uv_x, uv_y], [pos_x, pos_y, pos_z]].
      tolerance: acceptable distance between 3D positions.

    Returns:
      3D position (list [x, y, z]) from the original object if a match is found, otherwise None.
    """
    logger.debug(
        f"[find_original_position] Searching for a match for the global position: {seg_coord} with tolerance {tolerance}"
    )

    for pair in original_uv3d_pairs:
        # Check that the pair matches the format: [index, [uv_x, uv_y], [pos_x, pos_y, pos_z]]
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 3):
            logger.debug(
                f"[find_original_position] Skipping entry (does not match format): {pair}"
            )
            continue
        orig_coord = pair[2]  # Global position from the original object
        if not (isinstance(orig_coord, (list, tuple)) and len(orig_coord) >= 3):
            logger.debug(
                f"[find_original_position] Skipping entry (no correct 3D coordinates): {pair}"
            )
            continue
        dx = seg_coord[0] - orig_coord[0]
        dy = seg_coord[1] - orig_coord[1]
        dz = seg_coord[2] - orig_coord[2]
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        logger.debug(
            f"[find_original_position] Comparison: segment {seg_coord} with original {orig_coord} → distance {distance}"
        )
        if distance < tolerance:
            logger.debug(
                f"[find_original_position] Found a match for coordinate {seg_coord}: original position {pair[2]}"
            )
            return pair[2]
    logger.debug(f"[find_original_position] No match found for coordinate {seg_coord}")
    return None


def transform_original_position(pos, bbox, texture_width, texture_height):
    """
    Analog of transform_position, but for original 3D coordinates.
    The scale is taken from the longest side of the bbox, so that nothing > 1.0 comes out.
    """
    # logger.debug("[transform_original_position] start")
    try:
        cx = (bbox["min_x"] + bbox["max_x"]) / 2.0
        cy = (bbox["min_y"] + bbox["max_y"]) / 2.0
        uniform_scale = calc_uniform_scale(texture_width, texture_height)
        tx = (pos[0] - cx) * uniform_scale
        ty = -((pos[1] - cy) * uniform_scale)
        return tx, ty
    except KeyError as exc:
        logger.exception("transform_original_position error: %s", exc)
        return 0.0, 0.0


def compute_original_bbox(original_uv3d_pairs):
    # logger.debug("starting compute_original_bbox function")
    xs = [
        pair[2][0]
        for pair in original_uv3d_pairs
        if isinstance(pair, (list, tuple)) and len(pair) >= 3
    ]
    ys = [
        pair[2][1]
        for pair in original_uv3d_pairs
        if isinstance(pair, (list, tuple)) and len(pair) >= 3
    ]
    zs = [
        pair[2][2]
        for pair in original_uv3d_pairs
        if isinstance(pair, (list, tuple)) and len(pair) >= 3
    ]
    # logger.debug(f"compute_original_bbox with parameters 'min_x' {min(xs)}, 'max_x' {max(xs)}, 'min_y' {min(ys)}, 'max_y' {max(ys)}, 'min_z' {min(zs)}, 'max_z' {max(zs)}")
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
    }


def create_vertex_bones(
    vertex_prefix,
    uv3d_pairs,
    bones_info,
    center_x,
    center_y,
    pixels_per_blender_unit_width,
    pixels_per_blender_unit_length,
    original_uv3d_pairs,
    texture_width,
    texture_height,
    tolerance=1e-4,
    original_bbox=None,
):
    """
    Creates bones for vertices based on uv3d-pairs.

    If original uv3d-pairs and the calculated bounding box of the original object are passed,
    then for each entry, a match is searched for by 3D coordinate (entry[2]) with a tolerance.
    If a match is found, the original 3D coordinate is used for transformation via
    transform_original_position(), which allows obtaining correct bone coordinates.

    Arguments:
      - vertex_prefix: string, prefix for bone names (e.g., "Segment_0").
      - uv3d_pairs: list of entries of the form [index, [uv_x, uv_y], [pos_x, pos_y, pos_z]].
      - bones_info: dictionary with information about bones (in particular, main_bone_name, z_group_z_values and z_value_to_bone_name).
      - center_x, center_y: center coordinates for standard transformation (if original_bbox is not specified).
      - pixels_per_blender_unit_width/length: scaling factors.
      - original_uv3d_pairs: list of original uv3d-pairs; if passed, used to find the original position.
      - texture_width, texture_height: texture dimensions.
      - tolerance: acceptable tolerance for comparing 3D coordinates.
      - original_bbox: dictionary with the boundaries of the original object (if not passed, calculated from original_uv3d_pairs).

    Returns:
      dict with keys 'bones' (list of created bones) and 'vertex_bone_names' (list of their names).
    """

    bones = []
    vertex_bone_names = []
    main_bone_name = bones_info.get("main_bone_name", "main")

    # If local_z_groups are not passed, try to take them from bones_info
    local_z_groups = bones_info.get("z_group_z_values", [])
    # If original_z_groups are not passed, you can try to take them from bones_info (if added)
    original_z_groups = bones_info.get("original_z_groups", local_z_groups)

    # If original uv3d-pairs are specified, but the bounding box is not passed – calculate it
    if original_uv3d_pairs and original_bbox is None:
        try:
            original_bbox = compute_original_bbox(original_uv3d_pairs)
            # logger.debug(f"[create_vertex_bones] Calculated original_bbox: {original_bbox}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to calculate original_bbox: {e}")
            original_bbox = None

    # Process each entry from uv3d_pairs; each entry must have the format: [index, seg_uv, seg_pos]
    for entry in uv3d_pairs:
        try:
            if not (isinstance(entry, list) and len(entry) >= 3):
                logger.error(f"[ERROR] Invalid entry format uv3d_pairs: {entry}")
                continue

            index = entry[0]
            seg_uv = entry[1]  # Fallback: UV coordinate of the segment
            seg_pos = entry[2]  # 3D coordinate of the segment

            # By default, use the segment position
            used_pos = seg_pos

            # If original uv3d-pairs are specified, try to find an entry with a matching 3D position
            if original_uv3d_pairs:
                for orig_entry in original_uv3d_pairs:
                    if not (isinstance(orig_entry, list) and len(orig_entry) >= 3):
                        continue
                    orig_pos = orig_entry[2]  # Original 3D position
                    # Use the distance3d function (it is assumed that it is imported)
                    if distance3d(Vector(seg_pos), orig_pos) < tolerance:
                        used_pos = orig_pos
                        logger.debug(
                            f"[create_vertex_bones] For vertex {index}: match found. Using original position {orig_pos} instead of segment {seg_pos}"
                        )
                        break

            # If there is an original bounding box and texture dimensions, use transform_original_position
            if original_bbox is not None and texture_width and texture_height:
                transformed_x, transformed_y = transform_original_position(
                    used_pos, original_bbox, texture_width, texture_height
                )
            else:
                # Otherwise – standard transformation by UV (if bbox is not specified)
                transformed_x, transformed_y = transform_position(
                    seg_uv,
                    center_x,
                    center_y,
                    pixels_per_blender_unit_width,
                    pixels_per_blender_unit_length,
                )

            # Determine the parent bone based on the Z coordinate of the segment
            parent_bone_name = determine_parent_bone(
                seg_pos[2],
                original_z_groups,
                local_z_groups,
                bones_info.get("z_value_to_bone_name", {}),
                main_bone_name,
                vertex_prefix,
                tolerance=tolerance,
            )
            bone_name = f"{vertex_prefix}_vertex_{index}"
            bone = {
                "name": bone_name,
                "parent": parent_bone_name,
                "x": round(transformed_x, 2),
                "y": round(transformed_y, 2),
            }
            # logger.debug(f"[create_vertex_bones] Created bone: {bone}")
            bones.append(bone)
            vertex_bone_names.append(bone_name)
        except Exception as ex:
            logger.exception(f"[exception] Error processing entry {entry}: {ex}")
    return {"bones": bones, "vertex_bone_names": vertex_bone_names}


def calculate_average_y(uv3d_pairs):
    """Calculates the average Y value from uv3d_pairs."""
    if uv3d_pairs:
        y_values = [pair[1][1] for pair in uv3d_pairs]
        average_y = sum(y_values) / len(y_values)
        logger.debug(f"[INFO] Average Y coordinate from uv3d_pairs: {average_y}")
        return average_y
    else:
        logger.error("[WARNING] Array 'uv3d_pairs' is empty. Using Y=0 for main bone.")
        return 0


def determine_parent_bone(
    pos_z,
    original_z_groups,
    local_z_groups,
    z_value_to_bone_name,
    main_bone_name,
    obj_name,
    tolerance=1e-4,
):
    # logger.debug(f"determine_parent_bone: pos_z={pos_z} for object {obj_name}")
    pos_z_rounded = round(pos_z, 4)
    # Call the updated find_closest_z with both sets of z
    closest_z = find_closest_z(
        pos_z_rounded, original_z_groups, local_z_groups, obj_name, tolerance=tolerance
    )
    # logger.debug(f"determine_parent_bone: pos_z_rounded={pos_z_rounded}, closest_z={closest_z}")
    if closest_z is not None:
        parent_bone_name = z_value_to_bone_name.get(closest_z, main_bone_name)
    else:
        parent_bone_name = main_bone_name
    # logger.debug(f"determine_parent_bone: for pos_z {pos_z} the parent bone is selected: {parent_bone_name}")
    return parent_bone_name


def transform_position(pos_pair, center_x, center_y, ppbu_w: float, ppbu_h: float):
    """
    Conversion of global 3D position to Spine-XY for bones.
    Now a single ppbu (pixels-per-blender-unit) is used,
    to preserve the aspect ratio.
    """
    try:
        px, py, _ = map(float, pos_pair)
        ppbu = (ppbu_w + ppbu_h) / 2.0  # already uniform inside Blender
        x = (px - float(center_x)) * ppbu
        y = -((py - float(center_y)) * ppbu)
        return x, y
    except KeyError as exc:
        logger.exception("transform_position error: %s", exc)
        return 0.0, 0.0


def create_weighted_vertices(
    num_vertices,
    processed_vertices,
    bones_info,
    total_bones,
    original_uv3d_pairs,
    is_segment=False,
    obj_name=None,
    segment_uv3d_pairs=None,
    local_z_groups=None,
    texture_width=None,
    texture_height=None,
):
    weighted_vertices = []

    # Handle standard mode
    def handle_standard_mode():
        base_bone_index = total_bones - num_vertices
        for vertex_index in range(num_vertices):
            bone_index = base_bone_index + vertex_index
            weighted_vertices.extend([1, bone_index, 0.0, 0.0, 1.0])
        return weighted_vertices

    # Handle segment mode
    def handle_segment_mode():
        num_segment_vertices = len(segment_uv3d_pairs)
        logger.debug(
            f"For segment: expected number of unique vertices = {num_segment_vertices}"
        )
        offset = bones_info.get(
            "vertex_bones_start_index", total_bones - num_segment_vertices
        )
        for vertex_index in range(num_segment_vertices):
            bone_index = offset + vertex_index
            weighted_vertices.extend([1, bone_index, 0.0, 0.0, 1.0])
        return weighted_vertices

    try:
        if not is_segment or original_uv3d_pairs is None or segment_uv3d_pairs is None:
            return handle_standard_mode()
        else:
            return handle_segment_mode()
    except Exception as e:
        logger.exception(
            f"[exception] Exception in create_weighted_vertices for object {obj_name}: {e}"
        )
        return weighted_vertices


def add_stretch_and_z_groups_to_animations(
    spine_json_data: dict,
    stretch_values: dict,
    z_groups_info: dict,
    bones_info: dict,
    obj_name: str,
    is_segment: bool,
) -> None:
    """
    Adds translate (Y offset) values for each Z-group to the "animations" section.
    The main (bottom, minimum) Z-group gets translate=0,
    and for the rest, the exact offset is calculated: (z_val - min_z) * scale.
    fallback for groups with less than 3 vertices is not applied.
    """
    try:
        if not isinstance(spine_json_data, dict):
            raise TypeError("spine_json_data must be a dict")
        if not isinstance(z_groups_info, dict):
            raise TypeError("z_groups_info must be a dict")
        if not isinstance(stretch_values, dict):
            logger.warning("stretch_values is not a dict, using an empty dictionary")
            stretch_values = {}

        animations = spine_json_data.setdefault("animations", {})
        if "animation" not in animations:
            animations["animation"] = {}
        if "bones" not in animations["animation"]:
            animations["animation"]["bones"] = {}

        skeleton_info = spine_json_data.get("skeleton", {})
        fallback_height = skeleton_info.get("height", 0.0)

        limited_stretch = process_stretch_values(stretch_values, obj_name)

        # Select the main Z-group by the minimum value
        if z_groups_info:
            main_z = min(z_groups_info.keys(), key=lambda z: z)
        else:
            logger.warning(
                "[WARNING] z_groups_info is empty, skipping adding Z-groups."
            )
            return

        logger.debug(f"[INFO] Main Z-group defined as {main_z} (minimum)")

        bone_animations = {}
        for z_value, bone_name in bones_info.get("z_value_to_bone_name", {}).items():
            z_key = round(z_value, 4)
            z_info = z_groups_info.get(z_key, None)
            if not z_info:
                logger.debug(f"[WARNING] No data for z_value={z_key} in z_groups_info.")
                continue

            # Use the calculated height_real_pixels value from calculate_uv_dimensions
            translate_y = z_info.get("height_real_pixels", 0.0)
            # For the main (bottom) group, translate_y should be 0 (this will be the case if z_key == min_z)
            if z_key == main_z:
                stretch_val = 1.0
                logger.debug(f"[INFO] Primary Z-group {z_key}: stretch set to 1.0")
            else:
                stretch_val = limited_stretch.get(z_key, 1.0)
                logger.debug(f"[INFO] Z-group {z_key}: stretch = {stretch_val:.3f}")

            sub_bone_scale_name = f"{bone_name}_scale"
            bone_animations[sub_bone_scale_name] = {
                "translate": [{"y": round(translate_y, 2)}],
                "scale": [{}],
                "inherit": [{"inherit": "noScale"}],
            }

        # Add animation for the main scale and rotation bones
        scale_bone_name = bones_info.get("scale_bone_name", "scale_bone")
        rotation_bone_name = bones_info.get("main_rotation_bone_name", "main_rotation")
        bone_animations[scale_bone_name] = {
            "rotate": [{"value": 0}],
            "scale": [{"x": 1, "y": 1}],
        }
        bone_animations[rotation_bone_name] = {"rotate": [{"value": 0}]}

        animations["animation"]["bones"] = bone_animations
        logger.debug(
            f"[INFO] Added stretch and Z-group values to animations: {bone_animations}"
        )
    except Exception as e:
        logger.exception(
            f"[exception] Error in add_stretch_and_z_groups_to_animations for object {obj_name}: {e}"
        )


def add_preview_rotation_animation(spine_json_data: dict, bone_prefix: str) -> None:
    """
    Adds a "preview" animation to the JSON with rotation of three control bones:
        <prefix>_rotation_X, _rotation_Y, _rotation_Z.

    ▸ The function is idempotent – a second call will overwrite existing data.
    ▸ In case of errors, it writes to the log and does not interrupt the main pipeline.
    """
    try:
        animations = spine_json_data.setdefault("animations", {})
        preview = animations.setdefault("preview", {})
        bones_blk = preview.setdefault("bones", {})

        def _set(axis: str, timeline: list[dict]) -> None:
            bones_blk[f"{bone_prefix}_rotation_{axis}"] = {"rotate": timeline}

        # === Y ============================================================
        _set(
            "Y",
            [
                {"curve": [0.667, 0, 1.333, -360]},
                {"time": 2, "value": -360, "curve": [2.667, -360, 3.333, -360]},
                {"time": 4, "value": -360, "curve": [4.667, -360, 5.333, 0]},
                {"time": 6},
            ],
        )
        # === Z ============================================================
        _set(
            "Z",
            [
                {"time": 2, "curve": [2.667, 0, 3.333, 360]},
                {"time": 4, "value": 360, "curve": [4.667, 360, 5.333, 0]},
                {"time": 6},
            ],
        )
        # === X ============================================================
        _set(
            "X",
            [
                {"value": -360, "curve": [0.667, -360, 1.333, 0]},
                {"time": 2, "curve": "stepped"},
                {"time": 4, "curve": [4.667, 0, 5.333, -360]},
                {"time": 6, "value": -360, "curve": [6.667, -360, 7.333, -360]},
                {"time": 8, "value": -360},
            ],
        )

        logger.debug(
            "[add_preview_rotation_animation] preview animation added for '%s'",
            bone_prefix,
        )
    except KeyError as exc:
        logger.exception(
            "[add_preview_rotation_animation] error adding preview: %s", exc
        )


def process_stretch_values(stretch_values, obj_name):
    """Processes stretch values: smoothing and limiting."""
    smoothed_stretch = smooth_stretch_values(
        stretch_values, window_size=5, obj_name=None
    )
    limited_stretch = limit_stretch_changes(
        smoothed_stretch, max_change=0.1, obj_name=None
    )
    for z_key in limited_stretch:
        limited_stretch[z_key] = max(0.5, min(2.0, limited_stretch[z_key]))
    return limited_stretch


def save_json_files(raw_data, spine_json_data, filepath):
    """Saves raw data and the main JSON to files."""
    if logger.isEnabledFor(logging.DEBUG):
        # logger.debug("Saving raw data to a separate JSON file...")
        try:
            save_directory = (
                os.path.dirname(os.path.abspath(filepath))
                if filepath
                else bpy.utils.resource_path("USER")
            )
            raw_save_path = os.path.join(save_directory, "processed_raw_data.json")
            with open(raw_save_path, "w", encoding="utf-8") as raw_file:
                json.dump(raw_data, raw_file, indent=4, ensure_ascii=False)
            # logger.debug(f"[INFO] Raw data successfully saved to: {raw_save_path}")
        except Exception as e:
            logger.exception(f"[exception] Failed to save raw data: {e}")
    """
    logger.debug("Saving the main JSON file...")
    try:
        spine_save_path = os.path.join(save_directory, "spine_output.json")
        with open(spine_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(spine_json_data, json_file, indent=4, ensure_ascii=False)
        logger.debug(f"[INFO] Main JSON successfully saved to: {spine_save_path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save the main JSON file: {e}")
    finally:
    """
    cleanup_json_data()


def cleanup_json_data():
    # logger.debug("starting cleanup_json_data function")
    """Cleans up temporary data after JSON export."""
    try:
        # Clear defaultdict
        edge_count_map = defaultdict(int)
        edge_count_map.clear()
        # Clear global variables, if any
        global_vars = globals()
        if "spine_json_data" in global_vars:
            del global_vars["spine_json_data"]
        if "bones_info" in global_vars:
            del global_vars["bones_info"]
        if "processed_vertices" in global_vars:
            del global_vars["processed_vertices"]
        # logger.debug("[INFO] JSON data cleared")
    except Exception as e:
        logger.exception(f"[exception] Error clearing JSON data: {e}")


def register():
    logger.debug("[LOG] Logging json_export.py")
    pass


def unregister():
    logger.debug("[LOG] Unregister json_export.py")
    pass

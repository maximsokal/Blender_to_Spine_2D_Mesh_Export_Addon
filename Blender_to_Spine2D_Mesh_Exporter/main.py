# main.py
# pylint: disable=import-error
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
This file acts as the main orchestrator for the entire 'Blender_to_Spine2D_Mesh_Exporter' export pipeline.
It coordinates the process from the initial object selection to the final merged JSON output.

Its key responsibilities are:
1.  Main Pipeline Execution: The `save_uv_as_json` function defines the complete, step-by-step export process. This includes preprocessing the object, creating a working copy, performing UV unwrapping, cutting the mesh into segments, baking textures, exporting each segment individually, and finally merging the results.
2.  Object Preprocessing: It calls functions from other modules to prepare the source object. This involves applying transformations, creating a clean and optimized UV layout (`main_preprocessing`), and calculating initial mesh dimensions.
3.  Mesh Segmentation: It integrates the `plane_cut` module to slice the mesh into manageable segments based on user-defined seams or automatic angle detection. It then applies these cuts as seams to ensure the UV unwrap respects the segment boundaries.
4.  Texture Baking Integration: It orchestrates the texture baking process by calling `bake_textures_for_object`. It prepares a dedicated "texturing object" with a new UV layout, bakes the textures from the original object onto it, and then transfers the newly baked UVs back to all the individual mesh segments.
5.  Segment-by-Segment Export: It iterates through the original object and all its generated segments, calling the `_export_segment` function for each one. This function generates an intermediate JSON file for every piece.
6.  Final JSON Merging: After all segments are exported, it uses the `json_merger` module to combine the main object's JSON with all the segment JSONs into a single, final file.
7.  Blender Operator: It defines the `OBJECT_OT_SaveUVAsJSON` operator, which is the button the user clicks in the UI. This operator reads settings from the UI (like texture size), sets up the environment, and triggers the main `save_uv_as_json` pipeline.

ATTENTION: - This file is the "brain" of the addon, and the sequence of operations within `save_uv_as_json` is critical. Altering the order of preprocessing, cutting, baking, and exporting will break the workflow. The process relies on creating and managing several temporary objects (e.g., `_copy_for_uv`, `_texturing_copy`); ensuring these are correctly handled and cleaned up is essential to avoid cluttering the user's scene. The logic for transferring UVs between the original, the texturing copy, and the final segments is a key and delicate part of the process.
Author: Maxim Sokolenko
"""
import bpy
import logging

logger = logging.getLogger(__name__)
import json
import os
import bmesh
from collections import defaultdict, deque
from mathutils import Vector, Euler
from math import radians, sqrt

import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # for older Python versions – ignore

from . import plane_cut  # Import the plugin for cutting the mesh
from .config import (
    TEXTURE_WIDTH,
    TEXTURE_HEIGHT,
    get_default_output_dir,
)  # Import the configuration
from .uv_operations import (
    reset_uv,
    calculate_uv_dimensions,
    recenter_uv,
    smart_uv_project,
    unwrap_respecting_seams,
    rescale_uv_islands,
    transfer_baked_uvs_to_segments,
    transfer_uv_islands_between_objects,
)
from .json_export import check_euler_characteristic, process_raw_data
from .utils import smooth_stretch_values, limit_stretch_changes
from .json_merger import merge_spine_json_dicts, write_json
from .seam_marker import mark_seams_on_copy
from .texture_baker_integration import bake_textures_for_object


# -----------------------------------------------------------------------------
# Helper: apply segmentation seams to an object
# -----------------------------------------------------------------------------
def apply_segmentation_seams(
    obj: bpy.types.Object, segmentation_data: list[tuple[int, int]]
):
    """
    Marks seam edges on the given mesh object using segmentation edge data.

    Plane cutting produces a list of edge pairs (vertex indices) representing
    the borders between segments.  To ensure subsequent UV unwrap operations
    respect these boundaries, seam flags must be applied on the object that
    will be unwrapped (e.g. the copy used for calculating UV dimensions).

    Args:
        obj: The mesh object on which to mark seams.
        segmentation_data: A list of pairs of vertex indices representing
            boundary edges between segments.
    """
    try:
        if not segmentation_data:
            return
        # Build a set of frozensets for fast membership checks
        seg_edges = {frozenset(pair) for pair in segmentation_data}
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        for edge in bm.edges:
            verts = frozenset([edge.verts[0].index, edge.verts[1].index])
            if verts in seg_edges:
                edge.seam = True
        bm.to_mesh(obj.data)
        bm.free()
    except Exception as exc:
        logger.exception(
            f"[apply_segmentation_seams] Failed to mark seams on {obj.name}: {exc}"
        )


def assign_face_ids(obj: bpy.types.Object):
    face_id_map = {str(poly.index): poly.index for poly in obj.data.polygons}
    obj.data["face_id_map"] = face_id_map
    logger.debug(
        f"[assign_face_ids] Assigned face_id to {len(obj.data.polygons)} object faces'{obj.name}'"
    )


# If your texturing object (texturing_obj) does not contain the orig_face_id layer,
# you can call this function immediately after creating a copy of the texturing object:
def copy_orig_face_id_layer(source_obj, target_obj):
    bm_source = bmesh.new()
    bm_source.from_mesh(source_obj.data)

    orig_layer = bm_source.faces.layers.int.get("orig_face_id")
    if orig_layer is None:
        logger.warning("The original BMesh is missing the orig_face_id layer")
        bm_source.free()
        return

    bm_target = bmesh.new()
    bm_target.from_mesh(target_obj.data)

    target_layer = bm_target.faces.layers.int.new("orig_face_id")

    for f_src, f_tgt in zip(bm_source.faces, bm_target.faces):
        f_tgt[target_layer] = f_src[orig_layer]

    bm_target.to_mesh(target_obj.data)
    bm_source.free()
    bm_target.free()

    logger.debug(
        "Layer orig_face_id successfully copied from source_obj to target_obj."
    )


def check_active_object():
    # Checks that the active object exists and is a mesh.
    obj = bpy.context.active_object
    if obj is None:
        raise Exception("There is no active object. Please select an object.")
    if obj.type != "MESH":
        raise Exception(f"The active object is not a mesh. Object type: {obj.type}")
    return obj


def apply_transformations(obj):
    """Applies the scale and rotation of the object."""
    bpy.ops.object.mode_set(mode="OBJECT")  # Make sure we are in object mode
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    logger.debug("Transformations (scale and rotation) applied.")


def set_top_view_ortho(obj):
    """Switches the view to top orthographic."""
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    region_3d = space.region_3d
                    region_3d.view_perspective = "ORTHO"
                    euler = Euler((radians(90), 0, 0), "XYZ")
                    region_3d.view_rotation = euler.to_quaternion()
                    bbox_corners = [
                        obj.matrix_world @ Vector(corner) for corner in obj.bound_box
                    ]
                    bbox_min = Vector(
                        (
                            min(v.x for v in bbox_corners),
                            min(v.y for v in bbox_corners),
                            min(v.z for v in bbox_corners),
                        )
                    )
                    bbox_max = Vector(
                        (
                            max(v.x for v in bbox_corners),
                            max(v.y for v in bbox_corners),
                            max(v.z for v in bbox_corners),
                        )
                    )
                    bbox_size = bbox_max - bbox_min
                    max_dimension = max(bbox_size.x, bbox_size.y, bbox_size.z)
                    region_3d.view_distance = max_dimension * 2
                    region_3d.view_location = (bbox_min + bbox_max) / 2
                    logger.debug("The view has been switched to top orthographic.")
                    return
    raise Exception("Could not find VIEW_3D region to switch view to.")


def get_texture_dimensions(obj, default_texture_width, default_texture_height):
    """Gets the texture dimensions from the object's material or uses default values."""
    texture_width = default_texture_width
    texture_height = default_texture_height
    try:
        if obj.active_material:
            nodes = obj.active_material.node_tree.nodes
            for node in nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    texture_width = node.image.size[0]
                    texture_height = node.image.size[1]
                    logger.debug(
                        f"[INFO] Texture size: {texture_width}x{texture_height}"
                    )
                    break
        else:
            logger.debug("[INFO] Texture not found, using default values.")
    except (AttributeError, KeyError, RuntimeError) as e:
        logger.debug("[exception] Failed to get texture size: %s", e)
        logger.debug("[INFO] Default values are used.")
    finally:
        return texture_width, texture_height


def triangulate_mesh(obj):
    """Creates a BMesh and triangulates it."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        bmesh.ops.triangulate(
            bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY"
        )
    except Exception as e:
        logger.exception(f"[exception] Failed to triangulate mesh: {e}")
        bm.free()
        raise e
    return bm


def collect_vertices(bm, obj_name):
    try:
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            logger.error("[ERROR] UV layer not found after triangulation.")
            bm.free()
            raise Exception("UV layer not found after triangulation.")
        logger.debug(f"[INFO] UV layer '{uv_layer.name}' found.")
        triangles = []
        vertex_map = {}
        vertex_list = []
        vertex_positions_list = []
        logger.debug(
            "Assembling unique vertices, UV coordinates and positions in 3D space..."
        )
        for face in bm.faces:
            face_verts = []
            for loop in face.loops:
                uv = loop[uv_layer].uv
                # Change the key: add the vertex index to distinguish vertices with the same UVs
                key = (round(uv.x, 6), round(uv.y, 6), loop.vert.index)
                if key not in vertex_map:
                    vertex_index = len(vertex_list)
                    vertex_map[key] = vertex_index
                    vertex_list.append((uv.x, uv.y))
                    vertex_positions_list.append(loop.vert.co.copy())
                face_verts.append(vertex_map[key])
            triangles.extend(face_verts)
        num_triangles = len(triangles) // 3
        logger.debug(f"Number of unique vertices: {len(vertex_list)}")
        logger.debug(f"Number of triangles: {num_triangles}")
        if len(triangles) % 3 != 0:
            logger.error(
                f"[ERROR] The list 'triangles' contains {len(triangles)} elements, which is not a multiple of 3."
            )
            bm.free()
            raise Exception(
                "The list 'triangles' contains an incorrect number of elements."
            )
        return vertex_list, vertex_positions_list, triangles, obj_name
    finally:
        bm.free()


def extract_and_classify_edges(triangles, obj_name):
    """Extracts triangular edges and classifies them into external and internal."""
    logger.debug(f"[INFO] Extracting edges for an object: {obj_name}")
    triangle_edges = []
    for i in range(0, len(triangles), 3):
        a, b, c = triangles[i], triangles[i + 1], triangles[i + 2]
        triangle_edges.extend([(a, b), (b, c), (c, a)])
    edge_count_map = defaultdict(int)
    for edge in triangle_edges:
        sorted_edge = tuple(sorted(edge))
        edge_count_map[sorted_edge] += 1
    outer_edges = [edge for edge, count in edge_count_map.items() if count == 1]
    inner_edges = [edge for edge, count in edge_count_map.items() if count > 1]
    logger.debug(f"External edges found: {len(outer_edges)}")
    logger.debug(f"Internal edges found: {len(inner_edges)}")
    return outer_edges, inner_edges, edge_count_map, obj_name


def check_mesh_manifoldness(edge_count_map):
    """Checks the manifoldness of the mesh."""
    manifold = True
    for edge, count in edge_count_map.items():
        if count > 2:
            logger.error(f"[ERROR] Edge {edge} belongs to {count} triangles.")
            manifold = False
    if not manifold:
        logger.error("[ERROR] Mesh is not manifold. Export cancelled.")
        raise Exception("The mesh is not manifold.")
    else:
        logger.debug("[INFO] The mesh is manifold.")


# Insert this code into main.py, replacing the existing function
def order_outer_contour_vertices(outer_edges, is_original_object=False, object_name=""):
    """
    the function does not crash with an error in case of complex topology, but tries to find
    the largest contour and work with it, issuing a warning.
    """
    logger.debug(f"Ordering the vertices of the outer contour for {object_name}...")

    if not outer_edges:
        logger.warning(
            f"[{object_name}] There are no external edges to construct the outline. "
        )
        return []

    adjacency = defaultdict(list)
    for a, b in outer_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    # Check for vertices with incorrect degree
    bad_degree_verts = [v for v, n in adjacency.items() if len(n) != 2]

    if bad_degree_verts:
        logger.warning(
            f"[{object_name}]: Incorrect contour topology detected!"
            f"Vertices with degree != 2: {[v for v, n in adjacency.items() if len(n) != 2]}. "
            f"The algorithm will try to find the largest outer contour."
        )

        # [NEW] Try to find all cycles and choose the longest one
        all_loops = []
        visited_nodes = set()

        for start_node in adjacency:
            if start_node not in visited_nodes:
                # Simple traversal to find one of the cycles
                q = deque([(start_node, [start_node])])
                visited_this_cycle = {start_node}

                found_cycle = False
                while q:
                    curr, path = q.popleft()

                    for neighbor in adjacency[curr]:
                        if neighbor == start_node and len(path) > 2:
                            all_loops.append(path)
                            visited_nodes.update(path)
                            found_cycle = True
                            break

                        if neighbor not in visited_this_cycle:
                            visited_this_cycle.add(neighbor)
                            new_path = path + [neighbor]
                            q.append((neighbor, new_path))
                    if found_cycle:
                        break

        if not all_loops:
            logger.error(
                f"[{object_name}] No closed loops could be extracted due to the complex topology."
            )
            # Return all vertices as a fallback to avoid a complete crash
            return list(adjacency.keys())

        # Choose the longest contour as the main one
        ordered_vertices = max(all_loops, key=len)
        logger.info(
            f"[{object_name}] The longest contour of {len(ordered_vertices)} vertices out of {len(all_loops)} found is selected."
        )

    else:
        # Old, fast logic for "good" contours
        start_vertex = min(adjacency.keys())
        ordered_vertices = []
        visited = set()
        current_vertex = start_vertex

        while len(ordered_vertices) < len(adjacency):
            ordered_vertices.append(current_vertex)
            visited.add(current_vertex)

            next_vertex = None
            for neighbor in adjacency[current_vertex]:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break

            if next_vertex is None:
                # Check for closure
                if start_vertex in adjacency[current_vertex] and len(
                    ordered_vertices
                ) == len(adjacency):
                    break
                else:
                    logger.error(
                        f"Traversal of contour '{object_name}' was aborted. The next unvisited node could not be found."
                    )
                    break

            current_vertex = next_vertex

    logger.debug(
        f"[{object_name}] Ordered vertices of the outer contour: {len(ordered_vertices)} pcs."
    )
    return ordered_vertices


def rebuild_vertex_lists(
    ordered_vertices, vertex_list, vertex_positions_list, triangles
):
    """Rebuilds the lists of vertices, UVs, and positions."""
    new_vertex_list = []
    new_uvs = []
    new_vertex_positions_list = []
    vertex_remap = {}
    index = 0
    # Add vertices of the outer contour
    for v in ordered_vertices:
        coord = vertex_list[v]
        position = vertex_positions_list[v]
        new_vertex_list.append(coord)
        new_uvs.extend(coord)
        new_vertex_positions_list.append(position)
        vertex_remap[v] = index
        index += 1
    hull_vertex_count = index
    # Add the remaining vertices
    for i, coord in enumerate(vertex_list):
        if i not in vertex_remap:
            new_vertex_list.append(coord)
            new_uvs.extend(coord)
            new_vertex_positions_list.append(vertex_positions_list[i])
            vertex_remap[i] = index
            index += 1
    # Update indices in triangles
    new_triangles = [vertex_remap[i] for i in triangles]
    return (
        new_vertex_list,
        new_uvs,
        new_vertex_positions_list,
        new_triangles,
        hull_vertex_count,
        vertex_remap,
    )


def transform_uv_to_vertices(
    uv_list: list, texture_width: float, texture_height: float, flip_uv: bool = True
) -> list:
    """
    From a list of [[u,v], …] makes [x,y,…] for intermediate calculations.

    The logic is 1-to-1 with json_export.process_uvs, so that all data
    lives in the same coordinate system.
    """
    if not uv_list:
        return []

    out: list[float] = []
    for idx, (u, v) in enumerate(uv_list):
        try:
            u = float(u)
            v = float(v)

            x = (u - 0.5) * texture_width
            y = (0.5 - v) * texture_height if flip_uv else (v - 0.5) * texture_height

            out.extend([round(x, 3), round(y, 3)])
        except (RuntimeError, ValueError) as exc:
            logger.warning("[transform_uv_to_vertices] bad pair #%d: %s", idx, exc)

    logger.debug("[transform_uv_to_vertices] done, verts=%d", len(out) // 2)
    return out


def update_edges_indices(outer_edges, inner_edges, vertex_remap):
    """Updates the edge indices."""
    edges_indices = []
    for edge in outer_edges + inner_edges:
        a, b = vertex_remap[edge[0]], vertex_remap[edge[1]]
        edges_indices.extend([a * 2, b * 2])
    return edges_indices


def compute_bounding_rectangle(vertices):
    """Calculates the bounding rectangle."""
    min_x, min_y = min(vertices[::2]), min(vertices[1::2])
    max_x, max_y = max(vertices[::2]), max(vertices[1::2])
    width, height = max_x - min_x, max_y - min_y
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    return center_x, center_y, width, height


def group_positions_by_z(new_vertex_positions_list, obj_name):
    # logger.debug("starting group_positions_by_z function")
    # logger.debug(f"[INFO] Grouping positions for object: {obj_name}")
    positions_by_z = defaultdict(list)
    for position in new_vertex_positions_list:
        x, y, z = position.x, position.y, position.z
        z_key = round(z, 4)
        positions_by_z[z_key].append([x, y, z])
    # Instead of forming a list, return a dict
    # logger.debug(f"Grouping by Z completed. {dict(positions_by_z)}")
    return dict(positions_by_z)


def form_uv3d_pairs(new_vertex_list, new_vertex_positions_list):
    """
    Forms an array of uv3d-pairs for the original object.
    For each vertex, an entry is created:
      [index, [uv_x, uv_y], [pos.x, pos.y, pos.z]]
    """
    # logger.debug("Starting form_uv3d_pairs function")

    # Check input data
    if new_vertex_list is None or new_vertex_positions_list is None:
        logger.error("One of the input lists is None.")
        return []
    if len(new_vertex_list) != len(new_vertex_positions_list):
        logger.error(
            "The number of UV coordinates does not match the number of positions."
        )
        return []

    uv3d_pairs = []
    for i, (uv, pos) in enumerate(zip(new_vertex_list, new_vertex_positions_list)):
        # Check UV format
        if not (isinstance(uv, (list, tuple)) and len(uv) == 2):
            logger.warning(f"Invalid UV format for vertex {i}: {uv}")
            continue
        try:
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
        except Exception:
            logger.exception(f"Error extracting coordinates for vertex {i}: {pos}")
            continue
        uv3d_pairs.append([i, [uv[0], uv[1]], [x, y, z]])

    # logger.debug(f"Formed uv3d_pairs array: {uv3d_pairs}")
    return uv3d_pairs


def update_global_uv3d_pairs(global_uv3d_pairs, segment_uv3d_pairs, tolerance=1e-4):
    # logger.debug("Starting update_global_uv3d_pairs function")
    # [FIXED] Sort segment_uv3d_pairs by index so that the order corresponds to the original numbering
    segment_uv3d_pairs = sorted(segment_uv3d_pairs, key=lambda pair: pair[0])
    for seg_entry in segment_uv3d_pairs:
        if not (isinstance(seg_entry, list) and len(seg_entry) >= 3):
            continue
        seg_uv = seg_entry[1]
        seg_pos = seg_entry[2]
        for global_entry in global_uv3d_pairs:
            if not (isinstance(global_entry, list) and len(global_entry) >= 3):
                continue
            global_uv = global_entry[1]
            if not (isinstance(global_uv, (list, tuple)) and len(global_uv) >= 2):
                continue
            if (
                abs(global_uv[0] - seg_uv[0]) < tolerance
                and abs(global_uv[1] - seg_uv[1]) < tolerance
            ):
                if len(global_entry) < 4:
                    global_entry.append(seg_pos)
                else:
                    if seg_pos not in global_entry[3:]:
                        global_entry.append(seg_pos)
                break
    logger.debug(f"[DEBUG] Global array uv3d_pairs updated: {global_uv3d_pairs}")
    return global_uv3d_pairs


def form_raw_data(
    new_uvs,
    new_triangles,
    vertices,
    hull_vertex_count,
    edges_indices,
    z_groups,
    uv3d_pairs,
    obj_name,
):
    # logger.debug("Starting form_raw_data function")
    raw_data = {
        "object_name": obj_name,  # Object name
        "uvs": new_uvs,
        "triangles": new_triangles,
        "vertices": vertices,
        "hull": hull_vertex_count,
        "edges": edges_indices,
        "z_groups": z_groups,
        "uv3d_pairs": uv3d_pairs,
    }
    # logger.debug(f"Raw data formed: {raw_data}")
    return raw_data


def calculate_stretch_values(obj):
    # logger.debug("starting calculate_stretch_values function")
    """Calculates Stretch values for the mesh."""
    # logger.debug("=== Start of Stretch value calculation ===")
    bm_stretch = bmesh.new()
    bm_stretch.from_mesh(obj.data)
    uv_layer_stretch = bm_stretch.loops.layers.uv.active
    if uv_layer_stretch is None:
        logger.debug("No active UV layer found for Stretch calculation.")
        bm_stretch.free()
        return None
    else:
        vertex_groups = defaultdict(list)
        precision = 4  # Reduced precision to 4 decimal places
        for vert in bm_stretch.verts:
            z_key = round(vert.co.z, precision)
            vertex_groups[z_key].append(vert)
        processed_faces = set()
        group_stretch_values = defaultdict(lambda: {"area_3d": 0, "area_uv": 0})
        min_area_threshold = 1e-4  # Increased minimum area threshold
        for z_key, vertices_list in vertex_groups.items():
            for vert in vertices_list:
                for face in vert.link_faces:
                    if face.index in processed_faces:
                        continue
                    processed_faces.add(face.index)
                    face_z_values = [round(v.co.z, precision) for v in face.verts]
                    if all(z == face_z_values[0] for z in face_z_values):
                        face_z_key = face_z_values[0]
                    else:
                        face_z_key = round(
                            sum(face_z_values) / len(face_z_values), precision
                        )
                    face_area_3d = face.calc_area()
                    face_area_uv = 0.5 * abs(
                        sum(
                            (
                                face.loops[i][uv_layer_stretch].uv.x
                                * face.loops[i - 1][uv_layer_stretch].uv.y
                                - face.loops[i - 1][uv_layer_stretch].uv.x
                                * face.loops[i][uv_layer_stretch].uv.y
                            )
                            for i in range(len(face.loops))
                        )
                    )
                    if (
                        face_area_3d > min_area_threshold
                        and face_area_uv > min_area_threshold
                    ):
                        group_stretch_values[face_z_key]["area_3d"] += face_area_3d
                        group_stretch_values[face_z_key]["area_uv"] += face_area_uv
                    else:
                        logger.warning(
                            f"[WARNING] Face {face.index} has too small areas: 3D={face_area_3d}, UV={face_area_uv}"
                        )
                        continue
        final_stretch_values = {}
        for z_key, areas in group_stretch_values.items():
            if (
                areas["area_uv"] > min_area_threshold
                and areas["area_3d"] > min_area_threshold
            ):
                stretch = sqrt(areas["area_3d"] / areas["area_uv"])
                final_stretch_values[z_key] = stretch
                logger.debug(
                    f"[DEBUG] Group Z = {z_key}: area_uv = {areas['area_uv']:.6f}, area_3d = {areas['area_3d']:.6f}, raw_stretch = {stretch:.3f}"
                )
            else:
                logger.warning(
                    f"[WARNING] Z-group {z_key} has too small areas. stretch is set to 1.0"
                )
                final_stretch_values[z_key] = 1.0
        smoothed_stretch = smooth_stretch_values(final_stretch_values, window_size=5)
        limited_stretch = limit_stretch_changes(smoothed_stretch, max_change=0.1)
        for z_key in limited_stretch:
            limited_stretch[z_key] = max(0.5, min(2.0, limited_stretch[z_key]))
            logger.debug(
                f"Final Group Z = {z_key}: Stretch = {limited_stretch[z_key]:.3f}"
            )
        for z_key, stretch in sorted(limited_stretch.items()):
            logger.debug(
                f"Smoothed & Limited Group Z = {z_key}: Stretch = {stretch:.3f}"
            )
        bm_stretch.free()
        logger.info("=== End of Stretch value calculation ===")
        return limited_stretch


def main_preprocessing(obj):
    # logger.debug("starting main_preprocessing_v3 function")
    """
    Main preprocessing function.
    Creates a correct base UV unwrap, and then applies custom
    scaling to equalize texel density.
    """
    try:
        obj_name = obj.name
        apply_transformations(obj)

        # 1. Start with a clean UV layer.
        reset_uv(obj, obj_name)

        # 2. Create a quality base unwrap using Smart UV Project.
        #    This is the most reliable way to get undistorted islands for further work.
        angle_limit = getattr(bpy.context.scene, "spine2d_angle_limit", 66.0)
        smart_uv_project(obj, obj_name, angle_limit=angle_limit)

        # 3. Now that we have good islands, call YOUR IMPROVED
        #    function to equalize texel density.
        #    We no longer pass pre_unwrap_info, as it is not needed.
        rescale_uv_islands(obj)

        # 4. Center the final result.
        recenter_uv(obj)

        # 5. After all UV manipulations, calculate the final dimensions once.
        # logger.debug("\nPerforming calculations after final UV unwrap...")
        post_unwrap_info = calculate_uv_dimensions(
            obj, obj_name, TEXTURE_WIDTH, TEXTURE_HEIGHT, pre_unwrap=False
        )

        if not post_unwrap_info:
            logger.error(
                f"Unable to calculate dimensions for {obj.name} after unwrapping."
            )
            return None

        for z_val, info in sorted(post_unwrap_info.get("z_groups_info", {}).items()):
            logger.debug(
                f"Z-coordinate: {z_val}, "
                f"Real width (px): {info['width_real_pixels']:.2f}, "
                f"Real length (px): {info['length_real_pixels']:.2f}"
            )

        return post_unwrap_info
    except Exception as e:
        logger.exception(f"exception Error in object preprocessing {obj.name}: {e}")
        return None


def merge_uv3d_pairs(original_uv3d_pairs, new_uv3d_pairs, tolerance=1e-4):
    """
    Merges two lists of uv3d-pairs.
    For each element from new_uv3d_pairs, it checks if there is already a pair with similar UV coordinates in original_uv3d_pairs.
    If not, it adds it.
    Returns the merged list.
    Format of each entry: [index, [uv_x, uv_y], [original pos], (additional segment pos)...]
    """
    from math import sqrt

    def uv_distance(uv1, uv2):
        du = uv1[0] - uv2[0]
        dv = uv1[1] - uv2[1]
        return sqrt(du * du + dv * dv)

    if original_uv3d_pairs is None:
        return new_uv3d_pairs.copy()

    merged = original_uv3d_pairs.copy()
    for new_pair in new_uv3d_pairs:
        new_uv = new_pair[1]  # use the second position, where the uvs are
        found = False
        for orig_pair in merged:
            orig_uv = orig_pair[1]
            if uv_distance(new_uv, orig_uv) < tolerance:
                found = True
                break
        if not found:
            merged.append(new_pair)
    return merged


def delete_segment_artifacts(
    segment_objs: list[bpy.types.Object], original_obj: bpy.types.Object
) -> None:
    """
    Deletes all temporary segment objects and associated JSON files/debug
    data created during the export pipeline.

    Parameters
    ---------
    segment_objs : list[bpy.types.Object]
        A list of segment objects obtained immediately after cutting (plane-cut).
        Includes the original object (or its copy), which should not be deleted.
    original_obj : bpy.types.Object
        An instance of the original Blender object (or its UV copy).
    """
    logger = logging.getLogger(__name__)
    try:
        orig_name = original_obj.name
        # remove the standard suffix to get "Plane", "Cone", etc.
        base_name = (
            orig_name[: -len("_copy_for_uv")]
            if orig_name.endswith("_copy_for_uv")
            else orig_name
        )

        # ------------------------------------------------------------------
        # 1) Delete all segment objects in the scene
        # ------------------------------------------------------------------
        deleted_count = 0
        for seg in list(segment_objs):
            # the original object should not be touched
            if seg.name == original_obj.name:
                continue
            try:
                bpy.data.objects.remove(seg, do_unlink=True)
                deleted_count += 1
                # logger.debug(f"[delete_segment_artifacts] Object deleted: {seg.name}")
            except Exception as obj_err:
                logger.exception(
                    f"[delete_segment_artifacts] Failed to delete "
                    f"object {seg.name}: {obj_err}"
                )

        logger.info(
            f"[delete_segment_artifacts] Segment objects removed: {deleted_count}"
        )

        # ------------------------------------------------------------------
        # 2) Delete file artifacts (*.json, *_DEBUG.json, *_raw.json)
        # ------------------------------------------------------------------
        blend_filepath = bpy.data.filepath
        search_dir = (
            os.path.dirname(blend_filepath)
            if blend_filepath
            else os.path.dirname(__file__)
        )

        removed_files = 0
        patterns = (
            "_spine_output.json",  # main JSON of each segment
            "_DEBUG.json",  # debug JSONs
            "_raw.json",  # raw data files
        )

        for fname in os.listdir(search_dir):
            # search for files containing "<base>_Segment_" and one of the patterns
            if f"{base_name}_Segment_" not in fname:
                continue
            if not any(fname.endswith(pat) for pat in patterns):
                continue
            full_path = os.path.join(search_dir, fname)
            try:
                os.remove(full_path)
                removed_files += 1
                logger.debug(f"[delete_segment_artifacts] File removed: {fname}")
            except Exception as file_err:
                logger.exception(
                    f"[delete_segment_artifacts] Failed to delete "
                    f"file {fname}: {file_err}"
                )

        logger.info(f"[delete_segment_artifacts] Files removed: {removed_files}")

    except RuntimeError as exc:
        logger.exception(f"[delete_segment_artifacts] Common error: {exc}")


# =============================================================================
# Export one (any) segment/object → dict
# =============================================================================
def _export_segment(
    seg_obj,
    original_object_name,
    textured_obj,
    TEXTURE_WIDTH,
    TEXTURE_HEIGHT,
    original_uv3d_pairs,
    textured_uv3d_pairs_global,
    z_groups_info,
    original_z_groups,
    output_dir,
    local_z_groups=None,
    world_location=None,
):
    """
    Exports the given object (either a segment) to an intermediate Spine JSON (dict).
    If it is the original object, we use z_groups_info to write animations.
    If it is a segment, we pass z_groups_info={} (or None) so as not to call the Z animation logic.
    """
    # logger.debug("Starting _export_segment")
    output_filepath = os.path.join(output_dir, f"{seg_obj.name}_spine_output.json")
    logger.debug(f"[SegmentExport] Start of segment export: {seg_obj.name}")
    # logger.debug(f"[SegmentExport] original_z_groups: {original_z_groups}")

    if not seg_obj.data.polygons:
        logger.warning(
            f"[SegmentExport] Object {seg_obj.name} does not contain polygons. Skipping."
        )
        return None

    # Flag: is this the original object?
    is_original = seg_obj.name == original_object_name

    # Precaution: if a dictionary was not passed, make an empty one
    if not isinstance(z_groups_info, dict):
        z_groups_info = {}
    # If this is not the original — do Smart UV Project so that the segment has UVs
    if not is_original:
        smart_uv_project(seg_obj, seg_obj.name)

    obj_name_override = None if is_original else original_object_name

    bm = triangulate_mesh(seg_obj)
    textured_bm = triangulate_mesh(textured_obj)
    if not bm:
        logger.error(f"[SegmentExport] Failed to get BMesh for {seg_obj.name}")
        return None

    try:
        vertex_list, vertex_positions_list, triangles, obj_name = collect_vertices(
            bm, seg_obj.name
        )
        # logger.debug(f"[DEBUG] For object {seg_obj.name} vertex_list was formed (len={len(vertex_list)}): {vertex_list}")
        # logger.debug(f"[DEBUG] For object {seg_obj.name} vertex_positions_list was formed (len={len(vertex_positions_list)}): {vertex_positions_list}")
        # logger.debug(f"[DEBUG] For object {seg_obj.name} triangles were formed (len={len(triangles)}): {triangles}")
        outer_edges, inner_edges, edge_count_map, _ = extract_and_classify_edges(
            triangles, seg_obj.name
        )
        ordered_vertices = order_outer_contour_vertices(
            outer_edges, is_original_object=is_original, object_name=seg_obj.name
        )
        (
            new_vertex_list,
            new_uvs,
            new_vertex_positions_list,
            new_triangles,
            hull_vertex_count,
            vertex_remap,
        ) = rebuild_vertex_lists(
            ordered_vertices, vertex_list, vertex_positions_list, triangles
        )
        # logger.debug(f"[DEBUG] For object {seg_obj.name} new_vertex_list was formed: {new_vertex_list} (len={len(new_vertex_list)})")
        # logger.debug(f"[DEBUG] For object {seg_obj.name} new_vertex_positions_list was formed: {new_vertex_positions_list} (len={len(new_vertex_positions_list)})")
        vertices_2d = transform_uv_to_vertices(
            new_vertex_list, TEXTURE_WIDTH, TEXTURE_HEIGHT, flip_uv=True
        )
        edges_indices = update_edges_indices(outer_edges, inner_edges, vertex_remap)
        e_val = check_euler_characteristic(vertices_2d, edge_count_map, new_triangles)
        # if e_val != 1:
        # logger.warning(f"[SegmentExport] Euler characteristic={e_val}, mesh {seg_obj.name} may be incorrect")
        cx, cy, _, _ = compute_bounding_rectangle(vertices_2d)
        # Calculate local z groups for the segment
        local_z_groups_dict = group_positions_by_z(
            new_vertex_positions_list, seg_obj.name
        )
        local_z_groups_computed = sorted(local_z_groups_dict.keys())
        if not is_original and z_groups_info:
            z_groups = z_groups_info
        # calculations for textured_uv3d_pairs
        (
            textured_vertex_list,
            textured_vertex_positions_list,
            textured_triangles,
            textured_obj_name,
        ) = collect_vertices(textured_bm, textured_obj.name)

        (
            textured_outer_edges,
            textured_inner_edges,
            textured_edge_count_map,
            _,
        ) = extract_and_classify_edges(textured_triangles, textured_obj.name)
        textured_ordered_vertices = order_outer_contour_vertices(
            textured_outer_edges,
            is_original_object=is_original,
            object_name=textured_obj.name,
        )
        (
            textured_new_vertex_list,
            textured_new_uvs,
            textured_new_vertex_positions_list,
            textured_new_triangles,
            textured_hull_vertex_count,
            textured_vertex_remap,
        ) = rebuild_vertex_lists(
            textured_ordered_vertices,
            textured_vertex_list,
            textured_vertex_positions_list,
            textured_triangles,
        )
        textured_vertices_2d = transform_uv_to_vertices(
            textured_new_vertex_list, TEXTURE_WIDTH, TEXTURE_HEIGHT, flip_uv=False
        )  # added flip_uv=False to not flip textured UVs but it didn't help
        textured_edges_indices = update_edges_indices(
            textured_outer_edges, textured_inner_edges, textured_vertex_remap
        )
        textured_local_z_groups_dict = group_positions_by_z(
            textured_new_vertex_positions_list, seg_obj.name
        )

        if is_original:
            # For the original object, form a global array of uv3d-pairs
            global_uv3d_pairs = form_uv3d_pairs(
                new_vertex_list, new_vertex_positions_list
            )
            # For the original object, form a global array of textured uv3d-pairs
            global_textured_uv3d_pairs = form_uv3d_pairs(
                textured_new_vertex_list, textured_new_vertex_positions_list
            )
            segment_uv3d_pairs = None
            # local_z_groups_computed = local_z_groups_computed  # can be forced to be equal to the original ones
            original_z_groups = local_z_groups_computed
        else:
            # For the segment, form a local array of uv3d-pairs
            segment_uv3d_pairs = form_uv3d_pairs(
                new_vertex_list, new_vertex_positions_list
            )
            # logger.debug(f"[DEBUG] For segment {seg_obj.name} uv3d_pairs was formed: {segment_uv3d_pairs}")
            # Do not update the global array – use the original uv3d-pairs passed as a parameter
            global_uv3d_pairs = original_uv3d_pairs
            global_textured_uv3d_pairs = textured_uv3d_pairs_global
        logger.debug(
            f"[SegmentExport] Variable 'original_uv3d_pairs' before calling form_raw_data (len={len(global_uv3d_pairs)}): {global_uv3d_pairs}..."
        )
        raw_data = form_raw_data(
            new_uvs,
            new_triangles,
            vertices_2d,
            hull_vertex_count,
            edges_indices,
            local_z_groups_dict,
            global_uv3d_pairs,
            seg_obj.name,
        )
        # Log the variable BEFORE calling form_raw_data
        # logger.debug(f"[SegmentExport] Variable 'textured_uv3d_pairs_global' before calling form_raw_data (len={len(global_textured_uv3d_pairs)}): {global_textured_uv3d_pairs}...")
        textured_raw_data = form_raw_data(
            textured_new_uvs,
            textured_new_triangles,
            textured_vertices_2d,
            textured_hull_vertex_count,
            textured_edges_indices,
            textured_local_z_groups_dict,
            global_textured_uv3d_pairs,
            textured_obj.name,
        )
        # Here we save the uv3d-pairs for internal calculations in raw_data (if needed)
        raw_data["uv3d_pairs"] = global_uv3d_pairs
        textured_raw_data["uv3d_pairs"] = global_textured_uv3d_pairs
    finally:
        bm.free()
        textured_bm.free()
    stretch_values = calculate_stretch_values(seg_obj) or {}
    uv_pairs_to_use = global_uv3d_pairs
    textured_uv_pairs_to_use = global_textured_uv3d_pairs
    # Here we pass both arrays: global and local (segment_uv3d_pairs)
    # IMPORTANT: we pass exactly z_groups_info (if the object is original) or {} (if it is a segment)

    spine_data = process_raw_data(
        raw_data=raw_data,
        textured_raw_data=textured_raw_data,
        obj=seg_obj,
        textured_obj=textured_obj,
        TEXTURE_WIDTH=TEXTURE_WIDTH,
        TEXTURE_HEIGHT=TEXTURE_HEIGHT,
        z_groups_info=z_groups_info,
        stretch_values=stretch_values,
        obj_name_override=obj_name_override,
        is_segment=(not is_original),
        original_uv3d_pairs=uv_pairs_to_use,
        textured_uv3d_pairs=textured_uv_pairs_to_use,
        segment_uv3d_pairs=segment_uv3d_pairs,
        filepath=output_filepath,
        original_z_groups=original_z_groups,
        local_z_groups=local_z_groups_computed,  # pass the calculated local z groups
        world_location=world_location,  # pass the world_location if needed
    )
    if not spine_data:
        logger.error(
            f"[SegmentExport] process_raw_data returned None for {seg_obj.name}"
        )
        return None

    if is_original:
        # Step 1: Check the control icons setting
        should_preserve_icons = True
        try:
            # Use .get() for safe access to the scene property
            should_preserve_icons = bool(
                bpy.context.scene.get("spine2d_control_icons", True)
            )
        except Exception:
            # In case of an error (e.g., when running outside the UI), leave the default value
            logger.warning(
                "Failed to read setting 'spine2d_control_icons'. Icons will be preserved."
            )
            should_preserve_icons = True

        # Step 2: Execute the logic depending on the setting
        if should_preserve_icons:
            # Logic for preserving control elements (your existing code)
            control_slots = []
            control_attachments = {}

            for slot in spine_data.get("slots", []):
                slot_name = slot.get("name", "")
                if "_rotation_" in slot_name or "_main" in slot_name:
                    control_slots.append(slot)

            for skin in spine_data.get("skins", []):
                attachments = skin.get("attachments", {})
                for slot_name, slot_data in attachments.items():
                    if "_rotation_" in slot_name or "_main" in slot_name:
                        control_attachments[slot_name] = slot_data

            spine_data["slots"] = []
            for skin in spine_data.get("skins", []):
                skin["attachments"] = {}

            spine_data["slots"] = control_slots
            if spine_data.get("skins") and control_attachments:
                spine_data["skins"][0]["attachments"] = control_attachments

            logger.info(
                f"[SegmentExport] {seg_obj.name} – is original. Regular slots/attachments stripped, "
                f"control icons preserved: {len(control_slots)} slots."
            )
        else:
            # Logic for complete cleanup if the checkbox is unchecked
            spine_data["slots"] = []
            for skin in spine_data.get("skins", []):
                skin["attachments"] = {}
            logger.info(
                f"[SegmentExport] {seg_obj.name} – is original. All slots/attachments stripped as per user setting."
            )
    else:
        spine_data["transform"] = []
        spine_data["animations"] = {}

    # For debugging, save the intermediate JSON if needed
    if logger.isEnabledFor(logging.DEBUG):
        from os.path import dirname, join

        blend_filepath = bpy.data.filepath
        save_dir = (
            dirname(blend_filepath)
            if blend_filepath
            else bpy.utils.resource_path("USER")
        )
        debug_filename = f"{seg_obj.name}_DEBUG.json"
        debug_path = join(save_dir, debug_filename)
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(spine_data, f, indent=4, ensure_ascii=False)
            logger.debug(
                f"[SegmentExport] Intermediate JSON for {seg_obj.name} saved: {debug_path}"
            )
        except Exception as e:
            logger.exception(
                f"[SegmentExport] exception Failed to save intermediate JSON {debug_filename}: {e}"
            )

    logger.debug(
        f"[SegmentExport] Segment export completed successfully:{seg_obj.name}"
    )
    return spine_data


# =============================================================================
# Main pipeline function: export one active object + its segments
# =============================================================================
def save_uv_as_json(
    obj: bpy.types.Object,
    TEXTURE_WIDTH: int,
    TEXTURE_HEIGHT: int,
    output_dir: str | None = None,
):
    # Temporary objects for subsequent cleanup
    temp_objects_to_clean = []
    segment_objs = []
    original_copy = None
    textured_obj = None
    try:
        original_world_location = obj.matrix_world.translation.copy()
        logger.info(
            f"The original Origin position has been saved for'{obj.name}': {original_world_location}"
        )
        if output_dir and os.path.isdir(bpy.path.abspath(output_dir)):
            save_directory = bpy.path.abspath(output_dir)
            logger.info(f"Using user-provided save directory: {save_directory}")
        else:
            save_directory = get_default_output_dir()
            logger.info(f"Using default save directory: {save_directory}")
        if isinstance(save_directory, bytes):
            save_directory = save_directory.decode("utf-8")
        # [CHANGE START] Check and create UVs if necessary
        if not obj.data.uv_layers:
            logger.warning(
                f"Object '{obj.name}' has no UV maps. Creating temporary UV map 'UVMap' using Smart UV Project."
            )
            smart_uv_project(obj, obj.name)
            logger.info(f"Temporary UV map 'UVMap' for '{obj.name}' created.")
        # Step 1. Get the original UV map of the original object
        if obj.data.uv_layers.active:
            source_uv_map = obj.data.uv_layers.active
        elif obj.data.uv_layers:
            source_uv_map = obj.data.uv_layers[0]
            logger.warning(
                f"No active UV map found for '{obj.name}'. Using first one: '{source_uv_map.name}'."
            )
        else:
            logger.error(f"Object '{obj.name}' has no UV maps. Baking is not possible.")
            raise ValueError(f"Object '{obj.name}' has no UV maps.")
        source_uv_map_name = source_uv_map.name
        logger.info(
            f"Original UV map for baking from '{obj.name}': '{source_uv_map_name}'"
        )
        # here we will assign face_id to the original for all polygons to then put the texture on the segments from the texture copy
        assign_face_ids(obj)
        # New step: create a copy of the original object so that all operations are performed on the copy,
        # and the original object remains unchanged.
        original_copy = obj.copy()
        original_copy.data = obj.data.copy()
        original_copy.name = obj.name + "_copy_for_uv"
        bpy.context.collection.objects.link(original_copy)
        temp_objects_to_clean.append(original_copy)

        # [FIXED] Key change: Copy seams from the original to the working copy BEFORE cutting.
        # Now plane_cut in CUSTOM mode will see the user seams.
        bmesh_orig = bmesh.new()
        bmesh_orig.from_mesh(obj.data)
        # [FIXED] Correct way to get the vertex indices of an edge
        seams_on_original = {
            tuple(sorted([v.index for v in e.verts]))
            for e in bmesh_orig.edges
            if e.seam
        }
        bmesh_orig.free()

        if seams_on_original:
            logger.info(
                f"{len(seams_on_original)} seams found on original object. Transferring to copy."
            )
            bmesh_copy = bmesh.new()
            bmesh_copy.from_mesh(original_copy.data)
            bmesh_copy.edges.ensure_lookup_table()
            for edge in bmesh_copy.edges:
                # [FIXED] Here the way to get indices is also fixed
                if tuple(sorted([v.index for v in edge.verts])) in seams_on_original:
                    edge.seam = True
            bmesh_copy.to_mesh(original_copy.data)
            bmesh_copy.free()
        logger.info(
            f"A copy of the object has been created for UV processing: {original_copy.name}"
        )
        for o in bpy.context.selected_objects:
            o.select_set(False)
        original_copy.select_set(True)
        # Apply basic transformations and reset UVs
        bpy.context.view_layer.objects.active = original_copy
        triangulate_mesh(original_copy)
        post_unwrap_info = main_preprocessing(original_copy)
        if not post_unwrap_info:
            logger.error("[ERROR] Preprocessing failed, exiting.")
            return
        # Extract the z_groups_info dictionary, which already contains height_real_pixels, etc.
        # This is needed specifically for the original object
        z_groups_info_for_original = post_unwrap_info.get("z_groups_info", {})
        logger.debug(
            f"[DEBUG] z_groups_info_for_original keys: {list(z_groups_info_for_original.keys())}"
        )
        original_z_groups = sorted(z_groups_info_for_original.keys())
        logger.debug(f"[DEBUG] original_z_groups keys: {original_z_groups}")
        # --- STAGE 2: CUTTING AND GETTING BOUNDARIES ---
        scene = bpy.context.scene
        seam_mode = scene.spine2d_seam_maker_mode
        angle_limit = float(scene.spine2d_angle_limit)

        # [CHANGE] Call a single smart cutting function
        segmentation_data = plane_cut.execute_smart_cut(
            original_copy,
            angle_limit=angle_limit,
            seam_mode=seam_mode,
            capture_uv_data=True,  # Enable UV capture
        )

        logger.info(
            f"Slicing complete. {len(segmentation_data)} boundary edges for seams obtained."
        )

        # Apply segmentation seams to the copy used for UV calculations.  This
        # ensures that subsequent unwrap operations on this copy respect the
        # boundaries between segments.  Without marking seams here, functions
        # like smart_uv_project may ignore the segmentation and produce
        # inconsistent UV islands.
        apply_segmentation_seams(original_copy, segmentation_data)

        # --- STAGE 3: COLLECTING SEGMENTS ---
        orig_base_name = original_copy.name.replace("_copy_for_uv", "")
        segment_objs = [
            o
            for o in bpy.context.scene.objects
            if o.name.startswith(f"{orig_base_name}_Segment_")
        ]
        temp_objects_to_clean.extend(segment_objs)

        if not segment_objs:
            logger.warning(
                "No segment objects found. The export will be performed for the entire object."
            )
            segment_objs = [original_copy]
        else:
            # Add original_copy as the main container object
            segment_objs.insert(0, original_copy)
        # ---part about the texture object copy of the original---
        # Create a copy of the object with seams, using segmentation_data:
        logger.debug(
            f"segmentation_data for original object {original_copy.name}: {segmentation_data}"
        )
        # Create a copy of the object with seams, using segmentation_data:
        textured_obj, seams_info = mark_seams_on_copy(obj, segmentation_data)
        # Transfer stored UV island data if available
        if "uv_island_segments" in original_copy:
            logger.info("Transferring stored UV island data to textured object")
            transfer_uv_islands_between_objects(original_copy, textured_obj)
        else:
            # Fallback to current method
            logger.info("No stored UV data, using standard unwrap")
            if segmentation_data and len(segmentation_data) > 0:
                uv_map_name = unwrap_respecting_seams(textured_obj)
            else:
                uv_map_name = smart_uv_project(textured_obj, textured_obj.name)
        if textured_obj is None:
            logger.error(
                "Failed to create a copy of the object with seams. Export aborted."
            )
            # Delete original_copy
            if original_copy.name in bpy.data.objects:
                bpy.data.objects.remove(original_copy, do_unlink=True)
            return
        copy_orig_face_id_layer(
            original_copy, textured_obj
        )  # copy face_id to the texture object
        # Now get the UV map for baking.
        # Instead of "if there are no seams — try to take the old UV" always
        # create (or re-unwrap) "UVMap_for_texturing".
        if segmentation_data and len(segmentation_data) > 0:
            # If there are seams, perform an unwrap so that Blender takes the seams into account
            uv_map_name = unwrap_respecting_seams(
                textured_obj, method="ANGLE_BASED", margin=0.001
            )
            if not uv_map_name:
                logger.warning(
                    "[save_uv_as_json] unwrap_respecting_seams returned None, using fallback smart_uv_project"
                )
                uv_map_name = smart_uv_project(textured_obj, textured_obj.name)
        else:
            # If there are no seams, just do smart_uv_project
            uv_map_name = smart_uv_project(textured_obj, textured_obj.name)

        # Just in case, rename the active UV layer to "UVMap_for_texturing"
        if textured_obj.data.uv_layers:
            active_uv = textured_obj.data.uv_layers.active
            active_uv.name = "UVMap_for_texturing"
            uv_map_name = active_uv.name
        else:
            logger.warning(
                "[save_uv_as_json] The object has no UV layers even after unwrap! We use the default name."
            )
            uv_map_name = "UVMap_for_texturing"

        logger.debug(f"[save_uv_as_json] Final UV map for baking: {uv_map_name}")
        # Perform baking
        logger.info("Calling bake_textures_for_object:")
        logger.info(f"  target_obj = {textured_obj.name}")
        logger.info(f"  target_uv_map_name = {uv_map_name}")
        logger.info(
            f"  source_obj = {obj.name}"
        )  # Use the original object passed to the function
        logger.info(f"  source_uv_map_name = {source_uv_map_name}")
        # Then the standard texture baking pipeline
        if not bake_textures_for_object(
            textured_obj,  # Object to bake TO (copy with new UV)
            uv_map_name,  # Name of the UV map to bake TO
            obj,  # Object to bake FROM (original)
            source_uv_map_name,  # Name of the UV map to bake FROM
        ):
            logger.error("Texture baking failed!")
            return
        # =============================================================================
        try:
            # After successful baking, textured_obj has the correct UV map.
            # We will transfer these UVs to all segment objects.
            # segment_objs is a list of all ..._Segment_N objects.
            transfer_baked_uvs_to_segments(textured_obj, segment_objs, uv_map_name)
        except Exception as e:
            logger.exception(f"Critical error when transferring UV to segments: {e}")
            # You can interrupt the export if the transfer failed
            return
        # =============================================================================
        # ----------------------------------------------------------------------------
        # 3) Export each segment
        all_results = []
        original_uv3d_pairs_global = (
            None  # Here the uv3d-pairs of the original object will be saved
        )
        textured_uv3d_pairs_global = (
            None  # Here the uv3d-pairs of the original object will be saved
        )
        for seg_obj in segment_objs:
            # Calculate whether the segment is the original object
            is_original = seg_obj.name == original_copy.name

            # Check if this is our "original" object
            if seg_obj.name == original_copy.name:
                uv_param = None
                # If the segment is the original, then use the original z-groups, otherwise you can pass {} (or another value)
                local_z_groups_info = z_groups_info_for_original if is_original else {}
            else:
                if original_uv3d_pairs_global is None:
                    logger.error(
                        f"[ERROR] For segment {seg_obj.name} original array uv3d_pairs is not formed yet. Skipping segment."
                    )
                    continue
                if textured_uv3d_pairs_global is None:
                    logger.error(
                        f"[ERROR] For segment {seg_obj.name} the original textured_uv3d_pairs array has not yet been formed. Skipping segment."
                    )
                    continue
                uv_param = original_uv3d_pairs_global
                local_z_groups_info = {}
            # We will always pass the z_groups_info_for_original of the original object
            local_z_groups = original_z_groups if is_original else None

            seg_res = _export_segment(
                seg_obj,
                original_copy.name,
                textured_obj=textured_obj,
                TEXTURE_WIDTH=TEXTURE_WIDTH,
                TEXTURE_HEIGHT=TEXTURE_HEIGHT,
                original_uv3d_pairs=original_uv3d_pairs_global
                if original_uv3d_pairs_global
                else None,
                textured_uv3d_pairs_global=textured_uv3d_pairs_global
                if textured_uv3d_pairs_global
                else None,
                z_groups_info=local_z_groups_info,  # <-- IMPORTANT: pass the dictionary
                original_z_groups=original_z_groups,
                output_dir=save_directory,
                local_z_groups=local_z_groups,  # add a parameter for local groups
                world_location=original_world_location,  # Pass the world position of the object
            )
            if seg_res:
                all_results.append(seg_res)
                if seg_obj.name == original_copy.name:
                    # Extract uv3d-pairs from our private key
                    original_uv3d_pairs_global = seg_res.get("_uv3d_pairs", None)
                    if not original_uv3d_pairs_global:
                        logger.error(
                            "[ERROR] Unable to generate uv3d_pairs for original object. Export aborted."
                        )
                        return
                if seg_obj.name == original_copy.name:
                    # Extract textured_uv3d-pairs from our private key
                    textured_uv3d_pairs_global = seg_res.get(
                        "textured_uv3d_pairs", None
                    )
                    if not textured_uv3d_pairs_global:
                        logger.error(
                            "[ERROR] Failed to generate textured_uv3d_pairs for the original object. Export aborted."
                        )
                        return
            else:
                logger.error(f"[ERROR] Segment {seg_obj.name} is not exported.")

        if not all_results:
            logger.error("[ERROR] There are no final segments to combine.")
            return
        # Merge: the first result is the main one, the rest are segments
        main_data = all_results[0]
        segment_data_list = all_results[1:] if len(all_results) > 1 else []
        merged_data = merge_spine_json_dicts(main_data, segment_data_list)

        # DEBUGGING: Log the final bone data before writing to file
        logger.debug("Final bone data before writing to file:")
        for bone in merged_data.get("bones", []):
            # Log only the Z-group scale bones to reduce noise
            if "_scale" in bone.get("name", "") and "constraint" not in bone.get(
                "name", ""
            ):
                logger.debug(f"  - Bone: {bone.get('name')}, Y: {bone.get('y')}")

        # Save the final JSON
        final_path = os.path.join(save_directory, f"{obj.name}_merged.json")
        write_json(merged_data, final_path)
        logger.info(f"[save_uv_as_json] The final JSON is saved: {final_path}")
        logger.info("[save_uv_as_json] Pipeline is complete.")
        return final_path
    except Exception as e:
        logger.exception(f"[FATAL ERROR exception in save_uv_as_json] {e}")
        return None
    finally:
        # CORRECTED: Complete cleanup of all temporary objects
        try:
            if not logger.isEnabledFor(logging.DEBUG):
                logger.info("[save_uv_as_json] Performing cleanup (non-debug mode)")

                # Phase 1: Segment cleanup and JSON file removal
                if segment_objs and original_copy:
                    try:
                        _ = original_copy.name  # Validate object reference
                        delete_segment_artifacts(segment_objs, original_copy)
                        logger.debug(
                            "[save_uv_as_json] delete_segment_artifacts completed"
                        )
                    except ReferenceError:
                        logger.warning(
                            "[cleanup] original_copy already removed, skipping delete_segment_artifacts"
                        )
                    except Exception as artifact_cleanup_err:
                        logger.exception(
                            f"[cleanup] Error in delete_segment_artifacts: {artifact_cleanup_err}"
                        )

                # Phase 2: Explicit cleanup of critical temporary objects
                critical_cleanup_objects = []

                # Add original_copy for explicit removal
                if original_copy is not None:
                    try:
                        original_copy_name = original_copy.name
                        if original_copy_name in bpy.data.objects:
                            critical_cleanup_objects.append(
                                (original_copy, original_copy_name)
                            )
                    except ReferenceError:
                        logger.debug("[cleanup] original_copy already removed")

                # Add textured_obj for explicit removal
                if textured_obj is not None:
                    try:
                        textured_obj_name = textured_obj.name
                        if textured_obj_name in bpy.data.objects:
                            critical_cleanup_objects.append(
                                (textured_obj, textured_obj_name)
                            )
                    except ReferenceError:
                        logger.debug("[cleanup] textured_obj already removed")

                # Phase 3: Process remaining temporary objects
                for temp_obj in temp_objects_to_clean:
                    if temp_obj is None:
                        continue

                    # Skip objects already queued for critical cleanup
                    if temp_obj == original_copy or temp_obj == textured_obj:
                        continue

                    try:
                        obj_name = temp_obj.name
                        if obj_name in bpy.data.objects:
                            critical_cleanup_objects.append((temp_obj, obj_name))
                    except ReferenceError:
                        logger.debug("[cleanup] Object already removed from Blender")
                        continue
                    except Exception as check_err:
                        logger.warning(
                            f"[cleanup] Error checking object validity: {check_err}"
                        )
                        continue

                # Execute complete object removal
                removed_count = 0
                for temp_obj, obj_name in critical_cleanup_objects:
                    try:
                        bpy.data.objects.remove(temp_obj, do_unlink=True)
                        logger.debug(f"[cleanup] Removed temp object: {obj_name}")
                        removed_count += 1
                    except Exception as remove_err:
                        logger.warning(
                            f"[cleanup] Failed to remove {obj_name}: {remove_err}"
                        )

                logger.info(
                    f"[save_uv_as_json] Cleanup completed - removed {removed_count} temporary objects"
                )

            else:
                logger.info(
                    "[save_uv_as_json] DEBUG mode - skipping cleanup to preserve debug data"
                )
                try:
                    debug_original = original_copy.name if original_copy else "None"
                    debug_textured = textured_obj.name if textured_obj else "None"
                    logger.debug(
                        f"[save_uv_as_json] Preserved objects: original_copy={debug_original}, textured_obj={debug_textured}"
                    )
                except ReferenceError:
                    logger.debug("[save_uv_as_json] Some debug objects already removed")

        except Exception as cleanup_error:
            logger.exception(f"[save_uv_as_json] Error during cleanup: {cleanup_error}")
            # Preserve primary exception handling by not re-raising


# ======= Operator for Blender =======
class OBJECT_OT_SaveUVAsJSON(bpy.types.Operator):
    """Export current object button"""

    bl_idname = "object.save_uv_as_json"
    bl_label = "Export current object"
    bl_description = "Exports the current object to Spine JSON."

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _scale_applied(obj, tol=1e-4) -> bool:
        """True if local Scale ≈ (1,1,1)."""
        return all(abs(v - 1.0) < tol for v in obj.scale)

    # ──────────────────────────────────────────────────────────
    def execute(self, context):
        """
        Exports the active object to Spine-JSON taking into account the texture size,
        which the user sets in Scene.spine2d_texture_size.
        """
        logger.info("[SaveUVAsJSON] ► Start export")
        try:
            # 1) Take values from the UI
            from .config import get_texture_size

            size_or_tuple = get_texture_size(context.scene)

            # 2) Normalize to two numbers
            if isinstance(size_or_tuple, (tuple, list)) and len(size_or_tuple) == 2:
                tex_w, tex_h = map(int, size_or_tuple)
            else:
                tex_w = tex_h = int(size_or_tuple)

            # 3) Check for a healthy range
            if tex_w <= 0 or tex_h <= 0:
                self.report({"ERROR"}, f"Incorrect texture size: {tex_w}×{tex_h}")
                return {"CANCELLED"}

            # 4) Patch global constants inside the addon
            global TEXTURE_WIDTH, TEXTURE_HEIGHT
            TEXTURE_WIDTH = tex_w
            TEXTURE_HEIGHT = tex_h
            logger.debug(f"[SaveUVAsJSON] TEXTURE set to {tex_w}×{tex_h}")

            # 4a) If necessary, synchronize the config-module as well, because other imports could
            #     have picked up the old value before the setter was called
            from . import json_export as _je, config as _cfg

            _cfg.TEXTURE_WIDTH = tex_w
            _cfg.TEXTURE_HEIGHT = tex_h
            _je.TEXTURE_WIDTH = tex_w
            _je.TEXTURE_HEIGHT = tex_h

            # 5) Make sure there is an active mesh object
            obj = context.active_object
            if obj is None:
                self.report({"ERROR"}, "There is no active object")
                return {"CANCELLED"}
            if obj.type != "MESH":
                self.report({"ERROR"}, f"The object {obj.name} is not a mesh.")
                return {"CANCELLED"}

            # 6) Start the main pipeline
            save_uv_as_json(
                obj, tex_w, tex_h, output_dir=context.scene.spine2d_json_path
            )

            # self.report({"INFO"}, f"[SaveUVAsJSON] ► Export finished ({tex_w}×{tex_h})")
            logger.info("[SaveUVAsJSON] ► Export finished")
            return {"FINISHED"}

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.exception(f"[SaveUVAsJSON] Export failed: {exc}")
            self.report({"ERROR"}, f"Export policy: {exc}")
            return {"CANCELLED"}


def register():
    bpy.utils.register_class(OBJECT_OT_SaveUVAsJSON)
    logger.debug("OBJECT_OT_SaveUVAsJSON registered.")


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_SaveUVAsJSON)
    logger.debug("OBJECT_OT_SaveUVAsJSON is registered.")

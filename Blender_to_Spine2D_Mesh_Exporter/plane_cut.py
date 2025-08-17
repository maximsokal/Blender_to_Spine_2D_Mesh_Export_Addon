# plane_cut.py
"""
This module is responsible for the intelligent segmentation of a 3D mesh into smaller, manageable pieces (segments or islands). This is a crucial step in the export pipeline, as it determines how the model will be broken down for UV unwrapping and texturing.

The key functionalities are:
1.  Hybrid Segmentation Strategy: The core function, `execute_smart_cut`, implements a sophisticated hybrid approach. In "AUTO" mode, it first separates the mesh based on existing UV island boundaries and then further subdivides those islands based on a user-defined angle limit. In "CUSTOM" mode, it respects user-defined seams to create the initial segments.
2.  Segment Creation: After determining the boundaries, it creates new, separate Blender objects for each segment. It carefully preserves the original face IDs (`orig_face_id`) on these new objects, which is essential for correctly mapping textures back to the segments later.
3.  Topological Analysis and Decomposition: It includes advanced functions to analyze the topology of each potential segment (`calculate_topology_invariants`). If a segment is found to be topologically complex (e.g., it contains holes), the `decompose_complex_segment` function is used to break it down further into simpler, hole-free islands using a k-means clustering approach.
4.  Boundary Edge Detection: It identifies and returns a list of all the edges that form the boundaries between the final segments. This information is used to create seams for the final UV unwrap.
5.  UV Island Data Capture: It has the capability to capture detailed information about the UV layout of each segment *before* the final bake, storing this data as a custom property on the object. This can be used for more advanced texture transfer workflows.

ATTENTION: - The quality of the segmentation directly impacts the final UV layout and the number of resulting objects. The `angle_limit` and `seam_mode` parameters are the primary user controls and can significantly change the outcome. The logic for decomposing complex segments is computationally intensive and relies on heuristics (like `cluster_search_aggressiveness`) that may not always find a perfect solution for highly complex meshes. The preservation of `orig_face_id` is critical for the rest of the pipeline to work correctly.
Author: Maxim Sokolenko
"""
import bpy
import bmesh
from mathutils import Vector
from math import degrees
from collections import defaultdict, deque
import logging
import random
import contextlib
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def safe_bmesh(mesh_data=None, from_edit_mesh=False, edit_mesh_object=None):
    """
    Professional BMesh context manager with comprehensive error handling.

    Provides guaranteed cleanup and proper lifecycle management for BMesh objects.
    Supports both new BMesh creation and edit mesh access patterns.

    Args:
        mesh_data: Mesh data to load into BMesh (optional)
        from_edit_mesh: Whether to create from edit mesh mode
        edit_mesh_object: Object for edit mesh mode (required if from_edit_mesh=True)

    Yields:
        bmesh.types.BMesh: Properly initialized BMesh object

    Raises:
        RuntimeError: If BMesh creation fails or invalid parameters provided
    """
    bm = None
    created_new = False

    try:
        if from_edit_mesh:
            if edit_mesh_object is None or edit_mesh_object.data is None:
                raise RuntimeError("edit_mesh_object required for from_edit_mesh mode")
            bm = bmesh.from_edit_mesh(edit_mesh_object.data)
        else:
            bm = bmesh.new()
            created_new = True

            if mesh_data is not None:
                bm.from_mesh(mesh_data)

        # Ensure lookup tables for performance
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        yield bm

    except Exception as e:
        logger.error(f"BMesh operation failed: {e}")
        raise

    finally:
        # CRITICAL: Only free BMesh objects we created
        if bm is not None and created_new:
            try:
                bm.free()
            except ReferenceError:
                # BMesh already freed - this is acceptable
                pass
            except Exception as e:
                logger.warning(f"BMesh cleanup warning: {e}")


def validate_mesh_object(obj) -> Tuple[bool, str]:
    """
    Comprehensive mesh object validation with detailed diagnostic information.

    Performs multi-stage validation to ensure object is suitable for mesh processing.
    Provides specific error messages for different failure modes.

    Args:
        obj: Object to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if obj is None:
        return False, "Object is None"

    if not hasattr(obj, "type"):
        return (
            False,
            f"Object {type(obj)} lacks 'type' attribute (not a Blender object)",
        )

    if obj.type != "MESH":
        return False, f"Object type '{obj.type}' is not MESH"

    if not hasattr(obj, "data") or obj.data is None:
        return False, "Object lacks accessible mesh data"

    # Validate mesh data integrity
    try:
        polygon_count = len(obj.data.polygons)
        vertex_count = len(obj.data.vertices)

        if polygon_count == 0:
            return False, "Mesh contains no polygons"

        if vertex_count < 3:
            return False, f"Mesh has insufficient vertices: {vertex_count}"

    except Exception as e:
        return False, f"Mesh data access error: {e}"

    return True, "Valid mesh object"


def get_segments_from_uv_islands(
    bm: bmesh.types.BMesh, faces_subset: List[bmesh.types.BMFace]
) -> List[List[bmesh.types.BMFace]]:
    """
    Enhanced UV island segmentation with robust error handling.

    Groups faces into segments based on UV island boundaries using the most
    reliable method available in the current Blender version.

    Args:
        bm: BMesh instance with valid UV data
        faces_subset: List of faces to process

    Returns:
        List of face groups representing UV islands
    """
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        logger.warning("No active UV layer found, returning single segment")
        return [faces_subset] if faces_subset else []

    try:
        # Save current seam state for restoration
        original_seams = {e: e.seam for e in bm.edges}

        # Use Blender's built-in UV island detection
        bmesh.ops.uv_seams_from_islands(bm, uv_layer=uv_layer, faces=bm.faces)

        # Extract segments based on seams
        segments = get_segments_from_seams(bm)

        # Restore original seam state
        for edge, was_seam in original_seams.items():
            if edge.is_valid:
                edge.seam = was_seam

        logger.info(f"UV island segmentation: {len(segments)} segments found")
        return segments

    except AttributeError:
        logger.warning(
            "bmesh.ops.uv_seams_from_islands not available, using fallback method"
        )
        return _manual_uv_island_detection(bm, faces_subset, uv_layer)

    except Exception as e:
        logger.error(f"UV island segmentation error: {e}")
        return [faces_subset] if faces_subset else []


def _manual_uv_island_detection(
    bm: bmesh.types.BMesh, faces_subset: List[bmesh.types.BMFace], uv_layer
) -> List[List[bmesh.types.BMFace]]:
    """
    Fallback manual UV island detection for older Blender versions.

    Uses graph traversal to identify UV boundaries when built-in operations
    are not available.
    """
    segments = []
    visited_faces = set()
    faces_to_process = set(faces_subset)

    for face in faces_subset:
        if face in visited_faces or not face.is_valid:
            continue

        current_segment = []
        queue = deque([face])
        visited_faces.add(face)

        while queue:
            current_face = queue.popleft()
            current_segment.append(current_face)

            for edge in current_face.edges:
                if edge.is_boundary or len(edge.link_faces) < 2:
                    continue

                # Check for UV discontinuity
                is_uv_seam = _detect_uv_seam(edge, uv_layer)

                if not is_uv_seam:
                    for linked_face in edge.link_faces:
                        if (
                            linked_face in faces_to_process
                            and linked_face not in visited_faces
                            and linked_face.is_valid
                        ):
                            visited_faces.add(linked_face)
                            queue.append(linked_face)

        if current_segment:
            segments.append(current_segment)

    return segments


def _detect_uv_seam(edge: bmesh.types.BMEdge, uv_layer) -> bool:
    """
    Detect UV seam by comparing UV coordinates across edge.

    Returns True if the edge represents a UV boundary.
    """
    uv_tolerance = 1e-6

    try:
        uv_coords_per_vertex = defaultdict(list)

        for loop in edge.link_loops:
            vertex = loop.vert
            uv_coord = loop[uv_layer].uv
            uv_coords_per_vertex[vertex].append(uv_coord)

        # Check for UV discontinuity
        for vertex_uvs in uv_coords_per_vertex.values():
            if len(vertex_uvs) > 1:
                base_uv = vertex_uvs[0]
                for other_uv in vertex_uvs[1:]:
                    if (other_uv - base_uv).length > uv_tolerance:
                        return True

        return False

    except Exception as e:
        logger.warning(f"UV seam detection error: {e}")
        return False


def capture_segment_uv_islands(
    bm: bmesh.types.BMesh, segment_faces: List[bmesh.types.BMFace], segment_id: int
) -> Dict:
    """
    Enhanced UV island data capture with comprehensive error handling.

    Captures detailed UV layout information for advanced texture transfer workflows.
    Includes boundary detection, vertex mapping, and topology analysis.

    Args:
        bm: BMesh instance with UV layer
        segment_faces: Faces belonging to this segment
        segment_id: Unique identifier for this segment

    Returns:
        Dictionary containing complete UV island data
    """
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        logger.warning(f"No active UV layer for segment {segment_id}")
        return {"segment_id": segment_id, "error": "No UV layer"}

    try:
        # Collect segment vertices
        segment_verts = set()
        for face in segment_faces:
            if face.is_valid:
                segment_verts.update(face.verts)

        # Create vertex mapping for local indexing
        vert_to_local_index = {
            v: i for i, v in enumerate(sorted(segment_verts, key=lambda v: v.index))
        }

        # Initialize data structure
        uv_island_data = {
            "segment_id": segment_id,
            "vertex_count": len(segment_verts),
            "face_count": len(segment_faces),
            "vertex_uvs": {},
            "face_loops": [],
            "boundary_edges": [],
            "topology": calculate_topology_invariants(segment_faces),
        }

        # Process face UV data
        for face in segment_faces:
            if not face.is_valid:
                continue

            face_data = {"face_index": face.index, "loops": []}

            for loop in face.loops:
                if not loop.vert.is_valid:
                    continue

                vert_index = loop.vert.index
                local_vert_index = vert_to_local_index.get(loop.vert)
                uv_coords = tuple(loop[uv_layer].uv[:])

                loop_data = {
                    "vertex_index": vert_index,
                    "local_vertex_index": local_vert_index,
                    "uv": uv_coords,
                }
                face_data["loops"].append(loop_data)

                # Track UV variations per vertex
                if vert_index not in uv_island_data["vertex_uvs"]:
                    uv_island_data["vertex_uvs"][vert_index] = []
                uv_island_data["vertex_uvs"][vert_index].append(uv_coords)

            uv_island_data["face_loops"].append(face_data)

        # Identify boundary edges
        uv_island_data["boundary_edges"] = _extract_boundary_edges(segment_faces)

        return uv_island_data

    except Exception as e:
        logger.error(f"UV island capture error for segment {segment_id}: {e}")
        return {"segment_id": segment_id, "error": str(e)}


def _extract_boundary_edges(
    segment_faces: List[bmesh.types.BMFace]
) -> List[Tuple[int, int]]:
    """Extract boundary edges from a set of faces."""
    edge_face_count = defaultdict(int)

    for face in segment_faces:
        if face.is_valid:
            for edge in face.edges:
                edge_face_count[edge] += 1

    boundary_edges = []
    for edge, count in edge_face_count.items():
        if count == 1 and edge.is_valid:  # Boundary edge
            vert_indices = tuple(sorted(v.index for v in edge.verts if v.is_valid))
            if len(vert_indices) == 2:
                boundary_edges.append(vert_indices)

    return boundary_edges


def store_uv_islands_on_object(obj: bpy.types.Object, uv_islands_data: List[Dict]):
    """
    Store UV island data on object with enhanced serialization.

    Converts complex UV data to JSON-serializable format and stores as custom property.
    Includes data validation and error recovery.
    """
    try:
        import json

        if not uv_islands_data:
            logger.warning("No UV island data to store")
            return

        # Convert to JSON-serializable format
        serializable_data = []
        for island in uv_islands_data:
            if "error" in island:
                logger.warning(
                    f"Skipping island {island.get('segment_id', 'unknown')} due to error: {island['error']}"
                )
                continue

            island_copy = island.copy()

            # Convert vertex_uvs to serializable format
            if "vertex_uvs" in island_copy:
                island_copy["vertex_uvs"] = {
                    str(k): v for k, v in island["vertex_uvs"].items()
                }

            serializable_data.append(island_copy)

        # Store as custom property with error handling
        try:
            obj["uv_island_segments"] = json.dumps(serializable_data, indent=2)
            logger.info(
                f"Stored UV data for {len(serializable_data)} segments on {obj.name}"
            )
        except Exception as e:
            logger.error(f"Failed to store UV data on object: {e}")
            # Store basic fallback data
            obj["uv_segments_count"] = len(serializable_data)

    except Exception as e:
        logger.error(f"UV island storage error: {e}")


def execute_smart_cut(
    obj: bpy.types.Object,
    angle_limit: float,
    seam_mode: str,
    cluster_search_aggressiveness: int = 10,
    capture_uv_data: bool = True,
) -> List[Tuple[int, int]]:
    """
    Professional mesh segmentation with comprehensive error handling and memory safety.

    Implements hybrid cutting strategy with guaranteed resource cleanup and robust
    input validation. Supports both AUTO and CUSTOM segmentation modes with
    advanced topology analysis and UV data capture.

    Technical Implementation:
    - Multi-stage input validation with specific error reporting
    - Safe BMesh lifecycle management with context managers
    - Defensive programming against edge cases
    - Performance optimization for large meshes

    Args:
        obj: Blender mesh object for segmentation
        angle_limit: Maximum angle difference for segment grouping (degrees)
        seam_mode: Segmentation strategy ("AUTO" or "CUSTOM")
        cluster_search_aggressiveness: Decomposition iteration limit for complex segments
        capture_uv_data: Enable UV island data capture during processing

    Returns:
        List of boundary edge pairs as tuples of vertex indices
        Empty list for invalid inputs or processing errors

    Error Handling:
        - Graceful degradation for all error conditions
        - Detailed logging for diagnostic purposes
        - Guaranteed resource cleanup regardless of execution path
    """
    logger.info(
        f"=== Smart Cut: mode={seam_mode}, angle={angle_limit}°, capture_uv={capture_uv_data} ==="
    )

    # PHASE 1: COMPREHENSIVE INPUT VALIDATION
    is_valid, error_msg = validate_mesh_object(obj)
    if not is_valid:
        logger.warning(f"execute_smart_cut validation failed: {error_msg}")
        return []

    # PHASE 2: SAFE BMESH PROCESSING
    try:
        with safe_bmesh(obj.data) as bm:
            # Initialize processing
            ensure_material_sync(obj, bm)

            # Create face ID tracking layer
            orig_layer = bm.faces.layers.int.new("orig_face_id")
            for face in bm.faces:
                if face.is_valid:
                    face[orig_layer] = face.index

            # PHASE 3: SEGMENTATION STRATEGY EXECUTION
            final_segments = []

            if seam_mode == "CUSTOM":
                user_seams = [e for e in bm.edges if e.seam and e.is_valid]
                if user_seams:
                    logger.info(f"CUSTOM mode: Processing {len(user_seams)} user seams")
                    segments_to_process = get_segments_from_seams(bm)
                else:
                    logger.warning(
                        "CUSTOM mode: No seams found, falling back to AUTO mode"
                    )
                    segments_to_process = get_segments_from_uv_islands(
                        bm, list(bm.faces)
                    )
            else:
                logger.info(
                    "AUTO mode: UV island detection followed by angular segmentation"
                )
                segments_to_process = get_segments_from_uv_islands(bm, list(bm.faces))

            # PHASE 4: ANGULAR SEGMENTATION AND TOPOLOGY PROCESSING
            for segment_faces in segments_to_process:
                if not segment_faces:
                    continue

                # Apply angular cutting to each primary segment
                sub_segments = get_segments_by_angle_on_subset(
                    bm, segment_faces, angle_limit
                )

                for sub_segment in sub_segments:
                    if not sub_segment:
                        continue

                    # Analyze topology for holes
                    topology = calculate_topology_invariants(sub_segment)

                    if topology["holes"] > 0:
                        logger.warning(
                            f"Complex segment detected: {topology['holes']} holes, decomposing..."
                        )
                        decomposed_islands = decompose_complex_segment(
                            sub_segment, aggressiveness=cluster_search_aggressiveness
                        )
                        final_segments.extend(decomposed_islands)
                    else:
                        final_segments.append(sub_segment)

            # PHASE 5: RESULTS PROCESSING
            if not final_segments:
                logger.error("No final segments created during processing")
                return []

            # Process segments and capture UV data
            if capture_uv_data:
                process_final_segments_with_uv_capture(bm, final_segments, obj)
            else:
                process_final_segments(bm, final_segments, obj)

            # Extract boundary edges
            boundary_edges = get_boundary_edges_from_segments(bm, final_segments)

            # Update original mesh data
            bm.to_mesh(obj.data)

            logger.info(
                f"=== Segmentation complete: {len(final_segments)} segments, {len(boundary_edges)} boundary edges ==="
            )
            return boundary_edges

    except Exception as e:
        logger.error(f"execute_smart_cut processing error: {e}")
        return []


def get_segments_by_angle_on_subset(
    bm: bmesh.types.BMesh, faces_subset: List[bmesh.types.BMFace], angle_limit: float
) -> List[List[bmesh.types.BMFace]]:
    """
    Enhanced angular segmentation with improved performance and error handling.

    Groups faces within a subset based on normal angle differences using breadth-first
    traversal with material awareness and validity checking.
    """
    if not faces_subset:
        return []

    segments = []
    processed_faces = set()
    subset_face_set = set(faces_subset)

    for seed_face in faces_subset:
        if (
            not seed_face.is_valid
            or seed_face in processed_faces
            or seed_face.normal.length < 1e-6
        ):
            continue

        # Initialize segment with seed face
        visited_this_pass = set()
        queue = deque([seed_face])
        current_segment = []

        seed_normal = seed_face.normal.copy()
        seed_material_index = seed_face.material_index

        while queue:
            current_face = queue.popleft()

            if (
                current_face in visited_this_pass
                or not current_face.is_valid
                or current_face.normal.length < 1e-6
            ):
                continue

            visited_this_pass.add(current_face)
            current_segment.append(current_face)

            # Check adjacent faces
            for edge in current_face.edges:
                if not edge.is_valid:
                    continue

                for linked_face in edge.link_faces:
                    if (
                        linked_face in subset_face_set
                        and linked_face not in visited_this_pass
                        and linked_face.is_valid
                        and linked_face.material_index == seed_material_index
                        and linked_face.normal.length >= 1e-6
                    ):
                        # Check angle threshold
                        angle = degrees(seed_normal.angle(linked_face.normal))
                        if angle < angle_limit:
                            queue.append(linked_face)

        processed_faces.update(visited_this_pass)

        if current_segment:
            segments.append(current_segment)

    return segments


def get_boundary_edges_from_segments(
    bm: bmesh.types.BMesh, segments: List[List[bmesh.types.BMFace]]
) -> List[Tuple[int, int]]:
    """
    Enhanced boundary edge detection with validation and error handling.

    Identifies edges that form boundaries between different segments using
    efficient face-to-segment mapping.
    """
    if not segments or len(segments) < 2:
        return []

    try:
        # Create face-to-segment mapping
        face_to_segment_id = {}
        for segment_id, segment in enumerate(segments):
            for face in segment:
                if face.is_valid:
                    face_to_segment_id[face] = segment_id

        boundary_edges = set()

        # Find inter-segment edges
        for face in face_to_segment_id:
            current_segment_id = face_to_segment_id[face]

            for edge in face.edges:
                if not edge.is_valid:
                    continue

                for linked_face in edge.link_faces:
                    if (
                        linked_face in face_to_segment_id
                        and face_to_segment_id[linked_face] != current_segment_id
                    ):
                        # Create sorted edge tuple
                        if all(v.is_valid for v in edge.verts):
                            edge_tuple = tuple(sorted(v.index for v in edge.verts))
                            boundary_edges.add(edge_tuple)

        result = list(boundary_edges)
        logger.info(
            f"Boundary detection: {len(result)} edges between {len(segments)} segments"
        )
        return result

    except Exception as e:
        logger.error(f"Boundary edge detection error: {e}")
        return []


def get_segments_from_seams(bm: bmesh.types.BMesh) -> List[List[bmesh.types.BMFace]]:
    """
    Enhanced seam-based segmentation with improved connectivity analysis.

    Uses breadth-first traversal to group faces separated by seam edges,
    with comprehensive validation and error handling.
    """
    segments = []
    visited_faces = set()

    for seed_face in bm.faces:
        if not seed_face.is_valid or seed_face in visited_faces:
            continue

        current_segment = []
        queue = deque([seed_face])
        visited_faces.add(seed_face)

        while queue:
            current_face = queue.popleft()
            current_segment.append(current_face)

            for edge in current_face.edges:
                if not edge.is_valid or edge.seam:
                    continue

                for linked_face in edge.link_faces:
                    if linked_face.is_valid and linked_face not in visited_faces:
                        visited_faces.add(linked_face)
                        queue.append(linked_face)

        if current_segment:
            segments.append(current_segment)

    logger.info(f"Seam segmentation: {len(segments)} segments found")
    return segments


def ensure_material_sync(obj: bpy.types.Object, bm: bmesh.types.BMesh):
    """
    Enhanced material synchronization with comprehensive error handling.

    Synchronizes material indices between mesh object and BMesh representation
    with validation and fallback behavior.
    """
    try:
        if not obj.data.materials:
            logger.info(f"Object '{obj.name}' has no materials, using default indices")
            for face in bm.faces:
                if face.is_valid:
                    face.material_index = 0
            return

        mesh_polygon_count = len(obj.data.polygons)
        bmesh_face_count = len(bm.faces)

        if mesh_polygon_count != bmesh_face_count:
            logger.error(
                f"Face count mismatch: Mesh={mesh_polygon_count}, BMesh={bmesh_face_count}"
            )
            return

        # Synchronize material indices
        sync_count = 0
        for mesh_poly, bm_face in zip(obj.data.polygons, bm.faces):
            if bm_face.is_valid:
                bm_face.material_index = mesh_poly.material_index
                sync_count += 1

        logger.debug(f"Material sync completed: {sync_count} faces processed")

    except Exception as e:
        logger.error(f"Material synchronization error: {e}")


def calculate_topology_invariants(segment_faces: List[bmesh.types.BMFace]) -> Dict:
    """
    Enhanced topological analysis with comprehensive validation.

    Calculates Euler characteristic and geometric properties for topology
    analysis with robust error handling and edge case management.
    """
    if not segment_faces:
        return {"vertices": 0, "edges": 0, "faces": 0, "holes": 0, "euler_char": 0}

    try:
        unique_vertices = set()
        unique_edges = set()
        valid_faces = []

        for face in segment_faces:
            if not face.is_valid:
                continue

            valid_faces.append(face)

            # Collect vertices
            for vert in face.verts:
                if vert.is_valid:
                    unique_vertices.add(vert.index)

            # Collect edges
            for edge in face.edges:
                if edge.is_valid:
                    edge_key = tuple(sorted(v.index for v in edge.verts if v.is_valid))
                    if len(edge_key) == 2:
                        unique_edges.add(edge_key)

        V = len(unique_vertices)
        E = len(unique_edges)
        F = len(valid_faces)

        if V == 0:
            return {"vertices": 0, "edges": 0, "faces": 0, "holes": 0, "euler_char": 0}

        # Calculate Euler characteristic for planar graphs
        euler_char = V - E + F
        holes = max(0, 1 - euler_char)

        return {
            "vertices": V,
            "edges": E,
            "faces": F,
            "holes": holes,
            "euler_char": euler_char,
        }

    except Exception as e:
        logger.error(f"Topology calculation error: {e}")
        return {"vertices": 0, "edges": 0, "faces": 0, "holes": 0, "euler_char": 0}


def build_face_adjacency_graph(faces: List[bmesh.types.BMFace]) -> Dict:
    """Enhanced face adjacency graph construction with validation."""
    adjacency = defaultdict(list)
    face_set = set(faces)

    for face1 in faces:
        if not face1.is_valid:
            continue

        for edge in face1.edges:
            if not edge.is_valid:
                continue

            for face2 in edge.link_faces:
                if face1 != face2 and face2.is_valid and face2 in face_set:
                    adjacency[face1].append(face2)

    return adjacency


def find_connected_components(
    faces: List[bmesh.types.BMFace], adjacency_graph: Dict
) -> List[List[bmesh.types.BMFace]]:
    """Enhanced connected components detection with comprehensive validation."""
    components = []
    visited = set()
    face_set = set(faces)

    for face in faces:
        if not face.is_valid or face in visited:
            continue

        component = []
        queue = deque([face])
        visited.add(face)

        while queue:
            current_face = queue.popleft()
            component.append(current_face)

            for neighbor in adjacency_graph.get(current_face, []):
                if (
                    neighbor.is_valid
                    and neighbor in face_set
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)

        if component:
            components.append(component)

    return components


def partition_faces_into_islands(
    faces: List[bmesh.types.BMFace], k: int, adjacency_graph: Dict
) -> List[List[bmesh.types.BMFace]]:
    """
    Enhanced k-means face partitioning with connectivity guarantee.

    Partitions faces into k islands using spatial clustering while ensuring
    each island maintains topological connectivity.
    """
    valid_faces = [f for f in faces if f.is_valid]
    if not valid_faces or k <= 0:
        return []

    if k >= len(valid_faces):
        return [[f] for f in valid_faces]

    try:
        # Calculate face centers for clustering
        face_centers = {}
        for face in valid_faces:
            try:
                face_centers[face] = face.calc_center_median()
            except Exception:
                # Fallback to vertex average
                center = Vector()
                vert_count = 0
                for vert in face.verts:
                    if vert.is_valid:
                        center += vert.co
                        vert_count += 1
                if vert_count > 0:
                    face_centers[face] = center / vert_count

        if len(face_centers) < k:
            return [[f] for f in face_centers.keys()]

        # Initialize cluster centers
        initial_centroids = random.sample(list(face_centers.keys()), k)
        cluster_centers = [face_centers[f] for f in initial_centroids]

        # K-means iteration
        max_iterations = 10
        for iteration in range(max_iterations):
            new_clusters = [[] for _ in range(k)]

            # Assign faces to nearest cluster
            for face in valid_faces:
                if face not in face_centers:
                    continue

                center = face_centers[face]
                distances = [(center - cc).length_squared for cc in cluster_centers]
                best_cluster_idx = distances.index(min(distances))
                new_clusters[best_cluster_idx].append(face)

            # Update cluster centers
            converged = True
            for i in range(k):
                if new_clusters[i]:
                    new_center = sum(
                        (face_centers[f] for f in new_clusters[i] if f in face_centers),
                        Vector(),
                    ) / len(new_clusters[i])

                    if (new_center - cluster_centers[i]).length > 1e-6:
                        converged = False
                    cluster_centers[i] = new_center

            if converged:
                break

        # Ensure connectivity of each cluster
        final_islands = []
        for cluster in new_clusters:
            if cluster:
                connected_components = find_connected_components(
                    cluster, adjacency_graph
                )
                final_islands.extend(connected_components)

        return final_islands

    except Exception as e:
        logger.error(f"Face partitioning error: {e}")
        return [[f] for f in valid_faces]


def decompose_complex_segment(
    segment_faces: List[bmesh.types.BMFace], aggressiveness: int = 10
) -> List[List[bmesh.types.BMFace]]:
    """
    Enhanced complex segment decomposition with iterative refinement.

    Decomposes topologically complex segments using progressive k-means
    clustering with connectivity validation.
    """
    logger.info(f"Decomposing complex segment: {len(segment_faces)} faces")

    if not segment_faces:
        return []

    # Build adjacency graph for connectivity analysis
    adjacency_graph = build_face_adjacency_graph(segment_faces)

    # Check if already simple connected components
    components = find_connected_components(segment_faces, adjacency_graph)
    if len(components) > 1:
        all_simple = all(
            calculate_topology_invariants(comp)["holes"] == 0 for comp in components
        )
        if all_simple:
            logger.info(
                f"Natural decomposition into {len(components)} simple components"
            )
            return components

    # Progressive k-means decomposition
    n_faces = len(segment_faces)
    for k in range(2, min(n_faces + 1, 20)):  # Limit maximum k for performance
        best_partition = None
        for attempt in range(aggressiveness):
            logger.debug(f"Decomposition attempt: k={k}, iteration={attempt + 1}")

            try:
                islands = partition_faces_into_islands(
                    segment_faces, k, adjacency_graph
                )

                if not islands:
                    continue

                # Check if all islands are topologically simple
                all_simple = True
                for island in islands:
                    topology = calculate_topology_invariants(island)
                    if topology["holes"] > 0:
                        all_simple = False
                        break

                if all_simple:
                    best_partition = islands
                    break

            except Exception as e:
                logger.warning(f"Decomposition attempt failed: {e}")
                continue

        if best_partition:
            logger.info(
                f"Successful decomposition: {len(best_partition)} simple islands (k={k})"
            )
            return best_partition

    # Fallback: return individual faces
    logger.warning(
        f"Could not decompose segment of {n_faces} faces, returning individual faces"
    )
    return [[face] for face in segment_faces if face.is_valid]


def process_final_segments(
    bm: bmesh.types.BMesh,
    final_segments: List[List[bmesh.types.BMFace]],
    original_obj: bpy.types.Object,
):
    """
    Enhanced segment processing with comprehensive error handling and validation.

    Creates individual mesh objects for each segment with proper face ID preservation
    and resource management.
    """
    logger.info(f"Processing {len(final_segments)} final segments")

    if not bm.is_valid:
        logger.error("Invalid BMesh structure")
        return

    # Determine base name for segment objects
    orig_base_name = original_obj.name
    if orig_base_name.endswith("_copy_for_uv"):
        orig_base_name = orig_base_name[: -len("_copy_for_uv")]

    # Validate face ID layer
    orig_layer = bm.faces.layers.int.get("orig_face_id")
    if not orig_layer:
        logger.error("Missing 'orig_face_id' layer in BMesh")
        return

    successful_segments = 0

    for idx, segment_faces in enumerate(final_segments):
        try:
            # Filter valid faces
            valid_faces = [f for f in segment_faces if f.is_valid and f in bm.faces]
            if not valid_faces:
                logger.warning(f"Segment {idx}: No valid faces")
                continue

            # Collect unique vertices
            vertex_indices = set()
            for face in valid_faces:
                for vert in face.verts:
                    if vert.is_valid:
                        vertex_indices.add(vert.index)

            if not vertex_indices:
                logger.warning(f"Segment {idx}: No valid vertices")
                continue

            # Create new mesh object
            segment_name = f"{orig_base_name}_Segment_{idx:03d}"
            new_mesh = bpy.data.meshes.new(segment_name)
            new_obj = bpy.data.objects.new(segment_name, new_mesh)
            bpy.context.collection.objects.link(new_obj)

            # Process segment with safe BMesh
            with safe_bmesh() as bm_new:
                # Create vertex mapping
                vert_mapping = {}
                for v_idx in vertex_indices:
                    if v_idx < len(bm.verts) and bm.verts[v_idx].is_valid:
                        vert_mapping[v_idx] = bm_new.verts.new(bm.verts[v_idx].co)

                if not vert_mapping:
                    logger.warning(f"Segment {idx}: No valid vertex mapping")
                    continue

                # Create face ID layer
                face_layer_new = bm_new.faces.layers.int.new("face_id")

                # Create faces
                faces_created = 0
                for face in valid_faces:
                    face_vert_indices = [
                        v.index
                        for v in face.verts
                        if v.is_valid and v.index in vert_mapping
                    ]

                    if len(face_vert_indices) < 3:
                        continue

                    new_verts = [vert_mapping[v_idx] for v_idx in face_vert_indices]

                    try:
                        # Check for duplicate faces
                        if not bm_new.faces.get(new_verts):
                            face_new = bm_new.faces.new(new_verts)
                            face_new[face_layer_new] = face[orig_layer]
                            faces_created += 1
                    except ValueError as e:
                        logger.debug(f"Face creation failed: {e}")
                        continue

                if faces_created > 0:
                    bm_new.to_mesh(new_mesh)
                    successful_segments += 1
                    logger.info(f"Created '{segment_name}' with {faces_created} faces")
                else:
                    # Cleanup empty object
                    bpy.data.objects.remove(new_obj, do_unlink=True)
                    bpy.data.meshes.remove(new_mesh)
                    logger.warning(f"Segment {idx}: No faces created, object removed")

        except Exception as e:
            logger.error(f"Segment {idx} processing error: {e}")
            continue

    logger.info(
        f"Successfully created {successful_segments}/{len(final_segments)} segment objects"
    )


def process_final_segments_with_uv_capture(
    bm: bmesh.types.BMesh,
    final_segments: List[List[bmesh.types.BMFace]],
    original_obj: bpy.types.Object,
):
    """
    Enhanced segment processing with UV data capture and validation.

    Combines segment object creation with comprehensive UV island data capture
    for advanced texture workflows.
    """
    logger.info(f"Processing {len(final_segments)} segments with UV capture")

    # Validate UV layer availability
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        logger.warning("No active UV layer, creating default layer")
        uv_layer = bm.loops.layers.uv.new("UVMap_segments")

    # Capture UV data for all segments
    all_uv_islands = []

    for idx, segment_faces in enumerate(final_segments):
        if not segment_faces:
            continue

        valid_faces = [f for f in segment_faces if f.is_valid and f in bm.faces]
        if not valid_faces:
            continue

        # Capture UV island data
        uv_island_data = capture_segment_uv_islands(bm, valid_faces, idx)
        if "error" not in uv_island_data:
            all_uv_islands.append(uv_island_data)

    # Store UV data on original object
    if all_uv_islands:
        store_uv_islands_on_object(original_obj, all_uv_islands)

    # Continue with standard segment processing
    process_final_segments(bm, final_segments, original_obj)


def register():
    """Register module components."""
    logger.debug("Registering enhanced plane_cut module")


def unregister():
    """Unregister module components."""
    logger.debug("Unregistering enhanced plane_cut module")


if __name__ == "__main__":
    # Module test and diagnostic functionality
    logger.info("Enhanced plane_cut module loaded successfully")

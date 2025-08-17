# json_merger.py
# pylint: disable=import-error
"""
This module is responsible for merging multiple Spine2D JSON files into a single, coherent file.
Its primary purpose is to combine a "main" JSON file, which contains the base skeleton and animation structure, with several "segment" JSON files, each representing a piece of a larger mesh.

The key functionalities are:
1.  JSON Merging Logic: It takes a main JSON and a list of segment JSONs and intelligently combines them. It preserves the `skeleton`, `transform`, and `animations` sections from the main file while merging the `bones`, `slots`, and `skins` from all files.
2.  Bone Index Recalculation: When merging, it creates a global bone list. It then iterates through the mesh attachments from the segments and recalculates the bone indices in their `vertices` arrays to point to the correct indices in the new global list.
3.  Slot and Skin Combination: It adds new slots and skins from the segments to the main JSON, ensuring there are no duplicate slot names and correctly merging attachments into existing skins.
4.  Sequence Animation Generation: After merging, it processes any `sequence` metadata found in the mesh attachments and generates the corresponding keyframe animation timeline in the `animations` section, allowing for frame-by-frame animation playback in Spine.
5.  Vertex Bone Optimization: It includes a function to identify and remove duplicate vertex bones (bones at the exact same location within the same parent Z-group), optimizing the final bone count.
6.  Standalone Capability: The module can be run as a standalone script for testing or batch processing, using the `main()` function to load files from a specified directory and perform the merge.

ATTENTION: - The merging logic is based on a strict rule: the `skeleton`, `transform`, and `animations` sections are taken *only* from the main JSON file and are not modified by the segments. The core complexity lies in correctly re-mapping bone indices; any error in this process will result in a corrupted or incorrectly weighted mesh in Spine. The file paths and naming conventions for loading main vs. segment files are important for the standalone script to function correctly.
Author: Maxim Sokolenko
"""
import os
import json
import logging

logger = logging.getLogger(__name__)
from typing import Dict, List, Any, Tuple
from .config import SEQUENCE_FRAME_DELAY


# =============================================================================
# Helper functions for reading/writing JSON
# =============================================================================
def read_json(file_path: str) -> Dict[str, Any]:
    """
    Reads JSON data from the file_path and returns it as a dict.
    If the file is not found or a reading error occurs, returns an empty dict.
    """
    if not os.path.isfile(file_path):
        logger.error(f"File {file_path} not found.")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error reading JSON {file_path}: {e}")
        return {}


def write_json(data: Dict[str, Any], out_path: str) -> None:
    """
    Writes the dict data to the out_path file in JSON format (indent=4).
    """
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"File saved successfully: {out_path}")
    except OSError as e:
        logger.error(f"Failed to save file {out_path}: {e}")


# =============================================================================
# Main merge functions
# =============================================================================
def ensure_key(data_dict: Dict[str, Any], key: str, default_value: Any) -> Any:
    """
    Ensures the presence of the key in data_dict. If the key is not present, sets the default_value.
    Returns data_dict[key].
    """
    if key not in data_dict:
        data_dict[key] = default_value
    return data_dict[key]


def build_global_bone_index(main_bones: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Forms a name->index dictionary for bones,
    using their order (index) in the main_bones list.
    """
    bone_name_to_index: Dict[str, int] = {}
    for i, bone in enumerate(main_bones):
        bone_name_to_index[bone["name"]] = i
    return bone_name_to_index


def merge_bones_and_recalc_indices(
    global_bones: List[Dict[str, Any]],
    bone_name_to_index: Dict[str, int],
    seg_bones: List[Dict[str, Any]],
) -> Dict[int, int]:
    """
    For the bones of the segment seg_bones, we determine how their local indices (in seg_bones)
    map to global indices (in global_bones).

    - If a bone with this name already exists in global_bones, we take its index.
      (if desired, you can check for conflicts and the priority of the main one)
    - If not, we add it to global_bones and create a new index.

    Returns a mapping: segmentBoneIndex -> globalBoneIndex.
    """
    segment_index_to_global_index: Dict[int, int] = {}

    start_global_length = len(global_bones)

    for seg_index, bone in enumerate(seg_bones):
        bname = bone["name"]
        if bname in bone_name_to_index:
            # Such a bone already exists in the global list
            existing_global_idx = bone_name_to_index[bname]
            g_index = existing_global_idx
        else:
            # Add a new bone
            global_bones.append(bone)
            g_index = len(global_bones) - 1
            bone_name_to_index[bname] = g_index

        segment_index_to_global_index[seg_index] = g_index

    logger.debug(
        f"Before the bones of the segment were added {start_global_length}, now {len(global_bones)} bones."
    )
    return segment_index_to_global_index


def merge_slots(
    global_slots: List[Dict[str, Any]],
    seg_slots: List[Dict[str, Any]],
    seg_index_to_global_index: Dict[int, int],
    global_bones: List[Dict[str, Any]],
) -> None:
    """
    Adds segment slots to the global list and ensures
    that the "bone" field contains the _name_ of an existing bone.

    If in the segment the bone is specified by a number, we convert it to a name
    (via segment_index_map → global index → global_bones[...]['name']).
    """
    existing_slot_names = {s["name"] for s in global_slots}
    index_to_name = {i: b["name"] for i, b in enumerate(global_bones)}

    for slot in seg_slots:
        sname = slot.get("name")
        if sname in existing_slot_names:
            logger.debug(f"[merge_slots] Existing slot skipped «{sname}».")
            continue

        bone_field = slot.get("bone")
        bone_name: str

        # 1) bone is already a string → leave it
        if isinstance(bone_field, str):
            bone_name = bone_field
            if bone_name not in index_to_name.values():
                logger.warning(
                    f"[merge_slots] Slot «{sname}» specified bone «{bone_name}», "
                    f"which is not in the global list. I leave it as is."
                )

        # 2) bone is a number → convert it
        elif isinstance(bone_field, (int, float)):
            local_idx = int(bone_field)
            global_idx = seg_index_to_global_index.get(local_idx, local_idx)
            bone_name = index_to_name.get(global_idx, "root")
            logger.debug(
                f"[merge_slots] Slot «{sname}»: bone {local_idx}→{global_idx} "
                f"(«{bone_name}»)."
            )

        # 3) unknown format → fallback
        else:
            logger.warning(
                f"[merge_slots] Slot «{sname}» has unknown field type «bone» "
                f"({type(bone_field)}). Using «root»."
            )
            bone_name = "root"

        new_slot = json.loads(json.dumps(slot))
        new_slot["bone"] = bone_name
        global_slots.append(new_slot)
        existing_slot_names.add(sname)


def fix_attachment_bone_indices(
    attachment_data: Dict[str, Any], seg_index_to_global_index: Dict[int, int]
) -> None:
    """
    If attachment_data is a "mesh",
    then attachment_data["vertices"] contains an array of the form:
      [boneCount, boneIndex, x, y, weight, (repeat), ...]
    You need to go through this array and replace boneIndex using seg_index_to_global_index.
    """
    if attachment_data.get("type") != "mesh":
        return
    if "vertices" not in attachment_data:
        return

    old_vertices = attachment_data["vertices"]
    new_vertices = []
    i = 0
    while i < len(old_vertices):
        bone_count = old_vertices[i]
        new_vertices.append(bone_count)
        i += 1
        for _ in range(bone_count):
            old_bone_index = old_vertices[i]
            x = old_vertices[i + 1]
            y = old_vertices[i + 2]
            weight = old_vertices[i + 3]

            # Change the index to global
            if old_bone_index not in seg_index_to_global_index:
                logger.warning(
                    f"The segment encountered bone_index={old_bone_index}, "
                    "not found in the segment dictionary. Replace with 0 (root)."
                )
                new_bone_index = 0
            else:
                new_bone_index = seg_index_to_global_index[old_bone_index]

            new_vertices.extend([new_bone_index, x, y, weight])
            i += 4

    attachment_data["vertices"] = new_vertices


def fix_attachments_bone_indices(
    skin_data: Dict[str, Any], seg_index_to_global_index: Dict[int, int]
) -> None:
    """
    For the skin (skin_data), we go through all attachments (mesh) and fix their boneIndex,
    using fix_attachment_bone_indices.
    """
    attachments_dict = skin_data.get("attachments", {})
    for slot_name, attach_dict in attachments_dict.items():
        for att_name, att_data in attach_dict.items():
            fix_attachment_bone_indices(att_data, seg_index_to_global_index)


def merge_skins(
    main_skins: List[Dict[str, Any]],
    seg_skins: List[Dict[str, Any]],
    segment_index_to_global_index: Dict[int, int],
) -> None:
    """
    Merges the list of skins seg_skins into main_skins.
    For each mesh-attachment, it recalculates the bone indices.
    If a skin already exists (by name), then we merge its attachments.
    """
    for seg_skin in seg_skins:
        seg_skin_name = seg_skin["name"]
        # Search in main_skins for a skin with the same name
        main_skin = None
        for ms in main_skins:
            if ms["name"] == seg_skin_name:
                main_skin = ms
                break

        if main_skin is None:
            # There is no such skin — add it completely
            new_skin_copy = json.loads(json.dumps(seg_skin))  # deep copy
            fix_attachments_bone_indices(new_skin_copy, segment_index_to_global_index)
            main_skins.append(new_skin_copy)
        else:
            # Merge attachments
            seg_atts = seg_skin.get("attachments", {})
            main_atts = ensure_key(main_skin, "attachments", {})

            for slot_name, attach_dict in seg_atts.items():
                slot_main_atts = ensure_key(main_atts, slot_name, {})
                for att_name, att_data in attach_dict.items():
                    new_att = json.loads(json.dumps(att_data))  # copy
                    fix_attachment_bone_indices(new_att, segment_index_to_global_index)
                    # If attachment_name already exists, we can overwrite or somehow merge
                    slot_main_atts[att_name] = new_att


# =============================================================================
# Function for merging already LOADED (or generated) data
# =============================================================================
def merge_spine_json_dicts(
    main_data: Dict[str, Any], segment_data_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Takes an already prepared dict with the "main" JSON (main_data)
    and a list of dicts (segment_data_list),
    and merges them (bones, slots, skins).
    At the same time:
     - 'skeleton', 'transform', 'animations' are left from main_data (not supplemented).
     - New bones, slots, attachments from segments are added to main_data.
     - Recalculates bone indices in vertices (mesh attachments).

    Returns the modified main_data (and returns it).
    """
    if not main_data:
        logger.error("main_data is empty, returning as is.")
        return main_data

    # Make sure the necessary fields are present
    main_bones = ensure_key(main_data, "bones", [])
    main_slots = ensure_key(main_data, "slots", [])
    main_skins = ensure_key(main_data, "skins", [])

    # Build a map of bone name -> index (as of the beginning)
    bone_name_to_index = build_global_bone_index(main_bones)

    # For each segment, merge bones, slots, skins
    for seg_data in segment_data_list:
        seg_bones = seg_data.get("bones", [])
        seg_slots = seg_data.get("slots", [])
        seg_skins = seg_data.get("skins", [])

        # a) bones
        segment_index_map = merge_bones_and_recalc_indices(
            main_bones, bone_name_to_index, seg_bones
        )

        # b) slots
        merge_slots(main_slots, seg_slots, segment_index_map, main_bones)

        # c) skins (+recalculation of bone indices)
        merge_skins(main_skins, seg_skins, segment_index_map)

        # The rest (skeleton, transform, animations) — do not touch (according to the condition)
        # If desired, you could do a merge, but the condition says to take only from main.

        # After merging, delete the temporary key _uv3d_pairs if it is present
        if "_uv3d_pairs" in main_data:
            del main_data["_uv3d_pairs"]
            logger.debug("[INFO] _uv3d_pairs removed from concatenated JSON")
            # After merging, delete the temporary key textured_uv3d_pairs if it is present
        if "textured_uv3d_pairs" in main_data:
            del main_data["textured_uv3d_pairs"]
            logger.debug("[INFO] textured_uv3d_pairs removed from concatenated JSON")

    # d) Adding sequences to the animations section (if applicable)
    try:
        main_animations = ensure_key(main_data, "animations", {})
        # Take (or create) the main animation. Assume it is called "animation".
        main_animation = main_animations.get("animation")
        if main_animation is None:
            # If the "animation" key is not present, take the first available one or create a new one
            anim_name = (
                list(main_animations.keys())[0] if main_animations else "animation"
            )
            main_animation = main_animations.setdefault(anim_name, {})
        # Make sure the attachments sub-section exists
        attachments_timeline = ensure_key(main_animation, "attachments", {})
        default_timeline = ensure_key(
            attachments_timeline, "default", {}
        )  # timeline for the "default" skin

        # Go through all attachments in the "default" skin of the main JSON
        for skin in main_data.get("skins", []):
            if skin.get("name") != "default":
                continue  # consider only the main "default" skin
            attachments = skin.get("attachments", {})
            for slot_name, slot_attachments in attachments.items():
                for attachment_name, attachment_data in slot_attachments.items():
                    seq_info = attachment_data.get("sequence")
                    if seq_info:
                        # If the attachment has sequence information, generate a timeline for it
                        try:
                            frame_count = int(seq_info.get("count", 0))
                            digits = int(seq_info.get("digits", 0))
                        except Exception as e:
                            logger.warning(
                                f"[merge_spine_json_dicts] Invalid sequence data for {attachment_name}: {e}"
                            )
                            frame_count = 0
                        if frame_count > 0:
                            # Form a list of sequence events
                            events = []
                            delay = SEQUENCE_FRAME_DELAY
                            # First event: loop with a given delay
                            events.append({"mode": "loop", "delay": round(delay, 4)})
                            # Other events: fix the display of each frame at the corresponding time
                            for frame_index in range(1, frame_count):
                                events.append(
                                    {
                                        "time": round(delay * frame_index, 4),
                                        "mode": "loop",
                                        "index": frame_index,
                                    }
                                )
                            # Add the generated list of events to animations -> attachments
                            slot_timeline = ensure_key(default_timeline, slot_name, {})
                            attachment_timeline = ensure_key(
                                slot_timeline, attachment_name, {}
                            )
                            attachment_timeline["sequence"] = events
                            logger.info(
                                f"[merge_spine_json_dicts] Added sequence events for slot '{slot_name}', attachment '{attachment_name}' ({frame_count} frames)."
                            )
        if default_timeline:
            logger.info(
                "[merge_spine_json_dicts] Sequential timelines attachments generated successfully."
            )
    except Exception as e:
        logger.exception(
            f"[merge_spine_json_dicts] Error adding sequences to animations: {e}"
        )
    optimize_vertex_bones(main_data)
    return main_data


# =============================================================================
# Below is a test piece that reads the main JSON file + all segments from the disk
# and saves the result to output. You can call python json_merger.py
# =============================================================================
def load_main_and_segments_from_disk(
    base_path: str,
    main_filename_prefix: str = "output_1_object_n_meshes_main_object",
    segment_filename_prefix: str = "output_1_object_n_meshes_Cube",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Searches in the base_path folder:
      - the main file, starting with main_filename_prefix,
      - all segment files, starting with segment_filename_prefix.

    Returns (main_data, [segment_data_1, segment_data_2, ...]),
    where main_data and segment_data_i are dicts.

    If the main file is not found, returns an empty dict for main.
    If there are no segments, returns an empty list.
    """
    main_data: Dict[str, Any] = {}
    segment_data_list: List[Dict[str, Any]] = []

    if not os.path.isdir(base_path):
        logger.error(f"Folder {base_path} does not exist or is inaccessible.")
        return main_data, segment_data_list

    all_files = os.listdir(base_path)
    for fname in all_files:
        if not fname.lower().endswith(".json"):
            continue

        fullpath = os.path.join(base_path, fname)

        if fname.startswith(main_filename_prefix):
            logger.debug(f"Found main JSON: {fname}")
            main_data = read_json(fullpath)
        elif fname.startswith(segment_filename_prefix):
            logger.debug(f"Found JSON segment: {fname}")
            seg_data = read_json(fullpath)
            if seg_data:
                segment_data_list.append(seg_data)

    if not main_data:
        logger.warning("Main JSON file not found or empty!")
    if not segment_data_list:
        logger.warning("No segments found in folder!")

    return main_data, segment_data_list


def optimize_vertex_bones(spine_json: dict, xy_tol: float = 1e-4) -> None:
    """
    Removes duplicate vertex-bones within a single Z-group (parent-bone),
    rewrites indices in attachment.vertices and rebuilds the bones list.

    xy_tol — tolerance for comparing X/Y.
    """
    log = logging.getLogger(__name__)
    try:
        bones: list = spine_json.get("bones", [])
        if not bones:
            return

        # --- Group vertex-bones by parent ---
        groups: dict[str, dict[tuple[int, int], int]] = {}
        dup_map: dict[int, int] = {}  # old_idx → master_idx
        keep_set: set[int] = set()

        for idx, bone in enumerate(bones):
            name = bone.get("name", "")
            if "_vertex_" not in name:
                keep_set.add(idx)
                continue

            parent = bone.get("parent", "")
            gx = round(float(bone.get("x", 0)) / xy_tol)
            gy = round(float(bone.get("y", 0)) / xy_tol)
            key = (gx, gy)

            gdict = groups.setdefault(parent, {})
            if key in gdict:
                dup_map[idx] = gdict[key]  # duplicate
            else:
                gdict[key] = idx  # first → master
                keep_set.add(idx)

        if not dup_map:
            log.info("[opt-vtx] no duplicates.")
            return

        # --- Rebuild bones and form old→new ---
        new_bones: list = []
        old2new: dict[int, int] = {}
        for old_idx, bone in enumerate(bones):
            if old_idx in keep_set:
                new_idx = len(new_bones)
                new_bones.append(bone)
                old2new[old_idx] = new_idx

        for dup_idx, master_old in dup_map.items():
            old2new[dup_idx] = old2new[master_old]

        log.info(
            "[opt-vtx] %d duplicates removed, %d vertex bones remaining.",
            len(dup_map),
            len(new_bones),
        )

        # --- Patch vertices in all mesh-attachments ---
        def patch_vertices(arr: list) -> list:
            if not arr:
                return arr
            out, i, n = [], 0, len(arr)
            while i < n:
                cnt = int(arr[i])
                out.append(cnt)
                i += 1
                for _ in range(cnt):
                    old = int(arr[i])
                    x = arr[i + 1]
                    y = arr[i + 2]
                    w = arr[i + 3]
                    out.extend([old2new.get(old, old), x, y, w])
                    i += 4
            return out

        for skin in spine_json.get("skins", []):
            for slot_atts in skin.get("attachments", {}).values():
                for att in slot_atts.values():
                    if att.get("type") != "mesh":
                        continue
                    att["vertices"] = patch_vertices(att.get("vertices", []))

        spine_json["bones"] = new_bones

    except Exception as exc:
        log.exception("[opt-vtx] optimization error: %s", exc)


def main() -> None:
    """
    Example test function:
    1) Read the main JSON + segments from the folder
    2) Merge
    3) Save the result
    """
    # For the test, the path can be set manually:
    base_path = r"E:\muse\Spine2D_Mesh_Visualizer\Spine2D_Mesh_Visualizer\example_output_1_object_n_meshes"

    # Final JSON
    output_file = os.path.join(
        r"E:\muse\Spine2D_Mesh_Visualizer\Spine2D_Mesh_Visualizer",
        "output_1_object_n_meshes_TEST.json",
    )

    # 1) Load the main file + segments
    main_data, segment_data_list = load_main_and_segments_from_disk(base_path)

    if not main_data:
        logger.error("Main JSON not found, script stopped.")
        return

    # 2) If there are no segments — just save main as is
    if not segment_data_list:
        write_json(main_data, output_file)
        return

    # 3) Merge
    merged_data = merge_spine_json_dicts(main_data, segment_data_list)

    # 4) Save the result
    write_json(merged_data, output_file)
    logger.info("Done! JSON test merge script complete.")


def register():
    logger.debug("[LOG] Logging json_merger.py")


def unregister():
    logger.debug("[LOG] Unregister json_merger.py")

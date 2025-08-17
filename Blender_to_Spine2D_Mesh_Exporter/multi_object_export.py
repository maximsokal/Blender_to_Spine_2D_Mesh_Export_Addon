# multi_object_export.py
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
This module provides the functionality to export multiple selected Blender objects into a single, unified Spine2D JSON file. It is designed to handle complex scenes where several objects need to be part of the same skeleton, either as a connected rig or as separate entities within one file.

The key functionalities are:
1.  Group-Based Export: It intelligently groups the selected objects based on a "Connect" property. Objects marked as "connected" are processed together to form a single, complex rig. Objects not marked are treated as standalone entities.
2.  Global Rig Construction: For connected objects, it builds a global "all_objects" rig. This rig includes a master set of control bones (`all_objects_rotation_X/Y/Z`) and a hierarchy of Z-layer bones. This allows the entire group of objects to be animated (rotated, scaled) as a single unit.
3.  Offset and Constraint Management: It calculates the 3D spatial offsets of each object relative to a base object and applies these offsets to their main bones within the global rig. It also generates and re-numbers all the necessary IK and Transform constraints to make the global rig functional.
4.  Individual Object Export: It iterates through each selected object (or group) and calls the main `save_uv_as_json` pipeline for each one, generating an intermediate JSON file. This ensures that each object is properly unwrapped, baked, and processed.
5.  Final Merging: After all individual JSONs are created, it performs a final merge. It combines all the bones, slots, and skins, recalculating bone indices as needed. It also organizes all animations into named "folders" within the final JSON (e.g., `object_1/animation`, `object_2/animation`) to keep them distinct.
6.  Texture Size Synchronization: It ensures that the texture dimensions set by the user in the UI are correctly applied and synchronized across all modules for every object being exported.

ATTENTION: - This module's logic is highly dependent on the structure of the JSON files produced by `main.py` and `json_export.py`. The process of building the global rig and re-numbering constraints is complex; incorrect calculations can lead to a broken rig in Spine. The "Connect" checkbox on each object is the primary driver for how objects are grouped, so its state is critical. The final merging step assumes that the intermediate JSON files have been successfully created for all objects.
Author: Maxim Sokolenko
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple
import bpy
from mathutils import Vector

# ── internal modules ────────────────────────────────────────────────────────
from .main import save_uv_as_json  # pylint: disable=import-error
from .config import (
    get_default_output_dir,
    get_texture_size,
    TEXTURE_HEIGHT as CFG_H,
    TEXTURE_WIDTH as CFG_W,
)
from .json_merger import (
    build_global_bone_index,
    merge_bones_and_recalc_indices,
    merge_slots,
    merge_skins,
)
from . import json_export as _je  # needed for calc_uniform_scale()

# ADDED: Import necessary modules for patching globals
from . import main as _main
from . import config as _cfg


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# helpers: calculation of offsets, Z-groups, bone constructor
# ---------------------------------------------------------------------------


def _calc_offsets(src_objs: List[bpy.types.Object]) -> Dict[str, Dict[str, float]]:
    """Offsets of each object relative to the *first* in the list (active)."""
    base = src_objs[0].matrix_world.translation
    out: Dict[str, Dict[str, float]] = {}
    for ob in src_objs:
        d: Vector = ob.matrix_world.translation - base
        out[ob.name] = {"dx": d.x, "dy": d.y, "dz": d.z}
        logger.debug("[off] %s  dx=%.4f  dy=%.4f  dz=%.4f", ob.name, d.x, d.y, d.z)
    return out


def _group_by_z(
    offsets: Dict[str, Dict[str, float]], eps: float = 1e-4
) -> Tuple[Dict[float, int], Dict[str, int]]:
    """Clustering dz → layer index. Returns (z_val→idx, obj→idx)."""
    z_vals: List[float] = []
    for v in offsets.values():
        z_r = round(v["dz"], 4)
        if not any(abs(z_r - z) <= eps for z in z_vals):
            z_vals.append(z_r)
    z_vals.sort(reverse=True)
    z_map = {z: i for i, z in enumerate(z_vals)}
    obj_map = {name: z_map[round(v["dz"], 4)] for name, v in offsets.items()}
    logger.debug("[z-groups] %s", z_map)
    return z_map, obj_map


def _mk_bone(
    name: str, parent: str | None, length: float = 0.0, **extra
) -> Dict[str, Any]:
    """Convenient factory method for describing a bone."""
    b: Dict[str, Any] = {
        "name": name,
        "length": length,
        "x": round(extra.pop("x", 0.0), 2),
        "y": round(extra.pop("y", 0.0), 2),
    }
    if parent is not None:
        b["parent"] = parent
    # pass any valid spine-JSON fields (rotation, scaleX, …)
    for k, v in extra.items():
        b[k] = v
    return b


# ---------------------------------------------------------------------------
# building the global rig all_objects_*
# ---------------------------------------------------------------------------
def _build_global_rig(
    bones: List[Dict[str, Any]], z_map: Dict[float, int], half_px: float
) -> None:
    """
    Creates the bones of the "all_objects" group with the correct hierarchy:

    root
    └─ all_objects_main
       ├─ all_objects_rotation_X/Y/Z     (controllers under root)
       ├─ all_objects                    (holder)
       │  └─ all_objects_scale_rotate_X  (length = half_px)
       │      ├─ all_objects_rotate_X
       │      │   └─ all_objects_n_scale …  (one for each Z-layer)
       │      └─ …IK-chain…
       └─ …
    """
    px = half_px * 2  # "length" of controllers = 1 px

    # ensure the presence of root
    if not bones or bones[0]["name"] != "root":
        bones.insert(0, {"name": "root"})

    new: List[Dict[str, Any]] = []
    add = new.append  # compact alias

    # ── base holder-bones ───────────────────────────────────────────────
    add(_mk_bone("all_objects_main", "root", length=half_px))
    add(_mk_bone("all_objects", "all_objects_main"))

    # scale-bone now with length half_px
    add(_mk_bone("all_objects_scale_rotate_X", "all_objects", length=half_px))

    # ── rotation controllers X/Y/Z (child → root) ─────────────────────────
    add(
        _mk_bone(
            "all_objects_rotation_X",
            "root",
            length=px,
            x=px,
            y=half_px,
            color="ff0000ff",
        )
    )
    add(_mk_bone("all_objects_rotation_Y", "root", length=px, x=px, color="00ff18ff"))
    add(
        _mk_bone(
            "all_objects_rotation_Z",
            "root",
            length=px,
            x=px,
            y=-half_px,
            color="002cffff",
        )
    )

    # ── working rotation bone X, attached to scale-bone ────────────────
    add(
        _mk_bone(
            "all_objects_rotate_X", "all_objects_scale_rotate_X", length=half_px * 0.1
        )
    )

    # ── IK-chain under "all_objects" (unchanged) ──────────────────────
    add(
        _mk_bone(
            "all_objects_rotate_X_constraint",
            "all_objects",
            length=half_px,
            rotation=-90,
            color="abe323ff",
        )
    )
    add(
        _mk_bone(
            "all_objects_rotate_X_constraint_scale_IK", "all_objects", rotation=-90
        )
    )
    add(
        _mk_bone(
            "all_objects_rotate_X_constraint_rotate_IK",
            "all_objects_rotate_X_constraint_scale_IK",
        )
    )
    add(
        _mk_bone(
            "all_objects_rotate_X_constraint_IK",
            "all_objects_rotate_X_constraint_rotate_IK",
            rotation=90,
            color="ff3f00ff",
            icon="ik",
        )
    )

    # ── Z-layers: scale-bones → CHILDREN of all_objects_rotate_X ───────────────────
    for z_val, idx in z_map.items():
        scale_bone = f"all_objects_{idx}_scale"
        layer_bone = f"all_objects_layer_{idx}"
        add(_mk_bone(scale_bone, "all_objects_rotate_X", length=half_px * 0.1))
        add(_mk_bone(layer_bone, scale_bone, length=half_px * 0.1))
        logger.debug("[build-rig] add Z-layer %s (dz=%.4f)", layer_bone, z_val)

    # insert the block immediately after root
    bones[1:1] = new


# ---------------------------------------------------------------------------
# offset of the main bones of individual objects
# ---------------------------------------------------------------------------


def _apply_offsets(
    bones: List[Dict[str, Any]],
    offsets: Dict[str, Dict[str, float]],
    obj2grp: Dict[str, int],
    px_per_unit: float,
) -> None:
    by_name = {b["name"]: b for b in bones}
    for obj_name, off in offsets.items():
        main = f"{obj_name}_main"
        b = by_name.get(main)
        if not b:
            logger.warning("[apply-off] main bone %s not found", main)
            continue
        grp_idx = obj2grp[obj_name]
        b["parent"] = f"all_objects_layer_{grp_idx}"
        b["x"] = round(off["dx"] * px_per_unit, 2)
        b["y"] = round(b.get("y", 0) + off["dy"] * px_per_unit, 2)
        logger.debug(
            "[apply-off] %s parent→%s  x=%.2f  y=%.2f",
            main,
            b["parent"],
            b["x"],
            b["y"],
        )


# ---------------------------------------------------------------------------
# constraints for the global rig
# ---------------------------------------------------------------------------


def _add_unique(
    dst: List[Dict[str, Any]], itm: Dict[str, Any], key: str = "name"
) -> None:
    if not any(itm[key] == d[key] for d in dst):
        dst.append(itm)


def _renumber_object_constraints(
    merged: Dict[str, Any],
    obj2grp: Dict[str, int],
) -> None:
    """
    Renumbers the order of the transform/ik-constraints of each object so that
    they go in blocks by Z-layer (see the formula in _build_global_constraints).
    """
    if not obj2grp:
        return

    n_groups = max(obj2grp.values()) + 1
    rot_x_start = 3
    rot_y_start = rot_x_start + n_groups
    global_scale_ik_order = rot_y_start + n_groups
    scale_ik_start = global_scale_ik_order + 1
    global_scale_order = scale_ik_start + n_groups
    scale_start = global_scale_order + 1
    rot_z_start = scale_start + n_groups

    # --- quick access ---------------------------------------------------
    tf_by_name = {c["name"]: c for c in merged.get("transform", [])}
    ik_by_name = {c["name"]: c for c in merged.get("ik", [])}

    for obj, g in obj2grp.items():
        tf_by_name.get(f"{obj}_rotation_X", {})["order"] = rot_x_start + g
        tf_by_name.get(f"{obj}_rotation_Y", {})["order"] = rot_y_start + g
        tf_by_name.get(f"{obj}_rotation_Z", {})["order"] = rot_z_start + g
        tf_by_name.get(f"{obj}_scale_constraint", {})["order"] = scale_start + g
        ik_by_name.get(f"{obj}_scale_constraint_IK", {})["order"] = scale_ik_start + g

    # for reliability, make sure that order = int
    for coll in (merged["transform"], merged["ik"]):
        for c in coll:
            if "order" in c:
                c["order"] = int(c["order"])


def _build_global_constraints(
    merged: Dict[str, Any],
    z_map: Dict[float, int],
    px: float,
    half_px: float,
) -> None:
    """
    Creates global IK- and transform-constraints for the all_objects_* group.
    The order numbers are calculated automatically from the number of Z-layers.
    """
    n_groups = len(z_map)  # how many object-layers
    # -- global order formula --
    rot_x_start = 3  # first number for <grp>_rotation_X
    rot_y_start = rot_x_start + n_groups
    global_scale_ik_order = rot_y_start + n_groups
    scale_ik_start = global_scale_ik_order + 1
    global_scale_order = scale_ik_start + n_groups
    scale_start = global_scale_order + 1
    # rot_z_start will be needed when renumbering objects

    ik_list = merged.setdefault("ik", [])
    tf_list = merged.setdefault("transform", [])

    # --- IK (global only) ------------------------------------------
    _add_unique(
        ik_list,
        {
            "name": "all_objects_scale_constraint_IK",
            "order": global_scale_ik_order,
            "bones": ["all_objects_rotate_X_constraint"],
            "target": "all_objects_rotate_X_constraint_IK",
            "compress": True,
            "stretch": True,
        },
    )

    # --- helper lists ------------------------------------------
    scale_bones = [f"all_objects_{i}_scale" for i in range(n_groups)]
    bone_names = {b["name"] for b in merged["bones"]}
    root_bones = [
        base
        for base in (
            n[:-5]
            for n in bone_names
            if n.endswith("_main") and n != "all_objects_main"
        )
        if base in bone_names
    ]

    # --- global transform-constraints ---------------------------------
    _add_unique(
        tf_list,
        {
            "name": "all_objects_rotation_X",
            "order": 0,
            "bones": scale_bones + ["all_objects"],
            "target": "all_objects_rotation_X",
            "rotation": 90,
            "local": True,
            "relative": True,
            "x": -px * 2,
            "y": -half_px,
            "scaleX": -1,
            "scaleY": -1,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
    )

    _add_unique(
        tf_list,
        {
            "name": "all_objects_rotation_Y",
            "order": 1,
            "bones": [
                "all_objects_rotate_X",
                "all_objects_rotate_X_constraint_rotate_IK",
            ],
            "target": "all_objects_rotation_Y",
            "local": True,
            "relative": True,
            "x": px,
            "scaleX": -1,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
    )

    _add_unique(
        tf_list,
        {
            "name": "all_objects_rotation_Z",
            "order": 2,
            "bones": root_bones,
            "target": "all_objects_rotation_Z",
            "local": True,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
    )

    _add_unique(
        tf_list,
        {
            "name": "all_objects_scale_constraint",
            "order": global_scale_order,
            "bones": scale_bones,
            "target": "all_objects_rotate_X_constraint",
            "scaleX": -1,
            "mixX": 0,
            "mixScaleX": 0,
            "mixShearY": 0,
        },
    )


# ---------------------------------------------------------------------------
# merging JSON-files of individual objects
# ---------------------------------------------------------------------------


def _merge_jsons(
    paths: List[str], out_path: str, src_objects: List[bpy.types.Object]
) -> None:
    logger.info("[multi-merge] Join %d files → %s", len(paths), out_path)

    datasets = [json.load(open(p, "r", encoding="utf-8")) for p in paths]

    offsets = _calc_offsets(src_objects)
    z_map, obj2grp = _group_by_z(offsets)
    px_unit = _je.calc_uniform_scale(CFG_W, CFG_H)
    half_px = px_unit / 2.0

    merged: Dict[str, Any] = {
        "skeleton": datasets[0]["skeleton"],
        "bones": [{"name": "root"}],
        "slots": [],
        "skins": [],
        "ik": [],
        "transform": [],
        "animations": {},
    }

    # global rig before merging segments
    _build_global_rig(merged["bones"], z_map, half_px)

    for idx, (jd, _) in enumerate(zip(datasets, src_objects), start=1):
        seg_map = merge_bones_and_recalc_indices(
            merged["bones"],
            build_global_bone_index(merged["bones"]),
            jd.get("bones", []),
        )
        merge_slots(merged["slots"], jd.get("slots", []), seg_map, merged["bones"])
        merge_skins(merged["skins"], jd.get("skins", []), seg_map)
        merged["ik"].extend(jd.get("ik", []))
        merged["transform"].extend(jd.get("transform", []))
        # renamed animations
        folder = f"object_{idx}"
        for a_name, a_data in jd.get("animations", {}).items():
            merged["animations"][f"{folder}/{a_name}"] = json.loads(json.dumps(a_data))

    if len(src_objects) > 1:
        _apply_offsets(merged["bones"], offsets, obj2grp, px_unit)
        _build_global_constraints(merged, z_map, px_unit, half_px)
        _renumber_object_constraints(merged, obj2grp)
    # remove duplicates by name
    merged["ik"] = list({c["name"]: c for c in merged["ik"]}.values())
    merged["transform"] = list({c["name"]: c for c in merged["transform"]}.values())

    json.dump(
        merged, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2
    )
    logger.info(
        "[multi-merge] DONE  bones=%d  slots=%d  skins=%d  anim=%d",
        len(merged["bones"]),
        len(merged["slots"]),
        len(merged["skins"]),
        len(merged["animations"]),
    )


# ---------------------------------------------------------------------------
# external wrapper: export + merge
# ---------------------------------------------------------------------------


def _resolve_tex_size(
    scene: bpy.types.Scene, ow: int | None, oh: int | None
) -> Tuple[int, int]:
    if ow and oh:
        return int(ow), int(oh)
    return tuple(map(int, get_texture_size(scene)))


def _sync_tex_globals(w: int, h: int) -> None:
    """
    FIXED: Correctly patches global variables across modules without using
    the faulty `importlib.reload`. This ensures the user-defined texture
    dimensions are respected during the export of each object.
    """
    # Patch the global variables in each module that uses them.
    _cfg.TEXTURE_WIDTH = w
    _cfg.TEXTURE_HEIGHT = h

    _je.TEXTURE_WIDTH = w
    _je.TEXTURE_HEIGHT = h

    _main.TEXTURE_WIDTH = w
    _main.TEXTURE_HEIGHT = h

    # Also update the globals in the current module, just in case.
    globals()["CFG_W"] = w
    globals()["CFG_H"] = h
    logger.debug(f"Globals synced: W={w}, H={h}")


def export_selected_objects(
    texture_w: int | None = None,
    texture_h: int | None = None,
    output_dir: str | None = None,
) -> str | None:
    """
    Combined multi-export:
      • objects with the "Connect" checkbox → complex rig (all_objects_…)
      • others → simple placement side by side
    At the end, all .json files are glued together by merge_object_jsons(), so animations/skins
    are saved for each segment.
    """
    try:
        scene = bpy.context.scene
        tw, th = _resolve_tex_size(scene, texture_w, texture_h)
        _sync_tex_globals(tw, th)

        # [FIXED] Absolutely reliable determination of the output directory
        # 1. First, we try to use the path passed to the function (from the UI)
        #    and immediately convert it to an absolute path.
        resolved_path = bpy.path.abspath(output_dir) if output_dir else ""

        # 2. If the path after conversion turned out to be invalid (empty),
        #    we use our reliable default function.
        if not resolved_path:
            out_dir = get_default_output_dir()
        else:
            out_dir = resolved_path

        # This check ensures that we never call makedirs with an empty path.
        if not out_dir:
            logger.error(
                "[multi-export] Unable to determine output directory. Export cancelled."
            )
            # It might be worth showing an error to the user
            # self.report({'ERROR'}, "Could not determine output directory. Save the .blend file first.")
            return None
        os.makedirs(out_dir, exist_ok=True)

        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        if not meshes:
            logger.warning("[multi-export] no mesh objects selected")
            return None

        # ── 1. grouping by the "Connect" flag ──────────────────────────
        connected = [
            o
            for o in meshes
            if getattr(o, "spine2d_connect_settings", None)
            and o.spine2d_connect_settings.enabled
        ]
        standalone = [o for o in meshes if o not in connected]

        logger.info(
            "[multi-export] connected=%s | standalone=%s",
            [o.name for o in connected],
            [o.name for o in standalone],
        )

        produced_paths: List[str] = []
        src_for_merge: List[bpy.types.Object] = []

        # ── 2. export "connected" as a group (if there are ≥2) ─────────────────
        if len(connected) >= 2:
            grp_json = _export_group(connected, tw, th, out_dir)
            if grp_json:
                produced_paths.append(grp_json)
                src_for_merge.append(connected[0])  # represents the whole group
        elif len(connected) == 1:
            # a single "connected" → actually a regular object
            standalone.append(connected[0])
            connected.clear()

        # ── 3. export all standalone (and possibly a single connected) ─
        for ob in standalone:
            try:
                # [FIXED] Get the result of the call
                json_path = save_uv_as_json(ob, tw, th, output_dir=out_dir)
                # [FIXED] Check that the path was returned (i.e., there was no error)
                if json_path:
                    produced_paths.append(json_path)
                    src_for_merge.append(ob)
                else:
                    # Log that the export failed for this object
                    logger.error(
                        "[multi-export] export failed for %s, skipping merge.", ob.name
                    )
            except Exception:
                # This except now catches only errors in the loop itself, not inside save_uv_as_json
                logger.exception(
                    "[multi-export] critical error during export loop for %s", ob.name
                )

        if not produced_paths:
            logger.error("[multi-export] nothing to merge, abort")
            return None

        # ── 4. final merge of all segments ─────────────────────────────
        base_name = src_for_merge[0].name
        final_dst = os.path.join(
            out_dir,
            f"{base_name}_plus_{len(src_for_merge)-1}_objects.json"
            if len(src_for_merge) > 1
            else f"{base_name}_merged.json",
        )
        _merge_simple(produced_paths, final_dst, src_for_merge)
        logger.info("[multi-export] FINAL JSON → %s", final_dst)
        return final_dst

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[multi-export] fatal: %s", exc)
        return None


# ADD 2️⃣: helper function for exporting a group
def _export_group(
    objs: List[bpy.types.Object], tex_w: int, tex_h: int, out_dir: str
) -> str | None:
    """
    Exports several objects (objs) to Spine-JSON
    and merges them into a single file (_merge_jsons).
    Returns the path to the created JSON or None on error.
    """
    try:
        scene = bpy.context.scene
        produced: List[str] = []

        # backup of global scene bake-parameters
        g_frames = scene.spine2d_frames_for_render
        g_start = scene.spine2d_bake_frame_start

        for ob in objs:
            try:
                bake = getattr(ob, "spine2d_bake_settings", None)
                if bake:
                    scene.spine2d_frames_for_render = bake.frames_for_render
                    scene.spine2d_bake_frame_start = bake.bake_frame_start
                save_uv_as_json(ob, tex_w, tex_h)
                produced.append(
                    os.path.join(
                        os.path.dirname(bpy.data.filepath) or os.getcwd(),
                        f"{ob.name}_merged.json",
                    )
                )
            finally:
                scene.spine2d_frames_for_render = g_frames
                scene.spine2d_bake_frame_start = g_start

        if not produced:
            logger.error(
                "[_export_group] nothing produced for %s", [o.name for o in objs]
            )
            return None

        base = objs[0].name
        dst = os.path.join(
            out_dir,
            f"{base}_plus_{len(objs)-1}_objects.json"
            if len(objs) > 1
            else f"{base}_merged.json",
        )
        _merge_jsons(produced, dst, objs)
        logger.debug("[_export_group] DONE → %s", dst)
        return dst
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("[_export_group] fatal: %s", exc)
        return None


def _extend_unique(dst: list, src: list, *, key: str = "name") -> None:
    """Adds all elements from src to dst that are not yet present by the key field."""
    seen = {d[key] for d in dst}
    for itm in src:
        if itm[key] not in seen:
            dst.append(itm)
            seen.add(itm[key])


def _merge_simple(
    json_paths: List[str], output_path: str, src_objects: List[bpy.types.Object]
) -> None:
    """
    Glues several *_merged.json files, recalculates bone indices
    and arranges ALL animations into "folders":

        object_1/animation_object_1
        object_1/preview_object_1
        object_1/<OtherName>_object_1
        object_2/animation_object_2
        …

    where idx = position of the object in the src_objects list (starts from 1).
    """
    logger.info("[multi-merge] Join %d files → %s", len(json_paths), output_path)
    if not json_paths:
        logger.error("[multi-merge] Empty list – abort.")
        return

    # ── read everything ──────────────────────────────────────────────────────────
    datasets: List[Dict[str, Any]] = []
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as fp:
            datasets.append(json.load(fp))

    merged: Dict[str, Any] = {
        "skeleton": datasets[0]["skeleton"],
        "bones": [],
        "slots": [],
        "skins": [],
        "ik": [],
        "transform": [],
        "animations": {},
    }

    # ── merge in the order of objects ─────────────────────────────────────
    for idx, (jd, obj) in enumerate(zip(datasets, src_objects), start=1):
        folder = f"object_{idx}"

        # 1️⃣  bones  +  index map
        bone_map = merge_bones_and_recalc_indices(
            merged["bones"],
            build_global_bone_index(merged["bones"]),
            jd.get("bones", []),
        )

        # 2️⃣  slots / skins
        merge_slots(merged["slots"], jd.get("slots", []), bone_map, merged["bones"])
        merge_skins(merged["skins"], jd.get("skins", []), bone_map)

        # 3️⃣  IK / Transform without duplicates
        _extend_unique(merged["ik"], jd.get("ik", []))
        _extend_unique(merged["transform"], jd.get("transform", []))

        # 4️⃣  Animations  →  object_N/<localName>
        for orig, anim_data in jd.get("animations", {}).items():
            if orig.lower().startswith("animation"):
                local = f"animation_object_{idx}"
            elif orig.lower().startswith("preview"):
                local = f"preview_object_{idx}"
            else:
                local = f"{orig}_object_{idx}"

            new_key = f"{folder}/{local}"
            # if it happens to match (unlikely) – add a suffix
            dup = 1
            while new_key in merged["animations"]:
                dup += 1
                new_key = f"{folder}/{local}__dup{dup}"

            merged["animations"][new_key] = json.loads(json.dumps(anim_data))

    # ── write the result ───────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(merged, fout, ensure_ascii=False, indent=2)

    logger.info(
        "[multi-merge] DONE  bones=%d  slots=%d  skins=%d  anim=%d",
        len(merged["bones"]),
        len(merged["slots"]),
        len(merged["skins"]),
        len(merged["animations"]),
    )

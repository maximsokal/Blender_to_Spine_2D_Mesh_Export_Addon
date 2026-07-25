#!/usr/bin/env python3
"""Smoke an installed Blender extension and execute its production export operator."""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
import sys, traceback
import bpy

def _args():
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(values)

def _write_report(path: Path, payload: dict[str, object]) -> None:
    resolved = path.resolve(strict=False); resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(resolved)

def _create_source():
    mesh = bpy.data.meshes.new("Spine2D_InstallGateMesh")
    mesh.from_pydata(((-1.0,-1.0,0.0),(1.0,-1.0,0.0),(1.0,1.0,0.0),(-1.0,1.0,0.0)),(),((0,1,2,3),)); mesh.update(calc_edges=True)
    material = bpy.data.materials.new("Spine2D_InstallGateMaterial"); material.diffuse_color=(0.2,0.6,0.9,1.0); mesh.materials.append(material)
    obj=bpy.data.objects.new("Spine2D_InstallGate",mesh); bpy.context.scene.collection.objects.link(obj)
    for candidate in bpy.context.view_layer.objects: candidate.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active=obj; return obj

def _run(namespace):
    module=importlib.import_module(namespace.module); output_root=namespace.output_root.resolve(strict=False); output_root.mkdir(parents=True,exist_ok=True); obj=_create_source(); scene=bpy.context.scene
    scene.spine2d_json_path=str(output_root); scene.spine2d_images_path="images"; scene.spine2d_texture_size=64; scene.spine2d_angle_limit=30; scene.spine2d_seam_maker_mode="AUTO"; scene.spine2d_control_icons=False; scene.spine2d_export_preview_animation=False; scene.spine2d_frames_for_render=0; scene.spine2d_bake_frame_start=0
    operators={"single":bpy.types.Operator.bl_rna_get_subclass_py("OBJECT_OT_save_uv_as_json"),"multi":bpy.types.Operator.bl_rna_get_subclass_py("OBJECT_OT_spine2d_multi_export")}
    if any(value is None for value in operators.values()): raise RuntimeError(f"installed operators are missing: {operators}")
    result=set(bpy.ops.object.save_uv_as_json())
    if "FINISHED" not in result: raise RuntimeError(f"production smoke operator returned {sorted(result)}")
    json_files=tuple(sorted(output_root.glob("*.json")))
    if len(json_files)!=1 or json_files[0].stat().st_size<=2: raise RuntimeError(f"smoke export did not create one usable JSON: {json_files}")
    return {"status":"passed","module":module.__name__,"object":obj.name_full,"operator_result":sorted(result),"json_files":[str(path) for path in json_files],"handlers":tuple(callback.__name__ for callback in bpy.app.handlers.depsgraph_update_post if "spine2d" in getattr(callback,"__name__","").lower())}

def main():
    namespace=_args()
    try: _write_report(namespace.report_json,_run(namespace))
    except Exception as exc:
        _write_report(namespace.report_json,{"status":"failed","exception_type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}); raise
if __name__=="__main__": main()

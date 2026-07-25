#!/usr/bin/env python3
"""Compare Rewrite outputs against an immutable reviewed Legacy baseline."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any,Mapping,Sequence
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1ParitySettings,compare_a1_exports
from tools.blender_a1_image_compare import _compare_pair,_image_files
class ImmutableBaselineError(RuntimeError): pass
def _tree_digest(root):
    resolved=root.expanduser().resolve(strict=False)
    if not resolved.is_dir(): raise ImmutableBaselineError(f"directory does not exist: {resolved}")
    files=tuple(sorted(path for path in resolved.rglob("*") if path.is_file()))
    if not files: raise ImmutableBaselineError(f"directory is empty: {resolved}")
    digest=hashlib.sha256()
    for path in files:
        relative=path.relative_to(resolved).as_posix().encode("utf-8"); digest.update(len(relative).to_bytes(8,"big")); digest.update(relative)
        with path.open("rb") as stream:
            while chunk:=stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()
def _load_mapping(path,label):
    resolved=path.expanduser().resolve(strict=False)
    if not resolved.is_file(): raise ImmutableBaselineError(f"{label} JSON does not exist: {resolved}")
    try: value=json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError,json.JSONDecodeError) as exc: raise ImmutableBaselineError(f"unable to read {label} JSON: {exc}") from exc
    if not isinstance(value,Mapping): raise ImmutableBaselineError(f"{label} JSON root must be an object")
    return value
def compare_immutable_baseline(baseline_root,candidate_root,*,json_name,absolute_tolerance=1e-4,relative_tolerance=1e-6,compare_animations=True,strict_edges=False,image_absolute_tolerance=4.0/255.0,image_max_differing_pixel_ratio=0.02,image_max_mean_absolute_delta=0.002):
    baseline=baseline_root.expanduser().resolve(strict=False); candidate=candidate_root.expanduser().resolve(strict=False); before_digest=_tree_digest(baseline); expected=_load_mapping(baseline/json_name,"Legacy baseline"); actual=_load_mapping(candidate/json_name,"Rewrite candidate"); settings=A1ParitySettings(absolute_tolerance=absolute_tolerance,relative_tolerance=relative_tolerance,compare_animations=compare_animations,nonessential_mesh_edges_are_errors=strict_edges); json_report=compare_a1_exports(expected,actual,settings); expected_images=_image_files(baseline/"images"); actual_images=_image_files(candidate/"images"); image_names_match=set(expected_images)==set(actual_images); image_reports=[]
    if image_names_match:
        for relative in sorted(expected_images): image_reports.append(_compare_pair(relative,expected_images[relative],actual_images[relative],absolute_tolerance=image_absolute_tolerance,max_differing_pixel_ratio=image_max_differing_pixel_ratio,max_mean_absolute_delta=image_max_mean_absolute_delta))
    after_digest=_tree_digest(baseline)
    if after_digest!=before_digest: raise ImmutableBaselineError("Legacy baseline changed during comparison")
    images_compatible=image_names_match and all(bool(item.get("compatible")) for item in image_reports)
    return {"compatible":bool(json_report.compatible and images_compatible),"baseline_digest":before_digest,"json":{"compatible":json_report.compatible,"error_count":json_report.error_count,"warning_count":json_report.warning_count,"issues":[{"severity":issue.severity.value,"code":issue.code,"path":issue.path,"message":issue.message} for issue in json_report.issues]},"images":{"compatible":images_compatible,"names_match":image_names_match,"expected_names":sorted(expected_images),"actual_names":sorted(actual_images),"reports":image_reports}}
def run(arguments:Sequence[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--baseline-root",type=Path,required=True); parser.add_argument("--candidate-root",type=Path,required=True); parser.add_argument("--json-name",required=True); parser.add_argument("--report-json",type=Path,required=True); ns=parser.parse_args(arguments); report=compare_immutable_baseline(ns.baseline_root,ns.candidate_root,json_name=ns.json_name); destination=ns.report_json.expanduser().resolve(strict=False); destination.parent.mkdir(parents=True,exist_ok=True); temporary=destination.with_name(f".{destination.name}.tmp"); temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temporary.replace(destination); return 0 if report["compatible"] else 1
if __name__=="__main__": raise SystemExit(run())

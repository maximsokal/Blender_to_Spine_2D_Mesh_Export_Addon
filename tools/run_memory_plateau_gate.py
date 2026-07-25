#!/usr/bin/env python3
"""Run Blender memory stress and evaluate post-warmup RSS plateau."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import subprocess
from typing import Any,Mapping,Sequence
from tools.prepare_package import _resolve_blender_executable
ROOT=Path(__file__).resolve().parents[1]; WORKER=ROOT/"tools"/"blender_memory_stress.py"
class MemoryPlateauGateError(RuntimeError): pass
def build_command(blender,fixture,output_root,report_json,*,warmup,iterations,sample_every):
    return [blender,"--background","--factory-startup","--debug-memory","--log-show-memory",str(fixture),"--python-exit-code","1","--python",str(WORKER),"--","--output-root",str(output_root),"--report-json",str(report_json),"--warmup",str(warmup),"--iterations",str(iterations),"--sample-every",str(sample_every)]
def evaluate_plateau(payload:Mapping[str,Any],*,maximum_tail_growth_bytes:int,maximum_slope_bytes_per_sample:float):
    raw_samples=payload.get("samples")
    if not isinstance(raw_samples,list) or len(raw_samples)<3: raise MemoryPlateauGateError("memory report must contain at least 3 samples")
    values=[]
    for index,item in enumerate(raw_samples):
        if not isinstance(item,Mapping): raise MemoryPlateauGateError(f"samples[{index}] must be an object")
        value=item.get("rss_bytes")
        if isinstance(value,bool) or not isinstance(value,int) or value<=0: raise MemoryPlateauGateError(f"samples[{index}].rss_bytes is invalid")
        values.append(value)
    tail=values[len(values)//2:]; growth=max(tail)-min(tail); x_mean=(len(tail)-1)/2.; y_mean=sum(tail)/len(tail); denominator=sum((x-x_mean)**2 for x in range(len(tail))); slope=0. if denominator==0. else sum((x-x_mean)*(value-y_mean) for x,value in enumerate(tail))/denominator; compatible=growth<=maximum_tail_growth_bytes and slope<=maximum_slope_bytes_per_sample
    return {"compatible":compatible,"tail_sample_count":len(tail),"tail_growth_bytes":growth,"slope_bytes_per_sample":slope,"maximum_tail_growth_bytes":maximum_tail_growth_bytes,"maximum_slope_bytes_per_sample":maximum_slope_bytes_per_sample}
def run(arguments:Sequence[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--blender",default=None); parser.add_argument("--fixture",type=Path,required=True); parser.add_argument("--work-root",type=Path,required=True); parser.add_argument("--warmup",type=int,default=3); parser.add_argument("--iterations",type=int,default=50); parser.add_argument("--sample-every",type=int,default=5); parser.add_argument("--max-tail-growth-mib",type=float,default=64.); parser.add_argument("--max-slope-mib-per-sample",type=float,default=2.); args=parser.parse_args(arguments)
    blender=str(_resolve_blender_executable(None if args.blender is None else Path(args.blender))); fixture=args.fixture.expanduser().resolve(strict=True); work_root=args.work_root.expanduser().resolve(strict=False); work_root.mkdir(parents=True,exist_ok=True); worker_report=work_root/"memory-worker.json"; command=build_command(blender,fixture,work_root/"outputs",worker_report,warmup=args.warmup,iterations=args.iterations,sample_every=args.sample_every); completed=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False); (work_root/"blender-memory.log").write_text(completed.stdout,encoding="utf-8",errors="replace")
    if completed.returncode!=0 or not worker_report.is_file(): raise MemoryPlateauGateError(f"Blender memory worker failed with {completed.returncode}")
    payload=json.loads(worker_report.read_text(encoding="utf-8"))
    if not payload.get("success"): raise MemoryPlateauGateError(str(payload.get("error","worker failed")))
    evaluation=evaluate_plateau(payload,maximum_tail_growth_bytes=int(args.max_tail_growth_mib*1024*1024),maximum_slope_bytes_per_sample=args.max_slope_mib_per_sample*1024*1024); final={"worker":payload,"evaluation":evaluation,"command":command}; (work_root/"memory-plateau-report.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return 0 if evaluation["compatible"] else 1
def main(): raise SystemExit(run())
if __name__=="__main__": main()

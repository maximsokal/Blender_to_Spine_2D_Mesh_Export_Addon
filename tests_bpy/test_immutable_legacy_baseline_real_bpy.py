from pathlib import Path
import bpy
from tools.compare_immutable_legacy_baseline import compare_immutable_baseline
def _write_case(root:Path,*,color=(1.,0.,0.,1.)):
    (root/"images").mkdir(parents=True); (root/"Hero.json").write_text('{"skeleton":{"spine":"4.2.43"},"bones":[{"name":"root"}],"slots":[],"skins":[]}',encoding="utf-8"); image=bpy.data.images.new("Differential",width=2,height=2,alpha=True); image.pixels=color*4; image.filepath_raw=str(root/"images"/"Hero.png"); image.file_format="PNG"; image.save(); bpy.data.images.remove(image)
def test_immutable_baseline_comparison_passes_and_does_not_modify_oracle(tmp_path):
    baseline=tmp_path/"baseline"; candidate=tmp_path/"candidate"; _write_case(baseline); _write_case(candidate); before=tuple((path.relative_to(baseline),path.read_bytes()) for path in baseline.rglob("*") if path.is_file()); report=compare_immutable_baseline(baseline,candidate,json_name="Hero.json"); after=tuple((path.relative_to(baseline),path.read_bytes()) for path in baseline.rglob("*") if path.is_file()); assert report["compatible"] is True; assert before==after
def test_immutable_baseline_detects_candidate_pixel_difference(tmp_path):
    baseline=tmp_path/"baseline"; candidate=tmp_path/"candidate"; _write_case(baseline,color=(1.,0.,0.,1.)); _write_case(candidate,color=(0.,1.,0.,1.)); report=compare_immutable_baseline(baseline,candidate,json_name="Hero.json"); assert report["compatible"] is False; assert report["json"]["compatible"] is True; assert report["images"]["compatible"] is False

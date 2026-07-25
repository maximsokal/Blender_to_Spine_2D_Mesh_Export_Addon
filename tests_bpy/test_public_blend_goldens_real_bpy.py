"""Generate, reopen, export, and compare public .blend fixtures to reviewed goldens."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import bpy,pytest
import Blender_to_Spine2D_Mesh_Exporter as addon
from tools.create_public_blend_fixtures import create_all
ROOT=Path(__file__).resolve().parents[1]
GOLDEN=json.loads((ROOT/"tests"/"fixtures"/"public_blend_golden.json").read_text(encoding="utf-8"))["cases"]
@pytest.fixture(scope="session")
def generated_public_blend_fixtures(tmp_path_factory):
    fixture_root=tmp_path_factory.mktemp("spine2d-public-blend-fixtures"); created=create_all(fixture_root); assert tuple(sorted(path.stem for path in created))==tuple(sorted(GOLDEN))
    for path in created: assert path.is_file() and path.stat().st_size>1024
    return fixture_root
def _register_steps():
    completed=[]
    try:
        for step in addon.REGISTRATION_STEPS: step[1](); completed.append(step)
        return completed
    except Exception:
        for step in reversed(completed): step[2]()
        raise
def _unregister_steps(completed):
    failures=[]
    for label,_register,unregister in reversed(completed):
        try: unregister()
        except Exception as exc: failures.append(f"{label}: {exc}")
    assert not failures,failures
def _source_fingerprint(obj):
    mesh=obj.data
    return (obj.name_full,tuple(round(float(value),7) for row in obj.matrix_world for value in row),tuple(round(float(value),7) for value in obj.scale),tuple((item.name,item.type) for item in obj.modifiers),tuple(tuple(int(value) for value in edge.vertices) for edge in mesh.edges),tuple(tuple(int(value) for value in polygon.vertices) for polygon in mesh.polygons),tuple((layer.name,tuple(tuple(round(float(value),7) for value in item.uv) for item in layer.data)) for layer in mesh.uv_layers),tuple(material.name_full if material else None for material in mesh.materials))
def _image_metrics(path):
    image=bpy.data.images.load(str(path),check_existing=False)
    try:
        width,height=int(image.size[0]),int(image.size[1]); channels=int(image.channels); values=tuple(float(value) for value in image.pixels[:]); count=width*height; means=[]
        for channel in range(4): means.append(sum(values[index*channels+channel] for index in range(count))/count if channel<channels else 1.0)
        alpha_index=3 if channels>=4 else None; coverage=sum(1 for index in range(count) if values[index*channels+alpha_index]>(1.0/255.0))/count if alpha_index is not None else 1.0
        return {"width":width,"height":height,"mean_rgba":tuple(means),"alpha_coverage":coverage}
    finally: bpy.data.images.remove(image)
def _configure(output_root):
    scene=bpy.context.scene; scene.spine2d_json_path=str(output_root); scene.spine2d_images_path="images"; scene.spine2d_texture_size=64; scene.spine2d_angle_limit=30; scene.spine2d_seam_maker_mode="AUTO"; scene.spine2d_control_icons=False; scene.spine2d_export_preview_animation=False; scene.spine2d_frames_for_render=0; scene.spine2d_bake_frame_start=0; obj=bpy.data.objects["Hero"]
    for candidate in bpy.context.view_layer.objects: candidate.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active=obj; return obj
@pytest.mark.parametrize("case_id",tuple(sorted(GOLDEN)))
def test_public_blend_fixture_matches_reviewed_golden(case_id,tmp_path,generated_public_blend_fixtures):
    fixture=generated_public_blend_fixtures/f"{case_id}.blend"; assert fixture.is_file() and fixture.stat().st_size>1024; assert "FINISHED" in bpy.ops.wm.open_mainfile(filepath=str(fixture),load_ui=False); completed=_register_steps()
    try:
        obj=_configure(tmp_path); source_before=_source_fingerprint(obj); expected=GOLDEN[case_id]
        if expected["status"]=="failed":
            with pytest.raises(RuntimeError,match="A1_PREPARE_GEOMETRY_FAILED"): bpy.ops.object.save_uv_as_json()
            assert not tuple(tmp_path.rglob("*.json")); assert not tuple(tmp_path.rglob("*.png"))
        else:
            result=set(bpy.ops.object.save_uv_as_json()); assert "FINISHED" in result; json_files=tuple(tmp_path.glob("*.json")); assert len(json_files)==1; assert hashlib.sha256(json_files[0].read_bytes()).hexdigest()==expected["json_sha256"]; images=tuple(sorted((tmp_path/"images").glob("*"))); assert len(images)==len(expected["images"])
            for path,expected_image in zip(images,expected["images"],strict=True):
                actual=_image_metrics(path); assert path.name==expected_image["name"]; assert actual["width"]==expected_image["width"]; assert actual["height"]==expected_image["height"]; assert actual["alpha_coverage"]==pytest.approx(expected_image["alpha_coverage"],abs=0.01); assert actual["mean_rgba"]==pytest.approx(expected_image["mean_rgba"],abs=0.03)
        assert _source_fingerprint(obj)==source_before; assert not tuple(tmp_path.glob("*.spine2d.lock")); assert not tuple(tmp_path.glob(".spine2d-journal-*.json"))
    finally: _unregister_steps(completed)

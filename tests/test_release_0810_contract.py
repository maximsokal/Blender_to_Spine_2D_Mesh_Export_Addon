"""Current release-scope contracts for Depth Camera Projection 0.81.0."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
MANIFEST = PACKAGE / "blender_manifest.toml"
SCENE_MIGRATION = PACKAGE / "blender_adapter" / "scene_settings_migration.py"
RELEASE_NOTE = ROOT / "docs" / "releases" / "0.81.0.md"
SINGLE_RUNNER = ROOT / "tests" / "blender_headless" / "run_depth_camera_projection_integration.py"
MULTI_RUNNER = ROOT / "tests" / "blender_headless" / "run_depth_camera_projection_multi_object_integration.py"
SAMPLE_GENERATOR = ROOT / "tests" / "blender_headless" / "generate_depth_camera_projection_samples.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return " ".join(
        value.lower().replace("-", " ").replace("`", " ").split()
    )


def test_current_manifest_and_scene_schema_are_0810() -> None:
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    migration = _read(SCENE_MIGRATION)

    assert manifest["version"] == "0.81.0"
    assert manifest["blender_version_min"] == "5.2.0"
    assert "CURRENT_SETTINGS_SCHEMA_VERSION = 7" in migration
    assert "spine2d_depth_smoothing" in migration
    assert "spine2d_depth_edge_threshold" in migration
    assert "spine2d_depth_mesh_error_pixels" in migration
    assert "spine2d_depth_max_points" in migration
    assert "spine2d_depth_base_mode" in migration


def test_release_note_records_exact_three_mode_user_contract() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert note.lstrip().startswith("# Release 0.81.0")
    assert "Normal / UV Segments" in note
    assert "Camera Projection" in note
    assert "Depth Camera Projection" in note
    assert "third public export mode" in normalized
    assert "does not export blender camera animation" in normalized
    assert "does not require a new user facing rig selector" in normalized


def test_release_note_records_shared_camera_zero_and_local_relief_base() -> None:
    normalized = _normalized(_read(RELEASE_NOTE))

    assert "active camera is the single depth zero for every exported object" in normalized
    assert "positive camera distance" in normalized
    assert "shared zero is not reset independently for every object" in normalized
    assert "farthest visible point" in normalized
    assert "back surface of that object's one sided relief" in normalized
    assert "local relief policy never replaces the shared camera zero" in normalized
    assert "projected blender object origin x/y is retained" in normalized
    assert "object origin is implemented as a hidden fail closed relief base policy" in normalized


def test_release_note_records_depth_surface_and_quality_controls() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    assert "evaluated dependency graph mesh" in normalized
    assert "front most visible surface" in normalized
    assert "bounded screen lattice" in normalized
    assert "generated meshsnapshot" in normalized
    assert "topology aware" in normalized
    assert "steep continuous source face" in normalized
    assert "unrelated overlapping surfaces still remain disconnected" in normalized
    for label in (
        "Depth Smoothing",
        "Depth Edge Threshold",
        "Depth Mesh Error (px)",
        "Max Depth Points",
    ):
        assert label in note
    assert "smoothing never crosses a protected depth discontinuity" in normalized
    assert "structured diagnostic" in normalized


def test_release_note_records_one_compensated_attachment() -> None:
    normalized = _normalized(_read(RELEASE_NOTE))

    assert "exactly one visual slot and one weighted mesh attachment" in normalized
    assert "<object>_segment_0" in normalized
    assert "does not create a stray <object>_segment_1" in normalized
    assert "vertex_local_y + parent_depth_y = projected_screen_y" in normalized
    assert "prevents camera depth from being applied twice" in normalized


def test_release_note_records_hybrid_texture_and_crop_pipeline() -> None:
    normalized = _normalized(_read(RELEASE_NOTE))

    assert "active camera render" in normalized
    assert "generated weighted vertex bones" in normalized
    assert "direct full frame camera uv" in normalized
    assert "crop local uv" in normalized
    assert "without changing weighted vertices, triangles, hull, or bone indices" in normalized
    assert "source mesh" in normalized
    assert "source uv layers" in normalized
    assert "transactional state guards" in normalized


def test_release_note_records_material_only_depth_sequence() -> None:
    normalized = _normalized(_read(RELEASE_NOTE))

    assert "advances the blender timeline only to evaluate animated materials" in normalized
    assert "must not bake blender object movement" in normalized
    assert "active camera movement" in normalized
    assert "evaluated source mesh proxy" in normalized
    assert "active camera proxy" in normalized
    assert "alpha silhouette and render crop remain identical" in normalized
    assert "flat camera projection keeps its established timeline behavior" in normalized


def test_release_note_records_target_and_multi_object_acceptance_matrix() -> None:
    note = _read(RELEASE_NOTE)
    normalized = _normalized(note)

    for version in ("3.8", "4.0", "4.1", "4.2", "4.3"):
        assert version in note
    assert "perspective depth camera projection" in normalized
    assert "orthographic depth camera projection" in normalized
    assert "two frame spine 4.2 depth camera projection material sequence" in normalized
    assert "one sequence depth object" in normalized
    assert "one static depth object" in normalized
    assert "distinct projected main bone positions" in normalized
    assert "distinct positive camera distance ranges" in normalized
    assert "exactly one weighted attachment per depth object" in normalized
    assert "static siblings remain static" in normalized
    assert "texture namespaces" in normalized
    assert "atomic output reservations" in normalized


def test_release_runners_and_persistent_sample_generator_exist() -> None:
    for path in (SINGLE_RUNNER, MULTI_RUNNER, SAMPLE_GENERATOR):
        assert path.is_file(), path

    single = _read(SINGLE_RUNNER)
    multi = _read(MULTI_RUNNER)
    samples = _read(SAMPLE_GENERATOR)

    assert "SpineJsonTarget.SPINE_3_8" in single
    assert "SpineJsonTarget.SPINE_4_3" in single
    assert '_Case(SpineJsonTarget.SPINE_4_2, "ORTHO")' in single
    assert "_SEQUENCE_COUNT = 2" in single
    assert "sequence_frame_count=case.sequence_count" in single
    assert "all(distance > 0.0 for distance in distances)" in single
    assert "unexpected Segment_1 slot" in single

    assert "export_a1_multi_object(" in multi
    assert "A1MultiObjectMode.STANDALONE" in multi
    assert "_SEQUENCE_COUNT = 2" in multi
    assert '_Case(_TARGET, "PERSP", _SEQUENCE_COUNT)' in multi
    assert '_Case(_TARGET, "PERSP", 0)' in multi
    assert "_keyframe_sequence_source" in multi
    assert "_keyframe_active_camera" in multi
    assert "animated object/camera changed Depth sequence silhouette" in multi

    assert "_run_case(" in samples
    assert "_generate_multi_object(" in samples
    assert "_keyframe_active_camera" in samples
    assert "sample_manifest.json" in samples
    assert "sha256" in samples


def test_release_package_name_is_stable() -> None:
    note = _read(RELEASE_NOTE)
    assert "blender_to_spine2d_mesh_exporter-0.81.0.zip" in note

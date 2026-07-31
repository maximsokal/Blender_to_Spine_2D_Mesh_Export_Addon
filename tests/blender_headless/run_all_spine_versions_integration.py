"""Export every production-enabled Spine 3.8-4.3 target/profile/scope combination.

The worker exercises only public production export services. It never builds or repairs
Spine JSON directly. Unsupported target/profile/scope combinations are validated against
the fail-closed capability registry before any geometry or bake work starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Mapping

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
TOOLS_DIRECTORY = REPOSITORY_ROOT / "tools"
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT, TOOLS_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1MultiObjectSource,
    export_a1_mixed_object,
    export_a1_multi_object,
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.export_capabilities import (  # noqa: E402
    SpineJsonExportCapabilityError,
    SpineJsonExportScope,
    require_spine_json_export_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigProfile,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import (  # noqa: E402
    SpineJsonTarget,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import (  # noqa: E402
    PNG_SIGNATURE,
    _activate_only,
    _assert,
    _capture_context,
    _capture_scene_bake_state,
    _clear_scene,
    _configure_cycles_scene,
    _create_emission_material,
    _create_quad,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)
from spine_version_acceptance_matrix import (  # noqa: E402
    EXPECTED_CASE_COUNT_BY_TARGET,
    POSITIVE_CASES,
    SpineVersionAcceptanceCase,
)


POSITIONS = (
    (0.0, 0.0, 0.0),
    (2.0, 1.0, 0.5),
    (-1.5, 2.25, 1.0),
)
TARGET_TOKENS = {
    "SPINE_3_8": "Spine38",
    "SPINE_4_0": "Spine40",
    "SPINE_4_1": "Spine41",
    "SPINE_4_2": "Spine42",
    "SPINE_4_3": "Spine43",
}
PROFILE_TOKENS = {
    "THREE_AXIS_ROTATION": "ThreeAxis",
    "TWO_AXIS_ROTATION_SCALE": "TwoAxis",
}
SCOPE_TOKENS = {
    "SINGLE_OBJECT": "Single",
    "STANDALONE_MULTI_OBJECT": "Standalone",
    "CONNECTED_MULTI_OBJECT": "Connected",
    "MIXED_MULTI_OBJECT": "Mixed",
}
LEGACY_MIX_FIELDS = {
    "rotateMix",
    "translateMix",
    "scaleMix",
    "shearMix",
}
SPLIT_MIX_FIELDS = {
    "mixRotate",
    "mixX",
    "mixY",
    "mixScaleX",
    "mixScaleY",
    "mixShearY",
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _prepare_output_directory(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("output must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _case_output_directory(root: Path, case: SpineVersionAcceptanceCase) -> Path:
    directory = (root / case.key).resolve(strict=False)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Case output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _case_stem(case: SpineVersionAcceptanceCase) -> str:
    return "".join(
        (
            TARGET_TOKENS[case.target],
            PROFILE_TOKENS[case.profile],
            SCOPE_TOKENS[case.scope],
        )
    )


def _object_settings(
    output_directory: Path,
    case: SpineVersionAcceptanceCase,
    *,
    prefix: str,
    output_stem: str,
) -> A1SingleObjectExportSettings:
    target = SpineJsonTarget[case.target]
    profile = A1RigProfile[case.profile]
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=target.exact_version,
            rig_profile=profile.value,
            bake_margin=1,
        ),
        prefix=prefix,
        output_stem=output_stem,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
    )


def _build_sources(
    output_directory: Path,
    case: SpineVersionAcceptanceCase,
) -> tuple[tuple[A1MultiObjectSource, ...], tuple[object, ...]]:
    stem = _case_stem(case)
    sources: list[A1MultiObjectSource] = []
    materials: list[object] = []
    for index in range(case.object_count):
        suffix = chr(ord("A") + index)
        prefix = f"{stem}{suffix}"
        source_object = _create_quad(f"{stem}Source{suffix}")
        source_object.location = POSITIONS[index]
        material = _create_emission_material(source_object)
        materials.append(material)
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=f"{case.key}_component_{index + 1}",
                animation_namespace=f"{case.key}_object_{index + 1}",
                settings=_object_settings(
                    output_directory,
                    case,
                    prefix=prefix,
                    output_stem=prefix,
                ),
            )
        )
    return tuple(sources), tuple(materials)


def _contains_sequence(value: object) -> bool:
    if isinstance(value, dict):
        if "sequence" in value:
            return True
        return any(_contains_sequence(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_sequence(item) for item in value)
    return False


def _json_array(document: Mapping[str, object], field_name: str) -> list[object]:
    value = document.get(field_name, [])
    _assert(isinstance(value, list), f"{field_name} must be a JSON array")
    return value


def _constraint_counts(document: Mapping[str, object], target: SpineJsonTarget) -> dict[str, int]:
    if target is SpineJsonTarget.SPINE_4_3:
        constraints = _json_array(document, "constraints")
        result = {"ik": 0, "transform": 0, "path": 0, "constraints": len(constraints)}
        for index, value in enumerate(constraints):
            _assert(isinstance(value, dict), f"constraints[{index}] must be an object")
            constraint_type = value.get("type")
            _assert(
                constraint_type in {"ik", "transform", "path"},
                f"constraints[{index}] has invalid type: {constraint_type!r}",
            )
            result[str(constraint_type)] += 1
            _assert("order" not in value, "Spine 4.3 constraint retained legacy order")
        return result

    ik = _json_array(document, "ik")
    transform = _json_array(document, "transform")
    path = _json_array(document, "path")
    return {
        "ik": len(ik),
        "transform": len(transform),
        "path": len(path),
        "constraints": len(ik) + len(transform) + len(path),
    }


def _assert_constraint_schema(
    document: Mapping[str, object],
    target: SpineJsonTarget,
) -> None:
    if target is SpineJsonTarget.SPINE_4_3:
        _assert("constraints" in document, "Spine 4.3 unified constraints are missing")
        for field_name in ("ik", "transform", "path", "physics", "slider"):
            _assert(
                field_name not in document,
                f"Spine 4.3 retained legacy root field: {field_name}",
            )
        return

    _assert("constraints" not in document, "Unified constraints leaked into legacy target")
    transform = _json_array(document, "transform")
    if target is SpineJsonTarget.SPINE_3_8:
        for index, raw_constraint in enumerate(transform):
            _assert(isinstance(raw_constraint, dict), f"transform[{index}] invalid")
            _assert(
                LEGACY_MIX_FIELDS.issubset(raw_constraint),
                f"Spine 3.8 transform[{index}] lacks legacy mix fields",
            )
            _assert(
                not SPLIT_MIX_FIELDS.intersection(raw_constraint),
                f"Spine 3.8 transform[{index}] retained split mix fields",
            )
        return

    split_fields_seen = False
    for index, raw_constraint in enumerate(transform):
        _assert(isinstance(raw_constraint, dict), f"transform[{index}] invalid")
        _assert(
            not LEGACY_MIX_FIELDS.intersection(raw_constraint),
            f"{target.label} transform[{index}] retained Spine 3.8 mix fields",
        )
        split_fields_seen = split_fields_seen or bool(
            SPLIT_MIX_FIELDS.intersection(raw_constraint)
        )
    _assert(split_fields_seen or not transform, f"{target.label} has no split mix fields")


def _assert_bone_schema(document: Mapping[str, object], target: SpineJsonTarget) -> None:
    bones = _json_array(document, "bones")
    _assert(bones, "bones must be non-empty")
    names: set[str] = set()
    for index, value in enumerate(bones):
        _assert(isinstance(value, dict), f"bones[{index}] must be an object")
        name = value.get("name")
        _assert(isinstance(name, str) and name, f"bones[{index}] has no name")
        _assert(name not in names, f"duplicate bone name: {name}")
        names.add(name)
        if target in {
            SpineJsonTarget.SPINE_3_8,
            SpineJsonTarget.SPINE_4_0,
            SpineJsonTarget.SPINE_4_1,
        }:
            _assert("inherit" not in value, f"4.2+ bone.inherit leaked: {name}")
        else:
            _assert("transform" not in value, f"legacy bone.transform leaked: {name}")


def _assert_document(
    document: dict[str, object],
    case: SpineVersionAcceptanceCase,
) -> dict[str, object]:
    target = SpineJsonTarget[case.target]
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata is missing")
    _assert(
        skeleton.get("spine") == case.exact_version,
        f"version mismatch: expected={case.exact_version}, actual={skeleton.get('spine')!r}",
    )
    _assert(_json_array(document, "slots"), "slots must be non-empty")
    _assert(_json_array(document, "skins"), "skins must be non-empty")
    _assert_bone_schema(document, target)
    _assert_constraint_schema(document, target)
    if target in {SpineJsonTarget.SPINE_3_8, SpineJsonTarget.SPINE_4_0}:
        _assert(not _contains_sequence(document), f"sequence leaked into {target.label}")

    bone_names = {
        str(value.get("name"))
        for value in _json_array(document, "bones")
        if isinstance(value, dict)
    }
    wrapper_present = any(name.startswith("all_objects") for name in bone_names)
    expected_wrapper = case.scope in {"CONNECTED_MULTI_OBJECT", "MIXED_MULTI_OBJECT"}
    _assert(
        wrapper_present is expected_wrapper,
        f"connected wrapper mismatch for {case.key}: {wrapper_present}",
    )
    counts = _constraint_counts(document, target)
    _assert(counts["constraints"] > 0, f"{case.key} contains no runtime constraints")
    return {
        "bones": len(_json_array(document, "bones")),
        "slots": len(_json_array(document, "slots")),
        "skins": len(_json_array(document, "skins")),
        **counts,
        "connectedWrapperPresent": wrapper_present,
        "sequencePresent": _contains_sequence(document),
    }


def _assert_state_restored(
    *,
    context_before: object,
    scene_before: object,
    materials: tuple[object, ...],
    material_fingerprints: tuple[object, ...],
) -> None:
    _assert(_capture_context() == context_before, "export changed Blender context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        "export changed scene bake state",
    )
    _assert(
        tuple(_material_fingerprint(material) for material in materials)
        == material_fingerprints,
        "export mutated source materials",
    )
    _assert(not _temporary_datablock_names(), "export leaked temporary datablocks")


def _export_case(
    output_root: Path,
    case: SpineVersionAcceptanceCase,
) -> dict[str, object]:
    case_directory = _case_output_directory(output_root, case)
    _clear_scene()
    _configure_cycles_scene()
    sources, materials = _build_sources(case_directory, case)
    sentinel = _create_sentinel()
    _activate_only(sentinel)
    for source in sources:
        source.source_object.select_set(False)

    context_before = _capture_context()
    scene_before = _capture_scene_bake_state()
    material_fingerprints = tuple(_material_fingerprint(item) for item in materials)
    output_stem = _case_stem(case)

    if case.scope == "SINGLE_OBJECT":
        result = export_a1_single_object(sources[0].source_object, sources[0].settings)
        expected_json = (case_directory / f"{sources[0].settings.output_stem}.json").resolve()
    else:
        mode_by_scope = {
            "STANDALONE_MULTI_OBJECT": A1MultiObjectMode.STANDALONE,
            "CONNECTED_MULTI_OBJECT": A1MultiObjectMode.CONNECTED,
            "MIXED_MULTI_OBJECT": A1MultiObjectMode.MIXED,
        }
        settings = A1MultiObjectExportSettings(
            output_directory=case_directory,
            output_stem=output_stem,
            mode=mode_by_scope[case.scope],
            anchor_component_id=sources[0].component_id,
        )
        if case.scope == "MIXED_MULTI_OBJECT":
            result = export_a1_mixed_object(sources[:2], sources[2:], settings)
        else:
            result = export_a1_multi_object(sources, settings)
        expected_json = (case_directory / f"{output_stem}.json").resolve()

    _assert(result.success, f"{case.key} export failed: {result.issues}")
    _assert(result.output_files, f"{case.key} returned no output files")
    _assert(result.output_files[0] == expected_json, f"{case.key} JSON order changed")
    _assert(expected_json.is_file(), f"{case.key} JSON was not created")
    texture_paths = tuple(Path(path).resolve() for path in result.output_files[1:])
    _assert(
        len(texture_paths) == case.object_count,
        f"{case.key} texture count changed: {len(texture_paths)}",
    )
    for texture_path in texture_paths:
        _assert(texture_path.read_bytes()[:8] == PNG_SIGNATURE, f"invalid PNG: {texture_path}")

    document = json.loads(expected_json.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), f"{case.key} JSON root must be an object")
    structural = _assert_document(document, case)
    _assert_state_restored(
        context_before=context_before,
        scene_before=scene_before,
        materials=materials,
        material_fingerprints=material_fingerprints,
    )
    return {
        "status": "passed",
        "key": case.key,
        "target": case.target,
        "version": case.exact_version,
        "profile": case.profile,
        "scope": case.scope,
        "objectCount": case.object_count,
        "jsonPath": str(expected_json),
        "texturePaths": [str(path) for path in texture_paths],
        "outputFiles": [str(path) for path in result.output_files],
        **structural,
    }


def _validate_capability_matrix() -> dict[str, object]:
    positive = {(case.target, case.profile, case.scope) for case in POSITIVE_CASES}
    accepted: list[str] = []
    blocked: list[str] = []
    for target in SpineJsonTarget:
        for profile in A1RigProfile:
            for scope in SpineJsonExportScope:
                key = (target.name, profile.name, scope.name)
                label = "/".join(key)
                if key in positive:
                    require_spine_json_export_capability(target, profile, scope)
                    accepted.append(label)
                    continue
                try:
                    require_spine_json_export_capability(target, profile, scope)
                except SpineJsonExportCapabilityError:
                    blocked.append(label)
                else:
                    raise AssertionError(f"Unsupported capability unexpectedly accepted: {label}")
    _assert(len(accepted) == 20, f"accepted capability count changed: {len(accepted)}")
    _assert(len(blocked) == 20, f"blocked capability count changed: {len(blocked)}")
    return {"accepted": accepted, "blocked": blocked}


def run(output_directory: Path) -> Path:
    output_root = _prepare_output_directory(output_directory)
    capability_report = _validate_capability_matrix()
    cases = [_export_case(output_root, case) for case in POSITIVE_CASES]
    by_target = {
        target: sum(1 for case in cases if case["target"] == target)
        for target in EXPECTED_CASE_COUNT_BY_TARGET
    }
    _assert(by_target == dict(EXPECTED_CASE_COUNT_BY_TARGET), "target case counts changed")
    report = {
        "status": "passed",
        "caseCount": len(cases),
        "expectedCaseCountByTarget": dict(EXPECTED_CASE_COUNT_BY_TARGET),
        "capabilities": capability_report,
        "cases": cases,
    }
    report_path = output_root / "blender_acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[SPINE_ALL_VERSIONS] RUN 20 production export cases")
    report_path = run(arguments.output)
    print(f"[SPINE_ALL_VERSIONS] REPORT {report_path}")
    print("[SPINE_ALL_VERSIONS] PASS 20 production export cases")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

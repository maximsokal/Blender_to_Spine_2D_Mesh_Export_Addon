"""Generate and validate real Blender 5.2 Spine 3.8 standalone exports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback
from typing import Iterable, Mapping

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
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
    export_a1_multi_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
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


POSITIONS = (
    (0.0, 0.0, 0.0),
    (2.0, 1.0, 0.5),
    (-1.5, 2.25, 1.0),
)
LEGACY_MIX_FIELDS = (
    "rotateMix",
    "translateMix",
    "scaleMix",
    "shearMix",
)
FORBIDDEN_NEW_MIX_FIELDS = (
    "mixRotate",
    "mixX",
    "mixY",
    "mixScaleX",
    "mixScaleY",
    "mixShearY",
)


@dataclass(frozen=True, slots=True)
class ProfileCase:
    key: str
    profile: A1RigProfile
    output_stem: str
    prefixes: tuple[str, str, str]
    expected_bones: int
    expected_ik: int
    expected_transform: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("key must be a non-empty string")
        if not isinstance(self.profile, A1RigProfile):
            raise TypeError("profile must be A1RigProfile")
        if not isinstance(self.output_stem, str) or not self.output_stem.strip():
            raise ValueError("output_stem must be a non-empty string")
        if len(self.prefixes) != 3 or not all(self.prefixes):
            raise ValueError("prefixes must contain three non-empty names")
        for field_name in ("expected_bones", "expected_ik", "expected_transform"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


CASES = (
    ProfileCase(
        key="two_axis",
        profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        output_stem="Spine38TwoAxisStandaloneMulti",
        prefixes=("Spine38TwoA", "Spine38TwoB", "Spine38TwoC"),
        expected_bones=55,
        expected_ik=3,
        expected_transform=12,
    ),
    ProfileCase(
        key="three_axis",
        profile=A1RigProfile.THREE_AXIS_ROTATION,
        output_stem="Spine38ThreeAxisStandaloneMulti",
        prefixes=("Spine38ThreeA", "Spine38ThreeB", "Spine38ThreeC"),
        expected_bones=52,
        expected_ik=3,
        expected_transform=15,
    ),
)


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


def _object_settings(
    output_directory: Path,
    *,
    prefix: str,
    profile: A1RigProfile,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=SpineJsonTarget.SPINE_3_8.exact_version,
            rig_profile=profile.value,
            bake_margin=1,
        ),
        prefix=prefix,
        output_stem=prefix,
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
    )


def _build_sources(
    output_directory: Path,
    case: ProfileCase,
) -> tuple[tuple[A1MultiObjectSource, ...], tuple[object, ...]]:
    sources: list[A1MultiObjectSource] = []
    materials: list[object] = []
    for index, (prefix, position) in enumerate(
        zip(case.prefixes, POSITIONS, strict=True),
        start=1,
    ):
        source_object = _create_quad(f"{case.output_stem}Source{index}")
        source_object.location = position
        material = _create_emission_material(source_object)
        materials.append(material)
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=f"{case.key}_component_{index}",
                animation_namespace=f"{case.key}_object_{index}",
                settings=_object_settings(
                    output_directory,
                    prefix=prefix,
                    profile=case.profile,
                ),
            )
        )
    return tuple(sources), tuple(materials)


def _json_array(
    document: Mapping[str, object],
    field_name: str,
    *,
    required: bool = False,
) -> list[object]:
    if field_name not in document:
        _assert(not required, f"required JSON array is missing: {field_name}")
        return []
    value = document[field_name]
    _assert(isinstance(value, list), f"{field_name} must be an array")
    return value


def _owner_prefix(name: object, prefixes: tuple[str, ...]) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    for prefix in prefixes:
        if name == prefix or name.startswith(f"{prefix}_"):
            return prefix
    return None


def _assert_same_owner(
    owner: str,
    names: Iterable[object],
    *,
    prefixes: tuple[str, ...],
    label: str,
) -> None:
    for name in names:
        _assert(isinstance(name, str) and name, f"{label} has invalid name")
        if name == "root":
            continue
        _assert(
            _owner_prefix(name, prefixes) == owner,
            f"{label} crosses object rigs: owner={owner}, reference={name!r}",
        )


def _contains_sequence(value: object) -> bool:
    if isinstance(value, dict):
        if "sequence" in value:
            return True
        return any(_contains_sequence(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_sequence(item) for item in value)
    return False


def _assert_profile_markers(
    bone_names: set[str],
    case: ProfileCase,
) -> None:
    for prefix in case.prefixes:
        scale_control = f"{prefix}_scale"
        rotation_z = f"{prefix}_rotation_Z"
        bridge = f"{prefix}_1_scale_spine41_bridge"
        if case.profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
            _assert(scale_control in bone_names, f"missing 2-Axis scale control: {prefix}")
            _assert(rotation_z not in bone_names, f"unexpected 3-Axis control: {prefix}")
            _assert(bridge in bone_names, f"missing 3.8 scale bridge: {prefix}")
        else:
            _assert(rotation_z in bone_names, f"missing 3-Axis Z control: {prefix}")
            _assert(scale_control not in bone_names, f"unexpected 2-Axis control: {prefix}")
            _assert(bridge not in bone_names, f"unexpected scale bridge: {prefix}")


def _assert_document(
    document: dict[str, object],
    case: ProfileCase,
) -> dict[str, object]:
    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata is missing")
    _assert(
        skeleton.get("spine") == SpineJsonTarget.SPINE_3_8.exact_version,
        f"unexpected Spine version: {skeleton.get('spine')!r}",
    )
    _assert("constraints" not in document, "unified constraints leaked into Spine 3.8")
    _assert("physics" not in document, "physics leaked into Spine 3.8")
    _assert(not _contains_sequence(document), "sequence data leaked into Spine 3.8")

    bones = _json_array(document, "bones", required=True)
    slots = _json_array(document, "slots", required=True)
    skins = _json_array(document, "skins", required=True)
    ik = _json_array(document, "ik", required=True)
    transform = _json_array(document, "transform", required=True)
    path = _json_array(document, "path")

    _assert(len(bones) == case.expected_bones, f"{case.key} bone count changed")
    _assert(len(slots) == 3, f"{case.key} slot count changed")
    _assert(len(skins) == 1, f"{case.key} skin count changed")
    _assert(len(ik) == case.expected_ik, f"{case.key} IK count changed")
    _assert(
        len(transform) == case.expected_transform,
        f"{case.key} transform count changed",
    )
    _assert(not path, f"{case.key} path constraints changed")

    bone_names: set[str] = set()
    for index, raw_bone in enumerate(bones):
        _assert(isinstance(raw_bone, dict), f"bones[{index}] must be an object")
        name = raw_bone.get("name")
        _assert(isinstance(name, str) and name, f"bones[{index}] has no name")
        _assert(name not in bone_names, f"duplicate bone: {name}")
        _assert("inherit" not in raw_bone, f"4.x bone.inherit leaked: {name}")
        _assert("referenceScale" not in raw_bone, f"referenceScale leaked: {name}")
        _assert("color" not in raw_bone, f"bone color leaked: {name}")
        _assert("icon" not in raw_bone, f"bone icon leaked: {name}")
        transform_mode = raw_bone.get("transform")
        _assert(
            transform_mode is None or isinstance(transform_mode, str),
            f"bone transform mode is invalid: {name}",
        )
        if name != "root":
            owner = _owner_prefix(name, case.prefixes)
            _assert(owner is not None, f"unowned bone: {name}")
            parent = raw_bone.get("parent")
            if parent is not None:
                _assert_same_owner(
                    owner,
                    (parent,),
                    prefixes=case.prefixes,
                    label=f"bone {name} parent",
                )
        bone_names.add(name)

    _assert_profile_markers(bone_names, case)

    constraint_names: set[str] = set()
    orders: list[int] = []
    for collection_name, values in (("ik", ik), ("transform", transform)):
        for index, raw_constraint in enumerate(values):
            _assert(
                isinstance(raw_constraint, dict),
                f"{collection_name}[{index}] must be an object",
            )
            name = raw_constraint.get("name")
            _assert(isinstance(name, str) and name, f"constraint has no name")
            _assert(name not in constraint_names, f"duplicate constraint: {name}")
            owner = _owner_prefix(name, case.prefixes)
            _assert(owner is not None, f"unowned constraint: {name}")
            raw_bones = raw_constraint.get("bones")
            _assert(isinstance(raw_bones, list), f"constraint {name} bones invalid")
            _assert_same_owner(
                owner,
                raw_bones,
                prefixes=case.prefixes,
                label=f"constraint {name} bones",
            )
            _assert_same_owner(
                owner,
                (raw_constraint.get("target"),),
                prefixes=case.prefixes,
                label=f"constraint {name} target",
            )
            order = raw_constraint.get("order", 0)
            _assert(
                isinstance(order, int) and not isinstance(order, bool) and order >= 0,
                f"constraint {name} order invalid",
            )
            orders.append(order)
            constraint_names.add(name)

            if collection_name == "transform":
                for field_name in LEGACY_MIX_FIELDS:
                    value = raw_constraint.get(field_name)
                    _assert(
                        isinstance(value, (int, float)) and not isinstance(value, bool),
                        f"constraint {name} missing {field_name}",
                    )
                for field_name in FORBIDDEN_NEW_MIX_FIELDS:
                    _assert(
                        field_name not in raw_constraint,
                        f"constraint {name} retained {field_name}",
                    )

    _assert(len(set(orders)) == len(orders), "constraint orders are not unique")
    _assert(sorted(orders) == list(range(len(orders))), "orders must form 0..N-1")

    skin = skins[0]
    _assert(isinstance(skin, dict), "skin must be an object")
    _assert("constraints" not in skin, "skin.constraints leaked into Spine 3.8")
    _assert(isinstance(skin.get("attachments"), dict), "skin attachments are missing")
    for field_name, known in (
        ("ik", {item["name"] for item in ik if isinstance(item, dict)}),
        ("transform", {item["name"] for item in transform if isinstance(item, dict)}),
    ):
        membership = skin.get(field_name)
        if membership is None:
            continue
        _assert(isinstance(membership, list), f"skin.{field_name} must be an array")
        _assert(all(name in known for name in membership), f"skin.{field_name} invalid")

    return {
        "status": "passed",
        "profile": case.profile.value,
        "version": skeleton["spine"],
        "mode": A1MultiObjectMode.STANDALONE.value,
        "prefixes": list(case.prefixes),
        "bones": len(bones),
        "slots": len(slots),
        "skins": len(skins),
        "ik": len(ik),
        "transform": len(transform),
        "constraints": len(ik) + len(transform),
        "legacyMixFieldsPresent": True,
        "newMixFieldsPresent": False,
        "connectedWrapperPresent": False,
        "crossObjectReferencesPresent": False,
        "sequencePresent": False,
    }


def _assert_state_restored(
    *,
    context_before: object,
    scene_before: object,
    materials: tuple[object, ...],
    material_fingerprints: tuple[object, ...],
    label: str,
) -> None:
    _assert(_capture_context() == context_before, f"{label} changed context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        f"{label} changed scene bake state",
    )
    _assert(
        tuple(_material_fingerprint(material) for material in materials)
        == material_fingerprints,
        f"{label} mutated source materials",
    )
    _assert(
        not _temporary_datablock_names(),
        f"{label} leaked temporary Blender datablocks",
    )


def _run_case(output_root: Path, case: ProfileCase) -> dict[str, object]:
    case_directory = output_root / case.key
    case_directory.mkdir(parents=True, exist_ok=False)

    _clear_scene()
    _configure_cycles_scene()
    sources, materials = _build_sources(case_directory, case)
    sentinel = _create_sentinel()
    _activate_only(sentinel)
    for source in sources:
        source.source_object.select_set(False)

    context_before = _capture_context()
    scene_before = _capture_scene_bake_state()
    fingerprints = tuple(_material_fingerprint(item) for item in materials)

    result = export_a1_multi_object(
        sources,
        A1MultiObjectExportSettings(
            output_directory=case_directory,
            output_stem=case.output_stem,
            mode=A1MultiObjectMode.STANDALONE,
        ),
    )
    _assert(result.success, f"{case.key} export failed: {result.issues}")

    json_path = (case_directory / f"{case.output_stem}.json").resolve()
    textures = tuple(
        (case_directory / "images" / f"{prefix}_Baked.png").resolve()
        for prefix in case.prefixes
    )
    _assert(
        result.output_files == (json_path, *textures),
        f"{case.key} output order changed: {result.output_files}",
    )
    for texture in textures:
        _assert(texture.is_file(), f"missing texture: {texture}")
        _assert(texture.read_bytes()[:8] == PNG_SIGNATURE, f"invalid PNG: {texture}")

    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "generated JSON root must be an object")
    report = _assert_document(document, case)
    report.update(
        {
            "jsonPath": str(json_path),
            "texturePaths": [str(path) for path in textures],
            "outputFiles": [str(path) for path in result.output_files],
        }
    )
    _assert_state_restored(
        context_before=context_before,
        scene_before=scene_before,
        materials=materials,
        material_fingerprints=fingerprints,
        label=case.key,
    )
    return report


def run(output_directory: Path) -> Path:
    output_root = _prepare_output_directory(output_directory)
    reports: list[dict[str, object]] = []
    try:
        for case in CASES:
            reports.append(_run_case(output_root, case))
    finally:
        _clear_scene()

    report = {
        "status": "passed",
        "version": SpineJsonTarget.SPINE_3_8.exact_version,
        "mode": A1MultiObjectMode.STANDALONE.value,
        "profiles": reports,
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
    print("[SPINE38_STANDALONE] RUN 2-Axis and 3-Axis production exports")
    report_path = run(arguments.output)
    print(f"[SPINE38_STANDALONE] REPORT {report_path}")
    print("[SPINE38_STANDALONE] PASS production profile exports")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

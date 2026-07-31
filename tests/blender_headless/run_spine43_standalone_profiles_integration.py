"""Generate and structurally validate real Blender 5.2 Spine 4.3 exports.

The worker exercises the production standalone multi-object service for both supported
rig profiles. It never invokes a target codec directly, never constructs JSON manually,
and never enters connected composition.
"""

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
TRANSFORM_PROPERTIES = (
    "rotate",
    "x",
    "y",
    "scaleX",
    "scaleY",
    "shearY",
)
SOURCE_VERTEX_COUNT = 4
SOURCE_SEGMENT_INDEX = 0
SOURCE_Z_GROUP_INDEX = 1


@dataclass(frozen=True, slots=True)
class ProfileCase:
    """One deterministic production export case and its expected generated topology."""

    key: str
    profile: A1RigProfile
    output_stem: str
    prefixes: tuple[str, str, str]
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
            raise ValueError("prefixes must contain exactly three non-empty names")
        for field_name in ("expected_ik", "expected_transform"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


CASES = (
    ProfileCase(
        key="two_axis",
        profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE,
        output_stem="Spine43TwoAxisStandaloneMulti",
        prefixes=("Spine43TwoA", "Spine43TwoB", "Spine43TwoC"),
        expected_ik=3,
        expected_transform=12,
    ),
    ProfileCase(
        key="three_axis",
        profile=A1RigProfile.THREE_AXIS_ROTATION,
        output_stem="Spine43ThreeAxisStandaloneMulti",
        prefixes=("Spine43ThreeA", "Spine43ThreeB", "Spine43ThreeC"),
        expected_ik=3,
        expected_transform=15,
    ),
)


def _parse_arguments() -> argparse.Namespace:
    """Parse only arguments after Blender's ``--`` separator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Empty directory that receives both profile exports and reports.",
    )
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else ()
    return parser.parse_args(arguments)


def _prepare_output_directory(value: Path) -> Path:
    """Create one empty output directory without deleting caller-owned files."""

    if not isinstance(value, Path):
        raise TypeError("output must be pathlib.Path")
    resolved = value.expanduser().resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"Output directory must be empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_object_settings(
    output_directory: Path,
    *,
    prefix: str,
    profile: A1RigProfile,
) -> A1SingleObjectExportSettings:
    """Build exact production settings for one Spine 4.3 object."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    if not isinstance(profile, A1RigProfile):
        raise TypeError("profile must be A1RigProfile")

    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=SpineJsonTarget.SPINE_4_3.exact_version,
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
    """Create three independent real Blender mesh sources for one profile case."""

    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    if not isinstance(case, ProfileCase):
        raise TypeError("case must be ProfileCase")

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
                settings=_build_object_settings(
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
    """Resolve one JSON array and distinguish omitted optional collections."""

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be a non-empty string")
    if not isinstance(required, bool):
        raise TypeError("required must be bool")
    if field_name not in document:
        _assert(not required, f"required JSON array is missing: {field_name}")
        return []
    value = document[field_name]
    _assert(isinstance(value, list), f"{field_name} must be a JSON array")
    return value


def _owner_prefix(name: str, prefixes: tuple[str, ...]) -> str | None:
    """Resolve one generated identifier to its owning standalone object prefix."""

    if not isinstance(name, str) or not name:
        return None
    for prefix in prefixes:
        if name == prefix or name.startswith(f"{prefix}_"):
            return prefix
    return None


def _assert_same_owner_reference(
    owner: str,
    names: Iterable[object],
    *,
    prefixes: tuple[str, ...],
    label: str,
) -> None:
    """Reject connected wrappers and cross-object generated references."""

    for raw_name in names:
        _assert(isinstance(raw_name, str) and raw_name, f"{label} has invalid name")
        if raw_name == "root":
            continue
        _assert(
            _owner_prefix(raw_name, prefixes) == owner,
            f"{label} crosses object rigs: owner={owner}, reference={raw_name!r}",
        )


def _expected_profile_bone_names(
    prefix: str,
    profile: A1RigProfile,
) -> frozenset[str]:
    """Derive the exact final bone inventory for the deterministic quad fixture.

    Both profiles share the legacy physical hierarchy. Their semantic distinction is
    the third public control: 2-Axis owns ``*_scale`` while 3-Axis owns
    ``*_rotation_Z``. Vertex-bone names are derived from the actual fixture geometry
    instead of maintaining a hand-written total bone count.
    """

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    if not isinstance(profile, A1RigProfile):
        raise TypeError("profile must be A1RigProfile")

    third_control = (
        f"{prefix}_scale"
        if profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE
        else f"{prefix}_rotation_Z"
    )
    fixed = {
        f"{prefix}_main",
        prefix,
        f"{prefix}_scale_rotate_X",
        f"{prefix}_rotate_X",
        f"{prefix}_{SOURCE_Z_GROUP_INDEX}_scale",
        f"{prefix}_{SOURCE_Z_GROUP_INDEX}",
        f"{prefix}_rotation_X",
        f"{prefix}_rotation_Y",
        third_control,
        f"{prefix}_rotate_X_constraint",
        f"{prefix}_rotate_X_constraint_scale_IK",
        f"{prefix}_rotate_X_constraint_rotate_IK",
        f"{prefix}_rotate_X_constraint_IK",
    }
    vertex_bones = {
        f"{prefix}_Segment_{SOURCE_SEGMENT_INDEX}_vertex_{vertex_index}"
        for vertex_index in range(SOURCE_VERTEX_COUNT)
    }
    return frozenset((*fixed, *vertex_bones))


def _assert_transform_properties(
    constraint: Mapping[str, object],
    *,
    name: str,
) -> None:
    """Require exact same-property mapping for all canonical transform channels."""

    properties = constraint.get("properties")
    _assert(isinstance(properties, dict), f"constraint {name} properties must be an object")
    _assert(
        tuple(properties) == TRANSFORM_PROPERTIES,
        f"constraint {name} property order/content changed: {tuple(properties)}",
    )
    for property_name in TRANSFORM_PROPERTIES:
        from_entry = properties.get(property_name)
        _assert(
            isinstance(from_entry, dict),
            f"constraint {name} properties.{property_name} must be an object",
        )
        _assert(
            from_entry == {"to": {property_name: {}}},
            f"constraint {name} properties.{property_name} mapping changed",
        )


def _assert_skin_membership(
    skin: Mapping[str, object],
    *,
    ik_names: set[str],
    transform_names: set[str],
) -> None:
    """Validate typed skin membership after unified root-constraint conversion."""

    _assert("constraints" not in skin, "canonical skin.constraints leaked into Spine 4.3")
    for field_name, known_names in (
        ("ik", ik_names),
        ("transform", transform_names),
    ):
        membership = skin.get(field_name)
        if membership is None:
            continue
        _assert(isinstance(membership, list), f"skin.{field_name} must be an array")
        _assert(
            all(isinstance(name, str) and name in known_names for name in membership),
            f"skin.{field_name} references an unknown constraint",
        )


def _assert_standalone_document(
    document: dict[str, object],
    case: ProfileCase,
) -> dict[str, object]:
    """Validate exact 4.3 unified schema and independent per-object ownership."""

    if not isinstance(document, dict):
        raise TypeError("document must be dict")
    if not isinstance(case, ProfileCase):
        raise TypeError("case must be ProfileCase")

    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata is missing")
    _assert(
        skeleton.get("spine") == SpineJsonTarget.SPINE_4_3.exact_version,
        f"unexpected Spine version: {skeleton.get('spine')!r}",
    )

    for legacy_collection in ("ik", "transform", "path", "physics", "slider"):
        _assert(
            legacy_collection not in document,
            f"legacy root constraint collection leaked: {legacy_collection}",
        )

    bones = _json_array(document, "bones", required=True)
    slots = _json_array(document, "slots", required=True)
    skins = _json_array(document, "skins", required=True)
    constraints = _json_array(document, "constraints", required=True)

    _assert(len(slots) == 3, f"{case.key} expected 3 slots, got {len(slots)}")
    _assert(len(skins) == 1, f"{case.key} expected exactly one skin")
    _assert(
        len(constraints) == case.expected_ik + case.expected_transform,
        f"{case.key} unexpected unified constraint count: {len(constraints)}",
    )

    bone_by_name: dict[str, dict[str, object]] = {}
    for raw_bone in bones:
        _assert(isinstance(raw_bone, dict), "bones contains a non-object")
        name = raw_bone.get("name")
        _assert(isinstance(name, str) and name, "bone has no name")
        _assert(name not in bone_by_name, f"duplicate bone name: {name}")
        _assert(not name.startswith("all_objects"), f"connected wrapper leaked: {name}")
        _assert("transform" not in raw_bone, f"legacy bone.transform leaked: {name}")
        inherit = raw_bone.get("inherit")
        _assert(
            inherit is None or isinstance(inherit, str),
            f"bone {name} has invalid inherit value",
        )
        bone_by_name[name] = raw_bone

    expected_bone_names = {"root"}
    for prefix in case.prefixes:
        expected_bone_names.update(_expected_profile_bone_names(prefix, case.profile))
    actual_bone_names = set(bone_by_name)
    _assert(
        actual_bone_names == expected_bone_names,
        f"{case.key} bone inventory differs: "
        f"missing={sorted(expected_bone_names - actual_bone_names)}, "
        f"unexpected={sorted(actual_bone_names - expected_bone_names)}",
    )

    for prefix in case.prefixes:
        scale_control = f"{prefix}_scale"
        z_control = f"{prefix}_rotation_Z"
        if case.profile is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
            _assert(scale_control in bone_by_name, f"missing 2-Axis scale control: {prefix}")
            _assert(z_control not in bone_by_name, f"3-Axis Z control leaked: {prefix}")
        else:
            _assert(z_control in bone_by_name, f"missing 3-Axis Z control: {prefix}")
            _assert(scale_control not in bone_by_name, f"2-Axis scale control leaked: {prefix}")

    for name, bone in bone_by_name.items():
        if name == "root":
            continue
        owner = _owner_prefix(name, case.prefixes)
        _assert(owner is not None, f"standalone document has unowned bone: {name}")
        parent = bone.get("parent")
        if parent is not None:
            _assert_same_owner_reference(
                owner,
                (parent,),
                prefixes=case.prefixes,
                label=f"bone {name} parent",
            )

    names: set[str] = set()
    ik_names: set[str] = set()
    transform_names: set[str] = set()
    type_sequence: list[str] = []
    for index, raw_constraint in enumerate(constraints):
        _assert(isinstance(raw_constraint, dict), "constraints contains a non-object")
        name = raw_constraint.get("name")
        constraint_type = raw_constraint.get("type")
        _assert(isinstance(name, str) and name, f"constraint[{index}] has no name")
        _assert(name not in names, f"duplicate unified constraint name: {name}")
        _assert(
            constraint_type in {"ik", "transform"},
            f"constraint {name} has unsupported type: {constraint_type!r}",
        )
        _assert("order" not in raw_constraint, f"constraint {name} retained legacy order")
        _assert("local" not in raw_constraint, f"constraint {name} retained legacy local")
        _assert("relative" not in raw_constraint, f"constraint {name} retained legacy relative")
        _assert(not name.startswith("all_objects"), f"connected constraint leaked: {name}")

        owner = _owner_prefix(name, case.prefixes)
        _assert(owner is not None, f"constraint has no object owner: {name}")
        raw_bones = raw_constraint.get("bones")
        _assert(isinstance(raw_bones, list), f"constraint {name} bones must be an array")
        _assert_same_owner_reference(
            owner,
            raw_bones,
            prefixes=case.prefixes,
            label=f"constraint {name} bones",
        )

        if constraint_type == "ik":
            target = raw_constraint.get("target")
            _assert("source" not in raw_constraint, f"IK {name} has transform source")
            _assert("properties" not in raw_constraint, f"IK {name} has transform properties")
            _assert_same_owner_reference(
                owner,
                (target,),
                prefixes=case.prefixes,
                label=f"IK {name} target",
            )
            ik_names.add(name)
        else:
            source = raw_constraint.get("source")
            _assert("target" not in raw_constraint, f"transform {name} retained target")
            _assert_same_owner_reference(
                owner,
                (source,),
                prefixes=case.prefixes,
                label=f"transform {name} source",
            )
            for boolean_name in ("localSource", "localTarget", "additive"):
                value = raw_constraint.get(boolean_name)
                _assert(
                    value is None or isinstance(value, bool),
                    f"transform {name} {boolean_name} must be bool when present",
                )
            _assert_transform_properties(raw_constraint, name=name)
            transform_names.add(name)

        names.add(name)
        type_sequence.append(constraint_type)

    _assert(
        len(ik_names) == case.expected_ik,
        f"{case.key} expected {case.expected_ik} IK constraints, got {len(ik_names)}",
    )
    _assert(
        len(transform_names) == case.expected_transform,
        f"{case.key} expected {case.expected_transform} transform constraints, "
        f"got {len(transform_names)}",
    )

    for raw_slot in slots:
        _assert(isinstance(raw_slot, dict), "slots contains a non-object")
        name = raw_slot.get("name")
        bone_name = raw_slot.get("bone")
        _assert(isinstance(name, str) and name, "slot has no name")
        owner = _owner_prefix(name, case.prefixes)
        _assert(owner is not None, f"slot has no object owner: {name}")
        _assert_same_owner_reference(
            owner,
            (bone_name,),
            prefixes=case.prefixes,
            label=f"slot {name} bone",
        )

    skin = skins[0]
    _assert(isinstance(skin, dict), "skin is not a JSON object")
    _assert(isinstance(skin.get("attachments"), dict), "skin attachments are missing")
    _assert_skin_membership(
        skin,
        ik_names=ik_names,
        transform_names=transform_names,
    )

    return {
        "status": "passed",
        "profile": case.profile.value,
        "version": skeleton["spine"],
        "mode": A1MultiObjectMode.STANDALONE.value,
        "prefixes": list(case.prefixes),
        "bones": len(bones),
        "slots": len(slots),
        "skins": len(skins),
        "constraints": len(constraints),
        "ik": len(ik_names),
        "transform": len(transform_names),
        "constraintTypes": type_sequence,
        "profileBoneInventoryExact": True,
        "connectedWrapperPresent": False,
        "crossObjectReferencesPresent": False,
        "legacyRootConstraintCollectionsPresent": False,
        "legacyConstraintOrderPresent": False,
    }


def _assert_state_restored(
    *,
    context_before: object,
    scene_before: object,
    materials: tuple[object, ...],
    material_fingerprints: tuple[object, ...],
    label: str,
) -> None:
    """Verify context, scene, materials, and temporary datablock ownership."""

    _assert(_capture_context() == context_before, f"{label} export changed context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        f"{label} export changed scene bake state",
    )
    _assert(
        tuple(_material_fingerprint(material) for material in materials)
        == material_fingerprints,
        f"{label} export mutated source materials",
    )
    _assert(
        not _temporary_datablock_names(),
        f"{label} export leaked temporary Blender datablocks",
    )


def _run_case(output_root: Path, case: ProfileCase) -> dict[str, object]:
    """Run one complete production profile export and return its evidence report."""

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
    material_fingerprints = tuple(
        _material_fingerprint(material) for material in materials
    )

    settings = A1MultiObjectExportSettings(
        output_directory=case_directory,
        output_stem=case.output_stem,
        mode=A1MultiObjectMode.STANDALONE,
    )
    result = export_a1_multi_object(sources, settings)
    _assert(result.success, f"{case.key} Spine 4.3 export failed: {result.issues}")

    json_path = (case_directory / f"{case.output_stem}.json").resolve()
    expected_textures = tuple(
        (case_directory / "images" / f"{prefix}_Baked.png").resolve()
        for prefix in case.prefixes
    )
    _assert(
        result.output_files == (json_path, *expected_textures),
        f"{case.key} unexpected output file order: {result.output_files}",
    )
    for texture_path in expected_textures:
        _assert(texture_path.is_file(), f"missing texture: {texture_path}")
        _assert(
            texture_path.read_bytes()[:8] == PNG_SIGNATURE,
            f"invalid PNG: {texture_path}",
        )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "generated JSON root is not an object")
    report = _assert_standalone_document(document, case)
    report.update(
        {
            "jsonPath": str(json_path),
            "texturePaths": [str(path) for path in expected_textures],
            "outputFiles": [str(path) for path in result.output_files],
        }
    )

    _assert_state_restored(
        context_before=context_before,
        scene_before=scene_before,
        materials=materials,
        material_fingerprints=material_fingerprints,
        label=case.key,
    )
    return report


def run(output_directory: Path) -> Path:
    """Run both profile exports and return the combined evidence report path."""

    output_root = _prepare_output_directory(output_directory)
    reports: list[dict[str, object]] = []
    try:
        for case in CASES:
            reports.append(_run_case(output_root, case))
    finally:
        _clear_scene()

    report = {
        "status": "passed",
        "version": SpineJsonTarget.SPINE_4_3.exact_version,
        "mode": A1MultiObjectMode.STANDALONE.value,
        "profiles": reports,
        "runtimeValidated": False,
        "manualEditorImportRequired": True,
    }
    report_path = output_root / "blender_acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    """CLI entry point used by Blender's ``--python`` option."""

    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[SPINE43_STANDALONE] RUN 2-Axis and 3-Axis production exports")
    report_path = run(arguments.output)
    print(f"[SPINE43_STANDALONE] REPORT {report_path}")
    print("[SPINE43_STANDALONE] PASS production profile exports")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

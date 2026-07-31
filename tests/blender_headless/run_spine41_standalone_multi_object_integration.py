"""Generate and validate a real Blender 5.2 Spine 4.1 standalone multi-object export.

This worker exercises the production A1 multi-object output service. It never constructs
or repairs Spine JSON manually and never enters connected composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Iterable

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


OUTPUT_STEM = "Spine41StandaloneMulti"
PREFIXES = ("Spine41A", "Spine41B", "Spine41C")
COMPONENT_IDS = ("component_a", "component_b", "component_c")
POSITIONS = (
    (0.0, 0.0, 0.0),
    (2.0, 1.0, 0.5),
    (-1.5, 2.25, 1.0),
)


def _parse_arguments() -> argparse.Namespace:
    """Parse only arguments after Blender's ``--`` separator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Empty directory that receives generated JSON, PNG files, and reports.",
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
) -> A1SingleObjectExportSettings:
    """Build exact production settings for one independent Spine 4.1 object rig."""

    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=32,
            texture_height=32,
            output_directory=output_directory,
            images_relative_path="images",
            spine_version=SpineJsonTarget.SPINE_4_1.exact_version,
            rig_profile=A1RigProfile.TWO_AXIS_ROTATION_SCALE.value,
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
) -> tuple[tuple[A1MultiObjectSource, ...], tuple[object, ...]]:
    """Create three independent real Blender Mesh sources with distinct placement."""

    sources: list[A1MultiObjectSource] = []
    materials: list[object] = []
    for index, (prefix, component_id, position) in enumerate(
        zip(PREFIXES, COMPONENT_IDS, POSITIONS, strict=True),
        start=1,
    ):
        source_object = _create_quad(f"Spine41Source{index}")
        source_object.location = position
        material = _create_emission_material(source_object)
        materials.append(material)
        sources.append(
            A1MultiObjectSource(
                source_object=source_object,
                component_id=component_id,
                animation_namespace=f"object_{index}",
                settings=_build_object_settings(
                    output_directory,
                    prefix=prefix,
                ),
            )
        )
    return tuple(sources), tuple(materials)


def _owner_prefix(name: str) -> str | None:
    """Resolve the owning object prefix for one generated identifier."""

    if not isinstance(name, str) or not name:
        return None
    for prefix in PREFIXES:
        if name == prefix or name.startswith(f"{prefix}_"):
            return prefix
    return None


def _json_array(
    document: dict[str, object],
    field_name: str,
) -> list[object]:
    """Return one optional JSON array while rejecting malformed present fields.

    Spine omits empty top-level collections. Missing ``path`` therefore means an empty
    path-constraint collection, while an explicitly present non-array remains invalid.
    """

    if not isinstance(document, dict):
        raise TypeError("document must be dict")
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name must be a non-empty string")
    if field_name not in document:
        return []
    collection = document[field_name]
    _assert(
        isinstance(collection, list),
        f"{field_name} must be a JSON array when present",
    )
    return collection


def _all_constraints(document: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Return all ordered runtime constraint collections as JSON objects."""

    result: list[dict[str, object]] = []
    for collection_name in ("ik", "transform", "path"):
        for item in _json_array(document, collection_name):
            _assert(
                isinstance(item, dict),
                f"{collection_name} contains a non-object",
            )
            result.append(item)
    return tuple(result)


def _assert_same_owner_reference(
    owner: str,
    names: Iterable[object],
    *,
    label: str,
) -> None:
    """Reject any cross-object generated bone or target reference."""

    for raw_name in names:
        _assert(isinstance(raw_name, str) and raw_name, f"{label} has invalid name")
        if raw_name == "root":
            continue
        actual_owner = _owner_prefix(raw_name)
        _assert(
            actual_owner == owner,
            f"{label} crosses object rigs: owner={owner}, reference={raw_name!r}",
        )


def _assert_optional_skin_membership(
    skin: dict[str, object],
    document: dict[str, object],
    *,
    field_name: str,
) -> None:
    """Validate optional skin-required constraint membership without requiring it."""

    membership = skin.get(field_name)
    if membership is None:
        return
    _assert(
        isinstance(membership, list),
        f"skin.{field_name} must be a JSON array when present",
    )
    known_names = {
        item["name"]
        for item in _json_array(document, field_name)
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    _assert(
        all(isinstance(name, str) and name in known_names for name in membership),
        f"skin.{field_name} references an unknown constraint",
    )


def _assert_standalone_document(document: dict[str, object]) -> dict[str, object]:
    """Validate target metadata and independent per-object rig ownership."""

    skeleton = document.get("skeleton")
    _assert(isinstance(skeleton, dict), "skeleton metadata is missing")
    _assert(
        skeleton.get("spine") == SpineJsonTarget.SPINE_4_1.exact_version,
        f"unexpected Spine version: {skeleton.get('spine')!r}",
    )

    bones = _json_array(document, "bones")
    slots = _json_array(document, "slots")
    skins = _json_array(document, "skins")
    ik = _json_array(document, "ik")
    transform = _json_array(document, "transform")
    path = _json_array(document, "path")
    _assert(bones, "bones must be a non-empty array")
    _assert(slots, "slots must be a non-empty array")
    _assert(len(skins) == 1, "exactly one skin is required")
    _assert(len(ik) == 3, "expected one IK constraint per object")
    _assert(len(transform) == 12, "expected four transform constraints per object")
    _assert(len(path) == 0, "standalone acceptance expects no path constraints")

    bone_by_name: dict[str, dict[str, object]] = {}
    for bone in bones:
        _assert(isinstance(bone, dict), "bones contains a non-object")
        name = bone.get("name")
        _assert(isinstance(name, str) and name, "bone has no name")
        _assert(name not in bone_by_name, f"duplicate bone name: {name}")
        _assert(
            not name.startswith("all_objects"),
            f"connected wrapper leaked: {name}",
        )
        _assert("inherit" not in bone, f"Spine 4.2 inherit field leaked: {name}")
        bone_by_name[name] = bone

    _assert(
        tuple(name for name in bone_by_name if name == "root") == ("root",),
        "root count changed",
    )
    for prefix in PREFIXES:
        _assert(f"{prefix}_main" in bone_by_name, f"missing main bone for {prefix}")
        _assert(
            float(bone_by_name[f"{prefix}_rotation_X"].get("rotation", 0.0)) == 0.0,
            f"{prefix}_rotation_X must have a neutral setup rotation",
        )
        _assert(
            float(bone_by_name[f"{prefix}_rotation_Y"].get("rotation", 0.0)) == 0.0,
            f"{prefix}_rotation_Y must have a neutral setup rotation",
        )

    for name, bone in bone_by_name.items():
        if name == "root":
            continue
        owner = _owner_prefix(name)
        _assert(owner is not None, f"standalone document has an unowned bone: {name}")
        parent = bone.get("parent")
        if parent is not None:
            _assert_same_owner_reference(
                owner,
                (parent,),
                label=f"bone {name} parent",
            )

    constraints = _all_constraints(document)
    _assert(
        len(constraints) == 15,
        f"unexpected constraint count: {len(constraints)}",
    )

    orders: list[int] = []
    for constraint in constraints:
        name = constraint.get("name")
        _assert(isinstance(name, str) and name, "constraint has no name")
        _assert(
            not name.startswith("all_objects"),
            f"connected constraint leaked: {name}",
        )
        owner = _owner_prefix(name)
        _assert(owner is not None, f"constraint has no object owner: {name}")
        raw_bones = constraint.get("bones", [])
        _assert(
            isinstance(raw_bones, list),
            f"constraint {name} bones must be a list",
        )
        _assert_same_owner_reference(
            owner,
            raw_bones,
            label=f"constraint {name} bones",
        )
        _assert_same_owner_reference(
            owner,
            (constraint.get("target"),),
            label=f"constraint {name} target",
        )
        order = constraint.get("order", 0)
        _assert(
            isinstance(order, int) and not isinstance(order, bool),
            f"invalid order: {name}",
        )
        orders.append(order)

    _assert(
        tuple(sorted(orders)) == tuple(range(len(constraints))),
        f"constraint orders must be unique and contiguous: {orders}",
    )

    transform_by_name = {
        item["name"]: item
        for item in transform
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    for prefix in PREFIXES:
        _assert(
            transform_by_name[f"{prefix}_rotation_X_constraint"].get("rotation")
            == -134.67,
            f"{prefix} X reference angle was not preserved as a constraint offset",
        )
        _assert(
            transform_by_name[f"{prefix}_rotation_Y"].get("rotation") == -17.43,
            f"{prefix} Y reference angle was not preserved as a constraint offset",
        )

    slot_names: list[str] = []
    for slot in slots:
        _assert(isinstance(slot, dict), "slots contains a non-object")
        name = slot.get("name")
        bone = slot.get("bone")
        _assert(isinstance(name, str) and name, "slot has no name")
        owner = _owner_prefix(name)
        _assert(owner is not None, f"slot has no object owner: {name}")
        _assert_same_owner_reference(
            owner,
            (bone,),
            label=f"slot {name} bone",
        )
        slot_names.append(name)

    skin = skins[0]
    _assert(isinstance(skin, dict), "skin is not a JSON object")
    _assert("constraints" not in skin, "Spine 4.2 skin constraint field leaked")
    _assert(isinstance(skin.get("attachments"), dict), "skin attachments are missing")
    _assert_optional_skin_membership(skin, document, field_name="ik")
    _assert_optional_skin_membership(skin, document, field_name="transform")

    return {
        "version": skeleton["spine"],
        "mode": A1MultiObjectMode.STANDALONE.value,
        "prefixes": list(PREFIXES),
        "bones": len(bones),
        "slots": len(slots),
        "ik": len(ik),
        "transform": len(transform),
        "path": len(path),
        "constraints": len(constraints),
        "constraintOrderMinimum": min(orders),
        "constraintOrderMaximum": max(orders),
        "constraintOrdersContiguous": True,
        "connectedWrapperPresent": False,
        "crossObjectReferencesPresent": False,
        "slotNames": slot_names,
    }


def _assert_state_restored(
    *,
    context_before: object,
    scene_before: object,
    materials: tuple[object, ...],
    material_fingerprints: tuple[object, ...],
) -> None:
    """Verify Blender state and temporary datablock ownership after export."""

    _assert(_capture_context() == context_before, "Spine 4.1 export changed context")
    _assert(
        _capture_scene_bake_state() == scene_before,
        "Spine 4.1 export changed scene bake state",
    )
    _assert(
        tuple(_material_fingerprint(material) for material in materials)
        == material_fingerprints,
        "Spine 4.1 export mutated source materials",
    )
    _assert(
        not _temporary_datablock_names(),
        "Spine 4.1 export leaked temporary Blender datablocks",
    )


def run(output_directory: Path) -> Path:
    """Run the complete production export and return the generated JSON path."""

    output_directory = _prepare_output_directory(output_directory)
    _clear_scene()
    _configure_cycles_scene()
    sources, materials = _build_sources(output_directory)
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
        output_directory=output_directory,
        output_stem=OUTPUT_STEM,
        mode=A1MultiObjectMode.STANDALONE,
    )
    result = export_a1_multi_object(sources, settings)
    _assert(result.success, f"Spine 4.1 standalone export failed: {result.issues}")

    json_path = (output_directory / f"{OUTPUT_STEM}.json").resolve()
    expected_textures = tuple(
        (output_directory / "images" / f"{prefix}_Baked.png").resolve()
        for prefix in PREFIXES
    )
    _assert(
        result.output_files == (json_path, *expected_textures),
        f"unexpected output file order: {result.output_files}",
    )
    for texture_path in expected_textures:
        _assert(
            texture_path.read_bytes()[:8] == PNG_SIGNATURE,
            f"invalid PNG: {texture_path}",
        )

    document = json.loads(json_path.read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "generated JSON root is not an object")
    report = _assert_standalone_document(document)
    report.update(
        {
            "status": "passed",
            "jsonPath": str(json_path),
            "texturePaths": [str(path) for path in expected_textures],
            "outputFiles": [str(path) for path in result.output_files],
        }
    )
    report_path = output_directory / "blender_acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _assert_state_restored(
        context_before=context_before,
        scene_before=scene_before,
        materials=materials,
        material_fingerprints=material_fingerprints,
    )
    return json_path


def main() -> None:
    """CLI entry point used by Blender's ``--python`` option."""

    arguments = _parse_arguments()
    print(f"Blender version: {bpy.app.version_string}")
    print("[SPINE41_STANDALONE] RUN production multi-object export")
    json_path = run(arguments.output)
    print(f"[SPINE41_STANDALONE] JSON {json_path}")
    print("[SPINE41_STANDALONE] PASS production multi-object export")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

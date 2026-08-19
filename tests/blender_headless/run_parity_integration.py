"""Validate the A1 parity gate against JSON produced by Blender 5.2.

The fixture exports a real object through the complete service, confirms a copied
result is compatible despite volatile metadata changes, then corrupts one weighted
bone index and requires a semantic incompatibility report.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    BakeMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (  # noqa: E402
    compare_a1_exports,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (  # noqa: E402
    UvUnwrapSettings,
)
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _create_emission_material,
    _create_quad,
    _create_sentinel,
    _temporary_datablock_names,
)


def _build_settings(output_directory: Path) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=16,
            texture_height=16,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=1,
        ),
        prefix="ParityFixture",
        output_stem="ParityFixture",
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name="SpineBakeUV"),
        diffuse_mode=BakeMode.EMIT,
        procedural_mode=BakeMode.EMIT,
        bake_execution=BakeExecutionSettings(samples=1),
    )


def _first_mesh_attachment(document: dict):
    skin = document["skins"][0]
    slot_attachments = next(iter(skin["attachments"].values()))
    return next(iter(slot_attachments.values()))


def test_parity_gate_reads_real_service_output() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-parity-") as directory:
        output_directory = Path(directory)
        source = _create_quad("ParitySource")
        _create_emission_material(source)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        source.select_set(False)

        result = export_a1_single_object(
            source,
            _build_settings(output_directory),
        )
        _assert(result.success, f"fixture export failed: {result.issues}")
        json_path = output_directory / "ParityFixture.json"
        expected = json.loads(json_path.read_text(encoding="utf-8"))

        equivalent = deepcopy(expected)
        equivalent["skeleton"]["hash"] = "different-volatile-hash"
        equivalent["skeleton"]["images"] = "other/images"
        compatible = compare_a1_exports(expected, equivalent)
        _assert(
            compatible.compatible,
            f"equivalent real export rejected: {compatible.issues}",
        )

        corrupted = deepcopy(expected)
        attachment = _first_mesh_attachment(corrupted)
        original_bone_index = int(attachment["vertices"][1])
        attachment["vertices"][1] = original_bone_index + 1
        incompatible = compare_a1_exports(expected, corrupted)
        _assert(not incompatible.compatible, "corrupted weighted index was accepted")
        matching = tuple(
            issue
            for issue in incompatible.issues
            if issue.code == "WEIGHTED_BONE_INDEX_MISMATCH"
        )
        _assert(len(matching) == 1, f"unexpected parity issues: {incompatible.issues}")
        _assert(
            matching[0].expected == original_bone_index,
            "expected weighted index was not recorded",
        )
        _assert(
            matching[0].actual == original_bone_index + 1,
            "actual weighted index was not recorded",
        )
        _assert(
            not _temporary_datablock_names(),
            "parity fixture leaked temporary Blender datablocks",
        )


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    print(f"[PARITY] RUN {test_parity_gate_reads_real_service_output.__name__}")
    test_parity_gate_reads_real_service_output()
    print(f"[PARITY] PASS {test_parity_gate_reads_real_service_output.__name__}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

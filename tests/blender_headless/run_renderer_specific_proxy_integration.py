"""Blender 4.4 regression for renderer-specific copied-material proxies."""

from __future__ import annotations

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

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    execute_bake_plan,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
)
from run_alpha_bake_integration import (  # noqa: E402
    _assert_alpha_band,
    _prepare_plan,
    _read_pixels,
)
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _create_quad,
    _create_sentinel,
    _material_fingerprint,
    _temporary_datablock_names,
)


def _renderer_specific_alpha_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    # Create Eevee first so the historical "first active output" fallback selects the
    # wrong graph. The fixed proxy must explicitly select the Cycles output.
    eevee_output = nodes.new(type="ShaderNodeOutputMaterial")
    eevee_output.name = "Eevee Material Output"
    eevee_output.target = "EEVEE"
    cycles_output = nodes.new(type="ShaderNodeOutputMaterial")
    cycles_output.name = "Cycles Material Output"
    cycles_output.target = "CYCLES"

    eevee = nodes.new(type="ShaderNodeBsdfPrincipled")
    eevee.name = "Eevee Blue Alpha"
    eevee.inputs["Base Color"].default_value = (0.02, 0.08, 0.9, 1.0)
    eevee.inputs["Roughness"].default_value = 1.0
    eevee.inputs["Alpha"].default_value = 0.82

    cycles = nodes.new(type="ShaderNodeBsdfPrincipled")
    cycles.name = "Cycles Red Alpha"
    cycles.inputs["Base Color"].default_value = (0.9, 0.03, 0.01, 1.0)
    cycles.inputs["Roughness"].default_value = 1.0
    cycles.inputs["Alpha"].default_value = 0.27

    material.node_tree.links.new(eevee.outputs["BSDF"], eevee_output.inputs["Surface"])
    material.node_tree.links.new(cycles.outputs["BSDF"], cycles_output.inputs["Surface"])
    return material, eevee_output, cycles_output


def test_cycles_alpha_proxy_uses_cycles_material_output() -> None:
    _clear_scene()
    bpy.context.scene.render.engine = "CYCLES"
    with tempfile.TemporaryDirectory(prefix="spine2d-renderer-proxy-") as directory:
        obj = _create_quad("RendererProxy")
        material, eevee_output, cycles_output = _renderer_specific_alpha_material(
            "RendererProxyMaterial"
        )
        obj.data.materials.append(material)
        sentinel = _create_sentinel()
        _activate_only(sentinel)
        obj.select_set(False)

        target, analysis, plan = _prepare_plan(
            obj,
            Path(directory),
            "RendererProxy",
        )
        graph = analysis.slots[0].graph
        _assert(graph is not None, "renderer proxy graph is missing")
        _assert(
            graph.active_output_node_id == "Cycles Material Output",
            f"analysis selected wrong output: {graph.active_output_node_id}",
        )

        original_active = (
            bool(eevee_output.is_active_output),
            bool(cycles_output.is_active_output),
        )
        material_before = _material_fingerprint(material)
        result = execute_bake_plan(
            obj,
            target,
            plan,
            BakeExecutionSettings(render_engine="CYCLES", samples=1),
        )
        pixels = _read_pixels(result.representative_artifact.output_path)

        _assert_alpha_band(pixels, 0.27, tolerance=0.08)
        covered = [
            (
                pixels[offset],
                pixels[offset + 1],
                pixels[offset + 2],
            )
            for offset in range(0, len(pixels), 4)
            if pixels[offset + 3] > 0.15
        ]
        _assert(covered, "renderer-specific proxy produced no covered pixels")
        mean_red = sum(value[0] for value in covered) / len(covered)
        mean_blue = sum(value[2] for value in covered) / len(covered)
        _assert(
            mean_red > mean_blue * 2.0,
            f"proxy used Eevee blue graph instead of Cycles red: red={mean_red}, blue={mean_blue}",
        )
        _assert(
            (
                bool(eevee_output.is_active_output),
                bool(cycles_output.is_active_output),
            )
            == original_active,
            "renderer-specific output state was not restored",
        )
        _assert(_material_fingerprint(material) == material_before, "source material mutated")
        _assert(not _temporary_datablock_names(), "renderer proxy leaked temporary data")


def main() -> None:
    test_cycles_alpha_proxy_uses_cycles_material_output()
    print("[PASS] test_cycles_alpha_proxy_uses_cycles_material_output")
    print("Renderer-specific material proxy integration passed: 1 test")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

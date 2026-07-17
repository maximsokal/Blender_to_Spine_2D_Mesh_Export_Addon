"""Blender 4.4 integration checks for the A1 renderer contract."""

from __future__ import annotations

from dataclasses import replace
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
    A1SingleObjectStage,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    A1ObjectPreparationError,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.render_engine_contract import (  # noqa: E402
    RenderEngineContractError,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    BakeExecutionSettings,
    CameraProjectionPlan,
)
from run_bake_integration import _assert  # noqa: E402
from run_camera_projection_integration import (  # noqa: E402
    _create_quad,
    _prepare_scene_with_sentinel,
    _settings,
)


def _renderer_specific_material(name: str):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    cycles_output = nodes.new(type="ShaderNodeOutputMaterial")
    cycles_output.name = "Cycles Material Output"
    cycles_output.target = "CYCLES"
    eevee_output = nodes.new(type="ShaderNodeOutputMaterial")
    eevee_output.name = "Eevee Material Output"
    eevee_output.target = "EEVEE"

    layer_weight = nodes.new(type="ShaderNodeLayerWeight")
    cycles_emission = nodes.new(type="ShaderNodeEmission")
    material.node_tree.links.new(
        layer_weight.outputs["Facing"],
        cycles_emission.inputs["Color"],
    )
    material.node_tree.links.new(
        cycles_emission.outputs["Emission"],
        cycles_output.inputs["Surface"],
    )

    eevee_diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    eevee_diffuse.inputs["Color"].default_value = (0.7, 0.15, 0.03, 1.0)
    material.node_tree.links.new(
        eevee_diffuse.outputs["BSDF"],
        eevee_output.inputs["Surface"],
    )
    return material


def test_cycles_analysis_uses_cycles_output() -> None:
    _prepare_scene_with_sentinel()
    with tempfile.TemporaryDirectory(prefix="spine2d-renderer-cycles-") as directory:
        source = _create_quad("CyclesRendererSource")
        source.data.materials.append(_renderer_specific_material("RendererMaterial"))

        prepared = prepare_a1_object(
            source,
            _settings(Path(directory), "CyclesRenderer"),
        )
        graph = prepared.material_analysis.slots[0].graph
        _assert(graph is not None, "Cycles graph snapshot is missing")
        _assert(
            graph.active_output_node_id == "Cycles Material Output",
            f"wrong Cycles Material Output: {graph.active_output_node_id}",
        )
        _assert(
            isinstance(prepared.bake_plan, CameraProjectionPlan),
            "Cycles Layer Weight did not route to B4",
        )
        _assert(prepared.statistics["render_engine"] == "CYCLES", "wrong engine statistic")
        _assert(
            prepared.statistics["shader_render_target"] == "CYCLES",
            "wrong shader target statistic",
        )


def test_eevee_analysis_and_execution_contract_force_b4() -> None:
    _prepare_scene_with_sentinel()
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    with tempfile.TemporaryDirectory(prefix="spine2d-renderer-eevee-") as directory:
        source = _create_quad("EeveeRendererSource")
        source.data.materials.append(_renderer_specific_material("RendererMaterial"))
        settings = replace(
            _settings(Path(directory), "EeveeRenderer"),
            bake_execution=BakeExecutionSettings(
                render_engine="BLENDER_EEVEE_NEXT",
                samples=2,
            ),
        )

        prepared = prepare_a1_object(source, settings)
        graph = prepared.material_analysis.slots[0].graph
        _assert(graph is not None, "Eevee graph snapshot is missing")
        _assert(
            graph.active_output_node_id == "Eevee Material Output",
            f"wrong Eevee Material Output: {graph.active_output_node_id}",
        )
        _assert(
            isinstance(prepared.bake_plan, CameraProjectionPlan),
            "Eevee local material was incorrectly assigned to object bake",
        )
        _assert(
            prepared.statistics["render_engine"] == "BLENDER_EEVEE_NEXT",
            "wrong Eevee engine statistic",
        )
        _assert(
            prepared.statistics["shader_render_target"] == "EEVEE",
            "wrong Eevee shader target statistic",
        )


def test_renderer_mismatch_fails_before_execution() -> None:
    _prepare_scene_with_sentinel()
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    with tempfile.TemporaryDirectory(prefix="spine2d-renderer-mismatch-") as directory:
        source = _create_quad("RendererMismatchSource")
        source.data.materials.append(_renderer_specific_material("RendererMaterial"))

        try:
            prepare_a1_object(
                source,
                _settings(Path(directory), "RendererMismatch"),
            )
        except A1ObjectPreparationError as exc:
            _assert(
                exc.stage is A1SingleObjectStage.PLAN_BAKE,
                f"renderer mismatch failed at wrong stage: {exc.stage}",
            )
            _assert(
                isinstance(exc.cause, RenderEngineContractError),
                f"wrong mismatch cause: {type(exc.cause).__name__}",
            )
        else:
            raise AssertionError("renderer mismatch was silently accepted")


def main() -> None:
    tests = (
        test_cycles_analysis_uses_cycles_output,
        test_eevee_analysis_and_execution_contract_force_b4,
        test_renderer_mismatch_fails_before_execution,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"Renderer contract integration passed: {len(tests)} tests")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

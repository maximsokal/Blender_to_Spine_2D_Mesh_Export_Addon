"""Real Blender 5.2 custom Compositor isolation and restoration fixture for B4."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback
from unittest import mock

import bpy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_single_object,
)
import Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_execution as render_module  # noqa: E402
from run_bake_integration import (  # noqa: E402
    _assert,
    _capture_context,
    _temporary_datablock_names,
)
from run_camera_projection_integration import (  # noqa: E402
    _create_layer_weight_material,
    _create_quad,
    _prepare_scene_with_sentinel,
    _read_pixels,
    _scene_render_fingerprint,
    _settings,
    _visible_and_transparent_counts,
)


def _create_destructive_compositor(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    constant = tree.nodes.new(type="CompositorNodeRGB")
    constant.name = "Destructive Full Frame Magenta"
    constant.outputs["RGBA"].default_value = (1.0, 0.0, 1.0, 1.0)
    composite = tree.nodes.new(type="CompositorNodeComposite")
    composite.name = "Destructive Composite Output"
    tree.links.new(constant.outputs["RGBA"], composite.inputs["Image"])
    return tree


def _node_tree_fingerprint(tree):
    nodes = tuple(
        sorted(
            (
                node.name,
                node.bl_idname,
                tuple(
                    tuple(float(value) for value in socket.default_value)
                    if hasattr(socket, "default_value")
                    and hasattr(socket.default_value, "__iter__")
                    else None
                    for socket in node.outputs
                ),
            )
            for node in tree.nodes
        )
    )
    links = tuple(
        sorted(
            (
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            for link in tree.links
        )
    )
    return nodes, links


def _magenta_visible_count(pixels):
    count = 0
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset : offset + 4]
        if alpha > 0.95 and red > 0.95 and blue > 0.95 and green < 0.05:
            count += 1
    return count


def test_custom_compositor_is_bypassed_during_b4_and_restored_unchanged() -> None:
    _prepare_scene_with_sentinel()
    scene = bpy.context.scene
    tree = _create_destructive_compositor(scene)
    scene.render.use_compositing = True
    scene.render.use_sequencer = True
    tree_before = _node_tree_fingerprint(tree)

    with tempfile.TemporaryDirectory(prefix="spine2d-custom-compositor-") as directory:
        output_directory = Path(directory)
        source = _create_quad("CustomCompositorSource")
        source.scale = (0.62, 0.48, 1.0)
        source.data.materials.append(
            _create_layer_weight_material("CustomCompositorMaterial")
        )
        context_before = _capture_context()
        render_before = _scene_render_fingerprint()
        observations = []
        original_render = render_module._call_render_operator

        def guarded_render(bpy_module):
            observations.append(
                {
                    "use_compositing": bool(scene.render.use_compositing),
                    "use_sequencer": bool(scene.render.use_sequencer),
                    "use_nodes": bool(scene.use_nodes),
                    "tree": _node_tree_fingerprint(scene.node_tree),
                }
            )
            _assert(
                observations[-1]["use_compositing"] is False,
                "B4 did not disable custom Compositor execution",
            )
            _assert(
                observations[-1]["use_sequencer"] is False,
                "B4 did not disable Sequencer postprocessing",
            )
            _assert(
                observations[-1]["use_nodes"] is True,
                "B4 mutated scene.use_nodes instead of isolating execution",
            )
            _assert(
                observations[-1]["tree"] == tree_before,
                "B4 mutated the custom Compositor node tree before render",
            )
            return original_render(bpy_module)

        with mock.patch.object(
            render_module,
            "_call_render_operator",
            side_effect=guarded_render,
        ):
            result = export_a1_single_object(
                source,
                _settings(output_directory, "CustomCompositorIsolation"),
            )

        _assert(result.success, f"custom Compositor B4 failed: {result.issues}")
        _assert(observations, "B4 render operator was not called")
        _assert(scene.render.use_compositing, "Compositor flag was not restored")
        _assert(scene.render.use_sequencer, "Sequencer flag was not restored")
        _assert(scene.use_nodes, "scene.use_nodes was not preserved")
        _assert(
            _node_tree_fingerprint(scene.node_tree) == tree_before,
            "custom Compositor node tree was not restored byte-for-structure",
        )

        pixels = _read_pixels(result.image_paths[0])
        visible, transparent = _visible_and_transparent_counts(pixels)
        magenta = _magenta_visible_count(pixels)
        pixel_count = len(pixels) // 4
        _assert(visible > 20, "isolated B4 image has no visible source pixels")
        _assert(transparent > 20, "isolated B4 image lost transparent background")
        _assert(
            magenta < pixel_count // 4,
            "custom Compositor constant output leaked into staged B4 image",
        )

        _assert(_capture_context() == context_before, "custom Compositor B4 changed context")
        _assert(
            _scene_render_fingerprint() == render_before,
            "custom Compositor B4 changed render/visibility state",
        )
        _assert(not _temporary_datablock_names(), "custom Compositor B4 leaked data")


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    test_custom_compositor_is_bypassed_during_b4_and_restored_unchanged()
    print("[CUSTOM-COMPOSITOR] PASS isolation and restoration")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

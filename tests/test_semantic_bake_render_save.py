from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_execution import (
    _bake_composed_frame,
    _bake_single_frame,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_render_save import (
    _save_render_managed_image,
    save_semantic_bake_image,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import TextureFormat


class _Image:
    def __init__(self) -> None:
        self.file_format = ""
        self.save_render_calls: list[tuple[str, object]] = []

    def save_render(self, filepath: str, *, scene) -> None:
        self.save_render_calls.append((filepath, scene))
        Path(filepath).write_bytes(b"render-managed")


def _scene(*, file_format: str = "PNG"):
    return SimpleNamespace(
        render=SimpleNamespace(
            image_settings=SimpleNamespace(file_format=file_format),
        ),
        view_settings=SimpleNamespace(
            view_transform="AgX",
            look="None",
            exposure=0.0,
            gamma=1.0,
        ),
    )


def test_render_managed_save_uses_exact_scene_and_output_format(tmp_path):
    image = _Image()
    scene = _scene()
    output = tmp_path / "scene-aware.png"

    _save_render_managed_image(
        image,
        output,
        texture_format=TextureFormat.PNG,
        scene=scene,
    )

    assert output.read_bytes() == b"render-managed"
    assert image.file_format == "PNG"
    assert image.save_render_calls == [(str(output), scene)]


def test_render_managed_save_fails_closed_when_scene_format_drifted(tmp_path):
    image = _Image()

    with pytest.raises(BakeExecutionError, match="output format differs"):
        _save_render_managed_image(
            image,
            tmp_path / "wrong-format.png",
            texture_format=TextureFormat.PNG,
            scene=_scene(file_format="OPEN_EXR"),
        )

    assert image.save_render_calls == []


def test_semantic_save_policy_keeps_local_texture_data_and_scene_render_paths_separate():
    source = inspect.getsource(save_semantic_bake_image)

    assert "if not plan.scene_aware:" in source
    assert "_save_texture_data_image(image, reservation, plan)" in source
    assert "_flip_image_rows_for_spine(image)" in source
    assert "_save_render_managed_image(" in source


def test_runtime_passes_exact_scene_to_single_and_composed_save_paths():
    single = inspect.getsource(_bake_single_frame)
    composed = inspect.getsource(_bake_composed_frame)

    assert "save_semantic_bake_image(" in single
    assert "scene=runtime.scene" in single
    assert "save_semantic_bake_image(" in composed
    assert "scene=runtime.scene" in composed

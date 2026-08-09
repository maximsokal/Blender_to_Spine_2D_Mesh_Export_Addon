from __future__ import annotations

from pathlib import Path

import bpy
import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_render_save import (
    _save_render_managed_image,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import TextureFormat


_HDR_RGBA = (4.0, 0.25, 0.05, 1.0)


def _new_hdr_image(name: str):
    image = bpy.data.images.new(
        name=name,
        width=1,
        height=1,
        alpha=True,
        float_buffer=True,
    )
    image.alpha_mode = "STRAIGHT"
    image.file_format = "PNG"
    image.pixels.foreach_set(_HDR_RGBA)
    image.update()
    return image


def _load_first_rgba(path: Path) -> tuple[float, float, float, float]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        values = tuple(float(image.pixels[index]) for index in range(4))
        return values[0], values[1], values[2], values[3]
    finally:
        bpy.data.images.remove(image, do_unlink=True)


def test_scene_aware_hdr_png_uses_render_view_transform_instead_of_raw_clipping(tmp_path):
    scene = bpy.context.scene
    image_settings = scene.render.image_settings
    view_settings = scene.view_settings
    original = (
        str(image_settings.file_format),
        str(image_settings.color_mode),
        str(image_settings.color_depth),
        str(view_settings.view_transform),
        str(view_settings.look),
        float(view_settings.exposure),
        float(view_settings.gamma),
    )

    local_image = None
    render_image = None
    try:
        image_settings.file_format = "PNG"
        image_settings.color_mode = "RGBA"
        image_settings.color_depth = "8"
        view_settings.view_transform = "AgX"
        view_settings.exposure = 0.0
        view_settings.gamma = 1.0

        local_path = tmp_path / "local_texture_data.png"
        render_path = tmp_path / "scene_render_appearance.png"

        local_image = _new_hdr_image("__Spine2D_Test_Local_HDR")
        local_image.filepath_raw = str(local_path)
        local_image.save()

        render_image = _new_hdr_image("__Spine2D_Test_Render_HDR")
        _save_render_managed_image(
            render_image,
            render_path,
            texture_format=TextureFormat.PNG,
            scene=scene,
        )

        assert local_path.is_file() and local_path.stat().st_size > 0
        assert render_path.is_file() and render_path.stat().st_size > 0
        assert local_path.read_bytes() != render_path.read_bytes()

        local_rgba = _load_first_rgba(local_path)
        render_rgba = _load_first_rgba(render_path)

        # The old texture-data save clips scene-linear HDR red to the PNG ceiling.
        # AgX render saving compresses that highlight into display range instead.
        assert local_rgba[0] > 0.98
        assert render_rgba[0] < local_rgba[0] - 0.01
        assert render_rgba[0] > render_rgba[1] > render_rgba[2] >= 0.0
        assert render_rgba[3] == pytest.approx(1.0, abs=1.0 / 255.0)
    finally:
        if local_image is not None:
            bpy.data.images.remove(local_image, do_unlink=True)
        if render_image is not None:
            bpy.data.images.remove(render_image, do_unlink=True)

        image_settings.file_format = original[0]
        image_settings.color_mode = original[1]
        image_settings.color_depth = original[2]
        view_settings.view_transform = original[3]
        view_settings.look = original[4]
        view_settings.exposure = original[5]
        view_settings.gamma = original[6]

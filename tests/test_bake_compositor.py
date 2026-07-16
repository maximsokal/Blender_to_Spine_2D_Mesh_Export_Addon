from array import array

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_compositor import (
    BakeCompositeError,
    BakePixelBuffer,
    compose_bake_passes,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeCompositeMode,
    BakeCompositePlan,
)


def buffer(values, *, width=1, height=1):
    return BakePixelBuffer(
        width=width,
        height=height,
        channels=4,
        pixels=array("f", values),
    )


def test_single_composition_preserves_buffer_without_copy():
    source = buffer((0.2, 0.3, 0.4, 0.5))

    result = compose_bake_passes(
        (source,),
        BakeCompositePlan(mode=BakeCompositeMode.SINGLE),
    )

    assert result is source


def test_add_rgb_max_alpha_combines_surface_and_emission():
    surface = buffer((0.7, 0.1, 0.0, 0.6))
    emission = buffer((0.1, 0.0, 0.8, 0.9))

    result = compose_bake_passes(
        (surface, emission),
        BakeCompositePlan(
            mode=BakeCompositeMode.ADD_RGB_MAX_ALPHA,
            clamp_rgb=True,
        ),
    )

    assert tuple(float(value) for value in result.pixels) == pytest.approx(
        (0.8, 0.1, 0.8, 0.9)
    )


def test_additive_composition_clamps_export_rgb():
    first = buffer((0.8, 0.8, 0.8, 0.2))
    second = buffer((0.7, 0.6, 0.5, 0.4))

    result = compose_bake_passes(
        (first, second),
        BakeCompositePlan(
            mode=BakeCompositeMode.ADD_RGB_MAX_ALPHA,
            clamp_rgb=True,
        ),
    )

    assert tuple(float(value) for value in result.pixels) == pytest.approx(
        (1.0, 1.0, 1.0, 0.4)
    )


def test_incompatible_pass_dimensions_are_rejected():
    with pytest.raises(BakeCompositeError, match="dimensions"):
        compose_bake_passes(
            (
                buffer((0.0, 0.0, 0.0, 0.0)),
                buffer((0.0,) * 8, width=2),
            ),
            BakeCompositePlan(mode=BakeCompositeMode.ADD_RGB_MAX_ALPHA),
        )


def test_single_mode_rejects_multiple_buffers():
    with pytest.raises(BakeCompositeError, match="exactly one"):
        compose_bake_passes(
            (
                buffer((0.0, 0.0, 0.0, 0.0)),
                buffer((0.0, 0.0, 0.0, 0.0)),
            ),
            BakeCompositePlan(mode=BakeCompositeMode.SINGLE),
        )

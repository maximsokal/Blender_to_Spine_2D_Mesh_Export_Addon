from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_shared_pivot import (
    supports_a1_shared_pivot,
    validate_a1_shared_pivot_world,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    A1TextureExportMode,
    BakeExecutionSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection


SIGNED_AXES = (
    A1ProjectionDirection.POSITIVE_X,
    A1ProjectionDirection.NEGATIVE_X,
    A1ProjectionDirection.POSITIVE_Y,
    A1ProjectionDirection.NEGATIVE_Y,
    A1ProjectionDirection.POSITIVE_Z,
    A1ProjectionDirection.NEGATIVE_Z,
)


@pytest.mark.parametrize("direction", SIGNED_AXES)
def test_shared_pivot_capability_accepts_only_multi_object_signed_axis_normal(
    direction: A1ProjectionDirection,
) -> None:
    assert supports_a1_shared_pivot(
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
        direction,
        2,
    )
    assert not supports_a1_shared_pivot(
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
        direction,
        1,
    )


@pytest.mark.parametrize(
    "texture_mode,direction",
    (
        (
            A1TextureExportMode.CAMERA_PROJECTION,
            A1ProjectionDirection.POSITIVE_Z,
        ),
        (
            A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
            A1ProjectionDirection.POSITIVE_Z,
        ),
        (
            A1TextureExportMode.NORMAL_UV_SEGMENTS,
            A1ProjectionDirection.ACTIVE_CAMERA,
        ),
        (
            A1TextureExportMode.NORMAL_UV_SEGMENTS,
            A1ProjectionDirection.ACTIVE_CAMERA_CAMERA_ROOT,
        ),
    ),
)
def test_shared_pivot_capability_rejects_camera_routes(
    texture_mode: A1TextureExportMode,
    direction: A1ProjectionDirection,
) -> None:
    assert not supports_a1_shared_pivot(texture_mode, direction, 3)


def test_multi_object_api_keeps_legacy_shared_pivot_default_disabled() -> None:
    settings = A1MultiObjectExportSettings(
        output_directory=Path("shared-pivot-output"),
        output_stem="assembly",
    )

    assert settings.shared_pivot_enabled is False


def test_multi_object_shared_pivot_is_standalone_only() -> None:
    standalone = A1MultiObjectExportSettings(
        output_directory=Path("shared-pivot-output"),
        output_stem="assembly",
        mode=A1MultiObjectMode.STANDALONE,
        shared_pivot_enabled=True,
    )
    assert standalone.shared_pivot_enabled is True

    with pytest.raises(ValueError, match="STANDALONE"):
        A1MultiObjectExportSettings(
            output_directory=Path("shared-pivot-output"),
            output_stem="assembly",
            mode=A1MultiObjectMode.CONNECTED,
            shared_pivot_enabled=True,
        )


def _single_settings(
    *,
    mode: A1TextureExportMode = A1TextureExportMode.NORMAL_UV_SEGMENTS,
    direction: A1ProjectionDirection = A1ProjectionDirection.POSITIVE_Z,
    shared_pivot_world=None,
) -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=128,
            texture_height=128,
            output_directory=Path("shared-pivot-output"),
        ),
        bake_execution=BakeExecutionSettings(texture_export_mode=mode),
        projection_direction=direction,
        shared_pivot_world=shared_pivot_world,
    )


@pytest.mark.parametrize("direction", SIGNED_AXES)
def test_single_object_internal_shared_pivot_accepts_every_signed_axis(
    direction: A1ProjectionDirection,
) -> None:
    settings = _single_settings(
        direction=direction,
        shared_pivot_world=(1, -2.5, 3.0),
    )

    assert settings.shared_pivot_world == (1.0, -2.5, 3.0)


@pytest.mark.parametrize(
    "mode,direction",
    (
        (
            A1TextureExportMode.NORMAL_UV_SEGMENTS,
            A1ProjectionDirection.ACTIVE_CAMERA,
        ),
        (
            A1TextureExportMode.CAMERA_PROJECTION,
            A1ProjectionDirection.POSITIVE_Z,
        ),
        (
            A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
            A1ProjectionDirection.POSITIVE_Z,
        ),
    ),
)
def test_single_object_internal_shared_pivot_rejects_unsupported_routes(
    mode: A1TextureExportMode,
    direction: A1ProjectionDirection,
) -> None:
    with pytest.raises(ValueError, match="signed-axis"):
        _single_settings(
            mode=mode,
            direction=direction,
            shared_pivot_world=(1.0, 2.0, 3.0),
        )


def test_shared_pivot_world_rejects_invalid_values() -> None:
    assert validate_a1_shared_pivot_world((1, 2.0, -0.0)) == (1.0, 2.0, 0.0)

    with pytest.raises(TypeError, match="three-value tuple"):
        validate_a1_shared_pivot_world([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match=r"\[1\]"):
        validate_a1_shared_pivot_world((1.0, True, 3.0))
    with pytest.raises(ValueError, match="finite"):
        validate_a1_shared_pivot_world((1.0, float("inf"), 3.0))

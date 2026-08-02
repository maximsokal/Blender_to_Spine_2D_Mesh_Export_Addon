import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import TextureSequenceTiming


def test_scene_fps_uses_blender_fps_base() -> None:
    timing = TextureSequenceTiming(
        scene_fps=30000,
        scene_fps_base=1001.0,
    )

    assert timing.scene_fps_value == pytest.approx(29.97002997002997)
    assert timing.resolved_fps == pytest.approx(29.97002997002997)
    assert timing.frame_duration == pytest.approx(1001.0 / 30000.0)
    assert timing.time_for_frame_index(1) == 0.033367
    assert timing.time_for_frame_index(2) == 0.066733
    assert timing.duration_for_frame_count(3) == 0.1001


def test_positive_override_replaces_scene_fps() -> None:
    timing = TextureSequenceTiming(
        scene_fps=24,
        scene_fps_base=1.0,
        override_fps=12.5,
    )

    assert timing.scene_fps_value == 24.0
    assert timing.resolved_fps == 12.5
    assert timing.frame_duration == 0.08
    assert timing.time_for_frame_index(7) == 0.56


@pytest.mark.parametrize(
    "scene_fps,scene_fps_base",
    (
        (0, 1.0),
        (30, 0.0),
        (0, 0.0),
    ),
)
def test_invalid_scene_timing_uses_thirty_fps_fallback(
    scene_fps: int,
    scene_fps_base: float,
) -> None:
    timing = TextureSequenceTiming(
        scene_fps=scene_fps,
        scene_fps_base=scene_fps_base,
    )

    assert timing.scene_fps_value == 30.0
    assert timing.resolved_fps == 30.0
    assert timing.frame_duration == pytest.approx(1.0 / 30.0)


def test_timestamps_are_derived_directly_from_frame_index() -> None:
    timing = TextureSequenceTiming(
        scene_fps=24000,
        scene_fps_base=1001.0,
    )

    assert timing.time_for_frame_index(10000) == round(
        10000.0 / timing.resolved_fps,
        6,
    )
    assert timing.duration_for_frame_count(10001) == round(
        10001.0 / timing.resolved_fps,
        6,
    )


@pytest.mark.parametrize("frame_index", (-1, True, 1.5))
def test_invalid_frame_index_is_rejected(frame_index: object) -> None:
    timing = TextureSequenceTiming()

    with pytest.raises((TypeError, ValueError)):
        timing.time_for_frame_index(frame_index)  # type: ignore[arg-type]

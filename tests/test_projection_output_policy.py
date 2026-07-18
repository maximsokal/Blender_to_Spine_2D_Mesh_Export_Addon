from array import array

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    ProjectionAlphaRepresentation,
    ProjectionDynamicRange,
    ProjectionOutputPolicy,
    ProjectionOutputPolicyError,
    ProjectionToneMapping,
    TextureFormat,
    convert_rgba_alpha_representation,
    resolve_projection_output_policy,
)


def test_auto_policy_resolves_png_to_sdr_scene_view_straight():
    resolved = resolve_projection_output_policy(
        ProjectionOutputPolicy(),
        TextureFormat.PNG,
    )

    assert resolved.dynamic_range is ProjectionDynamicRange.DISPLAY_REFERRED_SDR
    assert resolved.tone_mapping is ProjectionToneMapping.SCENE_VIEW_TRANSFORM
    assert resolved.alpha_representation is ProjectionAlphaRepresentation.STRAIGHT
    assert resolved.color_depth == "8"
    assert not resolved.float_buffer
    assert resolved.blender_alpha_mode == "STRAIGHT"


def test_auto_policy_resolves_webp_to_sdr_straight():
    resolved = resolve_projection_output_policy(
        ProjectionOutputPolicy(),
        TextureFormat.WEBP,
    )

    assert resolved.dynamic_range is ProjectionDynamicRange.DISPLAY_REFERRED_SDR
    assert resolved.alpha_representation is ProjectionAlphaRepresentation.STRAIGHT


def test_auto_policy_resolves_exr_to_scene_linear_hdr_premultiplied():
    resolved = resolve_projection_output_policy(
        ProjectionOutputPolicy(),
        TextureFormat.OPEN_EXR,
    )

    assert resolved.dynamic_range is ProjectionDynamicRange.SCENE_LINEAR_HDR
    assert resolved.tone_mapping is ProjectionToneMapping.NONE
    assert resolved.alpha_representation is ProjectionAlphaRepresentation.PREMULTIPLIED
    assert resolved.color_depth == "32"
    assert resolved.float_buffer
    assert resolved.blender_alpha_mode == "PREMUL"


@pytest.mark.parametrize(
    ("policy", "texture_format", "message"),
    (
        (
            ProjectionOutputPolicy(
                dynamic_range=ProjectionDynamicRange.SCENE_LINEAR_HDR,
            ),
            TextureFormat.PNG,
            "requires TextureFormat.OPEN_EXR",
        ),
        (
            ProjectionOutputPolicy(
                dynamic_range=ProjectionDynamicRange.SCENE_LINEAR_HDR,
                tone_mapping=ProjectionToneMapping.SCENE_VIEW_TRANSFORM,
            ),
            TextureFormat.OPEN_EXR,
            "requires tone_mapping=NONE",
        ),
        (
            ProjectionOutputPolicy(
                dynamic_range=ProjectionDynamicRange.DISPLAY_REFERRED_SDR,
            ),
            TextureFormat.OPEN_EXR,
            "cannot be written as OPEN_EXR",
        ),
        (
            ProjectionOutputPolicy(
                dynamic_range=ProjectionDynamicRange.DISPLAY_REFERRED_SDR,
                tone_mapping=ProjectionToneMapping.NONE,
            ),
            TextureFormat.PNG,
            "requires SCENE_VIEW_TRANSFORM",
        ),
    ),
)
def test_incompatible_output_combinations_fail_before_render(
    policy,
    texture_format,
    message,
):
    with pytest.raises(ProjectionOutputPolicyError, match=message):
        resolve_projection_output_policy(policy, texture_format)


def test_b4_rejects_jpeg_even_with_auto_policy():
    with pytest.raises(ProjectionOutputPolicyError, match="alpha-capable"):
        resolve_projection_output_policy(
            ProjectionOutputPolicy(),
            TextureFormat.JPEG,
        )


def test_straight_to_premultiplied_preserves_hdr_values_without_clamping():
    result = convert_rgba_alpha_representation(
        array("f", (4.0, 2.0, 1.0, 0.25, 8.0, 3.0, 2.0, 1.0)),
        source_alpha_mode="STRAIGHT",
        target=ProjectionAlphaRepresentation.PREMULTIPLIED,
    )

    assert tuple(result) == pytest.approx((1.0, 0.5, 0.25, 0.25, 8.0, 3.0, 2.0, 1.0))
    assert max(result[0::4]) > 1.0


def test_premultiplied_to_straight_restores_rgb_and_zero_alpha_is_black():
    result = convert_rgba_alpha_representation(
        (1.0, 0.5, 0.25, 0.25, 5.0, 4.0, 3.0, 0.0),
        source_alpha_mode="PREMUL",
        target=ProjectionAlphaRepresentation.STRAIGHT,
    )

    assert tuple(result) == pytest.approx((4.0, 2.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0))


def test_same_alpha_representation_keeps_hdr_buffer_unchanged():
    source = (3.0, 2.0, 1.0, 0.5)

    result = convert_rgba_alpha_representation(
        source,
        source_alpha_mode="STRAIGHT",
        target=ProjectionAlphaRepresentation.STRAIGHT,
    )

    assert tuple(result) == pytest.approx(source)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_alpha_conversion_rejects_non_finite_rgba(value):
    with pytest.raises(ProjectionOutputPolicyError, match="non-finite"):
        convert_rgba_alpha_representation(
            (value, 0.0, 0.0, 1.0),
            source_alpha_mode="STRAIGHT",
            target=ProjectionAlphaRepresentation.PREMULTIPLIED,
        )


def test_alpha_conversion_rejects_unknown_blender_mode():
    with pytest.raises(ProjectionOutputPolicyError, match="unsupported Blender image alpha"):
        convert_rgba_alpha_representation(
            (1.0, 1.0, 1.0, 1.0),
            source_alpha_mode="MYSTERY",
            target=ProjectionAlphaRepresentation.STRAIGHT,
        )

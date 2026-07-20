from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    ConnectedGroupSettings,
    UniformScaleMode,
)


def test_connected_settings_keep_historical_positional_animation_separator():
    settings = ConnectedGroupSettings(
        100,
        200,
        "all_objects",
        None,
        1e-4,
        UniformScaleMode.MAXIMUM,
        "::",
    )

    assert settings.animation_separator == "::"
    assert settings.namespace_animations is True


def test_connected_namespace_flag_is_an_additive_keyword_contract():
    settings = ConnectedGroupSettings(
        100,
        200,
        namespace_animations=False,
    )

    assert settings.animation_separator == "/"
    assert settings.namespace_animations is False

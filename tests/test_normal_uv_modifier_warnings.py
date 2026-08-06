"""Regression coverage for modifiers ignored by Normal / UV Segments."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.normal_uv_modifier_warnings import (
    IgnoredNormalUvModifier,
    collect_normal_uv_ignored_modifiers,
    group_ignored_modifiers_by_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import A1TextureExportMode


def _modifier(
    name: str,
    modifier_type: str,
    *,
    show_viewport: bool = True,
    show_render: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        type=modifier_type,
        show_viewport=show_viewport,
        show_render=show_render,
    )


def _mesh(name: str, *modifiers: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        name_full=name,
        type="MESH",
        modifiers=modifiers,
    )


def test_normal_uv_reports_enabled_bevel_and_preserves_stack_order() -> None:
    source = _mesh(
        "Game Gold Coin",
        _modifier("Bevel", "BEVEL"),
        _modifier("Subdivision", "SUBSURF", show_render=False),
    )

    result = collect_normal_uv_ignored_modifiers(
        (source,),
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
    )

    assert result == (
        IgnoredNormalUvModifier(
            object_name="Game Gold Coin",
            modifier_name="Bevel",
            modifier_type="BEVEL",
            show_viewport=True,
            show_render=True,
        ),
        IgnoredNormalUvModifier(
            object_name="Game Gold Coin",
            modifier_name="Subdivision",
            modifier_type="SUBSURF",
            show_viewport=True,
            show_render=False,
        ),
    )


def test_modifiers_disabled_for_viewport_and_render_do_not_warn() -> None:
    source = _mesh(
        "Coin",
        _modifier(
            "Disabled Bevel",
            "BEVEL",
            show_viewport=False,
            show_render=False,
        ),
    )

    assert collect_normal_uv_ignored_modifiers(
        (source,),
        A1TextureExportMode.NORMAL_UV_SEGMENTS,
    ) == ()


def test_camera_routes_do_not_claim_modifiers_are_ignored_by_normal() -> None:
    source = _mesh("Coin", _modifier("Bevel", "BEVEL"))

    assert collect_normal_uv_ignored_modifiers(
        (source,),
        A1TextureExportMode.CAMERA_PROJECTION,
    ) == ()
    assert collect_normal_uv_ignored_modifiers(
        (source,),
        A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
    ) == ()


def test_non_mesh_objects_are_ignored_and_groups_remain_deterministic() -> None:
    first = _mesh("Coin A", _modifier("Bevel", "BEVEL"))
    ignored = SimpleNamespace(
        name="Camera",
        name_full="Camera",
        type="CAMERA",
        modifiers=(_modifier("Unexpected", "BEVEL"),),
    )
    second = _mesh(
        "Coin B",
        _modifier("Mirror", "MIRROR"),
        _modifier("Solidify", "SOLIDIFY"),
    )

    descriptors = collect_normal_uv_ignored_modifiers(
        (first, ignored, second),
        A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
    )
    grouped = group_ignored_modifiers_by_object(descriptors)

    assert tuple(name for name, _items in grouped) == ("Coin A", "Coin B")
    assert tuple(item.modifier_name for item in grouped[1][1]) == (
        "Mirror",
        "Solidify",
    )


def test_invalid_collections_fail_loudly() -> None:
    with pytest.raises(TypeError, match="objects must be"):
        collect_normal_uv_ignored_modifiers(
            "not-an-object-collection",
            A1TextureExportMode.NORMAL_UV_SEGMENTS,
        )

    with pytest.raises(TypeError, match="IgnoredNormalUvModifier"):
        group_ignored_modifiers_by_object((object(),))

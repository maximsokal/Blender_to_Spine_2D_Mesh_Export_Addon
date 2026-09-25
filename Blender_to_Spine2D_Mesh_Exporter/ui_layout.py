# pylint: disable=import-error
"""Advisory Normal/UV modifier-mismatch warning shared by the canonical exporter panel.

An earlier implementation temporarily unregistered the main panel and replaced it with
another class during add-on registration. Blender Extensions review explicitly rejected
that lifecycle. A later revision kept the main panel stable by moving Rig, Generated
Materials, and Analysis into separate ordinary Blender child panels instead. That traded
one problem for another: Blender always renders a child panel's content after everything
the parent panel's own ``draw()`` produces, and it never gives child panels the same
boxed, triangle-foldout look as content drawn directly in ``draw()`` — so those sections
rendered as a flat, out-of-order tail below the Export button.

The main panel in :mod:`ui` now draws every section itself, including the modifier
warning below, through the one shared ``_draw_foldout`` box style. This module only
keeps the warning-drawing helper so ``ui`` does not have to duplicate it.
"""

from __future__ import annotations

from typing import Any

import bpy

from .blender_adapter.normal_uv_modifier_warnings import (
    collect_normal_uv_ignored_modifiers,
    group_ignored_modifiers_by_object,
)
from .domain.baking import A1TextureExportMode


_MAX_VISIBLE_MODIFIERS_PER_OBJECT = 8


def _texture_mode(scene: bpy.types.Scene) -> A1TextureExportMode:
    raw = str(
        getattr(
            scene,
            "spine2d_texture_export_mode",
            A1TextureExportMode.NORMAL_UV_SEGMENTS.value,
        )
    ).strip().upper()
    try:
        return A1TextureExportMode(raw)
    except ValueError:
        return A1TextureExportMode.NORMAL_UV_SEGMENTS


def _analysis_mesh_objects(context: bpy.types.Context) -> tuple[bpy.types.Object, ...]:
    """Resolve the Mesh request represented by the current single/multi UI state."""

    selected = tuple(
        candidate
        for candidate in getattr(context, "selected_objects", ())
        if getattr(candidate, "type", None) == "MESH"
        and getattr(candidate, "data", None) is not None
    )
    if len(selected) > 1:
        return selected

    active = getattr(context, "active_object", None)
    if (
        active is not None
        and getattr(active, "type", None) == "MESH"
        and getattr(active, "data", None) is not None
    ):
        return (active,)
    return selected


def _draw_modifier_analysis_warning(
    layout: bpy.types.UILayout,
    context: bpy.types.Context,
) -> None:
    """Warn when Normal / UV export cannot reproduce evaluated modifier geometry.

    The warning is advisory only. It never disables Analyze or Export and therefore
    remains compatible with the explicit, non-blocking readiness policy.
    """

    scene = getattr(context, "scene", None)
    if scene is None:
        return

    descriptors = collect_normal_uv_ignored_modifiers(
        _analysis_mesh_objects(context),
        _texture_mode(scene),
    )
    if not descriptors:
        return

    box = layout.box()
    box.alert = True
    box.label(
        text="Normal / UV Segments ignores active modifiers",
        icon="ERROR",
    )
    box.label(text="Viewport and Spine geometry can look different.")
    box.label(text="Apply or convert modifiers before export.", icon="INFO")

    for object_name, modifiers in group_ignored_modifiers_by_object(descriptors):
        box.separator()
        box.label(text=object_name, icon="MESH_DATA")
        visible = modifiers[:_MAX_VISIBLE_MODIFIERS_PER_OBJECT]
        for modifier in visible:
            states: list[str] = []
            if modifier.show_viewport:
                states.append("viewport")
            if modifier.show_render:
                states.append("render")
            box.label(
                text=(
                    f"{modifier.modifier_name} ({modifier.modifier_type}) — "
                    + "/".join(states)
                ),
                icon="MODIFIER",
            )
        hidden_count = len(modifiers) - len(visible)
        if hidden_count > 0:
            box.label(text=f"... and {hidden_count} more modifier(s)")


# Compatibility symbols only. Every section now draws inside the canonical main panel,
# so this module owns no panel classes and no RNA state of its own.
CLASSES: tuple[Any, ...] = ()
RNA_PROPERTIES: tuple[Any, ...] = ()


def register() -> None:
    """Register child panels in declaration order (none remain; kept for symmetry)."""

    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    """Unregister child panels in reverse declaration order."""

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


__all__ = [
    "CLASSES",
    "RNA_PROPERTIES",
    "register",
    "unregister",
]

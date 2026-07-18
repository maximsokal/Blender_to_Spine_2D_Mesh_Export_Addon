"""Derive immutable connected and standalone subgroup settings for mixed export."""

from __future__ import annotations

from dataclasses import replace

from ..application import A1MultiObjectExportSettings, A1MultiObjectMode


def build_connected_subgroup_settings(
    settings: A1MultiObjectExportSettings,
    anchor_component_id: str,
) -> A1MultiObjectExportSettings:
    """Return the connected subgroup settings used by mixed preparation and output."""

    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode is not A1MultiObjectMode.MIXED:
        raise ValueError("connected subgroup settings require MIXED parent settings")
    if not isinstance(anchor_component_id, str) or not anchor_component_id.strip():
        raise ValueError("anchor_component_id must be a non-empty string")
    return replace(
        settings,
        mode=A1MultiObjectMode.CONNECTED,
        output_stem=f"{settings.resolved_output_stem}__connected",
        anchor_component_id=anchor_component_id,
    )


def build_standalone_subgroup_settings(
    settings: A1MultiObjectExportSettings,
) -> A1MultiObjectExportSettings:
    """Return the standalone subgroup settings used by mixed document composition."""

    if not isinstance(settings, A1MultiObjectExportSettings):
        raise TypeError("settings must be A1MultiObjectExportSettings")
    if settings.mode is not A1MultiObjectMode.MIXED:
        raise ValueError("standalone subgroup settings require MIXED parent settings")
    return replace(
        settings,
        mode=A1MultiObjectMode.STANDALONE,
        output_stem=f"{settings.resolved_output_stem}__standalone",
        anchor_component_id=None,
    )


__all__ = ["build_connected_subgroup_settings", "build_standalone_subgroup_settings"]

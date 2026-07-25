"""Format-dependent Blender image channel policy for semantic bake outputs."""

from __future__ import annotations

from .model import TextureFormat


def resolve_texture_color_mode(
    texture_format: TextureFormat,
    requested: str,
) -> str:
    """Return a Blender-compatible channel mode for one texture format.

    Blender narrows ``ImageFormatSettings.color_mode`` after switching to JPEG,
    because JPEG has no alpha channel. The exporter-wide RGBA default therefore
    resolves to RGB for JPEG while all alpha-capable formats preserve RGBA.
    """

    if not isinstance(texture_format, TextureFormat):
        raise TypeError("texture_format must be TextureFormat")
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("requested color mode must be a non-empty string")

    resolved = requested.strip().upper()
    if resolved not in {"BW", "RGB", "RGBA"}:
        raise ValueError("requested color mode must be BW, RGB, or RGBA")
    if texture_format is TextureFormat.JPEG and resolved == "RGBA":
        return "RGB"
    return resolved


__all__ = ["resolve_texture_color_mode"]

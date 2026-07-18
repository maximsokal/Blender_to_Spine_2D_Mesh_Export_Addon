"""Compatibility facade for Blender UI RNA capture helpers.

Production modules import the physical selection and Scene-capture owners directly. These
private re-exports remain for focused tests and existing external callers.
"""

from .a1_ui_scene_capture import (
    _SceneExportProfile,
    _capture_scene_profile,
    _projection_alpha_threshold,
    _resolve_geometry_settings,
    _resolve_images_relative_path,
    _resolve_output_directory,
    _texture_size,
)
from .a1_ui_selection import (
    _ObjectExportProfile,
    _active_mesh,
    _capture_object_profile,
    _connect_enabled,
    _object_name,
    _ordered_selected_meshes,
    _rna_identity,
)


__all__ = [
    "_ObjectExportProfile",
    "_SceneExportProfile",
    "_active_mesh",
    "_capture_object_profile",
    "_capture_scene_profile",
    "_connect_enabled",
    "_object_name",
    "_ordered_selected_meshes",
    "_projection_alpha_threshold",
    "_resolve_geometry_settings",
    "_resolve_images_relative_path",
    "_resolve_output_directory",
    "_rna_identity",
    "_texture_size",
]

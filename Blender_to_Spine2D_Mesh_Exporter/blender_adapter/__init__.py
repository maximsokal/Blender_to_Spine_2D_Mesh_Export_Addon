"""Blender API adapters for the rewritten exporter."""

from .mesh_reader import MeshReadError, read_source_mesh_snapshot

__all__ = ["MeshReadError", "read_source_mesh_snapshot"]

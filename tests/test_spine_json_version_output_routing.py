"""Architecture checks for every production A1 Spine JSON output route."""

from __future__ import annotations

import ast
from pathlib import Path


ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _tree(filename: str) -> ast.Module:
    return ast.parse((ADAPTER / filename).read_text(encoding="utf-8"), filename=filename)


def _called_names(filename: str) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(_tree(filename))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _imported_names(filename: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_tree(filename)):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def test_all_a1_output_routes_use_the_version_codec_facade() -> None:
    for filename in (
        "a1_single_object_export.py",
        "a1_multi_object_output.py",
        "a1_mixed_object_output.py",
    ):
        calls = _called_names(filename)
        imports = _imported_names(filename)

        assert "serialize_spine_document" in calls
        assert "serialize_spine_document" in imports
        assert "SpineSerializer" not in calls
        assert "SpineSerializer" not in imports


def test_multi_and_mixed_resolve_target_from_immutable_sources() -> None:
    for filename in ("a1_multi_object_output.py", "a1_mixed_object_output.py"):
        assert "resolve_a1_sources_spine_target" in _called_names(filename)
        assert "resolve_a1_sources_spine_target" in _imported_names(filename)


def test_connected_multi_output_keeps_specialized_serialization_validation() -> None:
    source = (ADAPTER / "a1_multi_object_output.py").read_text(encoding="utf-8")

    assert "ConnectedGroupSerializationValidator" in source
    assert "_serialization_validator_for_composition" in source
    assert "validator=_serialization_validator_for_composition(composition)" in source

import ast
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    validate_a1_realized_output_namespace,
)


ADAPTER_ROOT = (
    Path(__file__).parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
)


def _call_lines(filename: str, function_name: str) -> list[int]:
    tree = ast.parse(
        (ADAPTER_ROOT / filename).read_text(encoding="utf-8"),
        filename=filename,
    )
    result: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == function_name:
            result.append(node.lineno)
        elif isinstance(target, ast.Attribute) and target.attr == function_name:
            result.append(node.lineno)
    return result


def test_realized_namespace_detects_grouped_case_collision(tmp_path):
    with pytest.raises(ValueError, match="collision"):
        validate_a1_realized_output_namespace(
            output_root=tmp_path,
            json_path=tmp_path / "combined.json",
            texture_paths=(tmp_path / "images" / "Hero_Baked.png",),
            additional_texture_paths=(
                tmp_path / "IMAGES" / "hero_baked.PNG",
            ),
        )


def test_realized_namespace_detects_grouped_json_collision(tmp_path):
    with pytest.raises(ValueError, match="collision"):
        validate_a1_realized_output_namespace(
            output_root=tmp_path,
            json_path=tmp_path / "Grouped.JSON",
            texture_paths=(tmp_path / "images" / "Hero_Baked.png",),
            additional_texture_paths=(tmp_path / "grouped.json",),
        )


def test_realized_namespace_accepts_distinct_grouped_paths(tmp_path):
    result = validate_a1_realized_output_namespace(
        output_root=tmp_path,
        json_path=tmp_path / "combined.json",
        texture_paths=(tmp_path / "images" / "Hero_Baked.png",),
        additional_texture_paths=(
            tmp_path / "images" / "combined_grouped_camera_Baked.png",
        ),
    )

    assert result == (
        (tmp_path / "combined.json").resolve(strict=False),
        (tmp_path / "images" / "Hero_Baked.png").resolve(strict=False),
        (
            tmp_path / "images" / "combined_grouped_camera_Baked.png"
        ).resolve(strict=False),
    )


@pytest.mark.parametrize(
    "filename",
    (
        "a1_multi_object_output.py",
        "a1_mixed_object_output.py",
    ),
)
def test_grouped_namespace_is_resolved_before_transaction_and_texture_staging(
    filename,
):
    grouped_resolution = _call_lines(
        filename,
        "resolve_grouped_camera_projection_request",
    )
    namespace_validation = _call_lines(
        filename,
        "validate_a1_realized_output_namespace",
    )
    transaction = _call_lines(filename, "atomic_file_transaction")
    texture_staging = _call_lines(filename, "stage_and_finalize_a1_objects")

    assert grouped_resolution
    assert namespace_validation
    assert transaction
    assert texture_staging
    assert max(grouped_resolution) < min(transaction)
    assert max(namespace_validation) < min(transaction)
    assert max(namespace_validation) < min(texture_staging)

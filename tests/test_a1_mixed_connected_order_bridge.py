from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_mixed_composition import (
    _compose_outer_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    SpineCompositionError,
    SpineDocument,
    TransformConstraint,
)


def _document(
    prefix: str,
    orders: tuple[int, ...],
    *,
    missing_target_index: int | None = None,
) -> SpineDocument:
    bone_name = f"{prefix}_bone"
    target_name = f"{prefix}_target"
    constraints = tuple(
        TransformConstraint(
            name=f"{prefix}_constraint_{index}",
            order=order,
            bones=(bone_name,),
            target=(
                f"{prefix}_missing_target"
                if missing_target_index == index
                else target_name
            ),
        )
        for index, order in enumerate(orders)
    )
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(
            Bone("root"),
            Bone(bone_name, parent="root"),
            Bone(target_name, parent="root"),
        ),
        slots=(),
        skins=(),
        transform=constraints,
    )


def _settings(output_directory: Path) -> A1MultiObjectExportSettings:
    return A1MultiObjectExportSettings(
        output_directory=output_directory,
        output_stem="MixedConstraintOrderBridge",
        mode=A1MultiObjectMode.MIXED,
    )


def test_mixed_outer_composition_rebases_connected_order_ties(
    tmp_path: Path,
) -> None:
    connected = _document("connected", (6, 6))
    standalone = _document("standalone", (0,))

    composition = _compose_outer_document(
        connected,
        standalone,
        _settings(tmp_path),
    )

    assert tuple(item.order for item in connected.transform) == (6, 6)
    assert tuple(item.order for item in composition.document.transform) == (0, 1, 2)
    assert composition.components[0].component_id == "connected_group"
    assert composition.components[0].document is connected
    assert composition.components[1].component_id == "standalone_group"
    assert composition.components[1].document is standalone

    assignments = {
        item.constraint_name: (
            item.component_id,
            item.original_order,
            item.global_order,
        )
        for item in composition.constraint_orders
    }
    assert assignments == {
        "connected_constraint_0": ("connected_group", 6, 0),
        "connected_constraint_1": ("connected_group", 6, 1),
        "standalone_constraint_0": ("standalone_group", 0, 2),
    }


def test_mixed_order_bridge_rejects_non_order_validation_errors(
    tmp_path: Path,
) -> None:
    connected = _document(
        "connected",
        (6, 6),
        missing_target_index=1,
    )
    standalone = _document("standalone", (0,))

    with pytest.raises(SpineCompositionError, match="MISSING_CONSTRAINT_TARGET"):
        _compose_outer_document(
            connected,
            standalone,
            _settings(tmp_path),
        )


def test_mixed_order_bridge_preserves_connected_component_first(
    tmp_path: Path,
) -> None:
    connected = _document("connected", (10, 2, 2))
    standalone = _document("standalone", (0, 1))

    composition = _compose_outer_document(
        connected,
        standalone,
        _settings(tmp_path),
    )

    connected_assignments = tuple(
        item
        for item in composition.constraint_orders
        if item.component_id == "connected_group"
    )
    standalone_assignments = tuple(
        item
        for item in composition.constraint_orders
        if item.component_id == "standalone_group"
    )

    assert tuple(item.constraint_name for item in connected_assignments) == (
        "connected_constraint_1",
        "connected_constraint_2",
        "connected_constraint_0",
    )
    assert tuple(item.original_order for item in connected_assignments) == (2, 2, 10)
    assert tuple(item.global_order for item in connected_assignments) == (0, 1, 2)
    assert tuple(item.global_order for item in standalone_assignments) == (3, 4)

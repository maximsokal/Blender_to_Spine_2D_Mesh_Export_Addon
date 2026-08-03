"""Runtime contracts for frame-evaluated Normal UV sequence transforms."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_error import (
    SceneBakeAnalysisError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_bake_runtime import (
    synchronize_runtime_object_transform,
    validate_runtime_object_transform,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import ObjectBakeContext


_IDENTITY_ROWS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_MOVED_ROWS = (
    (0.8, -0.2, 0.0, 1.25),
    (0.3, 1.1, 0.0, -0.75),
    (0.0, 0.0, 0.9, 0.5),
    (0.0, 0.0, 0.0, 1.0),
)


def _flatten(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


class FakeMatrix:
    def __init__(self, rows: tuple[tuple[float, ...], ...]):
        self._rows = tuple(tuple(float(value) for value in row) for row in rows)

    def __getitem__(self, index: int) -> tuple[float, ...]:
        return self._rows[index]

    def copy(self) -> "FakeMatrix":
        return FakeMatrix(self._rows)

    def as_rows(self) -> tuple[tuple[float, ...], ...]:
        return self._rows


class RejectingTarget:
    def __init__(self, matrix: FakeMatrix):
        self._matrix_world = matrix

    @property
    def matrix_world(self) -> FakeMatrix:
        return self._matrix_world

    @matrix_world.setter
    def matrix_world(self, value: FakeMatrix) -> None:
        del value
        # Simulate an RNA assignment that silently fails to apply the requested matrix.


@dataclass
class FakeCollection:
    name: str


class FakeObject:
    def __init__(self, name: str, matrix: FakeMatrix):
        self.name = name
        self.name_full = name
        self.type = "MESH"
        self.matrix_world = matrix
        self.users_collection = (FakeCollection("Collection"),)
        self.hide_render = False
        self.visible_camera = True
        self.visible_shadow = True
        self.animation_data = None
        self.data = type("FakeData", (), {"animation_data": None})()


class FakeViewLayer:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


class FakeContext:
    def __init__(self) -> None:
        self.view_layer = FakeViewLayer()


def _expected(name: str = "Crystal") -> ObjectBakeContext:
    return ObjectBakeContext(
        source_object_id=name,
        object_type="MESH",
        world_matrix=_flatten(_IDENTITY_ROWS),
        collection_names=("Collection",),
        hide_render=False,
        visible_camera=True,
        visible_shadow=True,
        animated=False,
    )


def test_static_runtime_validation_still_rejects_changed_matrix() -> None:
    source = FakeObject("Crystal", FakeMatrix(_MOVED_ROWS))

    with pytest.raises(SceneBakeAnalysisError, match="changed after planning"):
        validate_runtime_object_transform(
            source,
            _expected(),
            timeline_frame=None,
        )


def test_sequence_runtime_validation_allows_frame_evaluated_matrix() -> None:
    source = FakeObject("Crystal", FakeMatrix(_MOVED_ROWS))

    current = validate_runtime_object_transform(
        source,
        _expected(),
        timeline_frame=2,
        allow_sequence_transform=True,
    )

    assert current is not None
    assert current.world_matrix == _flatten(_MOVED_ROWS)


def test_allow_flag_does_not_relax_non_sequence_execution() -> None:
    source = FakeObject("Crystal", FakeMatrix(_MOVED_ROWS))

    with pytest.raises(SceneBakeAnalysisError, match="changed after planning"):
        validate_runtime_object_transform(
            source,
            _expected(),
            timeline_frame=None,
            allow_sequence_transform=True,
        )


def test_sequence_sync_copies_source_matrix_and_updates_view_layer() -> None:
    source_matrix = FakeMatrix(_MOVED_ROWS)
    source = FakeObject("Crystal", source_matrix)
    target = FakeObject("TemporaryTarget", FakeMatrix(_IDENTITY_ROWS))
    context = FakeContext()

    current = synchronize_runtime_object_transform(
        source,
        target,
        _expected(),
        context=context,
        timeline_frame=3,
    )

    assert current is not None
    assert target.matrix_world is not source_matrix
    assert target.matrix_world.as_rows() == _MOVED_ROWS
    assert context.view_layer.update_count == 1


def test_sequence_sync_rejects_source_identity_change() -> None:
    source = FakeObject("OtherObject", FakeMatrix(_MOVED_ROWS))
    target = FakeObject("TemporaryTarget", FakeMatrix(_IDENTITY_ROWS))

    with pytest.raises(SceneBakeAnalysisError, match="identity changed"):
        synchronize_runtime_object_transform(
            source,
            target,
            _expected(),
            context=FakeContext(),
            timeline_frame=1,
        )


def test_sequence_sync_verifies_matrix_after_assignment() -> None:
    source = FakeObject("Crystal", FakeMatrix(_MOVED_ROWS))
    target = RejectingTarget(FakeMatrix(_IDENTITY_ROWS))

    with pytest.raises(SceneBakeAnalysisError, match="differs from"):
        synchronize_runtime_object_transform(
            source,
            target,
            _expected(),
            context=FakeContext(),
            timeline_frame=1,
        )

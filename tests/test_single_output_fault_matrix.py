from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_single_object_export
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_single_object_export import (
    export_a1_single_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget
from Blender_to_Spine2D_Mesh_Exporter.infrastructure import AtomicFileCommitError


def _serialize(document, target, *, indent, validator=None):
    assert target is SpineJsonTarget.SPINE_4_2
    assert validator is None
    return f'{{"document": "{document}", "indent": {indent}}}'


def _prepared(tmp_path: Path):
    return SimpleNamespace(
        source_object=object(),
        bake_target_snapshot=object(),
        bake_plan=object(),
        object_id="FaultMatrix",
        output_paths=SimpleNamespace(
            json_path=tmp_path / "FaultMatrix.json",
        ),
        statistics={"prepared": 1},
        warnings=(),
    )


def _settings():
    return SimpleNamespace(
        bake_execution=object(),
        json_indent=2,
        export=SimpleNamespace(spine_target=SpineJsonTarget.SPINE_4_2),
    )


def _install_common_pipeline(monkeypatch, tmp_path: Path, *, fail_stage=None):
    prepared = _prepared(tmp_path)
    texture_path = tmp_path / "FaultMatrix.png"
    monkeypatch.setattr(
        a1_single_object_export,
        "prepare_a1_object",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        a1_single_object_export,
        "validate_staged_normal_bake_coverage",
        lambda *_args, **_kwargs: (),
    )

    def stage_outputs(
        _source,
        _snapshot,
        _plan,
        transaction,
        _execution,
        **_kwargs,
    ):
        reservation = transaction.reserve(texture_path)
        reservation.staged_path.write_bytes(b"deterministic-texture")
        if fail_stage == "texture":
            raise RuntimeError("forced texture staging failure")
        return SimpleNamespace(
            reservations=(reservation,),
            projection_layout=None,
        )

    monkeypatch.setattr(
        a1_single_object_export,
        "stage_texture_plan_outputs",
        stage_outputs,
    )

    finalized = SimpleNamespace(
        object_id=prepared.object_id,
        document="deterministic-document",
        statistics={"prepared": 1, "finalized": 1},
        warnings=(),
    )

    def finalize(_prepared, _layout):
        if fail_stage == "finalize":
            raise RuntimeError("forced finalization failure")
        return finalized

    monkeypatch.setattr(
        a1_single_object_export,
        "finalize_prepared_camera_projection",
        finalize,
    )
    monkeypatch.setattr(
        a1_single_object_export,
        "serialize_spine_document",
        _serialize,
    )
    return prepared.output_paths.json_path, texture_path


def _work_files(tmp_path: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in tmp_path.rglob("*")
            if path.is_file()
            and (".spine2d-stage-" in path.name or ".spine2d-backup-" in path.name)
        )
    )


@pytest.mark.parametrize(
    ("failure_stage", "message"),
    (
        ("texture", "forced texture staging failure"),
        ("finalize", "forced finalization failure"),
        ("serialize", "forced serialization failure"),
        ("write", "forced staged write failure"),
        ("commit", "forced commit failure"),
    ),
)
def test_single_output_fault_matrix_preserves_existing_outputs_and_cleans_work_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    message: str,
):
    json_path, texture_path = _install_common_pipeline(
        monkeypatch,
        tmp_path,
        fail_stage=failure_stage if failure_stage in {"texture", "finalize"} else None,
    )
    json_path.write_bytes(b"previous-json")
    texture_path.write_bytes(b"previous-texture")

    if failure_stage == "serialize":
        def fail_serialize(_document, target, *, indent, validator=None):
            assert target is SpineJsonTarget.SPINE_4_2
            assert indent == 2
            assert validator is None
            raise RuntimeError(message)

        monkeypatch.setattr(
            a1_single_object_export,
            "serialize_spine_document",
            fail_serialize,
        )
    elif failure_stage == "write":
        def fail_write(path, _text, *, ensure_trailing_newline):
            assert ensure_trailing_newline
            Path(path).write_bytes(b"partial-staged-json")
            raise RuntimeError(message)

        monkeypatch.setattr(a1_single_object_export, "write_staged_utf8_text", fail_write)
    elif failure_stage == "commit":
        real_factory = a1_single_object_export.atomic_file_transaction

        class FailingTransactionContext:
            def __init__(self, operation_name):
                self._owner = real_factory(operation_name=operation_name)
                self._transaction = None

            def __enter__(self):
                self._transaction = self._owner.__enter__()
                transaction = self._transaction

                def fail_commit():
                    raise AtomicFileCommitError(message)

                transaction.commit = fail_commit
                return transaction

            def __exit__(self, exc_type, exc, traceback):
                return self._owner.__exit__(exc_type, exc, traceback)

        monkeypatch.setattr(
            a1_single_object_export,
            "atomic_file_transaction",
            lambda *, operation_name: FailingTransactionContext(operation_name),
        )

    result = export_a1_single_object(object(), _settings())

    assert not result.success
    assert result.issues
    assert message in result.issues[-1].message
    assert json_path.read_bytes() == b"previous-json"
    assert texture_path.read_bytes() == b"previous-texture"
    assert _work_files(tmp_path) == ()


def test_repeated_single_output_is_byte_deterministic_and_leaves_no_work_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    json_path, texture_path = _install_common_pipeline(monkeypatch, tmp_path)
    observed = []

    for _iteration in range(8):
        result = export_a1_single_object(object(), _settings())
        assert result.success
        assert result.output_files == (json_path.resolve(), texture_path.resolve())
        observed.append((json_path.read_bytes(), texture_path.read_bytes()))
        assert _work_files(tmp_path) == ()

    assert len(set(observed)) == 1

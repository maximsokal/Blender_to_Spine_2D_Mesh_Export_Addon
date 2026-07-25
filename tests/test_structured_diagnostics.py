from __future__ import annotations
import ast
from pathlib import Path
import pytest
from Blender_to_Spine2D_Mesh_Exporter.infrastructure import AtomicFileCommitError, AtomicFileTransaction
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.export_events import ExportEventDispatcher, ExportEventKind
ROOT=Path(__file__).resolve().parents[1]; PRODUCTION=ROOT/"Blender_to_Spine2D_Mesh_Exporter"
def test_atomic_failure_emits_complete_diagnostics_and_preserves_cause(tmp_path):
    events=[]; dispatcher=ExportEventDispatcher(); dispatcher.subscribe(events.append); transaction=AtomicFileTransaction(operation_name="diagnostic-test",dispatcher=dispatcher); reservation=transaction.reserve(tmp_path/"result.json"); assert not reservation.staged_path.exists()
    with pytest.raises(AtomicFileCommitError) as captured: transaction.commit()
    assert captured.value.__cause__ is not None; failures=[event for event in events if event.kind is ExportEventKind.TRANSACTION_FAILED]; assert len(failures)==1; event=failures[0]; assert event.operation_id.startswith("diagnostic-test:"); assert event.context["stage"]=="commit"; assert event.context["exception_type"]; assert event.context["rollback_result"]=="SUCCEEDED"; assert event.context["rollback_failure_count"]==0; assert tuple(event.context["output_paths"])==(str((tmp_path/"result.json").resolve()),); assert isinstance(event.context["temporary_resources"],tuple); assert not any(item.kind is ExportEventKind.COMMIT_SUCCEEDED for item in events)
def test_production_contains_no_silently_swallowed_broad_exception():
    allowed_fallbacks={("Blender_to_Spine2D_Mesh_Exporter/blender_adapter/bake_material_preparation.py","_input_socket"),("Blender_to_Spine2D_Mesh_Exporter/blender_adapter/bake_material_preparation.py","_output_socket"),("Blender_to_Spine2D_Mesh_Exporter/blender_adapter/bake_material_preparation.py","_incoming_links")}; violations=[]
    for root in (PRODUCTION/"application",PRODUCTION/"domain",PRODUCTION/"blender_adapter",PRODUCTION/"infrastructure"):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts: continue
            tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path)); parents={}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent): parents[child]=parent
            relative=path.relative_to(ROOT).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node,ast.ExceptHandler): continue
                is_bare=node.type is None; is_exception=isinstance(node.type,ast.Name) and node.type.id in {"Exception","BaseException"}; only_pass=len(node.body)==1 and isinstance(node.body[0],ast.Pass)
                if (is_bare or is_exception) and only_pass:
                    owner=parents.get(node)
                    while owner is not None and not isinstance(owner,(ast.FunctionDef,ast.AsyncFunctionDef)): owner=parents.get(owner)
                    function_name=owner.name if owner is not None else "<module>"
                    if (relative,function_name) not in allowed_fallbacks: violations.append((relative,function_name,node.lineno))
    assert violations==[]

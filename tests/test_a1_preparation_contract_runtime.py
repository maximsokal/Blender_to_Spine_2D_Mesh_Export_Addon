import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class Base:
    pass


class A1SingleObjectStage(Base):
    value = "STAGE"


class IssueSeverity:
    WARNING = "WARNING"


class ExportIssue:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Settings(Base):
    def __init__(self):
        self.export = SimpleNamespace(
            spine_version="4.2.43",
            texture_width=128,
            texture_height=64,
        )


class Snapshot(Base):
    def __init__(self):
        self.source_object_id = "Object"
        self.world_matrix = (1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 1.0)


class GeneratedBakePlan(Base):
    pass


class BakePlan(Base):
    def __init__(self):
        self.source_object_id = "Object"


class Rig(Base):
    def __init__(self):
        self.request = SimpleNamespace(prefix="Object")


class DocumentAssembly(Base):
    def __init__(self):
        self.document = object()


class Unwrap(Base):
    def __init__(self):
        self.snapshot = object()


def _install_stub(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    sys.modules[name] = module
    return module


def _load_contracts():
    root = "contractpkg"
    _install_stub(root).__path__ = []
    _install_stub(root + ".blender_adapter").__path__ = []
    _install_stub(root + ".domain").__path__ = []
    _install_stub(
        root + ".application",
        A1DocumentAssemblyResult=DocumentAssembly,
        A1GeometryPreparationResult=type("A1GeometryPreparationResult", (Base,), {}),
        A1ResolvedOutputPaths=type("A1ResolvedOutputPaths", (Base,), {}),
        A1SingleObjectExportSettings=Settings,
        A1SingleObjectStage=A1SingleObjectStage,
        A1TexturingTopology=type("A1TexturingTopology", (Base,), {}),
        A1UvPropagationResult=type("A1UvPropagationResult", (Base,), {}),
        A1ZGroupAssignmentPlan=type("A1ZGroupAssignmentPlan", (Base,), {}),
        ExportIssue=ExportIssue,
        IssueSeverity=IssueSeverity,
    )
    baking = _install_stub(
        root + ".domain.baking",
        BakePlan=BakePlan,
        ObjectMaterialAnalysis=type("ObjectMaterialAnalysis", (Base,), {}),
    )
    baking.__path__ = []
    _install_stub(
        root + ".domain.baking.generated_materials",
        GeneratedBakePlan=GeneratedBakePlan,
    )
    _install_stub(root + ".domain.geometry", MeshSnapshot=Snapshot)
    _install_stub(
        root + ".domain.spine",
        LegacyRigBuildResult=Rig,
        SpineDocument=type("SpineDocument", (Base,), {}),
    )
    _install_stub(root + ".domain.uv", UvUnwrapResult=Unwrap)
    path = Path(__file__).resolve().parents[1] / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter" / "a1_preparation_contracts.py"
    name = root + ".blender_adapter.a1_preparation_contracts"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, sys.modules[root + ".application"]


def test_statistics_are_immutable_and_reject_bool():
    module, _ = _load_contracts()
    frozen = module.freeze_statistics({"count": 2}, {"name": "Object"})
    assert dict(frozen) == {"count": 2, "name": "Object"}
    with pytest.raises(TypeError):
        frozen["count"] = 3
    with pytest.raises(TypeError, match="statistics values"):
        module.freeze_statistics({"flag": True})


def test_metadata_and_prepared_result_share_strict_contract():
    module, application = _load_contracts()
    settings = Settings()
    assert module.build_skeleton_metadata(settings)["width"] == 128
    prepared = module.PreparedA1Object(
        source_object=object(),
        object_id="Object",
        prefix="Object",
        settings=settings,
        output_paths=application.A1ResolvedOutputPaths(),
        source_snapshot=Snapshot(),
        z_groups=application.A1ZGroupAssignmentPlan(),
        geometry=application.A1GeometryPreparationResult(),
        texturing_topology=application.A1TexturingTopology(),
        unwrap_result=Unwrap(),
        uv_regions=application.A1UvPropagationResult(),
        material_analysis=sys.modules["contractpkg.domain.baking"].ObjectMaterialAnalysis(),
        bake_plan=BakePlan(),
        rig=Rig(),
        document_assembly=DocumentAssembly(),
        warnings=(),
        statistics={"count": 1},
    )
    assert prepared.world_position == (2.0, 3.0, 4.0)
    with pytest.raises(TypeError):
        prepared.statistics["count"] = 2

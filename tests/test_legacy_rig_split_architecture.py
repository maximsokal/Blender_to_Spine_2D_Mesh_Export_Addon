import ast
from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_assembly import (
    build_legacy_rig as physical_build_legacy_rig,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_bones import (
    build_z_group_bones_for_request,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_builder import (
    _build_constraints,
    _build_z_group_bones,
    _main_position,
    build_legacy_rig,
    calculate_uniform_scale,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_constraints import (
    build_legacy_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_rig_scale import (
    calculate_uniform_scale as physical_calculate_uniform_scale,
    resolve_main_position,
)


ROOT = Path(__file__).parents[1] / "Blender_to_Spine2D_Mesh_Exporter"
SPINE = ROOT / "domain" / "spine"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_level_definitions(path: Path) -> tuple[str, ...]:
    tree = ast.parse(_source(path), filename=str(path))
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _top_level_relative_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(_source(path), filename=str(path))
    result = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            if node.module.startswith("legacy_rig_"):
                result.append(node.module)
    return tuple(result)


def test_legacy_rig_builder_is_a_facade_without_second_implementation():
    path = SPINE / "legacy_rig_builder.py"
    source = _source(path)

    assert _top_level_definitions(path) == ()
    for owner in (
        "legacy_rig_assembly",
        "legacy_rig_bones",
        "legacy_rig_constraints",
        "legacy_rig_contracts",
        "legacy_rig_error",
        "legacy_rig_scale",
    ):
        assert owner in source


def test_physical_owners_keep_separate_responsibilities():
    contracts = _source(SPINE / "legacy_rig_contracts.py")
    scale = _source(SPINE / "legacy_rig_scale.py")
    plan = _source(SPINE / "legacy_rig_plan.py")
    bones = _source(SPINE / "legacy_rig_bones.py")
    constraints = _source(SPINE / "legacy_rig_constraints.py")
    validation = _source(SPINE / "legacy_rig_validation.py")
    assembly = _source(SPINE / "legacy_rig_assembly.py")

    assert "Bone(" not in contracts
    assert "from .model" not in scale
    assert "Bone(" not in plan
    assert "IKConstraint(" not in plan
    assert "TransformConstraint(" not in plan
    assert "IKConstraint(" not in bones
    assert "TransformConstraint(" not in bones
    assert "Bone(" not in constraints
    assert "def build_legacy_rig(" not in validation
    assert "build_legacy_rig_plan" in assembly
    assert "build_legacy_rig_bones" in assembly
    assert "build_legacy_rig_constraints" in assembly
    assert "validate_legacy_rig_result" in assembly


def test_production_callers_use_physical_rig_owners_and_profile_router():
    expected = {
        ROOT / "blender_adapter" / "a1_document_preparation.py": (
            "domain.spine.rig_builder",
            "domain.spine.legacy_rig_contracts",
        ),
        ROOT / "application" / "a1_single_object.py": (
            "domain.spine.legacy_rig_contracts",
            "domain.spine.legacy_rig_scale",
        ),
        ROOT / "application" / "a1_z_groups.py": (
            "domain.spine.legacy_rig_contracts",
        ),
        SPINE / "legacy_attachment_builder.py": (
            "from .legacy_rig_contracts import LegacyRigBuildResult",
        ),
        SPINE / "connected_group_contracts.py": (
            "from .legacy_rig_contracts import UniformScaleMode",
        ),
        SPINE / "connected_group_assembly.py": (
            "from .legacy_rig_scale import calculate_uniform_scale",
        ),
        SPINE / "__init__.py": (
            "from .legacy_rig_assembly import build_legacy_rig",
            "from .legacy_rig_contracts import",
            "from .legacy_rig_error import LegacyRigBuildError",
            "from .legacy_rig_scale import calculate_uniform_scale",
        ),
    }
    for path, fragments in expected.items():
        source = _source(path)
        assert "legacy_rig_builder" not in source
        for fragment in fragments:
            assert fragment in source

    preparation = _source(ROOT / "blender_adapter" / "a1_document_preparation.py")
    assert "build_rig(" in preparation
    assert "domain.spine.legacy_rig_assembly" not in preparation


def test_historical_facade_aliases_point_to_physical_functions():
    assert build_legacy_rig is physical_build_legacy_rig
    assert calculate_uniform_scale is physical_calculate_uniform_scale
    assert _main_position is resolve_main_position
    assert _build_constraints is build_legacy_constraints
    assert _build_z_group_bones is build_z_group_bones_for_request


def test_physical_top_level_import_graph_is_acyclic():
    modules = {
        path.stem: path
        for path in SPINE.glob("legacy_rig_*.py")
        if path.stem != "legacy_rig_builder"
    }
    graph = {
        name: tuple(
            dependency
            for dependency in _top_level_relative_imports(path)
            if dependency in modules
        )
        for name, path in modules.items()
    }

    visiting = set()
    visited = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise AssertionError(f"legacy rig import cycle at {name}: {graph}")
        visiting.add(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for module_name in graph:
        visit(module_name)

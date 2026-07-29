from dataclasses import replace
import ast
from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    ConnectedGroupBuildError,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    LegacyRigProfile,
    SpineCompositionError,
    SpineValidator,
    build_connected_group_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_assembly import (
    apply_object_placements,
    build_connected_group_document as physical_build_connected_group_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_layout import (
    ordered_component_ids,
    resolve_anchor,
    resolve_layers_and_placements,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_schedule import (
    build_constraint_schedule,
    reorder_object_constraints,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_validation import (
    validate_connected_group_inputs,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.legacy_connected_group import (
    _anchor,
    _apply_object_placements,
    _build_constraint_schedule,
    _ordered_component_ids,
    _reorder_object_constraints,
    _resolve_layers_and_placements,
    _validate_inputs,
)

from test_connected_group_document import connected_objects, settings


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


def test_connected_group_facade_has_no_second_implementation():
    path = SPINE / "legacy_connected_group.py"
    source = _source(path)

    assert _top_level_definitions(path) == ()
    for owner in (
        "connected_group_assembly",
        "connected_group_contracts",
        "connected_group_error",
        "connected_group_global_rig",
        "connected_group_layout",
        "connected_group_schedule",
        "connected_group_validation",
    ):
        assert owner in source


def test_physical_owners_keep_separate_responsibilities():
    contracts = _source(SPINE / "connected_group_contracts.py")
    layout = _source(SPINE / "connected_group_layout.py")
    schedule = _source(SPINE / "connected_group_schedule.py")
    global_rig = _source(SPINE / "connected_group_global_rig.py")
    validation = _source(SPINE / "connected_group_validation.py")
    assembly = _source(SPINE / "connected_group_assembly.py")

    assert "compose_spine_documents" not in contracts
    assert "IKConstraint(" not in layout
    assert "TransformConstraint(" not in layout
    assert "SpineDocumentComponent" not in schedule
    assert "compose_spine_documents" not in global_rig
    assert "resolve_layers_and_placements" not in validation
    assert "compose_spine_documents" in assembly
    assert "namespace_animations=settings.namespace_animations" in assembly


def test_production_and_package_import_physical_connected_owners():
    caller = _source(ROOT / "blender_adapter" / "a1_multi_object_composition.py")
    package = _source(SPINE / "__init__.py")

    assert "domain.spine.connected_group_assembly" in caller
    assert "domain.spine.connected_group_contracts" in caller
    assert "legacy_connected_group" not in caller
    assert "namespace_animations=settings.namespace_animations" in caller
    assert "from .connected_group_assembly import" in package
    assert "from .connected_group_contracts import" in package
    assert "from .connected_group_error import" in package
    assert "from .legacy_connected_group import" not in package


def test_historical_private_aliases_point_to_physical_functions():
    assert build_connected_group_document is physical_build_connected_group_document
    assert _anchor is resolve_anchor
    assert _resolve_layers_and_placements is resolve_layers_and_placements
    assert _ordered_component_ids is ordered_component_ids
    assert _build_constraint_schedule is build_constraint_schedule
    assert _reorder_object_constraints is reorder_object_constraints
    assert _apply_object_placements is apply_object_placements
    assert _validate_inputs is validate_connected_group_inputs


def test_connected_namespace_setting_can_preserve_distinct_original_names():
    objects = connected_objects()[:2]
    renamed = (
        replace(
            objects[0],
            document=replace(objects[0].document, animations={"idle": {}}),
        ),
        replace(
            objects[1],
            document=replace(objects[1].document, animations={"run": {}}),
        ),
    )

    result = build_connected_group_document(
        renamed,
        settings(namespace_animations=False),
    )

    assert tuple(result.document.animations) == ("idle", "run")
    assert tuple(
        (item.component_id, item.original_name, item.global_name)
        for item in result.composition.animation_names
    ) == (
        ("first", "idle", "idle"),
        ("second", "run", "run"),
    )


def test_connected_namespace_setting_rejects_duplicate_original_names():
    with pytest.raises(SpineCompositionError, match="Animation name 'animation' is duplicated"):
        build_connected_group_document(
            connected_objects()[:2],
            settings(namespace_animations=False),
        )


def test_identity_strings_reject_boundary_whitespace_before_composition():
    first = connected_objects()[0]
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        replace(first, prefix=" First ")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        replace(first, component_id=" first ")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        settings(group_prefix=" all_objects ")
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        settings(anchor_component_id=" first ")


def test_internal_component_and_global_bone_namespace_collisions_fail_early():
    objects = connected_objects()
    internal_collision = replace(
        objects[1],
        component_id="__all_objects_rig__",
    )
    with pytest.raises(ConnectedGroupBuildError, match="component ID collides"):
        build_connected_group_document(
            (objects[0], internal_collision),
            settings(),
        )

    changed_document = replace(
        objects[0].document,
        bones=objects[0].document.bones
        + (Bone("all_objects_0_scale", parent="root"),),
    )
    SpineValidator().validate_or_raise(changed_document)
    bone_collision = replace(objects[0], document=changed_document)
    with pytest.raises(ConnectedGroupBuildError, match="bone namespace collides"):
        build_connected_group_document(
            (bone_collision, objects[1]),
            settings(),
        )


def test_constraint_collections_reject_wrong_typed_values_at_model_boundary():
    first = connected_objects()[0]
    with pytest.raises(TypeError, match="transform must contain only TransformConstraint"):
        replace(
            first.document,
            ik=(),
            transform=(first.document.ik[0], *first.document.transform),
        )


def test_explicit_falsy_profile_is_not_replaced_by_default():
    with pytest.raises(TypeError, match="profile must be LegacyRigProfile"):
        build_connected_group_document(
            connected_objects()[:2],
            settings(),
            profile=0,
        )


def test_bool_values_are_not_accepted_as_connected_numeric_contracts():
    first = connected_objects()[0]
    with pytest.raises(ValueError, match="texture_width"):
        ConnectedGroupSettings(True, 100)
    with pytest.raises(ValueError, match="z_tolerance"):
        ConnectedGroupSettings(100, 100, z_tolerance=True)
    with pytest.raises(ValueError, match=r"world_position\[0\]"):
        ConnectedObjectDocument(
            component_id=first.component_id,
            prefix=first.prefix,
            document=first.document,
            world_position=(True, 0.0, 0.0),
        )


def test_existing_connected_golden_layout_and_schedule_match_main():
    result = build_connected_group_document(connected_objects(), settings())

    assert tuple(
        (
            layer.layer_index,
            layer.component_ids,
            layer.scale_bone_name,
            layer.layer_bone_name,
        )
        for layer in result.layers
    ) == (
        (0, ("second",), "all_objects_0_scale", "all_objects_layer_0"),
        (
            1,
            ("first", "third"),
            "all_objects_1_scale",
            "all_objects_layer_1",
        ),
    )
    assert result.constraint_schedule.object_rotation_x == (
        ("first", 4),
        ("second", 3),
        ("third", 4),
    )
    assert result.constraint_schedule.unique_orders == tuple(range(15))
    assert len(result.constraint_schedule.all_orders) > len(
        result.constraint_schedule.unique_orders
    )

from math import radians

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import build_uv_operator_plan
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (
    UvMarginMethod,
    UvPackPinMethod,
    UvPackRotateMethod,
    UvPackShapeMethod,
    UvPackUdimSource,
    UvSmartRotateMethod,
    UvUnwrapMethod,
    UvUnwrapSettings,
)


def test_smart_project_plan_maps_every_documented_setting():
    settings = UvUnwrapSettings(
        method=UvUnwrapMethod.SMART_PROJECT,
        margin_method=UvMarginMethod.FRACTION,
        smart_angle_limit_degrees=45.0,
        smart_rotate_method=UvSmartRotateMethod.AXIS_ALIGNED_X,
        island_margin=0.02,
        area_weight=0.7,
        correct_aspect=False,
        scale_to_bounds=False,
        pack_udim_source=UvPackUdimSource.ACTIVE_UDIM,
        pack_rotate=False,
        pack_rotate_method=UvPackRotateMethod.CARDINAL,
        pack_scale=False,
        pack_merge_overlap=True,
        pack_margin=0.03,
        pack_pin=True,
        pack_pin_method=UvPackPinMethod.ROTATION_SCALE,
        pack_shape_method=UvPackShapeMethod.CONVEX,
    )

    plan = build_uv_operator_plan(settings)

    assert plan.unwrap_operator == "smart_project"
    assert dict(plan.unwrap_arguments) == {
        "angle_limit": radians(45.0),
        "margin_method": "FRACTION",
        "rotate_method": "AXIS_ALIGNED_X",
        "island_margin": 0.02,
        "area_weight": 0.7,
        "correct_aspect": False,
        "scale_to_bounds": False,
    }
    assert dict(plan.pack_arguments) == {
        "udim_source": "ACTIVE_UDIM",
        "rotate": False,
        "rotate_method": "CARDINAL",
        "scale": False,
        "merge_overlap": True,
        "margin_method": "FRACTION",
        "margin": 0.03,
        "pin": True,
        "pin_method": "ROTATION_SCALE",
        "shape_method": "CONVEX",
    }


def test_conformal_plan_uses_unwrap_arguments_and_can_skip_pack():
    settings = UvUnwrapSettings(
        method=UvUnwrapMethod.CONFORMAL,
        margin_method=UvMarginMethod.ADD,
        island_margin=0.01,
        fill_holes=False,
        correct_aspect=False,
        use_subsurf_data=True,
        no_flip=True,
        iterations=32,
        use_weights=True,
        weight_group="importance",
        weight_factor=2.5,
        pack_islands=False,
    )

    plan = build_uv_operator_plan(settings)

    assert plan.unwrap_operator == "unwrap"
    assert dict(plan.unwrap_arguments) == {
        "method": "CONFORMAL",
        "fill_holes": False,
        "correct_aspect": False,
        "use_subsurf_data": True,
        "margin_method": "ADD",
        "margin": 0.01,
        "no_flip": True,
        "iterations": 32,
        "use_weights": True,
        "weight_group": "importance",
        "weight_factor": 2.5,
    }
    assert plan.pack_arguments is None


def test_minimum_stretch_is_not_silently_converted_to_another_method():
    plan = build_uv_operator_plan(
        UvUnwrapSettings(
            method=UvUnwrapMethod.MINIMUM_STRETCH,
            pack_islands=False,
        )
    )
    assert plan.unwrap_operator == "unwrap"
    assert plan.unwrap_arguments["method"] == "MINIMUM_STRETCH"

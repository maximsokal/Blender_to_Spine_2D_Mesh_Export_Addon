from Blender_to_Spine2D_Mesh_Exporter.application.a1_grouped_camera_projection import (
    _strip_hidden_visual_timelines,
)


def test_grouped_overlay_removes_only_hidden_source_slot_timelines():
    animations = {
        "walk": {
            "bones": {"arm": {"rotate": [{"time": 0.0, "value": 5.0}]}},
            "slots": {
                "hidden_a": {"rgba": [{"time": 0.0, "color": "ffffffff"}]},
                "hidden_b": {"attachment": [{"time": 0.0, "name": "B"}]},
                "visible": {"rgba": [{"time": 0.0, "color": "ff00ffff"}]},
            },
            "deform": {"default": {"hidden_a": {"A": []}}},
            "events": [{"time": 0.0, "name": "step"}],
        },
        "raw": "preserve-non-mapping",
    }

    result = _strip_hidden_visual_timelines(
        animations,
        {"hidden_a", "hidden_b"},
    )

    assert set(result["walk"]["slots"]) == {"visible"}
    assert result["walk"]["bones"] == animations["walk"]["bones"]
    assert result["walk"]["deform"] == animations["walk"]["deform"]
    assert result["walk"]["events"] == animations["walk"]["events"]
    assert result["raw"] == "preserve-non-mapping"


def test_grouped_overlay_removes_empty_slots_section_only():
    animations = {
        "preview": {
            "slots": {
                "hidden": {"rgba": [{"time": 0.0, "color": "ffffffff"}]}
            },
            "bones": {"root": {"translate": []}},
        }
    }

    result = _strip_hidden_visual_timelines(animations, {"hidden"})

    assert "slots" not in result["preview"]
    assert result["preview"]["bones"] == animations["preview"]["bones"]

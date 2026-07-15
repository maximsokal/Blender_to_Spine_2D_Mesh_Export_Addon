from copy import deepcopy

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1ParityError,
    A1ParitySettings,
    A1ParitySeverity,
    compare_a1_exports,
)


def build_export():
    return {
        "skeleton": {
            "hash": "legacy-hash",
            "spine": "4.2.43",
            "x": 0,
            "y": 0,
            "width": 128,
            "height": 128,
            "images": "",
            "audio": "./audio",
        },
        "bones": [
            {"name": "root"},
            {"name": "Mesh_main", "parent": "root", "x": 10.0, "y": 20.0},
            {"name": "Mesh", "parent": "Mesh_main"},
            {"name": "Mesh_vertex_0", "parent": "Mesh", "x": -10.0},
            {"name": "Mesh_vertex_1", "parent": "Mesh", "x": 10.0},
            {"name": "Mesh_vertex_2", "parent": "Mesh", "y": 10.0},
        ],
        "slots": [
            {"name": "Mesh", "bone": "Mesh", "attachment": "Mesh"},
        ],
        "skins": [
            {
                "name": "default",
                "attachments": {
                    "Mesh": {
                        "Mesh": {
                            "type": "mesh",
                            "path": "images/Mesh_Baked",
                            "width": 128.0,
                            "height": 128.0,
                            "uvs": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                            "triangles": [0, 1, 2],
                            "vertices": [
                                1,
                                3,
                                0.0,
                                0.0,
                                1.0,
                                1,
                                4,
                                0.0,
                                0.0,
                                1.0,
                                1,
                                5,
                                0.0,
                                0.0,
                                1.0,
                            ],
                            "hull": 3,
                            "edges": [0, 1, 1, 2, 2, 0],
                        }
                    }
                },
            }
        ],
        "ik": [],
        "transform": [],
        "animations": {"idle": {"bones": {}}},
    }


def test_parity_ignores_volatile_metadata_and_accepts_numeric_tolerance():
    expected = build_export()
    actual = deepcopy(expected)
    actual["skeleton"]["hash"] = "new-hash"
    actual["skeleton"]["images"] = "different/path"
    actual["skeleton"]["audio"] = "other/audio"
    actual["bones"][1]["x"] += 0.00001
    actual["skins"][0]["attachments"]["Mesh"]["Mesh"]["uvs"][0] += 0.00001

    report = compare_a1_exports(expected, actual)

    assert report.compatible
    assert report.error_count == 0
    assert report.warning_count == 0
    report.require_compatible()


def test_structural_bone_parent_change_is_an_error():
    expected = build_export()
    actual = deepcopy(expected)
    actual["bones"][2]["parent"] = "root"

    report = compare_a1_exports(expected, actual)

    assert not report.compatible
    assert any(
        issue.path == "structure.bone_parents"
        and issue.severity is A1ParitySeverity.ERROR
        for issue in report.issues
    )
    with pytest.raises(A1ParityError) as exc_info:
        report.require_compatible()
    assert exc_info.value.report is report


def test_weighted_bone_index_mismatch_is_reported_semantically():
    expected = build_export()
    actual = deepcopy(expected)
    actual["skins"][0]["attachments"]["Mesh"]["Mesh"]["vertices"][1] = 4

    report = compare_a1_exports(expected, actual)

    matching = [
        issue
        for issue in report.issues
        if issue.code == "WEIGHTED_BONE_INDEX_MISMATCH"
    ]
    assert len(matching) == 1
    assert matching[0].path.endswith("[vertex=0].influences[0].bone_index")
    assert not report.compatible


def test_invalid_rewritten_weight_stream_becomes_report_issue_not_exception():
    expected = build_export()
    actual = deepcopy(expected)
    actual["skins"][0]["attachments"]["Mesh"]["Mesh"]["vertices"] = [1, 3]

    report = compare_a1_exports(expected, actual)

    assert any(
        issue.code == "ACTUAL_WEIGHT_STREAM_INVALID" for issue in report.issues
    )
    assert not report.compatible


def test_nonessential_edges_warn_by_default_and_can_be_strict():
    expected = build_export()
    actual = deepcopy(expected)
    actual["skins"][0]["attachments"]["Mesh"]["Mesh"]["edges"] = [
        0,
        2,
        2,
        1,
        1,
        0,
    ]

    default_report = compare_a1_exports(expected, actual)
    strict_report = compare_a1_exports(
        expected,
        actual,
        A1ParitySettings(nonessential_mesh_edges_are_errors=True),
    )

    assert default_report.compatible
    assert default_report.warning_count > 0
    assert all(
        issue.severity is A1ParitySeverity.WARNING
        for issue in default_report.issues
        if issue.path.startswith("attachments.") and ".edges" in issue.path
    )
    assert not strict_report.compatible
    assert any(
        issue.severity is A1ParitySeverity.ERROR and ".edges" in issue.path
        for issue in strict_report.issues
    )


def test_animations_are_optional_but_can_be_compared_explicitly():
    expected = build_export()
    actual = deepcopy(expected)
    actual["animations"] = {"other": {}}

    default_report = compare_a1_exports(expected, actual)
    animation_report = compare_a1_exports(
        expected,
        actual,
        A1ParitySettings(compare_animations=True),
    )

    assert default_report.compatible
    assert not animation_report.compatible
    assert any(issue.path.startswith("animations") for issue in animation_report.issues)


def test_uv_length_change_prevents_weight_comparison_and_reports_length():
    expected = build_export()
    actual = deepcopy(expected)
    actual["skins"][0]["attachments"]["Mesh"]["Mesh"]["uvs"].extend((0.5, 0.5))

    report = compare_a1_exports(expected, actual)

    assert any(
        issue.code == "LENGTH_MISMATCH" and issue.path.endswith(".uvs")
        for issue in report.issues
    )
    assert not report.compatible

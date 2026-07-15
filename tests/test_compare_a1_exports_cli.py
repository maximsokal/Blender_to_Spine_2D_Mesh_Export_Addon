import json

from tools.compare_a1_exports import (
    EXIT_COMPATIBLE,
    EXIT_INCOMPATIBLE,
    EXIT_INVALID_INPUT,
    run,
)


def build_export():
    return {
        "skeleton": {
            "hash": "hash",
            "spine": "4.2.43",
            "width": 64,
            "height": 64,
            "images": "",
            "audio": "./audio",
        },
        "bones": [
            {"name": "root"},
            {"name": "Mesh", "parent": "root"},
            {"name": "Mesh_vertex_0", "parent": "Mesh"},
            {"name": "Mesh_vertex_1", "parent": "Mesh"},
            {"name": "Mesh_vertex_2", "parent": "Mesh"},
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
                            "width": 64,
                            "height": 64,
                            "uvs": [0, 0, 1, 0, 0, 1],
                            "triangles": [0, 1, 2],
                            "vertices": [
                                1,
                                2,
                                0,
                                0,
                                1,
                                1,
                                3,
                                0,
                                0,
                                1,
                                1,
                                4,
                                0,
                                0,
                                1,
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
    }


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cli_returns_zero_for_compatible_exports(tmp_path, capsys):
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    expected_value = build_export()
    actual_value = build_export()
    actual_value["skeleton"]["hash"] = "different"
    write_json(expected, expected_value)
    write_json(actual, actual_value)

    exit_code = run((str(expected), str(actual), "--quiet"))

    assert exit_code == EXIT_COMPATIBLE
    assert "COMPATIBLE" in capsys.readouterr().out


def test_cli_returns_one_and_writes_machine_report_for_errors(tmp_path, capsys):
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    report_path = tmp_path / "reports" / "parity.json"
    expected_value = build_export()
    actual_value = build_export()
    actual_value["skins"][0]["attachments"]["Mesh"]["Mesh"]["vertices"][1] = 3
    write_json(expected, expected_value)
    write_json(actual, actual_value)

    exit_code = run(
        (
            str(expected),
            str(actual),
            "--report-json",
            str(report_path),
        )
    )

    assert exit_code == EXIT_INCOMPATIBLE
    assert "INCOMPATIBLE" in capsys.readouterr().out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["compatible"] is False
    assert payload["error_count"] >= 1
    assert any(
        issue["code"] == "WEIGHTED_BONE_INDEX_MISMATCH"
        for issue in payload["issues"]
    )


def test_cli_returns_two_for_invalid_json(tmp_path, capsys):
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    expected.write_text("{not-json", encoding="utf-8")
    write_json(actual, build_export())

    exit_code = run((str(expected), str(actual)))

    assert exit_code == EXIT_INVALID_INPUT
    assert "input error" in capsys.readouterr().err

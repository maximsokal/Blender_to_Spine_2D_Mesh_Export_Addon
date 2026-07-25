from pathlib import Path

from tools.run_extension_install_gate import extension_commands, isolated_environment


def test_extension_gate_uses_official_repository_install_smoke_remove_sequence(tmp_path):
    commands = extension_commands(
        "/opt/blender/blender",
        repository_id="gate",
        repository_directory=tmp_path / "repo",
        archive=tmp_path / "addon.zip",
        extension_id="blender_to_spine2d_mesh_exporter",
        smoke_output=tmp_path / "out",
        smoke_report=tmp_path / "report.json",
    )
    assert tuple(name for name, _ in commands) == (
        "repo-add", "install-enable", "smoke-export", "remove", "repo-remove",
    )
    assert commands[0][1][1:4] == ("--command", "extension", "repo-add")
    assert "--clear-all" in commands[0][1]
    assert commands[1][1][1:4] == ("--command", "extension", "install-file")
    assert "-e" in commands[1][1]
    assert commands[2][1][1] == "--background"
    assert "bl_ext.gate.blender_to_spine2d_mesh_exporter" in commands[2][1]
    assert commands[3][1][-1] == "blender_to_spine2d_mesh_exporter"
    assert commands[4][1][-1] == "gate"


def test_extension_gate_uses_isolated_blender_user_directories(tmp_path):
    environment = isolated_environment(tmp_path)
    for key in ("BLENDER_USER_CONFIG", "BLENDER_USER_SCRIPTS", "BLENDER_USER_DATAFILES", "BLENDER_SYSTEM_EXTENSIONS"):
        path = Path(environment[key]); assert path.is_dir(); assert tmp_path.resolve() in path.parents

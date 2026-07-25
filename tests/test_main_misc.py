from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _excluded_paths() -> tuple[str, ...]:
    with (PACKAGE / "blender_manifest.toml").open("rb") as stream:
        return tuple(tomllib.load(stream)["build"]["paths_exclude_pattern"])


def test_retired_legacy_runtime_file_is_not_shipped():
    assert "/main.py" in _excluded_paths()


def test_rewrite_startup_does_not_import_retired_runtime_file():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "from . import main" not in source

import importlib
import sys
from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.pipeline_trace import (
    PipelineTraceSession,
    discover_pipeline_modules,
)


def _make_package(tmp_path: Path):
    package = tmp_path / "tracepkg"
    (package / "application").mkdir(parents=True)
    (package / "domain").mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "application" / "__init__.py").write_text("", encoding="utf-8")
    (package / "domain" / "__init__.py").write_text("", encoding="utf-8")
    (package / "domain" / "maths.py").write_text(
        "def double(value):\n    return value * 2\n", encoding="utf-8"
    )
    (package / "application" / "service.py").write_text(
        "from tracepkg.domain.maths import double\n"
        "def run(values):\n"
        "    return [double(value) for value in values]\n"
        "def handled():\n"
        "    try:\n"
        "        raise ValueError('probe')\n"
        "    except ValueError:\n"
        "        return None\n",
        encoding="utf-8",
    )
    return package


def test_trace_reports_per_file_shapes_edges_and_unreached_modules(tmp_path: Path):
    package = _make_package(tmp_path)
    (package / "domain" / "unused.py").write_text("def no_call():\n    return 1\n")
    sys.path.insert(0, str(tmp_path))
    try:
        service = importlib.import_module("tracepkg.application.service")
        session = PipelineTraceSession(
            package,
            package_name="tracepkg",
            focus_modules=("application.service",),
            capture_values=True,
        )
        with session:
            assert service.run([1, 2, 3]) == [2, 4, 6]
            assert service.handled() is None
        report = session.build_report(
            run_success=True,
            scenario="unit",
            expected_calls=(("application.service", "run"), ("domain.maths", "double")),
        )
    finally:
        sys.path.remove(str(tmp_path))
        for name in tuple(sys.modules):
            if name == "tracepkg" or name.startswith("tracepkg."):
                sys.modules.pop(name, None)

    modules = {item["module"]: item for item in report["modules"]}
    assert modules["application.service"]["status"] == "executed"
    assert modules["domain.maths"]["status"] == "executed"
    assert modules["domain.unused"]["status"] == "not_imported"
    assert report["summary"]["missing_expected_call_count"] == 0
    assert report["focus"]["matched_modules"][0]["module"] == "application.service"
    assert [item["event"] for item in report["focus"]["timeline"]]
    assert any(item["event"] == "call" and item["function"] == "run" for item in report["focus"]["timeline"])
    assert any(
        edge["source"]["module"] == "application.service"
        and edge["target"]["module"] == "domain.maths"
        for edge in report["call_edges"]
    )
    assert report["summary"]["exception_event_count"] >= 1


def test_discovery_includes_nested_production_files(tmp_path: Path):
    package = _make_package(tmp_path)
    modules = discover_pipeline_modules(package)
    names = {metadata["module"] for metadata in modules.values()}
    assert "application.service" in names
    assert "domain.maths" in names

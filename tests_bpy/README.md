# Real Blender 5.2 tests through `bpy`

This test root uses the official Blender Foundation `bpy==5.2.0` wheel. It does
not require installation or startup of the Blender desktop application, but it does
load the real Blender core, `bmesh`, depsgraph, RNA, operators, and datablocks.

The directory is intentionally separate from `tests/`. The legacy
`tests/conftest.py` installs `MagicMock` replacements for `bpy`, `bmesh`, and
`mathutils`, so placing these tests there would create false-positive results.

## Windows setup

```bat
py -3.13 -m venv .venv-bpy
.venv-bpy\Scripts\python -m pip install --upgrade pip
.venv-bpy\Scripts\python -m pip install -r requirements-bpy.txt
.venv-bpy\Scripts\python scripts\run_bpy_tests.py
```

Additional pytest arguments can be appended:

```bat
.venv-bpy\Scripts\python scripts\run_bpy_tests.py -k uv -vv
```

The runner fails closed when:

- Python is not CPython 3.13;
- the installed distribution is not exactly `bpy==5.2.0`;
- `bpy.app.version` is not exactly Blender 5.2.0;
- real `bpy` or `bmesh` cannot be imported.

## Covered integration boundaries

- real `bmesh.new()` ownership and `bm.free()` cleanup;
- borrowed `bmesh.from_edit_mesh()` without `bm.free()`;
- actual OBJECT/EDIT mode switching and restoration;
- Blender 5.2 UV attribute collections;
- BOOLEAN/EDGE seam and sharp attributes;
- source mesh snapshot immutability;
- evaluated depsgraph snapshots and temporary datablock cleanup;
- real UV unwrap/pack operators on an isolated temporary object;
- two complete Rewrite registration/unregistration cycles.

These tests complement rather than replace the full Blender executable matrices.
GPU rendering, compositor behavior, external image codecs, and UI event-loop behavior
still require the existing Blender headless/manual fixtures.

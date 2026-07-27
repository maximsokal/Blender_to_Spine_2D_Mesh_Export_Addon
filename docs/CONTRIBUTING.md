# Contributing

Thank you for contributing to Blender to Spine2D Mesh Exporter.

## Supported baseline

Production changes must target:

- Blender 5.2 or newer;
- Spine 4.2.43 compatibility contracts;
- the current package under `Blender_to_Spine2D_Mesh_Exporter`;
- Windows-safe output behavior.

Legacy source files may be used as behavioral reference material, but new production code must not import or execute Legacy modules unless an explicitly approved compatibility boundary requires it.

## Development setup

Create a Blender-independent Python environment for `tests/` and a separate real-bpy environment for `tests_bpy/`.

Install the repository development requirements appropriate to each environment. Do not add fake Blender modules to the extension package or use them as proof of real Blender runtime compatibility.

A local Blender 5.2 executable is required for headless integration and package validation.

## Branch and pull request workflow

- Create a focused feature or fix branch.
- Keep the working tree clean before recording final validation.
- Write commit messages and pull request text in English.
- Describe the problem, ownership boundary, implementation, tests, and remaining risks.
- Do not claim validation that was not run on the candidate SHA.
- Keep private customer scenes, textures, outputs, and reports out of the public repository.

## Architecture rules

### Domain and application code

Modules below `domain/` and Blender-independent application contracts must not import `bpy` or `bmesh`.

Use immutable dataclasses and explicit enums for data crossing stage boundaries. Validate inputs and outputs at the owning boundary.

Geometry identity must use local and source IDs. UV correspondence uses `SourceLoopId`; do not introduce rounded-coordinate or nearest-point fallback matching.

### Blender adapters

Blender state is mutable and context-sensitive. Every adapter must define what it owns, borrows, temporarily changes, and restores.

- Prefer direct RNA, Mesh, and BMesh APIs over `bpy.ops`.
- Do not call operators inside geometry loops.
- Validate Object Mode requirements explicitly.
- Do not silently switch user modes unless the public contract says so.
- Capture and restore active object, selection, frame, renderer, camera, View Layer, visibility, and material state when changed.
- Remove temporary objects, meshes, collections, images, materials, node trees, and attributes on success and failure paths.
- Preserve the original exception as the primary failure and report cleanup failures separately.

### BMesh ownership

For every BMesh created with `bmesh.new()`:

```python
bm = bmesh.new()
try:
    # complete operation
    ...
finally:
    bm.free()
```

A BMesh returned by `bmesh.from_edit_mesh()` is borrowed:

- do not call `bm.free()` on it;
- update it through `bmesh.update_edit_mesh()` when the operation owns an edit update;
- do not double-free any BMesh;
- ensure exception paths preserve the ownership rule.

### Filesystem output

Use the atomic output infrastructure. Do not write final JSON or texture files directly from inner preparation or bake stages.

- reserve paths before installation;
- write complete stage files;
- commit in deterministic order;
- restore backups on partial failure when possible;
- respect interprocess work-file ownership;
- never silently swallow rollback or cleanup failures.

### Logging and diagnostics

Use module-level loggers and structured issues at user-visible boundaries.

- Include object identity, stage, path, renderer, or resource name when relevant.
- Preserve exception causes with `raise ... from exc`.
- Avoid broad `except Exception: pass` blocks.
- Do not log private source data unnecessarily.

## Coding style

- Use clear English names and comments.
- Add type annotations to public and stage-boundary functions.
- Prefer small functions with one owner and explicit input/output types.
- Use `try`, `except`, and `finally` to make cleanup and state restoration visible.
- Avoid hardcoded values that exist only to satisfy one test fixture.
- Avoid introducing compatibility aliases unless a real caller or explicit contract requires them.
- Keep deterministic ordering independent of Blender collection iteration where order is not guaranteed.

The repository uses `ruff`, formatting checks, and pytest-based architecture contracts. Follow the configuration committed to the repository.

## Tests

Choose the lowest valid boundary and add higher-level coverage when Blender behavior matters.

### Pure Python tests

Use for:

- domain algorithms;
- immutable contracts;
- serializers and validators;
- output naming and transactions;
- source architecture checks;
- documentation contracts.

### Real bpy tests

Use for:

- RNA registration and migration;
- Mesh and UV API behavior;
- handler ownership;
- resource lifecycle;
- adapter behavior available through the installed bpy runtime.

### Blender headless tests

Use for:

- real bake operators;
- render engines;
- active camera and View Layer behavior;
- image save/load orientation;
- context-dependent operators;
- complete export and cleanup flows.

Every Blender command must include `--python-exit-code 1`.

Run the complete relevant suite without `--maxfail=1` before requesting final review. See [Testing and Release Validation](testing.md).

## Documentation

Public documentation is maintained in English.

A documentation change must:

- contain no Cyrillic characters;
- keep root README cover, release badge, download counter, Blender badge, Patreon badge, YouTube preview, and UI image references;
- use valid relative links;
- describe current production behavior rather than development history;
- update Settings Reference when user-facing RNA changes;
- update Architecture when ownership or data flow changes;
- update Output Format when names, JSON, texture, or transaction behavior changes;
- update Testing when validation commands or release gates change;
- update Changelog for public releases.

Temporary milestone journals named `REWRITE_*.md` are not part of the maintained documentation set.

## Pull request checklist

- [ ] The change has one clear owner.
- [ ] Production behavior is implemented without test-only hardcoding.
- [ ] Input and output types are validated.
- [ ] Blender state and temporary resources are restored in failure paths.
- [ ] BMesh ownership is correct.
- [ ] No new operators run inside performance-sensitive loops.
- [ ] Structured diagnostics preserve the original cause.
- [ ] Pure Python tests pass.
- [ ] Required real-bpy tests pass.
- [ ] Required Blender headless tests pass.
- [ ] Documentation links and English-only contract pass.
- [ ] The extension builds and validates with Blender 5.2.
- [ ] Validation logs correspond to the candidate SHA.

## License

Contributions are accepted under GNU GPL v3.0 or later, consistent with the repository license.
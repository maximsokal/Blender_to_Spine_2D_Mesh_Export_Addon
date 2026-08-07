# Contributing

Thank you for contributing to Blender to Spine2D Mesh Exporter.

## Supported baseline

Production changes must target the current extension contract:

- Blender 5.2 or newer;
- supported Spine target/profile/scope combinations from the capability registry;
- the production package under `Blender_to_Spine2D_Mesh_Exporter`;
- Windows-safe output behavior;
- current maintained documentation in English.

Historical Git revisions may be used as behavioral evidence, but new production code must
not import retired runtime modules unless an explicit compatibility boundary requires it.

## Development setup

Use separate environments for Blender-independent `tests/` and real-bpy `tests_bpy/`.
A local Blender 5.2 executable is required for Blender-headless integration and extension
validation.

Do not ship fake Blender modules as proof of real runtime compatibility.

## Branch and pull request workflow

- Work in a focused feature/fix branch.
- Keep the worktree clean before final validation.
- Write commits, pull requests, code comments, and documentation in English.
- Describe ownership, implementation, tests, and remaining risks.
- Do not claim tests that were not run on the exact candidate SHA.
- Keep private customer assets and reports out of the public repository.

## Architecture rules

### Domain and application code

Blender-independent domain/application modules must not import `bpy` or `bmesh`.

Use immutable typed contracts and explicit enums across stage boundaries. Validate inputs
and outputs at their owning boundary.

Geometry identity uses local/source IDs. UV correspondence uses source loop identity; do
not introduce coordinate-rounding or nearest-point fallback matching.

### Blender adapters

Blender state is mutable and context-sensitive. Every adapter must define what it owns,
borrows, changes, and restores.

- Prefer direct RNA, Mesh, and BMesh APIs over `bpy.ops`.
- Do not call operators inside geometry loops.
- Validate Object Mode requirements explicitly.
- Do not silently change user modes unless the public contract requires it.
- Capture/restore active object, selection, frame, renderer, camera, View Layer,
  visibility, and material state when changed.
- Remove temporary objects, meshes, collections, images, materials, node trees, cameras,
  and attributes on success and failure paths.
- Preserve the original exception and report cleanup failures separately.

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

- never call `bm.free()` on it;
- update through `bmesh.update_edit_mesh()` only when the operation owns the edit update;
- never double-free a BMesh.

### Filesystem output

Use the atomic output infrastructure. Inner preparation/bake stages must not write final
JSON or textures directly.

- reserve final paths;
- write complete stage files;
- commit deterministically;
- restore backups after partial failure when possible;
- respect interprocess work-file ownership;
- do not suppress rollback/cleanup failures.

### Logging and diagnostics

Use module-level loggers and structured issues at user-visible boundaries.

- include object/stage/path/resource identity when relevant;
- preserve causes with `raise ... from exc`;
- avoid `except Exception: pass`;
- do not log private source data unnecessarily.

## Coding style

- Use clear English names and comments.
- Add type annotations to public and stage-boundary functions.
- Keep functions focused on one owner/responsibility.
- Make cleanup and state restoration explicit with `try`/`except`/`finally`.
- Do not hardcode values purely for one test fixture.
- Do not add compatibility aliases without a real caller/contract.
- Keep ordering deterministic when Blender collection order is not guaranteed.

## Tests

Choose the lowest valid boundary and add higher-level coverage when Blender behavior matters.

### Pure Python

Use for domain algorithms, immutable contracts, serializers, validators, output naming,
transactions, architecture checks, and documentation contracts.

### Real bpy

Use for RNA registration/migration, Mesh/UV API behavior, handlers, resource lifecycle, and
adapter behavior available through the installed bpy runtime.

### Blender headless

Use for real bake/render behavior, active camera/View Layer state, image orientation,
context-sensitive operators, and complete export/cleanup flows.

Every Blender command must include `--python-exit-code 1`.

Run the complete relevant suite without fail-fast before final review. See
[Testing](testing.md).

## Documentation policy

Maintained documentation is English-only and describes the current product.

A documentation change must:

- contain no Cyrillic characters;
- preserve valid relative links and required README visual assets;
- avoid release-history sections and milestone journals;
- update Settings Reference when public RNA changes;
- update Rig Profiles when setup hierarchy/constraints change;
- update Architecture when ownership/data flow changes;
- update Output Format when serialized/output behavior changes;
- update Testing when validation gates change;
- keep the documented extension version synchronized with the manifest.

Historical release notes are not maintained as current documentation. Use Git tags/history
when historical behavior must be inspected.

## Pull request checklist

- [ ] The change has a clear owner.
- [ ] Production behavior is not test-only hardcoding.
- [ ] Input/output types are validated.
- [ ] Blender state and temporary resources are restored on failure.
- [ ] BMesh ownership is correct.
- [ ] No performance-sensitive loop contains avoidable operators.
- [ ] Structured diagnostics preserve the original cause.
- [ ] Pure Python tests pass.
- [ ] Required real-bpy tests pass.
- [ ] Required Blender-headless tests pass.
- [ ] Documentation is English-only and current.
- [ ] The extension builds and validates with Blender 5.2.
- [ ] Validation evidence corresponds to the exact candidate SHA.

## License

Contributions are accepted under GNU GPL v3.0 or later, consistent with the repository
license.

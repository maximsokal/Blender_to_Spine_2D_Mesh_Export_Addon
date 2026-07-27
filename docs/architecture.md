# Architecture

## Purpose

Blender to Spine2D Mesh Exporter is split into explicit boundaries so Blender state, deterministic geometry logic, Spine document construction, and output transactions can be tested independently.

The production extension targets Blender 5.2+ and Spine 4.2.43.

## Package boundaries

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/
  blender_adapter/
  domain/
  infrastructure/
```

### `application`

Owns use-case orchestration and immutable request, settings, progress, readiness, composition, and result contracts.

Application code coordinates stages but does not own Blender RNA mutation, low-level geometry algorithms, or filesystem installation details.

### `blender_adapter`

Owns every boundary that reads or temporarily mutates Blender state:

- source and evaluated Mesh capture;
- UV and edge attributes;
- material and shader graph analysis;
- render-engine and View Layer validation;
- semantic object baking;
- camera projection;
- Scene and object settings capture;
- UI request routing;
- registration-facing Scene migration;
- cleanup and state restoration.

A Blender adapter may call `bpy` or `bmesh`. It must release owned temporary resources and restore borrowed Blender state in `finally` paths.

### `domain`

Contains Blender-independent immutable models and algorithms:

- geometry IDs, lineage, topology, segmentation, decomposition, and triangulation;
- UV ranges and layout contracts;
- bake plans, material capabilities, projection coverage, contour, layout, and output policy;
- Spine bones, slots, attachments, constraints, animations, composition, validation, and serialization.

Domain modules do not import `bpy` or `bmesh`.

### `infrastructure`

Owns cross-cutting technical services:

- transactional registration;
- durable atomic output;
- interprocess locking;
- stale stage and backup recovery;
- export diagnostics and events;
- logging discovery;
- pipeline auditing and tracing;
- performance budgets.

## Registration lifecycle

The root `__init__.py` registers owners transactionally in dependency order:

```text
add-on preferences
Scene RNA properties
Scene settings migration
UI
readiness invalidation
automatic readiness
re-polish child panel
generated material UI
single-object operator boundary
```

If a step fails, completed steps are rolled back in reverse order. Unregistration also runs in reverse order and reports cleanup failures rather than silently suppressing them.

The Scene settings migration runs after Scene RNA exists and before the main UI becomes active. Registration-time RNA callbacks are distinguished from deliberate user edits so persisted values cannot prematurely advance the settings schema.

## Export request flow

```text
Blender panel and operators
  -> capture Scene and selected-object profiles
  -> build immutable single or multi export plan
  -> run readiness or production export
  -> prepare each object
  -> compose the Spine document
  -> stage JSON and textures
  -> validate staged output
  -> atomically commit all files
```

Mutable Blender UI state is captured once. Later stages consume typed immutable settings instead of repeatedly reading Scene properties.

## Object preparation stages

Each source object follows four typed stages:

```text
source geometry
  -> UV preparation
  -> texture planning
  -> Spine document preparation
```

Stage-specific errors preserve the owning stage, object identity, statistics, warnings, and original exception cause.

### Source geometry

The adapter reads an isolated evaluated mesh, applies the supported world-transform normalization policy, validates source lineage, and builds an immutable `MeshSnapshot`.

Local IDs identify elements inside one snapshot. Source IDs preserve identity back to the original Blender mesh. `SourceLoopId` is the authoritative key for UV correspondence; rounded positions and nearest-point matching are not used.

### Segmentation and decomposition

Auto Seam Maker uses deterministic angular region growth. Custom mode uses user-marked seam boundaries and disables angular splitting.

Every final attachment region must be a valid manifold disk. Complex regions are decomposed deterministically with complete and disjoint face coverage.

### UV preparation

Generated bake UVs are created on isolated temporary meshes. The source UV layer set, active/render-active roles, coordinates, and source Mesh identity are fingerprinted before and after relevant stages.

Required malformed or missing UV dependencies block export. Unused malformed UV data can be reported without being treated as a required dependency.

### Texture planning

Material graphs are analyzed for the effective renderer output. Image dependencies, semantic channels, and renderer capabilities determine whether the selected user mode can execute safely.

Normal mode never silently switches to Camera Projection. Unsupported mode/material combinations produce explicit diagnostics.

## Normal - UV Segments pipeline

```text
prepared regions
  -> shared generated SpineBakeUV layout
  -> temporary material and image setup
  -> semantic Cycles bake
  -> row conversion to Spine PNG file space
  -> staged texture validation
  -> per-region attachment output
```

The source Scene may use Blender 5.2 EEVEE. The bake transaction temporarily configures the validated Cycles state and restores the original Scene state afterward.

The saved texture rows are converted to the file-space orientation expected by the exported Spine UV coordinates. Staged validation loads the image through Blender and applies the inverse loaded-image axis conversion when sampling Spine UVs.

## Camera Projection pipeline

```text
validated active camera and render context
  -> transparent full-frame render tasks
  -> sequence maximum-coverage union
  -> alpha threshold and conservative cleanup
  -> stable crop
  -> concave contour or safe convex fallback
  -> exact triangulation
  -> projection attachment and cropped texture output
```

Camera Projection is explicit. It produces a screen-space attachment rather than region-based Normal attachments.

## Generated materials

Generated material policy is independent of source geometry preparation:

```text
Require Source
Generate If Missing
Force Generated
```

Temporary generated material patterns can color one complete output, each region, or each exported polygon. Generated materials use temporary node trees and color attributes and are removed on every exit path.

## Multi-object composition

The UI partitions selected Mesh objects into connected and standalone groups.

- Standalone mode composes independent component rigs.
- Connected mode composes at least two connected sources under the shared connected contract.
- Mixed mode combines one connected subgroup with standalone sources.

Exactly one connected source falls back to standalone mode with a structured warning.

The outer multi-object transaction owns the final JSON and every individual or grouped texture. Inner stages do not commit independently.

## Readiness analysis

Readiness uses the production preparation path without final file commit. It stores an immutable report with:

- overall state;
- object reports;
- blockers and warnings;
- geometry, topology, material, texture, rig, and attachment statistics.

Selection, geometry, UV, material, Scene, renderer, camera, or setting changes invalidate or stale the cached report.

## Atomic output

Outputs are reserved before installation and written to unique stage files. Existing finals may be protected by backups.

Commit guarantees:

1. reservation order is deterministic;
2. every staged file is complete before installation;
3. partial installation attempts restore previous finals when possible;
4. rollback removes or preserves stage files according to diagnostics preferences;
5. stale stage and backup files are recovered on a later export;
6. one live process does not remove work files owned by another live process.

## Source immutability and cleanup

The production path must not permanently change:

- source Mesh topology;
- source UV layers or coordinates;
- source material graphs;
- active object, selection, mode, renderer, frame, camera, View Layer, or visibility state beyond the transaction scope.

A `bmesh` created with `bmesh.new()` must be freed exactly once in `finally`. A BMesh returned by `bmesh.from_edit_mesh()` is borrowed and must not be freed by the caller.

Temporary Blender images, meshes, objects, collections, materials, node trees, and attributes are removed on success and failure paths.

## Stable public surfaces

The user-facing production operators are exposed through the current UI as:

```text
object.spine2d_single_export
object.spine2d_multi_export
object.spine2d_refresh_info
spine2d.reset_settings
spine2d.reset_generated_materials
```

Internal module and helper names are not public compatibility guarantees unless a test or documented API explicitly states otherwise.

## Related documents

- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Output Format](output-format.md)
- [Testing](testing.md)
- [Contributing](CONTRIBUTING.md)
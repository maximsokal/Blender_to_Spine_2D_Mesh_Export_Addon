# Scene bake analysis and runtime parity split

## Purpose

The former `scene_bake_analyzer.py` mixed Blender RNA conversion, World graph
inspection, source/Light/Camera snapshots, Scene assembly, and planning/execution
parity. Those responsibilities now have separate physical owners while the old
module remains a compatibility facade.

## Physical ownership

```text
scene_bake_error.py
  -> SceneBakeAnalysisError

scene_bake_rna.py
  -> lazy bpy loading
  -> names, matrices, colors, finite numeric reads
  -> animation and render-visibility reads
  -> explicit Scene/Context resolution

scene_bake_world.py
  -> active World Output and socket lookup
  -> direct Background strength diagnostics
  -> WorldBakeSnapshot

scene_bake_resources.py
  -> ObjectBakeContext
  -> LightBakeSnapshot
  -> CameraBakeSnapshot
  -> ColorManagementSnapshot

scene_bake_capture.py
  -> deterministic Scene object freeze
  -> sorted visible Light snapshots
  -> visible-object and shadow-caster sets
  -> SceneBakeContext
  -> combined object/Scene planning capture

scene_bake_runtime.py
  -> planning/execution structural parity

scene_bake_analyzer.py
  -> compatibility re-exports only
```

## Capture flow

```text
resolve explicit Scene or Context without bpy
-> lazily load bpy only when neither was supplied
-> freeze scene.objects once
-> filter render-visible objects
-> capture sorted Lights
-> derive visible-object and shadow-caster IDs
-> capture World, Camera, color management, renderer, and frame
-> build immutable SceneBakeContext
```

`a1_texture_planning.py` imports `analyse_bake_contexts()` from the physical
capture owner. Public adapter exports are also assembled from the physical
error, resource, capture, and runtime modules.

## Fail-closed runtime parity

Object-bake and B4 preflight both import the physical runtime owner. Structural
values compared between planning and execution are:

```text
source object ID and type
source collection membership
source hide_render, visible_camera, and visible_shadow
source animation presence
Scene ID and render engine
World ID, use_nodes, node types, and animation presence
Camera ID, type, and animation presence
Light IDs, types, use_shadow, and animation presence
render-visible object IDs
shadow-caster IDs
color-management snapshot
```

The following frame-evaluated values deliberately remain outside parity:

```text
analysis frame
object, Light, and Camera transforms
Light energy and color
Camera lens, ortho scale, and clipping values
World color and Background strength
```

This permits animated sequence evaluation while rejecting a plan executed
against a structurally different Scene.

## Closed races

The old validator compared only source ID/type, Scene ID, World ID, Camera ID,
and Light IDs. A caller could therefore change an AO occluder, source render
visibility, World structure, Camera/Light type, or display color management
between planning and execution.

The new gate rejects those changes before reservation and before Blender Scene
mutation. In particular:

```text
plan B3 with AO occluder
-> hide occluder before execution
-> fail on render-visible and shadow-caster set changes

plan B4 under Standard view transform
-> switch to AgX before execution
-> fail on color-management change
```

## Numeric and error contract

Matrix, color, Light, Camera, and color-management reads reject non-finite
values through `SceneBakeAnalysisError`. Negative Light energy retains the
historical clamp to zero. Invalid Camera clipping is reported with the Camera
identity rather than leaking a domain `ValueError`.

Complex or linked World Background graphs remain diagnostic `None`; the adapter
does not mutate or simplify the World node tree.

## Compatibility

`scene_bake_analyzer.py` contains no physical implementation. It retains:

```text
SceneBakeAnalysisError
analyse_object_bake_context
analyse_scene_bake_context
analyse_bake_contexts
validate_runtime_scene_context

_load_bpy
_name
_matrix_tuple
_color_tuple
_animated
_visible_boolean
_object_render_visible
_active_world_output
_input_socket
_background_strength
_analyse_world
_analyse_light
_analyse_camera
_analyse_color_management
```

## Validation

Local focused validation for this slice:

- all split modules compile;
- fifteen focused capture/parity checks pass;
- the physical import graph is acyclic;
- explicit fake Scene and Context paths do not load `bpy`;
- frame and numeric changes remain accepted;
- source visibility, resource structure, Scene sets, and color management fail
  closed;
- compatibility alias identity is retained;
- all split modules are included in the no-`bpy.ops` architecture boundary.

The manual-only Blender 4.4 Scene workflow adds a separate parity process for
B3 AO-object-set and B4 color-management races. It remains `workflow_dispatch`
only and was not run by this change.

The complete pytest suite and real Blender matrices remain separate release
gates.

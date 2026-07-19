# Shader capability audit and production gate

## Purpose

The shader capability layer answers one question before a texture pipeline is selected:

```text
Can the current reconstructed UV-bake target reproduce every reachable Blender value?
```

`blender_adapter/shader_capability_audit.py` classifies immutable renderer-specific graph
snapshots. `blender_adapter/production_shader_capabilities.py` enriches that report with live
Blender facts that are intentionally absent from the domain model, including node mute state,
source UV availability, named UV references, and copied-material proxy boundaries.

Each used node material receives one strongest capability:

```text
LOCAL_UV_SAFE
SCENE_UV_SAFE
CAMERA_RENDER_REQUIRED
GROUP_RENDER_REQUIRED
UNSUPPORTED
```

Unknown reachable node types are deliberately `UNSUPPORTED`. New Blender nodes must receive an
explicit policy before production routing may treat them as safe.

## Physical shader-graph analysis ownership

The former `shader_graph_analyzer.py` mixed Blender RNA compatibility, renderer-specific output
selection, recursive group traversal, semantic classification and immutable snapshot assembly.
Physical ownership is now:

```text
shader_graph_error.py
  -> shared MaterialGraphAnalysisError

shader_graph_rna.py
  -> Blender RNA identity and names
  -> temporary-node filtering
  -> safe node/link/socket iteration
  -> renderer-target normalization
  -> renderer-specific Material Output selection
  -> cross-version group-interface socket matching

shader_graph_traversal.py
  -> RecursiveShaderGraphWalker
  -> muted internal_links bypass traversal
  -> nested Group Input/Output mapping
  -> instance-qualified node IDs
  -> recursive-cycle and maximum-depth handling
  -> frozen ShaderGraphTraversalResult

shader_graph_semantics.py
  -> semantic channels from frozen reachable nodes
  -> material dependencies from frozen reachable nodes and node trees
  -> Principled emission/alpha/reflection/transmission policies

shader_graph_snapshot.py
  -> deterministic node and link ordering
  -> ShaderNodeSnapshot / ShaderLinkSnapshot construction
  -> exact parallel ordering of snapshots and live Blender nodes

shader_graph_analysis.py
  -> renderer-specific analysis orchestration
  -> MaterialGraphAnalysisResult

shader_graph_analyzer.py
  -> compatibility re-exports only
```

`MaterialGraphAnalysisResult.reachable_nodes` remains exactly parallel to
`snapshot.reachable_nodes`. The production capability gate relies on this invariant when it
combines immutable snapshots with live `mute`, UV-map and socket state.

`material_analyzer.py`, `production_shader_capabilities.py` and the public adapter package import
the physical analysis/error/RNA owners directly. Historical public and private imports remain
available from `shader_graph_analyzer.py` without retaining a second implementation.

## Production routing

The production gate is now authoritative during `prepare_a1_object()`:

```text
renderer-specific reachable graph
        -> capability audit
        -> live source/proxy preflight
        |
        +-- LOCAL_UV_SAFE ----------> B1/B2 object UV bake
        +-- SCENE_UV_SAFE ----------> B3 scene-aware Cycles UV bake
        +-- CAMERA_RENDER_REQUIRED -> B4 source-only camera render
        +-- GROUP_RENDER_REQUIRED --> explicit PLAN_BAKE failure
        +-- UNSUPPORTED ------------> explicit PLAN_BAKE failure
```

The domain strategy registry remains Blender-independent. Live `bpy` inspection and capability
routing stay at the Blender adapter boundary.

## Renderer contract

One immutable renderer contract is shared by:

1. the explicit export Scene;
2. renderer-specific Material Output analysis;
3. capability auditing;
4. copied-material proxy output selection;
5. object-bake or camera-render execution.

Supported canonical engines are:

```text
CYCLES             -> ShaderNodeTree target CYCLES
BLENDER_EEVEE_NEXT -> ShaderNodeTree target EEVEE
```

Eevee materials never enter Blender object bake. They use B4 camera rendering. A Scene/execution
engine mismatch is an explicit error rather than a silent Material Output substitution.

Copied-material Alpha, straight-color, and slot-mask proxies temporarily select the exact
renderer-specific Material Output and restore every original `is_active_output` flag in
`finally`.

## Socket-level policies

Node type alone is insufficient for Texture Coordinate and Geometry. The audit uses
`ShaderLinkSnapshot.from_socket` and classifies only outputs that actually contribute to the
effective Material Output.

Current Texture Coordinate policy:

| Output | Capability | Reason |
| --- | --- | --- |
| UV | `LOCAL_UV_SAFE` when source render UV exists | reads the preserved source sampling UV |
| Camera | `CAMERA_RENDER_REQUIRED` | active camera coordinate system |
| Window | `CAMERA_RENDER_REQUIRED` | screen-space coordinate system |
| Reflection | `CAMERA_RENDER_REQUIRED` | view/reflection coordinate system |
| Object | `CAMERA_RENDER_REQUIRED` | original source/reference-object context |
| Generated | `CAMERA_RENDER_REQUIRED` | original undeformed source bounds |
| Normal | `CAMERA_RENDER_REQUIRED` | original source shading context |
| From Instancer | `GROUP_RENDER_REQUIRED` | instance context cannot be reconstructed locally |

Geometry outputs `Incoming`, `Backfacing`, `Pointiness`, and `Random Per Island` are source- or
camera-bound.

## Separate source and bake UV roles

`MeshSnapshot` now stores two independent roles:

```text
active_uv_layer -> Cycles bake destination, normally SpineBakeUV
render_uv_layer -> source shader sampling UV
```

The original render UV survives source/evaluated reading, triangulation, face extraction, UV
correspondence, and temporary Blender mesh materialization. The writer makes `SpineBakeUV`
active for writing while keeping the original layer `active_render` for unlinked Image Texture
Vector inputs and Texture Coordinate UV.

When a graph requires Blender's default render UV but the source mesh has no source render UV,
the whole object is routed to B4. The newly generated `SpineBakeUV` is never silently reused as
source texture coordinates.

Named UV Map, Normal Map, and Tangent references are also checked against live source UV layers.
Missing named layers require B4 instead of producing an incorrect object bake.

## Alpha proxy boundaries

Blender material copies share nested Shader Node Group datablocks. Flattening or rewiring a
nested group during Alpha extraction could therefore mutate the user's original group.

For an Alpha-bearing graph, reachable Group, Reroute, or muted bypass nodes are classified as
`CAMERA_RENDER_REQUIRED`. Native B4 rendering evaluates Blender's group interfaces and
`internal_links` directly without modifying shared node-group data.

## Explicit high-risk families

The production gate currently records:

- Camera Data and Object Info as `CAMERA_RENDER_REQUIRED`;
- Shader to RGB as Eevee B4-only and unsupported for Cycles;
- Attribute and Vertex Color as B4 until generic/color attributes are represented explicitly;
- Particle Info, Hair Info, Curves Info, Point Density, and From Instancer as
  `GROUP_RENDER_REQUIRED`;
- OSL Script as `UNSUPPORTED` until engine/device/source/compilation preflight exists;
- Volume and render displacement as `CAMERA_RENDER_REQUIRED`;
- incomplete recursive graph analysis as `UNSUPPORTED`;
- unclassified reachable node types as `UNSUPPORTED`.

Common local normal/tangent nodes such as Normal Map, Bump, Tangent, and Vector Rotate remain
local-safe when their required source UV exists. Vector Transform and camera-sensitive closure
families are routed to B4.

## B4 render isolation

B4 captures and restores Blender render state. During every source-only projection it now:

- validates execution engine against the analyzed renderer;
- disables `render.use_compositing`;
- disables `render.use_sequencer`;
- validates that the source has at least one direct path through the active View Layer;
- rejects source objects available only through excluded, Holdout, or Indirect Only Layer
  Collections.

This prevents compositor/VSE changes and View Layer matte state from silently changing alpha,
crop, or hull geometry.

## Legacy non-node materials

An opaque material with `use_nodes=False` is represented on an owned copied material as an
explicit Principled Base Color graph. The source material remains unchanged.

A transparent non-node `diffuse_color` is rejected with an actionable error requesting material
nodes, because its opacity cannot currently be represented in the immutable semantic graph and
must not be silently discarded.

## Validation fixtures

Pure tests cover capability precedence, common node families, source UV roles, renderer
selection, copied-material output restoration, production routing, postprocess state, non-node
fallback, View Layer rules and physical shader-graph ownership.

The split-specific checks cover:

- compatibility facade ownership;
- immutable traversal handoff;
- renderer-specific Material Output parity;
- nested and reused group instances;
- unused group-input isolation;
- muted `internal_links` bypass;
- recursive-cycle termination;
- deterministic snapshot/live-node parallel ordering;
- missing-output material-classification fallback.

Manual-only Blender 4.4 fixtures cover:

- real Texture Coordinate, Camera Data, Object Info, Attribute, Particle Info, and Shader to RGB;
- Cycles/Eevee renderer-specific Material Outputs;
- renderer-specific copied-material Alpha proxies;
- Group/Reroute/muted Alpha routing;
- source render UV versus `SpineBakeUV` sampling;
- opaque and transparent non-node materials;
- compositor/sequencer isolation;
- Holdout-only active View Layer rejection.

The workflows remain `workflow_dispatch` only and are not run automatically during active
rewrite development.

## Remaining boundaries

The capability gate deliberately rejects or defers:

- grouped particle/instance/strand rendering;
- OSL execution without a complete preflight;
- arbitrary connected-layer per-pixel depth intersections;
- panoramic projection semantics;
- full generic Geometry Nodes attribute preservation;
- transparent non-node materials without an explicit node graph.

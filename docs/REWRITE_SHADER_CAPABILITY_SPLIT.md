# Shader capability audit decomposition

## Scope

The former `blender_adapter/shader_capability_audit.py` mixed four independent
responsibilities:

1. renderer and dependency policy tables;
2. used-socket indexing and finding construction;
3. Texture Coordinate, Geometry and node-family policy application;
4. graph-level orchestration, deduplication, ordering and strongest-capability
   selection.

The split preserves every public capability, finding code, reason, ordering key
and B1-B4 routing decision.

## Physical ownership

```text
shader_capability_policy.py
  -> immutable renderer/dependency/node-family policy tables
  -> immutable Texture Coordinate and Geometry socket mappings
  -> strict ALL/CYCLES/EEVEE normalization

shader_capability_findings.py
  -> used output-socket index
  -> generic ShaderCapabilityFinding construction
  -> historical deduplication key
  -> deterministic final finding order

shader_capability_node_findings.py
  -> Texture Coordinate socket policy
  -> Geometry socket policy
  -> Shader to RGB renderer policy
  -> OSL/source-attribute/instance/camera/scene/local node policies

shader_capability_analysis.py
  -> graph type and render-target validation
  -> missing-output and graph-issue findings
  -> per-node finding orchestration
  -> camera-over-scene dependency precedence
  -> volume/displacement findings
  -> local fallback
  -> strongest capability and MaterialCapabilityAudit

shader_capability_audit.py
  -> compatibility re-exports only
```

## Compatibility contracts

The facade retains:

```text
_RENDER_TARGETS
_CAMERA_DEPENDENCIES
_SCENE_DEPENDENCIES
_LOCAL_SAFE_NODE_TYPES
_SCENE_NODE_TYPES
_CAMERA_NODE_TYPES
_GROUP_NODE_TYPES
_SOURCE_ATTRIBUTE_NODE_TYPES
_TEXTURE_COORD_CAPABILITIES
_GEOMETRY_OUTPUT_CAPABILITIES
_normalise_render_target
_used_outputs
_finding
_texture_coordinate_findings
_geometry_findings
_node_findings
audit_material_graph_capabilities
```

Existing package and production imports may continue to use the facade without
owning a second implementation.

## Determinism

Finding identity and ordering remain based on:

```text
capability.value
code
node_id
node_type
output_socket
```

Duplicate keys retain the first finding, including its original reason. The
ordered tuple is then used both for the returned audit and for strongest
capability selection.

## Preserved routing semantics

- unknown reachable nodes remain `UNSUPPORTED`;
- OSL remains `UNSUPPORTED` pending complete preflight;
- particle, strand, curve and instancer context remains
  `GROUP_RENDER_REQUIRED`;
- camera/source-dependent nodes remain `CAMERA_RENDER_REQUIRED`;
- scene-aware nodes remain `SCENE_UV_SAFE`;
- audited local nodes remain `LOCAL_UV_SAFE`;
- Texture Coordinate policy remains socket-specific;
- Geometry source-sensitive outputs remain camera-bound;
- camera graph dependencies take precedence over scene dependencies;
- volume and render displacement remain camera-render requirements;
- missing Material Output and graph-analysis issues remain fail-closed.

## Validation

Focused local validation covered:

- compilation of all new modules;
- import-cycle analysis;
- every node-family policy table;
- Texture Coordinate and Geometry socket policies;
- Cycles/Eevee normalization and Shader to RGB routing;
- graph dependency precedence;
- volume/displacement and incomplete-graph handling;
- deterministic deduplication and ordering;
- read-only policy mappings;
- facade alias identity and source ownership boundaries.

The complete repository pytest suite and real Blender integration matrices remain
manual release gates. No GitHub Actions workflow is triggered by this slice.

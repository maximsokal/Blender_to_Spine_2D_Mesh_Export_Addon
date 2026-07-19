# Production shader capability split

## Purpose

The production shader capability gate combines immutable graph analysis with live
Blender facts that are intentionally absent from the domain snapshot. The split
separates graph parity, Alpha-proxy policy, source-UV policy, object-slot auditing
and B1-B4 routing without changing any capability result or texture-plan contract.

## Physical ownership

```text
production_shader_capability_error.py
  -> ProductionShaderCapabilityError

production_shader_capability_merge.py
  -> extend_material_capability_audit()
  -> shared deterministic finding ordering

production_shader_capability_runtime.py
  -> renderer-specific live graph re-analysis
  -> immutable snapshot parity
  -> snapshot/live-node alignment
  -> live mute enrichment

production_shader_capability_proxy.py
  -> Alpha proxy findings
  -> Group/Reroute/muted bypass boundaries

production_shader_capability_uv.py
  -> source UV layer inspection
  -> source render UV selection
  -> Image Texture, Texture Coordinate, Normal Map, Tangent and UV Map findings

production_shader_capability_object_audit.py
  -> object and material-slot orchestration
  -> immutable audit plus live proxy/UV enrichment

production_shader_capability_routing.py
  -> strongest object capability
  -> deterministic failure message
  -> B1-B4 texture-plan routing

production_shader_capabilities.py
  -> compatibility re-exports only
```

`a1_texture_planning.py` imports the physical object-audit and routing owners.
The compatibility facade is not used by the production caller.

## Fail-closed graph parity

Material analysis and production planning are separate stages. A Blender material
may be changed between them by an operator, handler, driver or external add-on.
The live production re-analysis therefore validates:

```text
material name
renderer-specific Material Output
reachable node count and deterministic order
node_id
node_type
node_name
group_path
reachable links
semantic channels
dependencies
analysis issues
```

This is stricter than comparing only Blender node names. Reused node groups may
contain nodes with identical names, so `node_id` plus `group_path` is required to
prevent live `mute`, `uv_map` or `direction_type` from being applied to a different
immutable node instance.

After snapshot parity, the live tuple is checked again against the current
snapshot before mute enrichment. Count, name or node-type mismatches fail before
source-UV inspection.

## Shared finding ordering

Production enrichment no longer implements a second finding sort/deduplication
algorithm. `extend_material_capability_audit()` delegates to
`shader_capability_findings.order_unique_findings()` and preserves the historical
key:

```text
capability.value
code
node_id
node_type
output_socket
```

The first finding for one key retains its original reason.

## Alpha proxy boundary

For an Alpha-bearing graph, the following still require native B4 rendering:

```text
reachable GROUP or REROUTE
muted node internal_links bypass
```

The existing finding codes remain:

```text
ALPHA_PROXY_RECURSIVE_BOUNDARY
ALPHA_PROXY_MUTED_BYPASS
```

## Source UV boundary

The source sampling UV remains separate from the generated `SpineBakeUV` bake
destination. Live inspection still records:

```text
NAMED_NORMAL_UV_MISSING
NAMED_TANGENT_UV_MISSING
NAMED_UV_MISSING
SOURCE_RENDER_UV_MISSING
```

More than one `active_render` source UV layer is rejected explicitly. A live-node
count mismatch is rejected before UV collection or socket inspection.

## Object audit flow

```text
validate MESH object and ObjectMaterialAnalysis
-> resolve renderer contract
-> validate material-slot count
-> re-analyze each live material
-> validate immutable graph parity
-> enrich live mute state
-> run immutable shader capability audit
-> apply Alpha proxy boundary
-> apply source UV boundary
-> return audits in material-slot order
```

The object-audit owner never builds a texture plan.

## Routing matrix

```text
UNSUPPORTED
  -> BakePlanError

GROUP_RENDER_REQUIRED
  -> BakePlanError

BLENDER_EEVEE_NEXT
  -> B4 camera projection

CAMERA_RENDER_REQUIRED
  -> B4 camera projection

LOCAL_UV_SAFE / SCENE_UV_SAFE under Cycles
  -> object bake strategy planning
```

The routing owner performs no Blender node-tree, socket or UV inspection.

## Compatibility

Historical imports remain available from `production_shader_capabilities.py`,
including:

```text
ProductionShaderCapabilityError
_enriched_graph_with_live_mute
_rebuild_audit
_with_proxy_boundary
_source_uv_layers
_source_render_uv_name
_input_socket
_graph_uses_texture_coordinate_uv
_with_source_uv_boundary
audit_object_material_capabilities
strongest_object_capability
capability_failure_message
build_capability_checked_texture_plan
```

## Validation

Focused ordinary-Python validation covers:

- facade-only ownership;
- acyclic physical import graph;
- equal-name recursive group instance swaps;
- node type, link, channel, dependency and issue mismatches;
- live-node count and alignment;
- shared finding ordering and first-reason retention;
- Alpha proxy finding codes;
- named Normal Map, Tangent and UV Map findings;
- multiple active render UV rejection;
- production caller imports of physical owners;
- absence of `bpy.ops` access in all split modules.

No GitHub Actions workflow is triggered by this slice. The complete pytest suite,
manual Blender 4.4 matrices and private production release gate remain separate
release requirements.

# Material analysis physical ownership

## Purpose

The Blender material adapter previously combined several unrelated responsibilities in
`material_analyzer.py`:

- Blender RNA-compatible names, node types and renderer target resolution;
- Image datablock inspection;
- legacy `MaterialKind` classification;
- recursive reachable shader-graph resolution;
- missing-output compatibility fallback;
- material-slot and object-slot orchestration;
- public and historical private compatibility names.

Those responsibilities now have separate physical owners. The public material-analysis contract,
legacy classification behavior and production B1-B4 routing are unchanged.

## Physical modules

```text
material_analysis_error.py
  -> MaterialAnalysisError

material_analysis_rna.py
  -> material/object names
  -> node type and temporary-node compatibility
  -> root node and dense material-slot freezing
  -> renderer-target resolution

material_node_classification.py
  -> procedural node policy
  -> ImageDependency extraction
  -> deterministic dependency deduplication and ordering
  -> typed MaterialNodeClassification
  -> legacy four-item tuple adapter

material_graph_resolution.py
  -> recursive renderer-specific graph analysis
  -> effective reachable live-node selection
  -> missing-output root-node compatibility fallback
  -> graph-analysis diagnostics

material_slot_analysis.py
  -> one MaterialAnalysis construction
  -> empty and non-node material handling
  -> graph/classification issue merge

material_object_analysis.py
  -> Blender MESH validation
  -> dense material-slot traversal
  -> source object ID ownership
  -> ObjectMaterialAnalysis construction

material_analyzer.py
  -> compatibility re-exports only
```

## Physical flow

```text
analyse_object_materials
  -> validate Blender MESH object and object name
  -> resolve renderer target once
  -> freeze material slots once
  -> analyse_material_slot in dense slot order
       -> handle empty slot
       -> handle non-node diffuse-color fallback
       -> freeze root nodes
       -> resolve effective recursive graph
       -> select reachable nodes or root fallback
       -> classify node/image dependencies
       -> merge classification issues before graph issues
       -> build MaterialAnalysis
  -> build ObjectMaterialAnalysis
```

`a1_texture_planning.py` imports `material_object_analysis.analyse_object_materials` directly.
The public `blender_adapter` package exports `MaterialAnalysisError`, `analyse_material_slot` and
`analyse_object_materials` from their physical owners. Runtime production code does not depend on
the compatibility facade.

## Graph versus legacy fallback

An effective renderer-specific graph uses the exact reachable live-node tuple returned by
`shader_graph_analysis.analyse_material_graph_detailed()`. Unreachable editor nodes do not affect
legacy kind or image dependency classification.

The historical root-node fallback remains active when:

```text
active_output_node_id is None
and semantic_channels is empty
```

This preserves damaged/synthetic material behavior: orphaned group content is diagnostic data,
not an active material program. If recursive analysis raises `MaterialGraphAnalysisError`, root
nodes are also used and the issue remains:

```text
Shader graph analysis failed: <message>
```

Temporary bake/proxy nodes remain excluded from fallback classification.

## Legacy classification matrix

```text
invalid reachable Image Texture reference
  -> UNSUPPORTED

image dependency plus procedural node
  -> MIXED

image dependency only
  -> IMAGE

procedural node only
  -> PROCEDURAL

otherwise
  -> SOLID_COLOR
```

Empty slots remain `EMPTY`. Materials without a node tree remain `SOLID_COLOR` with the historical
issue:

```text
Material has no node tree; diffuse_color fallback is required
```

Muted nodes and temporary bake/proxy nodes do not change classification.

## Deterministic ImageDependency ordering

The domain permits `ImageDependency.filepath` to be `None`. The previous implementation sorted raw
keys containing the optional path. Two otherwise-equal dependencies with `None` and a string path
could therefore make Python compare `None` with `str` and raise `TypeError`.

Deduplication retains the historical key:

```text
image_name
source
filepath
frame_duration
```

Ordering now uses a separate total-order key:

```text
casefolded and original image name
casefolded and original source
None/string path discriminator
casefolded and original path, with None represented safely
frame duration
generated flag
```

The ordinary order remains deterministic, but every value allowed by the domain model can now be
sorted. Dependency order no longer depends on Blender node iteration order.

## Compatibility facade

`material_analyzer.py` retains:

```text
MaterialAnalysisError
analyse_material_slot
analyse_object_materials
render_target_from_engine

_PROCEDURAL_NODE_TYPES
_material_name
_node_type
_is_temporary_bake_node
_normalise_render_target
_resolve_render_target
_image_dependency
_classify_nodes
```

The private `_classify_nodes()` adapter still returns the historical four-item tuple:

```text
(kind, node_types, image_dependencies, issues)
```

It does not expose `MaterialNodeClassification` to old callers.

## Validation

Focused local validation covers:

- facade-only ownership;
- physical package and A1 caller imports;
- no direct `bpy.ops` access in any material-analysis module;
- optional `None`/string image filepath ordering;
- dependency deduplication and node-order independence;
- muted and temporary-node filtering;
- effective reachable graph selection;
- missing-output root fallback;
- recursive graph-analysis error fallback;
- classification-before-graph issue ordering and deduplication;
- dense slot order and source object ID override;
- material-slot collection failure wrapping;
- historical facade tuple and alias identity.

The complete repository pytest suite and real Blender 4.4 matrices remain separate manual release
gates. GitHub Actions remain manual-only and were not triggered by this slice.

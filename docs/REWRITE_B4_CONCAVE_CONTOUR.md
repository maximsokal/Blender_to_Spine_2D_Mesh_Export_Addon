# B4 simplified concave screen-space contour

## Purpose

B4 no longer requires every rendered object to use only a convex screen-space hull.
The sequence alpha union can now produce one deterministic simple concave outer contour,
which is triangulated into a valid Spine mesh.

The implementation remains Blender-independent after the staged render has been decoded.
No `bpy` or `bmesh` access is used during contour extraction or triangulation.

## Execution policy

The immutable output policy lives in `BakeExecutionSettings`:

```python
BakeExecutionSettings(
    projection_contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    projection_contour_simplify_tolerance_pixels=1.0,
)
```

Available modes:

- `SIMPLIFIED_CONCAVE` — the production default;
- `CONVEX_HULL` — explicit compatibility mode matching the previous geometry policy.

The simplify tolerance must be a finite non-negative number. Booleans, NaN,
infinities, strings, `None`, and negative values are rejected before rendering.

## Boundary extraction

Every visible union-mask pixel contributes only boundary edges whose neighboring pixel is
transparent or outside the frame. Edges are oriented so visible coverage remains on the
left side.

The tracer follows deterministic turn priority:

1. left;
2. straight;
3. right;
4. reverse.

This resolves diagonal pixel contacts as separate loops instead of accidentally joining
them through one shared corner.

Outer loops have positive signed area. Hole loops have negative signed area.

## Holes

The Spine attachment uses one outer polygon. Internal holes are deliberately not converted
into polygon holes. They remain represented by transparent texture pixels.

This keeps the mesh topology simple while preserving the rendered result.

## Multiple alpha islands

One simple polygon cannot represent several disconnected visible components without adding
an artificial bridge. Therefore:

```text
one outer component
    -> simplified concave contour

multiple outer components
    -> deterministic convex fallback
```

The fallback never clips visible pixels. `CameraProjectionLayout` records:

- `contour_mode`;
- `outer_component_count`;
- `contour_fallback_reason`;
- `source_contour_vertex_count`;
- final contour vertex count;
- simplification tolerance.

## Conservative simplification

The simplifier first removes exact collinear points.

It then removes only shallow reflex vertices. For a counter-clockwise polygon, removing a
reflex vertex fills a transparent notch rather than cutting into the visible region.
Convex corners are never removed.

A candidate is accepted only when:

- its distance from the replacement chord is within the configured tolerance;
- the replacement chord does not intersect a non-adjacent contour edge;
- the resulting contour remains simple and counter-clockwise;
- deterministic ear clipping can triangulate it exactly.

The policy therefore allows a small amount of transparent overdraw but does not remove
alpha-bearing coverage.

## Triangulation

Convex contours retain the historical deterministic triangle fan.

Concave contours use deterministic ear clipping. Every output is validated:

```text
triangle count = contour vertex count - 2
all triangle signed areas > 0
sum(triangle signed areas) = contour signed area
```

Duplicate points, collinear consecutive points, clockwise contours, self-intersections,
degenerate ears, and incomplete triangulations are rejected.

## Compatibility surface

`CameraProjectionLayout.hull` remains available as the compatibility field name. It now
contains the selected outer contour. New code may use the `contour` property.

Existing full-frame layouts remain four-vertex convex quads. Existing UV and screen-position
formulas are unchanged.

`build_camera_projection_mesh_snapshot()` consumes `layout.triangle_indices`, so the
application layer does not assume that every contour is a fan.

## Validation

Focused tests cover:

- an exact L-shaped concave contour;
- conservative removal of a one-pixel reflex notch;
- preservation of a deeper concavity;
- internal transparent holes;
- diagonal contacts and disconnected component fallback;
- explicit convex compatibility mode;
- exact-area triangulation;
- arbitrary concave `_edge_pairs` topology;
- a complete concave `MeshSnapshot` with UV and face-index parity;
- 250 deterministic randomized binary masks.

## Remaining boundary

Disconnected components still use one convex fallback mesh. Supporting several independent
polygon components inside one Spine attachment would require a generalized layout containing
multiple contours and explicit triangle groups. That extension is separate from the current
safe single-contour contract.

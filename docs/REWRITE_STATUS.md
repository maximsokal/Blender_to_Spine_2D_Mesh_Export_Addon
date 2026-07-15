# Staged rewrite status

## v0.24 resource-safety foundation

The public package keeps its existing import surface while export orchestration
moves out of `main.py` into `pipeline_v2.py`.  `__init__.py` installs the new
implementations before UI and multi-object modules import the legacy API.

This first slice intentionally does not change the Spine 4.2 JSON schema,
object/slot/bone naming, segmentation algorithms, UV correspondence format or
texture bake output.

### Corrected ownership rules

- A BMesh created by `bmesh.new()` has one owner and one `free()` call.
- `collect_vertices()` borrows a BMesh and never frees it.
- `_export_segment()` remains the owner of BMeshes returned by
  `triangulate_mesh()`.
- The discarded pre-processing triangulation call is removed; export
  triangulation remains local to the BMeshes owned by `_export_segment()`.
- Source/target face-ID transfer and segmentation seam application use managed
  BMeshes with guaranteed cleanup.

### Corrected Blender state rules

- The active object, selection and mode are captured once per export and
  restored in `finally`.
- Transform application activates exactly the requested object.
- Temporary objects are tracked by identity and removed through
  `bpy.data.objects.remove(..., do_unlink=True)`.
- Logging level no longer changes cleanup behaviour.  Temporary objects are
  preserved only when the scene custom property
  `spine2d_preserve_debug_artifacts` is explicitly true.
- A temporary source UV layer and `face_id_map` metadata are restored after the
  export.

## Next slices

1. Move `_export_segment` and mesh correspondence into a geometry/export
   service with typed result objects.
2. Split segmentation and UV operations from Blender operators so pure
   topology logic can be tested without `bpy`.
3. Replace global texture dimensions with immutable export settings.
4. Extract baking into a context-managed bake session and support mixed shader
   graphs.
5. Add Blender 4.4 headless integration fixtures for mode restoration,
   temporary datablock cleanup, triangulation and bake context.
6. Remove the compatibility installer after UI and multi-object export import
   the new services directly.

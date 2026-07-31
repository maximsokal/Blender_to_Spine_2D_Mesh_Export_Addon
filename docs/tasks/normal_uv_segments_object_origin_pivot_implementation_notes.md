# Object Origin pivot implementation notes

This note supplements `normal_uv_segments_object_origin_pivot.md` with one setup-pose dependency discovered during implementation.

## Single-object setup-pose route

The previous public single-object route always selected:

```text
A1RigSetupPoseMode.NORMALIZED_SINGLE
```

For `TWO_AXIS_ROTATION_SCALE`, that mode deliberately wrote `(0, 0)` to `<prefix>_main` and moved the captured object placement to the internal base bone. That behavior conflicts with the approved requirement that `<prefix>_main` itself represent Blender Object Origin.

The approved route-specific behavior is therefore:

```text
Normal / UV Segments + TWO_AXIS_ROTATION_SCALE
    -> PRESERVE_COMPOSITION

Camera Projection
    -> NORMALIZED_SINGLE
```

Public standalone multi-object export already uses `PRESERVE_COMPOSITION` for every component.

This is not a new scope expansion. It is required to satisfy acceptance criterion 1 of the approved task: `<prefix>_main` must correspond to Blender Object Origin.

## Public rig UI

Only `TWO_AXIS_ROTATION_SCALE` is exposed. `THREE_AXIS_ROTATION` remains available to internal builders and tests but is not offered by the public EnumProperty or public rig description panel.

Persisted historical `THREE_AXIS_ROTATION` values are normalized to `TWO_AXIS_ROTATION_SCALE` at public scene capture so old `.blend` data cannot silently re-enable the hidden profile.

## Version

The implementation release is `0.55.0`. Any user-visible correction after this release must increment the manifest version again.

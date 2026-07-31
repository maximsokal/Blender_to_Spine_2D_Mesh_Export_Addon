# Object Origin pivot implementation notes

This note supplements `normal_uv_segments_object_origin_pivot.md` with setup-pose and persisted-RNA dependencies discovered during implementation.

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

## Public rig UI and persisted `.blend` compatibility

Only `TWO_AXIS_ROTATION_SCALE` is shown by the public rig panel. The panel does not draw the persisted rig EnumProperty as a selector.

`THREE_AXIS_ROTATION` remains in the hidden RNA enum solely so Blender can bind and migrate historical Scene ID-properties without rejecting an old `.blend` file. It also remains available to internal builders and tests.

Public scene capture normalizes a persisted `THREE_AXIS_ROTATION` value to `TWO_AXIS_ROTATION_SCALE`, so the hidden compatibility value cannot silently re-enable Three Axis export.

## Version

The implementation release is `0.55.0`. Any user-visible correction after this release must increment the manifest version again.

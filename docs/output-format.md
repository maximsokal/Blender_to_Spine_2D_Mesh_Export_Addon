# Output Format

This document describes the current output contract for Blender to Spine2D Mesh Exporter
**0.150.0**.

## Spine targets

The exporter builds a canonical typed Spine document and adapts it to the selected target.
Supported standalone target metadata versions are:

```text
3.8.99
4.0.64
4.1.24
4.2.43
4.3.23
```

Target adaptation may change version-specific bone indices, sequence representation, or
other schema details while preserving the validated geometry/rig semantics.

## Output directories

The JSON setting selects the final output directory. `Images Subfolder` selects a relative
directory below it.

Example:

```text
JSON directory:   D:/project/export
Images Subfolder: images/
```

Typical result:

```text
D:/project/export/Hero_merged.json
D:/project/export/images/Hero_Baked.png
```

Texture resolution is controlled by the Scene-level `Texture size` setting in the **Bake**
foldout. Its UI location does not change output naming or path semantics.

## Naming

Single-object JSON:

```text
<ObjectName>_merged.json
```

Static texture:

```text
<stem>_Baked.png
```

Sequence textures:

```text
<stem>_Baked_0000.png
<stem>_Baked_0001.png
...
```

Multi-object JSON uses the first ordered source name plus the number of additional selected
objects.

Output stems are sanitized for ordinary Windows filesystem restrictions. Invalid filename
characters and reserved device-name collisions are normalized or rejected before staging.

## Normal / UV Segments attachments

Each final manifold region becomes a Spine mesh attachment containing:

- UVs;
- triangle indices;
- physical hull count;
- optional edges;
- target-relative texture path;
- weighted vertex stream;
- dimensions;
- optional sequence metadata.

UV identity is loop-aware. One geometric source vertex can therefore produce more than one
attachment vertex when UV seams require distinct UV values.

Setup-degenerate side geometry may remain present because deformable rig controls can make
it visible later.

## Generated vertex-bone weights

Before optional sharing, every attachment vertex owns one generated Spine bone and one
full-weight influence:

```text
influence count = 1
local x = 0
local y = 0
weight = 1
```

The generated bone owns the exported setup XY position.

### Signed-axis Normal

Generated vertex bones are parented to the matching depth rotation bone.

### Active Camera — Object Root Bone

Generated vertex bones are parented to the matching generated
`<depth-bone>_camera_setup` inverse child.

For each depth group:

```text
depth scale bone
-> depth rotation bone
-> camera setup inverse bone
-> generated vertex bone
```

The depth translation and inverse setup translation cancel in setup pose, so attachment
world XY remains the camera-projected XY. Live depth ownership remains in the ancestors for
later X/Y control deformation.

### Active Camera — Camera Root Bone

All generated vertex bones are parented below the object base under one rigid
camera-relative depth layer. The exported main bone represents camera-space zero.

## Shared generated vertex bones

Equivalent generated vertex bones can be compacted across segmented attachments of the
same object when their complete setup semantics match. Parent identity is part of the
comparison.

Compaction changes only weighted bone indices and generated-bone inventory. It does not
change:

- UV values;
- triangle order;
- hull;
- edges;
- local influence X/Y;
- influence weight;
- attachment texture path.

## Camera Projection attachment

Camera Projection produces a flat screen-space mesh from the active camera render.

The attachment is based on:

- usable alpha coverage;
- stable crop across sequence frames;
- contour simplification/fallback;
- deterministic triangulation;
- crop-local UV coordinates.

It is a separate representation and does not reuse Normal region attachments.

## Depth Camera Projection attachments

Depth Camera Projection emits weighted relief attachments.

At `Parallax Horizon Angle = 0°`, the object has the FRONT representation only.

With positive parallax:

- retained surfaces share one union geometry/rig;
- every non-empty reserve view receives its own texture namespace and attachment;
- each view owns an independent stable crop;
- reserve slots are serialized before FRONT;
- sequence frame tasks are shared while texture/crop ownership stays per view.

## Sequence encoding

Spine 3.8 and 4.0 use the supported attachment-swap representation.

Spine 4.1, 4.2, and 4.3 use native sequence metadata/timelines where supported by the
selected target contract.

A static object does not receive sequence metadata merely because another object in the
same export request is animated.

## Bones, slots, constraints, and skins

Before serialization the document validates:

- unique bone names and valid parents;
- slot-to-bone references;
- skin/attachment references;
- IK and Transform constraint references/order;
- weighted mesh bone indices;
- finite numeric payloads;
- target-specific sequence data;
- generated control references.

## Texture-space contract

Normal / UV semantic bake images are saved in the file-space orientation expected by the
exported Spine UV values.

Rendered-camera modes remap full-frame camera UV into the final crop without changing the
validated attachment topology unexpectedly.

## Atomic output

Export stages candidate files before installation. Temporary transaction files can look
like:

```text
.spine2d-stage-*
.spine2d-backup-*
```

They are not final Spine assets.

The transaction is responsible for:

1. deterministic output reservation;
2. complete staged JSON/textures before installation;
3. backup of replaced finals when required;
4. rollback/restoration after partial failure;
5. stale work recovery;
6. avoiding work owned by another live process.

## Related documents

- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Rig Profiles](rig-profiles.md)
- [Architecture](architecture.md)
- [Troubleshooting](troubleshooting.md)

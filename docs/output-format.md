# Output Format

This document describes the current output contract for Blender to Spine2D Mesh Exporter
**0.152.0**.

## Spine targets and exact project versions

The exporter builds a canonical typed Spine document and adapts it through the selected
schema-family codec. Supported standalone families and built-in default exact versions are:

```text
3.8 -> 3.8.99
4.0 -> 4.0.64
4.1 -> 4.1.24
4.2 -> 4.2.43
4.3 -> 4.3.23
```

The exact project patch is a separate Add-on Preference. A canonical same-family value such
as `4.2.35` still uses the 4.2 codec. The effective exact value is written to
`skeleton.spine` and included in the versioned JSON filename. Target adaptation may change
version-specific bone indices, sequence representation, or other schema details while
preserving validated geometry/rig semantics.

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
D:/project/export/Hero_merged_spine_4.2.35.json
D:/project/export/images/Hero_Baked.png
```

Texture resolution is controlled by the Scene-level `Texture size` setting in the **Bake**
foldout. Its UI location does not change output path semantics.

## Naming

Single-object JSON:

```text
<ObjectName>_merged_spine_<exact-version>.json
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
objects and the effective exact version. Output stems are sanitized for ordinary Windows
filesystem restrictions.

## Normal / UV Segments attachments

Each final manifold region becomes a Spine mesh attachment containing UVs, triangle indices,
physical hull count, optional edges, target-relative texture path, weighted vertex stream,
dimensions, and optional sequence metadata. UV identity is loop-aware, so one source vertex
can produce several attachment vertices when UV seams require distinct values.

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
`<depth-bone>_camera_setup` inverse child. The depth translation and inverse setup
translation cancel in setup pose while live depth ownership remains available for later X/Y
deformation.

### Active Camera — Camera Root Bone

All generated vertex bones are parented below the object base under one rigid
camera-relative depth layer. The exported main bone represents camera-space zero.

## Shared generated vertex bones

Equivalent generated vertex bones can be compacted across segmented attachments when their
complete setup semantics match. Parent identity is part of the comparison. Compaction changes
only weighted bone indices/generated-bone inventory, not UVs, triangles, hull, edges, local
influence coordinates, weight, or texture path.

## Camera Projection attachment

Camera Projection produces a flat screen-space mesh from the active camera render using
usable alpha coverage, stable crop, contour construction and deterministic triangulation.
It does not reuse Normal region attachments.

## Depth Camera Projection attachments

Depth Camera Projection emits weighted relief attachments. At
`Parallax Horizon Angle = 0°` the object has FRONT only. With positive parallax, retained
surfaces share one union geometry/rig, each non-empty reserve view receives its own texture
namespace/attachment/crop, and reserve slots are serialized before FRONT.

## Sequence encoding

Spine 3.8 and 4.0 use the supported attachment-swap representation. Spine 4.1, 4.2 and 4.3
use native sequence metadata/timelines where supported by the selected family contract. A
custom exact patch never changes the family's sequence encoding policy.

## Bones, slots, constraints, and skins

Before serialization the document validates unique bone names/parents, slot-to-bone
references, skin/attachment references, IK/Transform constraint references/order, weighted
mesh indices, finite numeric payloads, target-specific sequence data, and generated control
references.

## Texture-space contract

Normal / UV semantic bake images are saved in the file-space orientation expected by the
exported Spine UV values. Rendered-camera modes remap full-frame camera UV into the final
crop without changing validated attachment topology unexpectedly.

## Atomic output

Export stages candidate files before installation. Temporary transaction files such as
`.spine2d-stage-*` and `.spine2d-backup-*` are not final Spine assets. The transaction owns
deterministic reservation, staged JSON/textures, backup, rollback/restoration, stale-work
recovery and live-process ownership checks.

## Related documents

- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Rig Profiles](rig-profiles.md)
- [Architecture](architecture.md)
- [Troubleshooting](troubleshooting.md)

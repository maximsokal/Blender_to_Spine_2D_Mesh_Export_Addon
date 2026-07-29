# Output Format

## Compatibility target

The current exporter builds Spine JSON for the Spine 4.2.43 compatibility profile. Generated documents contain typed and validated skeleton, bone, slot, skin, attachment, constraint, and optional animation data before serialization.

The exporter does not guarantee byte-for-byte equality between runs when volatile metadata or external Blender rendering results change. Stable geometry, UV, rig, naming, ordering, and cross-reference contracts are validated independently.

## Output directories

The JSON setting selects the final output directory. Images Subfolder selects a relative directory below it.

Example:

```text
JSON directory:       D:/project/export
Images Subfolder:     images/
```

Result:

```text
D:/project/export/<name>.json
D:/project/export/images/<texture files>
```

The JSON stores texture paths relative to the configured output relationship.

## Single-object naming

For an object named `Hero`, the UI derives:

```text
JSON stem: Hero_merged
Texture stem: Hero
```

Typical static output:

```text
Hero_merged.json
images/Hero_Baked.png
```

The exact texture extension follows the resolved texture format policy.

## Multi-object naming

The output stem uses the first ordered selected object name and the number of additional selected objects.

For four ordered selected objects beginning with `Body`:

```text
Body_plus_3_objects.json
```

Associated texture stems remain deterministic for the individual or grouped texture plans that participate in the request.

## Filename sanitization

Output stems are sanitized for ordinary Windows file APIs:

- `< > : " / \ | ? *` and ASCII control characters become `_`;
- surrounding whitespace is removed;
- trailing spaces and periods are removed;
- empty sanitized stems are rejected;
- reserved DOS device names such as `CON`, `NUL`, `COM1`, and `LPT1` receive a safe suffix.

Output namespace preflight rejects collisions using case-insensitive Windows path identity even when tests run on another operating system.

## Static textures

A static bake uses:

```text
<stem>_Baked.<extension>
```

Examples:

```text
Hero_Baked.png
Projection_Baked.png
```

## Texture sequences

A sequence uses:

```text
<stem>_Baked_<frame>.<extension>
```

Frame numbers are zero padded according to the configured sequence digit contract.

Example:

```text
Hero_Baked_0000.png
Hero_Baked_0001.png
Hero_Baked_0002.png
```

The output frame number is `sequence_start_frame + task_index`.

## Normal - UV Segments attachments

Normal mode exports one or more mesh attachments derived from final manifold disk regions.

Each attachment contains:

- ordered UV coordinates;
- triangle indices;
- physical Spine hull size;
- weighted vertex data;
- texture path metadata;
- deterministic region and source lineage relationships used during assembly.

The physical `hull` prefix describes the convex hull of final attachment XY positions. Topological boundary order is remapped when required so Spine receives a valid physical hull.

The exported UVs retain the generated bake layout values. Before saving the semantic bake image, Blender pixel rows are converted to the top-down file-space orientation expected by those Spine UVs.

### Shared segment vertex bones

Segmentation can repeat one physical source point in several independent attachments. Version 0.47.0 keeps every attachment vertex and its own UV entry, but equivalent generated vertex bones are shared when all of the following match:

- the segments belong to the same exported object;
- the generated bones have the same Z-parent;
- their final serialized setup X and Y values are identical;
- their remaining setup properties are identical;
- removing the duplicate name cannot break a slot, constraint, child, skin, or animation reference.

Only weighted `boneIndex` values are compacted. The exporter does not alter attachment UVs, triangles, hull, edges, texture paths, local influence X/Y, or influence weights. Coincident XY points under different Z parents remain separate because they participate in different depth deformation.

## Camera Projection attachment

Camera Projection produces one screen-space mesh attachment for the selected render group.

The attachment is derived from:

- active camera render coverage;
- stable sequence crop;
- simplified concave contour or deterministic convex fallback;
- exact triangulation;
- cropped output dimensions and UV layout.

The source region attachments used by Normal mode are not substituted silently. Camera Projection is a separate explicit output contract.

## Connected and mixed output

A connected subgroup shares the connected rig and composition contract. Standalone sources retain independent component rigs.

Mixed output combines:

```text
connected subgroup
+ standalone components
-> one final Spine document
```

The final transaction owns all JSON and texture paths for the complete request.

## Bones, slots, and constraints

The compatibility profile preserves deterministic ordering and cross-references for:

- bones and parents;
- slots and slot-to-bone references;
- skins and attachment paths;
- IK and transform constraints;
- weighted mesh bone indices;
- optional control icons;
- optional preview animation.

Every serialized document is validated for finite numeric values, valid indices, valid names, and legal cross-references before commit.

## Generated material output

Generated patterns affect only temporary bake appearance. They do not change source materials.

- Solid Gray produces one opaque selected RGB value.
- One Region - One Color produces deterministic region colors.
- One Polygon - One Color produces deterministic final-triangle colors.

Generated output alpha is always 1.0.

## Atomic transaction files

During export, the output directory may temporarily contain files similar to:

```text
.spine2d-stage-*
.spine2d-backup-*
```

These are not final assets.

- Stage files hold complete candidate output before installation.
- Backup files protect existing finals during replacement.
- Successful commit installs every reserved final and removes obsolete work files.
- Failure restores previous finals when possible and removes or preserves stages according to preferences.
- A later export can recover stale work left by a hard process interruption.

Do not import stage or backup files into Spine.

## Staged texture validation

Normal mode validates staged textures before final commit. The validation checks that exported triangle samples:

- use finite UV coordinates inside the unit square;
- map through the saved Spine file-space orientation;
- reach non-empty alpha coverage in the staged image.

Directional Blender headless regression tests additionally verify that geometry vertices, JSON UV corners, and asymmetric baked image regions correspond correctly.

## Related documents

- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Architecture](architecture.md)
- [Troubleshooting](troubleshooting.md)

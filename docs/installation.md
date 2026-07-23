# Installation Guide

## Requirements

- Blender 5.2 or newer. Blender 4.x and Blender 5.0/5.1 are not supported.
- Windows is the currently tested desktop platform.
- 4 GB RAM minimum; 8 GB or more is recommended for complex meshes and baking.
- A writable output directory for Spine JSON, textures, staging files, diagnostics, and logs.

The minimum version is declared in
`Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml` and is checked again
before registration mutates Blender state.

## Install a release ZIP

1. Open **Edit > Preferences > Extensions** in Blender 5.2 or newer.
2. Open the Extensions menu and choose **Install from Disk**.
3. Select the release ZIP.
4. Enable **Blender to Spine2D Mesh Exporter**.
5. Open the 3D View sidebar and locate the exporter panel.

Do not unpack the release ZIP. Its root must contain `blender_manifest.toml` and
`__init__.py`.

## Build locally

The repository uses Blender's official extension validator and builder. The old
file-selection cleaner and `fake-bpy-module-4.1` dependency are no longer used.

From the repository root:

```text
python tools/prepare_package.py --blender <path-to-Blender-5.2-executable>
```

The script:

1. verifies the Blender runtime is 5.2 or newer;
2. validates the extension source and manifest;
3. invokes `blender --command extension build`;
4. verifies that a non-empty ZIP was created in `dist`.

Optional arguments:

```text
--source-dir <directory-containing-__init__.py-and-blender_manifest.toml>
--output <output-archive.zip>
```

The executable can also be supplied through `BLENDER_EXECUTABLE` or found as
`blender` on `PATH`.

## Validate manually

Validate the source directory:

```text
blender --command extension validate Blender_to_Spine2D_Mesh_Exporter
```

Validate a built ZIP:

```text
blender --command extension validate <archive.zip>
```

## Troubleshooting

### Incompatible extension

Upgrade to Blender 5.2 or newer. This branch intentionally has no Blender 4.x
compatibility path.

### Registration failure

Open Blender's system console and inspect the complete traceback. Registration
is transactional: completed classes and Scene properties are rolled back when a
later step fails.

### Export failure

Confirm that:

- the destination is writable;
- the selected object is a supported mesh;
- the active renderer is Cycles or Blender 5.2 EEVEE;
- source materials expose valid node trees;
- enough disk space exists for temporary and final textures.

### Remove the extension

Remove it from **Preferences > Extensions**. The add-on does not invoke legacy
`addon_disable` or `addon_remove` operators.

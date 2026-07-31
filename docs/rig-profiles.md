# Rig profiles implementation and reference

## Supported profiles

The exporter exposes rig generation as an explicit Scene-level category independent from texture, cutting, and baking settings.

- `LEGACY_ROTATABLE_MESH` — the existing three-axis rotation rig. Its builder and serialized output remain available for compatibility.
- `TWO_AXIS_ROTATION_SCALE` — X/Y pseudo-rotation with an independent uniform scale control, generalized from the complete Spine 4.2.43 reference embedded below.

A genuinely fresh Blender Scene defaults to `TWO_AXIS_ROTATION_SCALE`. Existing saved projects are not silently changed: pre-profile projects migrate to the compatibility three-axis rig, while schema-4 projects preserve the profile already selected by the user.

## UI order

All sections use the same main-panel foldout style in this exact order:

```text
Paths and Spine 2D version
Rig
Rewrite Generated Materials
Cut
Bake
Analysis
```

The Rig foldout contains the profile selector, control-icon toggle, preview-animation toggle, and a profile-specific description. Changing the profile invalidates cached readiness and schedules a new analysis. The dedicated Rig reset restores `TWO_AXIS_ROTATION_SCALE`.

## Immutable settings pipeline

The selected profile and setup-pose policy travel through the complete immutable path:

```text
Scene RNA
-> _SceneExportProfile
-> ExportSettings
-> A1SingleObjectExportSettings
-> LegacyRigBuildRequest
-> rig builder router
-> document assembly
-> Spine JSON
```

Single-object and multi-object exports use one Scene-level profile. Mixed profiles inside one document are rejected.

The setup-pose policy is separate from the selected profile:

- `NORMALIZED_SINGLE` is used only when one object owns the document.
- `PRESERVE_COMPOSITION` is used by every multi-object source.

The policy is explicit typed data. It is never inferred from object names, prefixes, or coordinate values.

## Builder architecture

The legacy bone and constraint builders remain unchanged. An explicit router selects the physical owner:

- `LEGACY_ROTATABLE_MESH -> build_legacy_rig`
- `TWO_AXIS_ROTATION_SCALE -> build_two_axis_scale_rig`

The two-axis implementation is split into contracts, plan, bones, constraints, validation, and assembly modules. It reuses the stable attachment contract: one generated vertex bone per exported mesh vertex, parented to its resolved Z-group rotation bone, with one full-weight influence. Reference bones `TOP1..TOP4` and `BOTTOM5..BOTTOM8` therefore generalize to existing per-vertex bones and are not hard-coded.

## Two-axis hierarchy

```text
root
├── <prefix>_main
│   ├── <prefix>
│   │   ├── <prefix>_scale_rotate_X
│   │   │   └── <prefix>_rotate_X
│   │   │       ├── <prefix>_1_scale
│   │   │       │   └── <prefix>_1
│   │   │       ├── <prefix>_2_scale
│   │   │       │   └── <prefix>_2
│   │   │       └── ... one pair per Z group
│   │   ├── <prefix>_rotate_X_constraint
│   │   └── <prefix>_rotate_X_constraint_scale_IK
│   │       └── <prefix>_rotate_X_constraint_rotate_IK
│   │           └── <prefix>_rotate_X_constraint_IK
│   ├── <prefix>_rotation_X
│   └── <prefix>_rotation_Y
└── <prefix>_scale
```

## Neutral rotation controls

Both setup-pose policies create neutral visible rotation controls:

```text
<prefix>_main.x = 0
<prefix>_main.y = 0
<prefix>_rotation_X.rotation = 0
<prefix>_rotation_Y.rotation = 0
```

For `NORMALIZED_SINGLE`, the previous main placement is transferred to the internal `<prefix>` base bone and to the control coordinates. This preserves the same visible object position while exposing a clean neutral setup pose to the animator. For `PRESERVE_COMPOSITION`, the calculated `<prefix>_main` placement remains unchanged so composition keeps each object in place.

The reference setup rotations `-134.67` and `-17.43` are always emitted as X/Y transform-constraint rotation offsets. They are not used as the visible control bones' defaults and are not discarded.

## Multi-object placement policy

For `PRESERVE_COMPOSITION`:

- `<prefix>_main` retains its calculated nonzero position;
- the internal base receives no duplicate placement;
- X/Y controls are created with `rotation = 0`;
- the reference angles are stored in the matching constraint offsets.

This keeps standalone and future connected compositions from flattening object placement or overlapping multiple exported components.

## Control layout

The visible two-axis controls form one editor column:

```text
<prefix>_rotation_X
    one control length
<prefix>_rotation_Y
    one control length
<prefix>_scale
```

All three controls share the same world-space X coordinate. Their spacing is derived from the generated control length rather than from one model-specific pixel value.

## Constraint order

The reference evaluation order is preserved semantically:

1. X transform constraint (`order=0`).
2. IK (`order=1`).
3. Uniform scale transform (`order=2`).
4. X depth scale transform (`order=3`).
5. Y transform constraint (`order=4`).

The scale transform targets `<prefix>_rotate_X` and every Z-group rotation bone. The two-axis profile does not generate a Z rotation control or the three-axis scale compensator.

## Generalization rules

The embedded JSON is a behavior reference, not a literal template:

- Names are always prefixed; global names such as `scale`, `IK`, and `null` are forbidden.
- Fixed model values (`500`, `600`, `818.36`, and similar) become centralized layout ratios derived from texture dimensions and Z-group offsets.
- The number of Z groups and exported vertices is unrestricted.
- The existing three-axis profile remains unchanged.
- Connected composition remains explicitly blocked for the two-axis profile until a dedicated five-phase connected schedule is implemented.

## Validation

Required regression coverage includes:

- Scene property registration, reset, readiness invalidation, and settings propagation.
- Exact UI foldout order and reversible main-panel replacement.
- Exact two-axis bone and constraint order.
- Neutral X/Y controls with normalized single-object and preserved multi-object placement.
- Arbitrary Z-group counts and vertex counts.
- No Z rotation control in the two-axis profile.
- One shared X control column and deterministic vertical spacing.
- Scale target set containing the X owner and all Z-group rotation bones.
- Spine cross-reference validation and weighted-index validation.
- Existing three-axis golden output remaining unchanged.
- Blender headless export and manual Spine 4.2.43 validation for X, Y, X+Y, Scale, and X+Y+Scale.

## Reference Spine 4.2.43 skeleton

This is the complete user-provided reference and must remain available in the repository for implementation and regression review. It is also stored as machine-readable fixture `tests/fixtures/two_axis_scale_reference.json`.

```json
{
"skeleton": {
	"hash": "6b/QcQ/2H2s",
	"spine": "4.2.43",
	"x": -251.28,
	"y": -201.47,
	"width": 502.57,
	"height": 501.47,
	"images": "./Images/",
	"audio": ""
},
"bones": [
	{ "name": "root" },
	{ "name": "BOX_RIG", "parent": "root", "color": "faff00ff", "icon": "square" },
	{ "name": "null", "parent": "BOX_RIG", "color": "faff00ff" },
	{ "name": "SCALE_ROTATE_X", "parent": "null", "scaleX": 0 },
	{ "name": "ROTATE_X", "parent": "SCALE_ROTATE_X" },
	{ "name": "TOP_SCALE", "parent": "ROTATE_X", "rotation": 90, "y": 300, "inherit": "onlyTranslation" },
	{ "name": "TOP_ROTATION", "parent": "TOP_SCALE", "rotation": -90 },
	{ "name": "TOP1", "parent": "TOP_ROTATION", "rotation": 90, "x": 249.95, "y": 250.41 },
	{ "name": "TOP2", "parent": "TOP_ROTATION", "x": 249.95, "y": -250.46 },
	{ "name": "TOP3", "parent": "TOP_ROTATION", "rotation": -90, "x": -249.99, "y": -250.46 },
	{ "name": "TOP4", "parent": "TOP_ROTATION", "x": -249.99, "y": 250.49 },
	{ "name": "BOTTOM_SCALE", "parent": "ROTATE_X", "rotation": 90, "y": -201.47, "inherit": "onlyTranslation" },
	{ "name": "BOTTOM_ROTATION", "parent": "BOTTOM_SCALE", "rotation": -90 },
	{ "name": "BOTTOM5", "parent": "BOTTOM_ROTATION", "rotation": 90, "x": 249.95, "y": 250.41 },
	{ "name": "BOTTOM6", "parent": "BOTTOM_ROTATION", "x": 249.95, "y": -250.46 },
	{ "name": "BOTTOM7", "parent": "BOTTOM_ROTATION", "rotation": -90, "x": -249.99, "y": -250.46 },
	{ "name": "BOTTOM8", "parent": "BOTTOM_ROTATION", "x": -249.99, "y": 250.49 },
	{ "name": "ROTATE_Y_CTRL", "parent": "BOX_RIG", "length": 200, "rotation": -17.43, "x": 818.36, "color": "1aff00ff" },
	{ "name": "SCALE_ROTATE_X_CONTRAINT", "parent": "null", "length": 600, "rotation": 90, "color": "abe323ff" },
	{ "name": "SCALE_IK", "parent": "null", "y": 600, "scaleX": 0 },
	{ "name": "ROTATE_IK", "parent": "SCALE_IK", "x": -600 },
	{ "name": "IK", "parent": "ROTATE_IK", "x": 600, "color": "ff3f00ff", "icon": "ik" },
	{
		"name": "ROTATE_X_CTRL",
		"parent": "BOX_RIG",
		"length": 200,
		"rotation": -134.67,
		"x": 818.36,
		"y": 100,
		"color": "ff0000ff"
	},
	{ "name": "scale", "parent": "root", "x": 1124.14, "y": 59.89, "color": "abe323ff", "icon": "square" }
],
"slots": [
	{ "name": "BOX_FRONT", "bone": "TOP_ROTATION", "attachment": "BOX" },
	{ "name": "BOX_BACK", "bone": "BOTTOM_ROTATION", "attachment": "BOX2" },
	{ "name": "BOX_SIDE1", "bone": "null", "attachment": "BOX" },
	{ "name": "BOX_SIDE2", "bone": "null", "attachment": "BOX" },
	{ "name": "BOX_SIDE3", "bone": "null", "attachment": "BOX" },
	{ "name": "BOX_SIDE4", "bone": "null", "attachment": "BOX" }
],
"ik": [
	{
		"name": "IK",
		"order": 1,
		"bones": [ "SCALE_ROTATE_X_CONTRAINT" ],
		"target": "IK",
		"compress": true,
		"stretch": true
	}
],
"transform": [
	{
		"name": "ROTATE_X_CONSTRAINT",
		"bones": [ "ROTATE_IK", "ROTATE_X" ],
		"target": "ROTATE_X_CTRL",
		"local": true,
		"relative": true,
		"x": -500,
		"y": 500,
		"scaleX": -1,
		"mixX": 0,
		"mixScaleX": 0,
		"mixShearY": 0
	},
	{
		"name": "ROTATE_Y",
		"order": 4,
		"bones": [ "TOP_ROTATION", "BOTTOM_ROTATION" ],
		"target": "ROTATE_Y_CTRL",
		"local": true,
		"relative": true,
		"x": -500,
		"y": 300,
		"mixX": 0,
		"mixScaleX": 0,
		"mixShearY": 0
	},
	{
		"name": "scale",
		"order": 2,
		"bones": [ "ROTATE_X", "TOP_ROTATION", "BOTTOM_ROTATION" ],
		"target": "scale",
		"relative": true,
		"mixRotate": 0,
		"mixX": 0,
		"mixShearY": 0
	},
	{
		"name": "SCALE_ROTATE_X_CONSTRAINT",
		"order": 3,
		"bones": [ "BOTTOM_SCALE", "TOP_SCALE" ],
		"target": "SCALE_ROTATE_X_CONTRAINT",
		"rotation": -90,
		"x": -201.47,
		"scaleX": -1,
		"mixRotate": 0,
		"mixX": 0,
		"mixShearY": 0
	}
],
"skins": [
	{
		"name": "default",
		"attachments": {
			"BOX_BACK": {
				"BOX2": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 16, 499.99, 0.01, 1, 1, 13, 0.09, 499.95, 1, 1, 14, -499.95, -0.04, 1, 1, 15, 0.03, 499.99, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_FRONT": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 10, -0.01, 0.01, 1, 1, 7, 0.09, -0.05, 1, 1, 8, 0.05, -0.04, 1, 1, 9, 0.03, -0.01, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE1": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 1, 2, 3, 0, 1 ],
					"vertices": [ 1, 9, 0.82, -0.01, 1, 1, 8, 0.05, -0.82, 1, 1, 14, 0.05, -0.35, 1, 1, 15, 0.35, -0.01, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE2": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 10, -1.3, -0.49, 1, 1, 9, -0.46, -1.3, 1, 1, 15, -0.47, -0.83, 1, 1, 16, -0.83, -0.49, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE3": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 7, 0.87, -0.05, 1, 1, 10, -0.01, 0.79, 1, 1, 16, -0.01, 0.32, 1, 1, 13, 0.41, -0.05, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			},
			"BOX_SIDE4": {
				"BOX": {
					"type": "mesh",
					"uvs": [ 0, 0, 1, 0, 1, 1, 0, 1 ],
					"triangles": [ 3, 0, 1, 3, 1, 2 ],
					"vertices": [ 1, 8, 1.33, 0.46, 1, 1, 7, -0.41, -1.33, 1, 1, 13, -0.41, -0.86, 1, 1, 14, 0.86, 0.46, 1 ],
					"hull": 4,
					"edges": [ 0, 2, 2, 4, 4, 6, 0, 6, 6, 2 ],
					"width": 500,
					"height": 501
				}
			}
		}
	}
],
"animations": {
	"rotate both": {
		"bones": {
			"ROTATE_X_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"ROTATE_Y_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{ "x": 0.981, "y": 0.981 }
				]
			}
		}
	},
	"rotate both2": {
		"bones": {
			"ROTATE_X_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"ROTATE_Y_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{
						"x": 0.75,
						"y": 0.75,
						"curve": [ 0.167, 0.75, 0.333, 1.5, 0.167, 0.75, 0.333, 1.5 ]
					},
					{
						"time": 0.5,
						"x": 1.5,
						"y": 1.5,
						"curve": [ 0.667, 1.5, 0.833, 0.75, 0.667, 1.5, 0.833, 0.75 ]
					},
					{
						"time": 1,
						"x": 0.75,
						"y": 0.75,
						"curve": [ 1.167, 0.75, 1.333, 1.5, 1.167, 0.75, 1.333, 1.5 ]
					},
					{
						"time": 1.5,
						"x": 1.5,
						"y": 1.5,
						"curve": [ 1.667, 1.5, 1.833, 0.75, 1.667, 1.5, 1.833, 0.75 ]
					},
					{ "time": 2, "x": 0.75, "y": 0.75 }
				]
			}
		}
	},
	"rotate x": {
		"bones": {
			"ROTATE_X_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{ "x": 2, "y": 2 }
				]
			}
		}
	},
	"rotate y": {
		"bones": {
			"ROTATE_Y_CTRL": {
				"rotate": [
					{},
					{ "time": 2, "value": 360 }
				]
			},
			"scale": {
				"scale": [
					{ "x": 2, "y": 2 }
				]
			}
		}
	}
}
}
```

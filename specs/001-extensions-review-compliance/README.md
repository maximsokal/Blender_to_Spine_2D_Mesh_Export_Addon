# Spec Kit — Blender Extensions Review Compliance

This directory is the planning, implementation, and release-evidence package for closing the current Blender Extensions moderation feedback against the Blender 5.2+ Rewrite.

**Base:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Working branch:** `001-extensions-review-compliance`  
**Target corrected version:** `0.155.0`

## Documents

| File | Purpose |
| --- | --- |
| `review-feedback.md` | Exact current seven-point reviewer ledger and closure evidence. |
| `spec.md` | Functional/non-functional compliance requirements and acceptance matrix. |
| `research.md` | Source/manifest/docs audit, Blender constraints, risks, and open research. |
| `plan.md` | Ordered implementation slices and release gates. |
| `tasks.md` | Fine-grained execution ledger. |
| `implementation-log.md` | Production changes actually made, with reasons and pending evidence. |
| `checklists/extensions-review.md` | Exact-release-candidate moderator closure checklist. |
| `quickstart.md` | Local test/build/install gate commands. |

## Non-negotiable current-review rules

1. `PipelineTraceSession`/development-only material must not be shipped unless it has a real user runtime purpose.
2. Shipped runtime must not use `threading` or `queue`.
3. Re-Polish advertising/integration must not be shipped; the extension must be self-contained.
4. Registration must use straightforward Blender ownership; no root state machine or `ui_layout.py` panel unregister/re-register/restore dance.
5. Manifest tags must contain only `Import-Export`.
6. Do not keep a Windows-only manifest restriction without an actual runtime requirement.
7. The public title must not contain `Blender`; technical `id` may stay unchanged.
8. Upload the corrected higher version to the **same existing submission**; do not create another extension listing.
9. Built ZIP inventory, Blender validation, and clean installed-extension tests are final evidence.
10. Retained legacy repository code is not modified merely for package cleanup; non-runtime legacy stays excluded from the ZIP.
11. Every implementation change is recorded in this Spec Kit before release.

## Implementation status

Already changed on this branch:

- manifest target moved to `0.155.0`, public name `Spine2D Mesh Exporter`, only `Import-Export` tag, no Windows-only platform declaration;
- `threading` removed from the exclusive-operation runtime guard;
- Re-Polish runtime panel removed;
- root registration state machine removed;
- `ui_layout.py` panel swapping replaced with ordinary child panels;
- rig UI no longer swaps the base Reset operator during registration;
- focused compliance tests added.

Still required before release:

- run focused/full/bpy suites and adapt tests that intentionally encoded the rejected architecture;
- audit/simplify remaining unnecessary module-local registration wrappers;
- synchronize public docs/submission/testing with 0.155.0 and same-submission workflow;
- build the exact ZIP with Blender, inspect its full inventory, validate it, install it in an isolated profile, and run lifecycle/export gates;
- record exact Git SHA + ZIP SHA256 and upload that ZIP to the existing Blender Extensions submission.

# Spec Kit — Blender Extensions Review Compliance

This directory is the planning/evidence package for closing the Blender Extensions moderation feedback against the current Blender 5.2+ Rewrite.

**Documentation baseline:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Documentation branch:** `001-extensions-review-compliance`  
**Production changes in this branch:** none.

## Documents

| File | Purpose |
| --- | --- |
| `review-feedback.md` | Reviewer-comment ledger, current baseline status, and required closure evidence. |
| `spec.md` | Mandatory functional/non-functional requirements and acceptance matrix. |
| `research.md` | Current-main source/manifest/docs audit, Blender constraints, risks, and open implementation research. |
| `plan.md` | Ordered implementation slices with test/release gates and rollback strategy. |
| `tasks.md` | Fine-grained execution ledger; this is the “every change” checklist during implementation. |
| `checklists/extensions-review.md` | Exact-release-candidate moderator closure checklist. |
| `quickstart.md` | How to start the future implementation branch and run the existing repository gates. |

## Governing rules

1. Reviewer intent wins over obsolete historical line numbers.
2. Current Rewrite source is the implementation target.
3. No persistent Python background thread/thread-timer may be left running while Blender resumes normal execution.
4. No `bpy`/Blender data access may occur from a non-main Python thread.
5. Trace/diagnostic persistence must have an explicit deterministic request owner.
6. Registration complexity must be justified by real Blender resource ownership and partial-failure behavior.
7. UI settings must have canonical owners; duplicate `_dup` drawing paths are forbidden.
8. Built ZIP inventory, not source exclusions alone, decides package hygiene.
9. Platform support is an evidence-backed release claim.
10. Public title can change without casually changing stable technical package identity.
11. Corrected package is uploaded as a new version of the **same existing declined submission**.
12. Retained legacy repository code is not modified merely to make the extension ZIP smaller; exclude non-runtime legacy from the package instead.
13. Every implementation slice updates `tasks.md` and maps to `RF-*` / `FR-*` IDs.
14. Any edit after a candidate ZIP is built invalidates that artifact and requires a rebuild/package revalidation.

## Baseline snapshot

At the baseline commit, the most important current facts are:

- extension version: `0.154.0`;
- public name: `Blender to Spine2D Mesh Exporter` — reviewer rename still open;
- tags: `Import-Export`, `Mesh`, `UV`, `Animation` — historical irrelevant tags are already gone;
- platform: `windows-x64` — reviewer platform question still open;
- build exclusions: substantially improved, but exact ZIP inventory is still mandatory;
- historical `spine_core/chat_persistence.py` is not a current Rewrite path;
- current `ui_layout.py` no longer contains the reviewer-named duplicate drawing functions;
- current root registration remains a non-trivial transactional lifecycle and needs resource-by-resource simplification review;
- current submission documentation still says “initial submission” and must be changed before moderator re-upload.

## Implementation handoff

When production work is approved, do not continue committing runtime changes to this documentation branch. Create a fresh implementation branch from the approved `main`, bring this Spec Kit package with it, and execute `tasks.md` in the order constrained by `plan.md`.

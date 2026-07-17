# Rewrite branch CI mode

Automatic GitHub Actions triggers are temporarily disabled on
`rewrite/a1-domain-foundation` to avoid consuming Actions minutes and sending notification
mail during active rewrite development.

The following workflows are retained but use only `workflow_dispatch`:

- `.github/workflows/ci.yml`;
- `.github/workflows/blender-alpha-headless.yml`;
- `.github/workflows/blender-scene-headless.yml`;
- `.github/workflows/blender-camera-projection.yml`;
- `.github/workflows/blender-headless.yml`.

They can still be started explicitly from **Actions → Run workflow** when a validation run is
actually needed.

## Required before merge

Before this branch is merged into `main`, restore the intended automatic triggers from the base
branch or from the last validated rewrite commit:

- `push`/`pull_request` for the pure Python workflow as required by repository policy;
- path-filtered `pull_request` triggers for the Blender workflows.

After restoring the triggers, run the complete validation matrix once on the final candidate head.
Manual-only CI must not be carried into the release merge by accident.

# Rewrite diagnostics, events, and per-file logging

## Problem fixed

The addon preferences previously exposed per-module logging, but the persisted list only contained a small legacy subset. Rewrite modules under `application`, `domain`, `blender_adapter`, and `infrastructure` were absent, and runtime logger names were hardcoded to one Blender extension prefix.

The atomic output transaction removed staged files during normal Python exceptions, but rollback failures were silently ignored. Work files left after an interrupted Blender process were not recovered on the next export.

## Per-file logging contract

The module list is now discovered from every Python source file below the addon package directory.

Persisted names are relative paths such as:

```text
blender_adapter.a1_ui_bridge
blender_adapter.a1_multi_object_output
domain.baking.projection_layout
infrastructure.atomic_files
infrastructure.export_events
```

The actual runtime logger name is resolved from the package's real `__package__` value. This supports both normal package imports and Blender extension names such as `bl_ext.user_default...`.

Each module level is independent. Changing the package/root level no longer overwrites every file. Existing levels survive a source-tree rescan.

Addon Preferences provide:

- `Enable file logging`;
- `Log file path`;
- `Filter modules`;
- one `ERROR/WARNING/INFO/DEBUG` selector per Python file;
- `Refresh Module List` for newly added files.

## Failed work-file policy

Two preferences control failed outputs:

```text
Preserve failed work files = false (default)
Recover stale work files   = true  (default)
```

When preservation is disabled, failed `.spine2d-stage-*` files are deleted during rollback. Existing final outputs are restored from `.spine2d-backup-*` files.

When preservation is enabled, failed stage files remain available for manual diagnostics. Backups are still restored because they are safety data rather than debug artifacts.

On the next reservation in an output directory, stale work recovery:

1. restores a backup when its final output is missing;
2. removes a stale backup when the final output already exists;
3. removes abandoned stage files unless preservation is enabled;
4. reports every cleanup failure instead of silently suppressing it.

A Python cleanup cannot run after an operating-system or Blender hard crash. The next-export recovery step covers that case.

## Event dispatcher

`infrastructure.export_events` provides a non-fatal, thread-safe dispatcher. Listener failures are logged and cannot interrupt the export.

Lifecycle events include:

- transaction started;
- output reserved;
- commit started/succeeded;
- rollback started/succeeded;
- work file removed/preserved;
- backup restored;
- stale work recovered;
- transaction failed;
- cleanup failed.

Set `infrastructure.export_events` or `infrastructure.atomic_files` to `DEBUG` in Addon Preferences to isolate this pipeline without enabling debug logging for the rest of the addon.

## Failure visibility

Rollback errors no longer use an empty `except` block. The original export exception remains primary, while cleanup failure is:

- emitted as `CLEANUP_FAILED`;
- written through the normal per-file logger;
- attached as an exception note when supported by the active Python runtime.

## Validation

Focused pure tests cover:

- nested module discovery;
- persisted level migration;
- runtime package-prefix resolution;
- normal failed-stage deletion;
- debug preservation;
- stale backup restoration;
- lifecycle event order;
- architecture guards against hardcoded module lists and silent cleanup handlers.

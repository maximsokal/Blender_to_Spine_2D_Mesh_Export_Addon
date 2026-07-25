from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.interprocess_lock import (
    InterprocessFileLock,
    InterprocessLockError,
    lock_path_for_resource,
)


ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), existing) if value
    )
    return environment


def test_two_processes_cannot_own_one_output_and_dead_owner_is_recovered(tmp_path):
    final_path = (tmp_path / "hero.json").resolve()
    script = r'''
import sys
import time
from pathlib import Path
from Blender_to_Spine2D_Mesh_Exporter.infrastructure.interprocess_lock import InterprocessFileLock
lock = InterprocessFileLock(Path(sys.argv[1]), minimum_stale_age_seconds=0.0)
lock.acquire()
print("READY", flush=True)
time.sleep(120)
'''
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(final_path)],
        cwd=ROOT,
        env=_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(InterprocessLockError, match="another active transaction/process"):
            InterprocessFileLock(
                final_path,
                minimum_stale_age_seconds=0.0,
            ).acquire()
    finally:
        child.terminate()
        child.wait(timeout=10)

    recovered = InterprocessFileLock(
        final_path,
        minimum_stale_age_seconds=0.0,
    )
    recovered.acquire()
    recovered.release()
    assert not lock_path_for_resource(final_path).exists()


def test_different_output_paths_can_be_locked_concurrently(tmp_path):
    first = InterprocessFileLock((tmp_path / "a.json").resolve())
    second = InterprocessFileLock((tmp_path / "b.json").resolve())
    first.acquire()
    second.acquire()
    assert first.acquired and second.acquired
    second.release()
    first.release()


def test_release_refuses_foreign_token(tmp_path):
    resource = (tmp_path / "hero.json").resolve()
    lock = InterprocessFileLock(resource)
    lock.acquire()
    path = lock.lock_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["token"] = "foreign-token"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InterprocessLockError, match="another token"):
        lock.release()
    path.unlink()


def test_fresh_malformed_lock_fails_closed_but_old_malformed_lock_is_replaced(tmp_path):
    resource = (tmp_path / "hero.json").resolve()
    path = lock_path_for_resource(resource)
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(InterprocessLockError, match="unreadable fresh lock"):
        InterprocessFileLock(resource, minimum_stale_age_seconds=60.0).acquire()

    old = time.time() - 120.0
    os.utime(path, (old, old))
    lock = InterprocessFileLock(resource, minimum_stale_age_seconds=60.0)
    lock.acquire()
    lock.release()
    assert not path.exists()

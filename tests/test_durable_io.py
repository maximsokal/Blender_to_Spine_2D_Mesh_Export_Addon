from __future__ import annotations

import os
from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.durable_io import fsync_file


def test_fsync_file_opens_a_writable_descriptor(tmp_path: Path, monkeypatch):
    target = tmp_path / "output.bin"
    target.write_bytes(b"payload")
    observed_flags: list[int] = []
    real_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        observed_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)

    fsync_file(target)

    assert observed_flags
    assert observed_flags[0] & os.O_RDWR == os.O_RDWR

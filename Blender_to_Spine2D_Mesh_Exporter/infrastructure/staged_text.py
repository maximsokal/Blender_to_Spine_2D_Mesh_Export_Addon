"""Write validated UTF-8 text into a caller-owned staged output path."""

from __future__ import annotations

from pathlib import Path


class StagedTextWriteError(RuntimeError):
    """Raised when a staged UTF-8 text artifact cannot be written completely."""


def write_staged_utf8_text(
    path: Path,
    text: str,
    *,
    ensure_trailing_newline: bool = False,
) -> None:
    """Write one non-empty text artifact and verify it exists before commit."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if not isinstance(ensure_trailing_newline, bool):
        raise TypeError("ensure_trailing_newline must be bool")

    payload = text
    if ensure_trailing_newline and not payload.endswith("\n"):
        payload += "\n"
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise StagedTextWriteError(
            f"Unable to write staged UTF-8 text '{path}': {exc}"
        ) from exc
    try:
        if not path.is_file():
            raise StagedTextWriteError(
                f"Staged UTF-8 text file was not created: {path}"
            )
        actual_size = path.stat().st_size
    except OSError as exc:
        raise StagedTextWriteError(
            f"Unable to verify staged UTF-8 text '{path}': {exc}"
        ) from exc
    if actual_size <= 0:
        raise StagedTextWriteError(f"Staged UTF-8 text file is empty: {path}")

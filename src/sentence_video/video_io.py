# -*- coding: utf-8 -*-
"""Video upload persistence helpers for sentence recognition."""

from pathlib import Path

from fastapi import UploadFile


async def save_upload_file(upload_file: UploadFile, target_path: Path) -> Path:
    """Persist a FastAPI upload to disk in chunks."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("wb") as output:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    return target_path

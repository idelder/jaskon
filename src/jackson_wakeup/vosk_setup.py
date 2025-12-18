from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests


def ensure_vosk_model(model_dir: Path, *, url: str) -> None:
    """Download and unpack a Vosk model zip if model_dir doesn't exist.

    Note: The default URL is configurable; model zips can be large.
    """

    if model_dir.exists() and any(model_dir.iterdir()):
        return

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = model_dir.parent / "vosk-model.zip"

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with tmp_zip.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(model_dir.parent)

    # Many Vosk zips extract to a nested folder (e.g., vosk-model-small-en-us-0.15)
    extracted = [p for p in model_dir.parent.iterdir() if p.is_dir() and p.name.startswith("vosk-model")]
    extracted = [p for p in extracted if p != model_dir]
    if len(extracted) == 1 and not model_dir.exists():
        extracted[0].rename(model_dir)
    elif len(extracted) == 1 and model_dir.exists() and not any(model_dir.iterdir()):
        # Move contents
        for item in extracted[0].iterdir():
            shutil.move(str(item), str(model_dir / item.name))
        shutil.rmtree(extracted[0], ignore_errors=True)

    tmp_zip.unlink(missing_ok=True)

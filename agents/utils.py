from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


def encode_image_to_jpeg_base64(np_image, quality: int = 85) -> str:
    """
    Convert a numpy image (H,W,3) uint8 to base64-encoded JPEG.
    """
    img = Image.fromarray(np_image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_image_hash(np_image) -> str:
    h = hashlib.sha256()
    h.update(np_image.tobytes())
    return h.hexdigest()


def compute_prompt_fingerprint(
    image_hash: str, links: List[Dict[str, Any]], meta: Dict[str, Any]
) -> str:
    payload = {
        "image": image_hash,
        "links": links,
        "meta": meta,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_get(cache_dir: Path | str | None, key: str) -> Dict[str, Any] | None:
    if cache_dir is None:
        return None

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_put(cache_dir: Path | str | None, key: str, value: Dict[str, Any]) -> None:
    if cache_dir is None:
        return

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    tmp.replace(path)

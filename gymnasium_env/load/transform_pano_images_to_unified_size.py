from __future__ import annotations

from pathlib import Path
from typing import Tuple

TARGET_SIZE: Tuple[int, int] = (2048, 1024)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Paths relative to the script location (resolved to absolute paths)
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = (BASE_DIR / "images").resolve()
OUTPUT_DIR = (BASE_DIR / "unifiedimages").resolve()


def _resize_with_pillow(src: Path, dst: Path) -> None:
    from PIL import Image

    with Image.open(src) as image:
        resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        suffix = src.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            # Ensure RGB for JPEG
            resized = resized.convert("RGB")
            resized.save(dst, format="JPEG", quality=95, subsampling=1, optimize=True)
        elif suffix == ".png":
            resized.save(dst, format="PNG", optimize=True)
        else:
            resized.save(dst)


def transform_all_images() -> None:
    images_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    processed = 0

    for entry in images_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        total += 1
        destination = output_dir / entry.name

        try:
            _resize_with_pillow(entry, destination)
            processed += 1
        except Exception as exc:  # pragma: no cover - best-effort batch process
            print(f"Failed to transform '{entry.name}': {exc}")

    print(
        f"Transformed {processed}/{total} image(s) to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} in '{output_dir}'."
    )


if __name__ == "__main__":
    transform_all_images()

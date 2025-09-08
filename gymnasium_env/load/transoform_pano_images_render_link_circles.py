import json
import math
import os
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def read_minimetadata_jsonl(jsonl_path: str) -> Dict[str, List[Tuple[float, str]]]:
    """
    Read a JSONL file where each line contains a panorama entry with its links.

    Returns a mapping from panorama_id -> list of link directions (radians).
    """
    pano_id_to_links: Dict[str, List[Tuple[float, str]]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pano_id = data.get("id")
            links = data.get("links", [])

            if not pano_id:
                continue

            entries: List[Tuple[float, str]] = []
            for link in links:
                # Link schema: { "pano": { "id": str, ... }, "direction": float }
                direction = link.get("direction")
                link_pano = link.get("pano", {}) or {}
                link_pano_id = link_pano.get("id")
                if isinstance(direction, (int, float)) and isinstance(link_pano_id, str):
                    entries.append((float(direction), link_pano_id))

            pano_id_to_links[pano_id] = entries

    return pano_id_to_links


def read_headings_jsonl(jsonl_path: str) -> Dict[str, float]:
    """
    Read a JSONL file and return a mapping from panorama_id -> heading (radians).
    If a pano has no heading, it will be omitted from the map.
    """
    pano_id_to_heading: Dict[str, float] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pano_id = data.get("id")
            heading = data.get("heading")
            if isinstance(pano_id, str) and isinstance(heading, (int, float)):
                pano_id_to_heading[pano_id] = float(heading)

    return pano_id_to_heading


def normalize_direction_radians(direction: float, heading: float = 0.0) -> float:
    """
    Normalize direction to [0, 2π].
    """
    tau = getattr(math, "tau", 2 * math.pi)
    # Adjust by original pano heading and wrap into [0, tau)
    print(f"DIRECTION: {direction}, HEADING:{heading}")
    adjusted = direction + heading
    wrapped = adjusted % tau
    # Keep exactly tau as tau (we'll clamp x later)
    if math.isclose(wrapped, 0.0, abs_tol=1e-12) and adjusted != 0.0:
        # If original was effectively 2π, prefer tau to place on right edge
        return tau
    print(f"WRapped: {wrapped:.6f}")
    return wrapped


def direction_to_x(direction: float, image_width: int, heading: float = 0.0) -> int:
    """
    Map a direction in radians to the horizontal pixel coordinate on an
    equirectangular image where:
    - 0 corresponds to the left edge (x = 0)
    - π corresponds to the image center (x = width / 2)
    - 2π corresponds to the right edge (x = width - 1)
    """
    tau = getattr(math, "tau", 2 * math.pi)
    d = normalize_direction_radians(direction, heading=heading)

    # Linear mapping across the width.
    # Note: if d == tau, x_float == width; clamp to width - 1 below.
    x_float = (d / tau) * float(image_width)
    x = int(round(x_float))
    # Clamp to valid pixel indices
    if x < 0:
        x = 0
    if x >= image_width:
        x = image_width - 1
    return x


def load_large_font(preferred_size: int):
    """
    Try to load a TrueType font at the requested size; fall back to a default font.
    Attempts common fonts on macOS/Linux/Windows.
    """
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Verdana.ttf",
        "Arial.ttf",
        "Verdana.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=preferred_size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=preferred_size)
    except Exception:
        return ImageFont.load_default()


def draw_circles_on_image(image_path: str, links: List[Tuple[float, str]], output_path: str, pano_id: str, *,
                          pano_heading: float = 0.0,
                          radius: int = 24, fill_color=(255, 0, 0), outline_color=(255, 255, 255), outline_width: int = 2) -> None:
    """
    Draw a circle at the vertical center for each link direction.
    Saves the result to output_path.
    """
    if not links:
        # Still copy the image to output to keep parity, or skip? We'll copy.
        with Image.open(image_path) as img:
            img.save(output_path)
        return

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size
        # Choose a readable font size that adapts to both height and width
        # Make it 4x smaller than the previous sizing
        preferred_size = max(14, int(min(height * 0.06, width * 0.03) / 4))
        font = load_large_font(preferred_size=preferred_size)
        y = height // 2

        for direction, link_pano_id in links:
            # Use the exact normalized direction for both x and label to keep them identical
            d_norm = normalize_direction_radians(direction, heading=pano_heading)
            tau = getattr(math, "tau", 2 * math.pi)
            x_float = (d_norm / tau) * float(width)
            x = int(round(x_float))
            if x < 0:
                x = 0
            if x >= width:
                x = width - 1
            left = x - radius
            top = y - radius
            right = x + radius
            bottom = y + radius

            # Build list of bounding boxes accounting for horizontal wrap-around.
            bboxes = [[left, top, right, bottom]]
            if left < 0:
                bboxes.append([left + width, top, right + width, bottom])
            if right >= width:
                bboxes.append([left - width, top, right - width, bottom])

            # Optional outline to improvecontrast; apply to all wrapped copies.
            for box in bboxes:
                if outline_width > 0:
                    for ow in range(outline_width, 0, -1):
                        draw.ellipse([box[0] - ow, box[1] - ow, box[2] + ow, box[3] + ow], outline=outline_color)
                draw.ellipse(box, fill=fill_color)

            # Prepare label text with the same normalized angle used for x
            label = f"{d_norm:.3f} rad | {link_pano_id}"
            padding = 4
            # Compute text size
            tb = draw.textbbox((0, 0), label, font=font, stroke_width=2)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]

            # Default place below the circle; if not enough space, place above.
            text_y = bottom + padding
            if text_y + text_h > height:
                text_y = top - padding - text_h

            # Render labels with horizontal wrap like the circles
            x_positions = [x]
            if left < 0:
                x_positions.append(x + width)
            if right >= width:
                x_positions.append(x - width)

            for xp in x_positions:
                text_x = int(round(xp - text_w / 2))
                draw.text(
                    (text_x, text_y),
                    label,
                    fill=(255, 255, 255),
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

        img.save(output_path)


def ensure_directory(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pano_id = "DyDhU3ixcGl-9BT_SNzHTQ"
    metadata_path = os.path.join(project_root, "load", "metadata", f"{pano_id}_minimetadata.jsonl")
    images_dir = os.path.join(project_root, "load", "images")
    output_dir = os.path.join(project_root, "load", "markedimages")

    ensure_directory(output_dir)

    pano_to_links = read_minimetadata_jsonl(metadata_path)
    pano_to_heading = read_headings_jsonl(metadata_path)

    total_processed = 0
    total_missing = 0

    for pano_id, links in pano_to_links.items():
        src_image = os.path.join(images_dir, f"{pano_id}.jpg")
        if not os.path.isfile(src_image):
            total_missing += 1
            continue

        dst_image = os.path.join(output_dir, f"{pano_id}.jpg")
        try:
            heading = float(pano_to_heading.get(pano_id, 0.0))
            draw_circles_on_image(src_image, links, dst_image, pano_id, pano_heading=heading)
            total_processed += 1
        except Exception as e:
            # Skip problematic images but continue processing others
            print(f"Failed to process {pano_id}: {e}")

    print(f"Processed {total_processed} images. Missing {total_missing} source images.")


if __name__ == "__main__":
    main()



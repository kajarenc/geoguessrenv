import logging

from PIL import Image, UnidentifiedImageError
from streetview import crop_bottom_and_right_black_border, get_panorama

logger = logging.getLogger(__name__)


def load_single_panorama(pano_id: str, image_path: str, zoom: int = 4):
    try:
        image = get_panorama(
            pano_id=pano_id,
            multi_threaded=True,
            zoom=zoom,
        )
    except UnidentifiedImageError as e:
        logger.error(
            "Failed to download panorama %s: received non-image content (ID may be invalid or deprecated)",
            pano_id,
        )
        raise ValueError(f"Cannot download image for panorama {pano_id}: {e}")
    except Exception as e:
        logger.error("Failed to download panorama %s: %s", pano_id, e)
        raise

    image = crop_bottom_and_right_black_border(image)
    original_width, original_height = image.size
    target_width = 1024
    scale = target_width / original_width
    target_height = int(round(original_height * scale))
    if target_height != 512:
        print(
            f"Warning: aspect-preserving resize to 1024px width results in height {target_height}px (not 512px) for pano_id {pano_id}"
        )
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    image = image.resize((target_width, target_height), resample=resample_filter)
    image.save(image_path, "jpeg", quality=95, optimize=True)

from streetview import get_panorama, crop_bottom_and_right_black_border



def load_single_panorama(pano_id: str, zoom: int = 4):
    image = get_panorama(
        pano_id=pano_id,
        multi_threaded=True,
        zoom=zoom,
    )
    image = crop_bottom_and_right_black_border(image)
    image.save(f"images/{pano_id}.jpg", "jpeg")

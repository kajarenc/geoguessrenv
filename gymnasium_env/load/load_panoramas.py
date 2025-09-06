import json
import os
from streetview import get_panorama
from pydantic_models import Panorama


def load_single_panorama(pano_id):
    image = get_panorama(
        pano_id=pano_id,
        multi_threaded=True,
        zoom=2,

    )
    image.save(f"images/{pano_id}.jpg", "jpeg")


if __name__ == "__main__":
    with open("metadata/minimetadata.jsonl", "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                metadata = json.loads(line)
                # Validate using Panorama model
                panorama = Panorama(**metadata)
                pano_id = panorama.id
                image_path = f"images/{pano_id}.jpg"

                if os.path.exists(image_path):
                    print(f"Image already exists, skipping: {pano_id}")
                else:
                    print(f"Downloading panorama: {pano_id}")
                    load_single_panorama(pano_id)
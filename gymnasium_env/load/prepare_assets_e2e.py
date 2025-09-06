import os

from streetview import search_panoramas
import json
from dataclasses import asdict
from enum import Enum
from streetlevel import streetview as streetlevel

from gymnasium_env.load.load_panoramas import load_single_panorama
from pydantic_models import Panorama

from gymnasium_env.load.transform_metadata_to_essential_only import transform_metadata_to_essential_only

NUMBER_OF_PANOS_TO_PROCESS = 42

def make_serializable(obj):
    """Convert non-serializable objects to serializable format"""
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, '_asdict'):  # namedtuple
        return obj._asdict()
    else:
        return str(obj)


def safe_json_dumps(data):
    """Safely serialize data to JSON, handling non-serializable objects"""

    def default_handler(obj):
        return make_serializable(obj)

    return json.dumps(data, default=default_handler, skipkeys=True)


def bfs(pano_id):
    print("STARTING BFS")
    queue = [pano_id]
    visited = set()
    number_of_processed_panos = 0
    with open(f"metadata/{pano_id}.jsonl", "w") as pano_f:
        with open("metadata/realmetadata.jsonl", "a") as f:
            while queue and number_of_processed_panos < NUMBER_OF_PANOS_TO_PROCESS:
                print("IN LOOP")

                current_pano_id = queue.pop(0)

                if current_pano_id not in visited:
                    visited.add(current_pano_id)
                    pano = streetlevel.find_panorama_by_id(current_pano_id)
                    print(f"NUMBER OF LINKS for {current_pano_id}: {len(pano.links)}")

                    # Convert to dict and write to JSONL file
                    pano_dict = asdict(pano)
                    f.write(safe_json_dumps(pano_dict) + "\n")
                    pano_f.write(safe_json_dumps(pano_dict) + "\n")

                    f.flush()  # Ensure data is written immediately
                    pano_f.flush()

                    for link in pano.links:
                        queue.append(link.pano.id)
                    number_of_processed_panos += 1



def get_nearest_pano_id(lat: float, lon: float) -> str | None:
    panos = search_panoramas(lat=lat, lon=lon)

    if not panos:
        return None

    def sort_key(pano):
        ds = getattr(pano, "date", None)
        if isinstance(ds, str):
            try:
                year_str, month_str = ds.split("-")
                return (int(year_str), int(month_str))
            except Exception:
                pass
        # Put undated or malformed entries at the beginning (they'll be lowest when sorting desc)
        return (-1, -1)

    # Sort by date descending (latest first)
    panos_sorted = sorted(panos, key=sort_key, reverse=True)

    print("PANOS SORTED!!!!!!!")
    for pano in panos_sorted:
        print(pano.date)

    return panos_sorted[0].pano_id


def load_panorama_images(root_pano_id: str):
    with open(f"metadata/{root_pano_id}_minimetadata.jsonl", "r") as f:
        for line in f:
            if line.strip():
                metadata = json.loads(line)
                panorama = Panorama(**metadata)
                pano_id = panorama.id
                image_path = f"images/{pano_id}.jpg"

                if os.path.exists(image_path):
                    print(f"Image already exists, skipping: {pano_id}")
                else:
                    print(f"Downloading panorama: {pano_id}")
                    load_single_panorama(pano_id)


def prepare_assets_e2e(lat: float, lon: float):
    root_pano_id = get_nearest_pano_id(lat, lon)

    if root_pano_id is not None:
        print(f"START BFS FOR ROOT PANO {root_pano_id}")
        bfs(root_pano_id)

        transform_metadata_to_essential_only(
            input_file="metadata/realmetadata.jsonl",
            output_file="metadata/minimetadata.jsonl"
        )
        transform_metadata_to_essential_only(
            input_file=f"metadata/{root_pano_id}.jsonl",\
            output_file=f"metadata/{root_pano_id}_minimetadata.jsonl"
        )
        load_panorama_images(root_pano_id)

def main():
    # lat, lon = 47.62145616847461, -122.34769137299278 # Needle space

    # lat, lon = 47.622118, -122.3459565
    # lat, lon = 52.5107515, 13.3768324
    # lat, lon = 52.50980, 13.37654
    lat, lon = 52.2296973, 20.9847014
    prepare_assets_e2e(lat, lon)


if __name__ == "__main__":
    main()
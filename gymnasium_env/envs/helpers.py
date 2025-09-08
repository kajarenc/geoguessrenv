import os
from pathlib import Path

from streetview import search_panoramas
import json
from dataclasses import asdict
from enum import Enum
from streetlevel import streetview as streetlevel

from gymnasium_env.envs.load_panorama_helper import load_single_panorama
from gymnasium_env.envs.pydantic_models import Panorama, PanoramaLink


NUMBER_OF_PANOS_TO_PROCESS = 8


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

    for pano in panos_sorted:
        print(pano.date)

    return panos_sorted[0].pano_id


def bfs(pano_id, metadata_dir: str):
    print("STARTING BFS")
    queue = [pano_id]
    visited = set()
    number_of_processed_panos = 0
    with open(f"{metadata_dir}/{pano_id}.jsonl", "w") as pano_f:
        while queue and number_of_processed_panos < NUMBER_OF_PANOS_TO_PROCESS:
            print("IN LOOP")
            current_pano_id = queue.pop(0)

            if current_pano_id not in visited:
                visited.add(current_pano_id)
                pano = streetlevel.find_panorama_by_id(current_pano_id)
                print(f"NUMBER OF LINKS for {current_pano_id}: {len(pano.links)}")

                # Convert to dict and write to JSONL file
                pano_dict = asdict(pano)
                pano_f.write(safe_json_dumps(pano_dict) + "\n")

                # Ensure data is written immediately
                pano_f.flush()

                for link in pano.links:
                    queue.append(link.pano.id)
                number_of_processed_panos += 1


def download_metadata(root_pano_id: str, metadata_dir: str):
    print(f"Start BFS for root pano {root_pano_id} metadata scraping...")
    bfs(root_pano_id, metadata_dir)
    transform_metadata_to_essential_only(
        input_file=f"{metadata_dir}/{root_pano_id}.jsonl",
        output_file=f"{metadata_dir}/{root_pano_id}_mini.jsonl"
    )


def download_images(root_pano_id: str, metadata_dir: str, images_dir: str):
    print(f"Start downloading images for root pano {root_pano_id}...")
    with open(f"{metadata_dir}/{root_pano_id}_mini.jsonl", "r") as f:
        for line in f:
            if line.strip():
                metadata = json.loads(line)
                panorama = Panorama(**metadata)
                pano_id = panorama.id
                image_path = f"{images_dir}/{pano_id}.jpg"

                if os.path.exists(image_path):
                    print(f"Image already exists, skipping: {pano_id}")
                else:
                    print(f"Downloading panorama: {pano_id}")
                    load_single_panorama(pano_id, image_path)


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


def transform_metadata_to_essential_only(input_file: str, output_file: str) -> None:
    """
    Transform raw metadata from realmetadata.jsonl to essential-only format
    based on the Panorama model structure and write to mymetadata.jsonl.

    Args:
        input_file: Path to the input metadata file (realmetadata.jsonl)
        output_file: Path to the output metadata file (mymetadata.jsonl)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the raw metadata
                raw_data = json.loads(line.strip())

                # Extract essential fields for Panorama model
                panorama_data = {
                    'id': raw_data['id'],
                    'lat': raw_data['lat'],
                    'lon': raw_data['lon'],
                    'heading': raw_data['heading'],
                    'pitch': raw_data.get('pitch'),
                    'roll': raw_data.get('roll'),
                    'elevation': raw_data.get('elevation')
                }

                # Handle date conversion from dict to string
                if 'date' in raw_data and raw_data['date']:
                    date_dict = raw_data['date']
                    if isinstance(date_dict, dict):
                        year = date_dict.get('year')
                        month = date_dict.get('month')
                        day = date_dict.get('day')
                        if year and month:
                            if day:
                                panorama_data['date'] = f"{year}-{month:02d}-{day:02d}"
                            else:
                                panorama_data['date'] = f"{year}-{month:02d}"
                        else:
                            panorama_data['date'] = str(date_dict)
                    else:
                        panorama_data['date'] = str(raw_data['date'])

                # Process links if they exist
                if 'links' in raw_data and raw_data['links']:
                    links = []
                    for link_data in raw_data['links']:
                        if 'pano' in link_data:
                            # Create PanoramaLink object
                            panorama_link = PanoramaLink(
                                pano=link_data['pano'],
                                direction=link_data['direction']
                            )
                            links.append(panorama_link)
                    panorama_data['links'] = links

                # Validate and create Panorama object
                panorama = Panorama(**panorama_data)

                # Write to output file as JSON line
                json.dump(panorama.model_dump(), outfile, ensure_ascii=False)
                outfile.write('\n')

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    print(f"Successfully transformed metadata from {input_file} to {output_file}")

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from streetlevel import streetview as streetlevel
from streetview import search_panoramas

from .load_panorama_helper import load_single_panorama
from .pydantic_models import Panorama, PanoramaLink

NUMBER_OF_PANOS_TO_PROCESS = 8


def get_nearest_pano_id(
    lat: float, lon: float, metadata_dir: str | None = None
) -> str | None:
    """Return nearest panorama id for given coordinates with optional file cache.

    If ``metadata_dir`` is provided, a JSON cache file ``nearest_pano_cache.json``
    will be used to store and retrieve previously resolved pano ids keyed by
    rounded coordinates. This avoids repeated network calls and enables
    deterministic offline replays for identical inputs.
    """
    cache_path = None
    cache_key = f"{round(float(lat), 6)},{round(float(lon), 6)}"
    cache_data: dict[str, str | None] = {}

    if isinstance(metadata_dir, str) and metadata_dir:
        try:
            os.makedirs(metadata_dir, exist_ok=True)
        except Exception:
            pass
        cache_path = os.path.join(metadata_dir, "nearest_pano_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cache_data = loaded
            except Exception:
                # Corrupt cache: ignore and rebuild lazily
                cache_data = {}

        if cache_key in cache_data:
            return cache_data.get(cache_key)

    # Cache miss or no cache directory supplied → perform network lookup
    panos = search_panoramas(lat=lat, lon=lon)

    if not panos:
        result: str | None = None
    else:

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

        # Get first not None pano_id
        for pano in panos_sorted:
            if pano.pano_id is not None:
                result = pano.pano_id
                break
        else:
            result = None
            print(
                "Could not find a valid pano_id in the search results for given coordinates"
            )
        return result

    # Persist into cache if applicable
    if cache_path is not None:
        try:
            cache_data[cache_key] = result
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
        except Exception:
            pass

    return result


def bfs(pano_id, metadata_dir: str):
    print("STARTING BFS")
    queue = [pano_id]
    visited = set()
    number_of_processed_panos = 0
    with open(f"{metadata_dir}/{pano_id}.jsonl", "w") as pano_f:
        while queue and number_of_processed_panos < NUMBER_OF_PANOS_TO_PROCESS:
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
    raw_path = f"{metadata_dir}/{root_pano_id}.jsonl"
    mini_path = f"{metadata_dir}/{root_pano_id}_mini.jsonl"

    raw_exists = os.path.exists(raw_path)
    mini_exists = os.path.exists(mini_path)

    if raw_exists and mini_exists:
        print(
            f"Metadata already cached for {root_pano_id}; skipping BFS and transform."
        )
        return

    if not raw_exists:
        print(f"Start BFS for root pano {root_pano_id} metadata scraping...")
        bfs(root_pano_id, metadata_dir)
    else:
        print(f"Raw metadata exists for {root_pano_id}; skipping BFS.")

    if not mini_exists:
        print(f"Transform raw metadata to essential-only for {root_pano_id}...")
        transform_metadata_to_essential_only(input_file=raw_path, output_file=mini_path)
    else:
        print(f"Mini metadata exists for {root_pano_id}; skipping transform.")


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
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif hasattr(obj, "_asdict"):  # namedtuple
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

    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the raw metadata
                raw_data = json.loads(line.strip())

                # Extract essential fields for Panorama model
                panorama_data = {
                    "id": raw_data["id"],
                    "lat": raw_data["lat"],
                    "lon": raw_data["lon"],
                    "heading": raw_data["heading"],
                    "pitch": raw_data.get("pitch"),
                    "roll": raw_data.get("roll"),
                    "elevation": raw_data.get("elevation"),
                }

                # Handle date conversion from dict to string
                if "date" in raw_data and raw_data["date"]:
                    date_dict = raw_data["date"]
                    if isinstance(date_dict, dict):
                        year = date_dict.get("year")
                        month = date_dict.get("month")
                        day = date_dict.get("day")
                        if year and month:
                            if day:
                                panorama_data["date"] = f"{year}-{month:02d}-{day:02d}"
                            else:
                                panorama_data["date"] = f"{year}-{month:02d}"
                        else:
                            panorama_data["date"] = str(date_dict)
                    else:
                        panorama_data["date"] = str(raw_data["date"])

                # Process links if they exist
                if "links" in raw_data and raw_data["links"]:
                    links = []
                    for link_data in raw_data["links"]:
                        if "pano" in link_data:
                            # Create PanoramaLink object
                            panorama_link = PanoramaLink(
                                pano=link_data["pano"], direction=link_data["direction"]
                            )
                            links.append(panorama_link)
                    panorama_data["links"] = links

                # Validate and create Panorama object
                panorama = Panorama(**panorama_data)

                # Write to output file as JSON line
                json.dump(panorama.model_dump(), outfile, ensure_ascii=False)
                outfile.write("\n")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    print(f"Successfully transformed metadata from {input_file} to {output_file}")

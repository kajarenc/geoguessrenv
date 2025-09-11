import json
from pathlib import Path

from pydantic_models import Panorama, PanoramaLink


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


if __name__ == "__main__":
    # Transform the metadata files
    transform_metadata_to_essential_only(
        input_file="metadata/realmetadata.jsonl",
        output_file="metadata/minimetadata.jsonl",
    )

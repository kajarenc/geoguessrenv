import asyncio
import json
from dataclasses import asdict
from enum import Enum
from streetlevel import streetview
from aiohttp import ClientSession


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


async def bfs(pano_id):
    print("STARTING BFS")
    queue = [pano_id]
    visited = set()
    number_of_processed_panos = 0

    with open("metadata/realmetadata.jsonl", "a") as f:
        while queue and number_of_processed_panos < 100:
            print("IN LOOP")
            async with ClientSession() as session:
                current_pano_id = queue.pop(0)

                if current_pano_id not in visited:
                    visited.add(current_pano_id)
                    pano = streetview.find_panorama_by_id(current_pano_id)
                    # await streetview.download_panorama_async(pano, f"images/{pano.id}.jpg", session)
                    print(f"NUMBER OF LINKS for {current_pano_id}: {len(pano.links)}")

                    # Convert to dict and write to JSONL file
                    pano_dict = asdict(pano)
                    f.write(safe_json_dumps(pano_dict) + "\n")
                    f.flush()  # Ensure data is written immediately

                    for link in pano.links:
                        queue.append(link.pano.id)
                    number_of_processed_panos += 1


async def main():
    print("MAIN STARTED")
    await bfs("47GDDoYyLTICi5F0eoLz-w")
    print("MAIN FINISHED!")

if __name__ == "__main__":
    asyncio.run(main())

    # vu7f9URDaFEkma5kLjchZg - Seattle Needle space
    # 47GDDoYyLTICi5F0eoLz-w - Berlin Podzdamer Platz


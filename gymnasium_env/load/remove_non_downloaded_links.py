import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def remove_external_links(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Remove links that point to pano IDs not present among the records.

    Returns (cleaned_records, total_links_before, total_links_after).
    """
    id_set = {rec.get("id") for rec in records if isinstance(rec.get("id"), str)}
    total_before = 0
    total_after = 0

    for rec in records:
        links = rec.get("links", [])
        if not isinstance(links, list):
            rec["links"] = []
            continue

        total_before += len(links)
        cleaned_links: List[Dict[str, Any]] = []
        for link in links:
            if not isinstance(link, dict):
                continue
            pano = link.get("pano") or {}
            if not isinstance(pano, dict):
                continue
            target_id = pano.get("id")
            if isinstance(target_id, str) and target_id in id_set:
                cleaned_links.append(link)

        rec["links"] = cleaned_links
        total_after += len(cleaned_links)

    return records, total_before, total_after


def default_paths(pano_id: str) -> Tuple[str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = os.path.join(
        project_root, "load", "metadata", f"{pano_id}_minimetadata_copy.jsonl"
    )
    out_path = os.path.join(
        project_root, "load", "metadata", f"{pano_id}_minimetadata_copy_output.jsonl"
    )
    return in_path, out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove links to panoramas not present as nodes in the input JSONL. "
            "Reads {pano_id}_minimetadata_copy.jsonl and writes {pano_id}_minimetadata_copy_output.jsonl by default."
        )
    )
    parser.add_argument("pano_id", type=str, help="Root panorama ID (filename stem)")
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default=None,
        help="Optional explicit input JSONL path",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=str,
        default=None,
        help="Optional explicit output JSONL path",
    )

    args = parser.parse_args()

    inferred_in, inferred_out = default_paths(args.pano_id)
    input_path = args.input_path or inferred_in
    output_path = args.output_path or inferred_out

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = read_jsonl(input_path)
    cleaned, total_before, total_after = remove_external_links(records)
    removed = total_before - total_after

    write_jsonl(output_path, cleaned)

    print(
        f"Wrote cleaned JSONL to {output_path}. Nodes: {len(cleaned)}. "
        f"Links before: {total_before}, after: {total_after}, removed: {removed}."
    )


if __name__ == "__main__":
    main()



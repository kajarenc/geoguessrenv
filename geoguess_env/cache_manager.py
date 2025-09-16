"""Cache manager for enforcing standardized cache layout and manifest handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .providers.base import PanoramaMetadata


class CacheManager:
    """Manage cached panorama assets following the repository specification.

    Responsibilities:
    - Enforce naming for cached images (``<provider>_<pano_id>[...].jpg``)
    - Maintain a ``manifest.jsonl`` file with metadata and SHA256 hashes
    - Ensure attribution information is written to ``attribution.md``
    """

    MANIFEST_FILENAME = "manifest.jsonl"
    ATTRIBUTION_FILENAME = "attribution.md"

    def __init__(
        self,
        cache_root: Path,
        provider_name: str,
        attribution: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.provider_name = provider_name
        self.images_dir = self.cache_root / "images"
        self.metadata_dir = self.cache_root / "metadata"
        self.replays_dir = self.cache_root / "replays"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.metadata_dir / self.MANIFEST_FILENAME
        self.attribution_path = self.metadata_dir / self.ATTRIBUTION_FILENAME

        self._manifest_index: Dict[str, Dict] = {}
        self._load_manifest()

        if attribution:
            self._ensure_attribution(attribution)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            self._manifest_index = {}
            return

        index: Dict[str, Dict] = {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pano_id = data.get("pano_id") or data.get("id")
                if not pano_id:
                    continue
                index[str(pano_id)] = data
        self._manifest_index = index

    def _write_manifest(self) -> None:
        if not self._manifest_index:
            # If there are no entries, remove the manifest file if it exists
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            return

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for pano_id in sorted(self._manifest_index.keys()):
                json.dump(self._manifest_index[pano_id], f, ensure_ascii=False)
                f.write("\n")

    def get_manifest_entry(self, pano_id: str) -> Optional[Dict]:
        return self._manifest_index.get(pano_id)

    def record_manifest_entry(
        self,
        metadata: PanoramaMetadata,
        image_path: Path,
        image_hash: str,
        request_params: Optional[Dict] = None,
        attribution: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record or update manifest entry for a panorama asset."""
        try:
            relpath = image_path.relative_to(self.cache_root)
        except ValueError:
            relpath = image_path.name

        entry = {
            "provider": self.provider_name,
            "pano_id": metadata.pano_id,
            "lat": self._to_float(metadata.lat),
            "lon": self._to_float(metadata.lon),
            "heading": self._to_float(metadata.heading),
            "pitch": self._to_float(metadata.pitch),
            "roll": self._to_float(metadata.roll),
            "date": metadata.date,
            "elevation": self._to_float(metadata.elevation),
            "links": metadata.links,
            "image_relpath": str(relpath),
            "image_sha256": image_hash,
        }
        if request_params:
            entry["request_params"] = request_params
        if attribution:
            entry["attribution"] = attribution

        self._manifest_index[metadata.pano_id] = entry
        self._write_manifest()

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Attribution helpers
    # ------------------------------------------------------------------
    def _ensure_attribution(self, attribution: Dict[str, str]) -> None:
        text_lines = ["# Attribution", ""]
        for key, value in attribution.items():
            text_lines.append(f"- **{key.capitalize()}**: {value}")
        content = "\n".join(text_lines) + "\n"

        if self.attribution_path.exists():
            existing = self.attribution_path.read_text(encoding="utf-8")
            if content == existing:
                return
        self.attribution_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Image path helpers
    # ------------------------------------------------------------------
    def get_image_filename(
        self,
        pano_id: str,
        heading: Optional[float] = None,
        pitch: Optional[float] = None,
        extension: str = "jpg",
    ) -> str:
        sanitized_id = pano_id.replace("/", "_").replace("\\", "_")
        components = [self.provider_name, sanitized_id]
        if heading is not None:
            components.append(f"h{int(round(heading))}")
        if pitch is not None:
            components.append(f"p{int(round(pitch))}")
        filename = "_".join(components) + f".{extension}"
        return filename

    def get_image_path(
        self,
        pano_id: str,
        heading: Optional[float] = None,
        pitch: Optional[float] = None,
        extension: str = "jpg",
    ) -> Path:
        return self.images_dir / self.get_image_filename(
            pano_id=pano_id, heading=heading, pitch=pitch, extension=extension
        )

    def get_existing_image_path(self, pano_id: str) -> Optional[Path]:
        entry = self.get_manifest_entry(pano_id)
        if entry:
            relpath = entry.get("image_relpath")
            if relpath:
                candidate = self.cache_root / relpath
                if candidate.exists():
                    return candidate
        default_path = self.get_image_path(pano_id)
        return default_path if default_path.exists() else None

    def image_exists(self, pano_id: str) -> bool:
        return self.get_existing_image_path(pano_id) is not None

    # ------------------------------------------------------------------
    # Convenience serialization helpers
    # ------------------------------------------------------------------
    def export_manifest(self) -> Dict[str, Dict]:
        """Return a shallow copy of manifest data for external consumers."""
        return dict(self._manifest_index)


__all__ = ["CacheManager"]


from pydantic import BaseModel
from typing import Optional, List


class PanoramaLink(BaseModel):
    """Represents a link to another panorama with direction information."""
    pano: 'Panorama'
    direction: float


class Panorama(BaseModel):
    """Google Street View Panorama model."""
    id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float] = None
    roll: Optional[float] = None
    date: Optional[str] = None
    elevation: Optional[float] = None
    links: Optional[List[PanoramaLink]] = None

    model_config = {
        "validate_assignment": True,
        "populate_by_name": True
    }

    @property
    def pano_id(self) -> str:
        """Alias for id for compatibility."""
        return self.id
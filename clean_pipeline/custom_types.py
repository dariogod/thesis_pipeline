from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from conversions import RGBColor255

class BBox(BaseModel):
    x1: int
    y1: int 
    x2: int
    y2: int
    
    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, coords: List[int]) -> "BBox":
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

class DominantColors(BaseModel):
    background: RGBColor255 | None
    jersey: RGBColor255 | None

class TrackInfo(BaseModel):
    """Represents tracking information for a detected object"""
    bbox: BBox
    age: int = 0
    confidence: float = Field(ge=0, le=1)
    last_seen: int = 0

class MinimapCoordinates(BaseModel):
    x: int
    y: int
    x_max: int
    y_max: int

class Detection(BaseModel):
    """Represents a single object detection"""
    bbox: BBox
    roi_bbox: BBox | None = None
    confidence: float = Field(ge=0, le=1)
    track_id: Optional[int] = None
    class_name: Literal["person", "sports ball"]
    jersey_color: RGBColor255 | None = None
    role: Literal["TEAM A", "TEAM B", "REF/GK", "REF", "GK", "UNK", "OOB"] | None = None
    minimap_coordinates: MinimapCoordinates | None = None

class FrameDetections(BaseModel):
    """Represents all detections in a single frame"""
    frame_id: int
    detections: List[Detection]

class TrackRole(BaseModel):
    track_id: int
    role: Literal["TEAM A", "TEAM B", "REF/GK", "REF", "GK", "UNK", "OOB"]
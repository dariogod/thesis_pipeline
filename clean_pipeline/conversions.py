from skimage import color
import numpy as np
from pydantic import BaseModel, Field


class RGBColor255(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)

    def to_normalized(self) -> "RGBColorNormalized":
        return RGBColorNormalized(r=self.r / 255, g=self.g / 255, b=self.b / 255)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> "RGBColor255":
        return cls(r=array[0], g=array[1], b=array[2])

class RGBColorNormalized(BaseModel):
    r: float = Field(ge=0, le=1)
    g: float = Field(ge=0, le=1)
    b: float = Field(ge=0, le=1)

    def to_255(self) -> "RGBColor255":
        return RGBColor255(r=int(self.r * 255), g=int(self.g * 255), b=int(self.b * 255))
    
    def to_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> "RGBColorNormalized":
        return cls(r=array[0], g=array[1], b=array[2])

class LABColor(BaseModel):
    l: float = Field(ge=0, le=100)
    a: float = Field(ge=-128, le=127)
    b: float = Field(ge=-128, le=127)

    def to_array(self) -> np.ndarray:
        return np.array([self.l, self.a, self.b])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> "LABColor":
        return cls(l=array[0], a=array[1], b=array[2])

## Conversions

def lab_to_rgb_255(lab: LABColor | np.ndarray) -> RGBColor255 | np.ndarray:
    if isinstance(lab, LABColor):
        rgb_normalized = color.lab2rgb(lab.to_array())
        return RGBColor255.from_array(np.clip(np.round(rgb_normalized * 255), 0, 255))
    
    else: # np.ndarray (image)
        rgb_normalized = color.lab2rgb(lab)
        return rgb_normalized * 255
    
def rgb_to_lab(rgb: RGBColor255 | RGBColorNormalized | np.ndarray) -> LABColor | np.ndarray:
    if isinstance(rgb, (RGBColor255, RGBColorNormalized)):
        if isinstance(rgb, RGBColor255):
            rgb_normalized = rgb.to_normalized()
        else:
            rgb_normalized = rgb
        lab_array = color.rgb2lab(rgb_normalized.to_array())
        return LABColor.from_array(lab_array)
    
    else: # np.ndarray (image)
        if rgb.max() > 1.0:
            rgb_normalized = rgb.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb.astype(np.float32)
        
        lab_image = color.rgb2lab(rgb_normalized)
        return lab_image

from pydantic import BaseModel
from typing import List, Any

class CocoImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int

class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: List[float] # [x_min, y_min, width, height]

class CocoDataset(BaseModel):
    images: List[CocoImage]
    annotations: List[CocoAnnotation]
    categories: List[Any] = [] # Opcional para a conversão de bbox
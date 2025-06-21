from pydantic import BaseModel
from typing import Optional

class ProductDTO(BaseModel):
    Id: int
    Title: Optional[str] = None
    SKU: Optional[str] = None
    Price: Optional[float] = None
    Stock: Optional[int] = None
    Manufacturer: Optional[str] = None
    Categories: Optional[str] = None
    Status: Optional[str] = None
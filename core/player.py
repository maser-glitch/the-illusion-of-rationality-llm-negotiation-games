from typing import Optional
from pydantic import BaseModel

class Player(BaseModel):
    model: str
    region: Optional[str] = None
    optimized: bool = False


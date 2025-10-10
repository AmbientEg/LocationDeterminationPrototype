from pydantic import BaseModel, Field
from typing import List



class Position(BaseModel):
    """
    Represents the position calculated by the server
    and sent back to the mobile client.
    """
    x: float = Field(..., description="X-coordinate of the device position")
    y: float = Field(..., description="Y-coordinate of the device position")

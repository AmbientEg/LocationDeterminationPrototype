from pydantic import BaseModel, Field
from typing import List


class SensorData(BaseModel):
    """
    Represents the sensor data sent from the Flutter mobile app.
    This includes beacon identification, RSSI signal strength,
    accelerometer, and gyroscope readings.
    """
    uuid: str = Field(..., description="Beacon UUID (identifier for the BLE beacon)")
    major: int = Field(..., description="Major value of the beacon")
    minor: int = Field(..., description="Minor value of the beacon")
    rssi: int = Field(..., description="Received Signal Strength Indicator")
    accel: List[float] = Field(
        ..., 
        min_items=3, 
        max_items=3, 
        description="Accelerometer readings [x, y, z]"
    )
    gyro: List[float] = Field(
        ..., 
        min_items=3, 
        max_items=3, 
        description="Gyroscope readings [x, y, z]"
    )



import pandas as pd
from pydantic import BaseModel, validator
from typing import Dict, List


class BeaconReading(BaseModel):
    location: str
    date: str  # could also be datetime, but your file uses custom format
    beacons: Dict[str, int]

    # validate RSSI range
    @validator("beacons")
    def check_rssi(cls, v):
        for beacon, rssi in v.items():
            if not (-200 <= rssi <= 0):
                raise ValueError(f"Invalid RSSI value {rssi} for {beacon}")
        return v


def read_csv_with_pydantic(filepath: str) -> List[BeaconReading]:
    df = pd.read_csv(filepath)

    readings = []
    for _, row in df.iterrows():
        beacon_values = {col: int(row[col]) for col in df.columns if col.startswith("b")}
        reading = BeaconReading(
            location=row["location"],
            date=row["date"],
            beacons=beacon_values
        )
        readings.append(reading)

    return readings


if __name__ == "__main__":
    filepath = "data/sample.csv"
    readings = read_csv_with_pydantic(filepath)

    # Example: print first parsed row
    print(readings[0])
    print(readings[0].beacons)  # dict of beacon RSSI values

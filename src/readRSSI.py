import pandas as pd
from pydantic import BaseModel, validator
from typing import Dict, List, Optional

class BeaconReading(BaseModel):
    location: str
    date: str
    beacons: Dict[str, int]

    @validator("beacons")
    def check_rssi(cls, v):
        for beacon, rssi in v.items():
            if not (-200 <= rssi <= 0):
                raise ValueError(f"Invalid RSSI value {rssi} for {beacon}")
        return v


def readRSSI(filepath: str, top_n: Optional[int] = 3) -> List[BeaconReading]:
    """
    Read CSV and return a list of BeaconReading with top_n strongest readings per row.
    Only includes readings > -200.
    """
    df = pd.read_csv(filepath)
    readings = []

    for _, row in df.iterrows():
        # Keep only beacon columns
        beacon_cols = [c for c in df.columns if c.startswith("b")]
        beacon_values = {c: row[c] for c in beacon_cols if row[c] > -200}

        # Skip rows with fewer than top_n readings
        if len(beacon_values) < top_n:
            continue

        # Keep top_n strongest readings (highest RSSI)
        sorted_beacons = dict(sorted(beacon_values.items(), key=lambda x: x[1], reverse=True)[:top_n])

        readings.append(
            BeaconReading(
                location=row["location"],
                date=row["date"],
                beacons=sorted_beacons
            )
        )

    return readings

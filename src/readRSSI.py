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
            if not isinstance(rssi, (int, float)):
                raise ValueError(f"RSSI for {beacon} must be numeric, got {type(rssi)}")
            if not (-200 <= rssi <= 0):
                raise ValueError(f"Invalid RSSI value {rssi} for {beacon}")
        return v


def readRSSI(filepath: str, top_n: Optional[int] = 3) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # --- ✅ Convert beacon columns to numeric to avoid dtype 'object' issues ---
    beacon_cols = [c for c in df.columns if c.startswith("b")]
    df[beacon_cols] = df[beacon_cols].apply(pd.to_numeric, errors="coerce")

    readings = []

    for _, row in df.iterrows():
        # extract valid beacon readings (numeric and > -200)
        beacon_values = {
            c: float(row[c])
            for c in beacon_cols
            if pd.notna(row[c]) and row[c] > -200
        }

        # skip rows with too few valid beacons
        if len(beacon_values) < top_n:
            continue

        # sort by strongest RSSI (less negative = stronger)
        sorted_beacons = dict(
            sorted(beacon_values.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        # validate with Pydantic model
        readings.append(
            BeaconReading(
                location=str(row["location"]),
                date=str(row["date"]),
                beacons=sorted_beacons
            )
        )

    # Convert list of models to a clean DataFrame
    return pd.DataFrame(
        [
            {"location": r.location, "date": r.date, **r.beacons}
            for r in readings
        ]
    )

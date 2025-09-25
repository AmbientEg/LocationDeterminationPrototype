from src.readRSSI import readRSSI
from src.trilateration import trilaterate
from src.kalman import RSSIKalman
from src.log_normal_propagation import rssi_to_distance
import numpy as np

if __name__ == "__main__":
    # Step 1: Load dataset (RSSI values) with top 3 strongest per row
    readings = readRSSI("data/kaggel/iBeacon_RSSI_Labeled.csv", top_n=3)
    print("Sample cleaned data (first 3 rows):")
    for r in readings[:3]:
        print(f"{r.location} | {r.date} | {r.beacons}")

    # Step 2: Define known beacon positions
    beacon_positions = {
        "b3001": (6, 9),
        "b3002": (10, 4),
        "b3003": (14, 4),
        "b3004": (18, 4),
        "b3005": (10, 7),
        "b3006": (14, 7),
        "b3007": (18, 7),
        "b3008": (10, 10),
        "b3009": (4, 15),
        "b3010": (10, 15),
        "b3011": (14, 15),
        "b3012": (18, 15),
        "b3013": (23, 15)
    }

    # Step 3: Iterate readings → RSSI → distance → trilateration
    positions = []
    for reading in readings:
        beacon_ids = list(reading.beacons.keys())
        rssi_values = list(reading.beacons.values())

        # Skip if any beacon not in known positions
        if any(b not in beacon_positions for b in beacon_ids):
            continue

        # Convert RSSI → distances
        distances = [rssi_to_distance(rssi) for rssi in rssi_values]

        try:
            pos = trilaterate(
                beacon_positions[beacon_ids[0]], distances[0],
                beacon_positions[beacon_ids[1]], distances[1],
                beacon_positions[beacon_ids[2]], distances[2]
            )
            positions.append(pos)
        except Exception as e:
            print(f"Trilateration failed on row: {e}")
            continue

    print("\nTrilateration results (first 5):")
    print(positions[:5])

    # Step 4: Apply Kalman smoothing
    kf = RSSIKalman()
    smoothed = []
    for x, y in positions:
        kf.predict()
        state = kf.update(x, y)
        smoothed.append((state[0], state[1]))  # X, Y

    print("\nSmoothed positions (first 5):")
    print(smoothed[:5])

from src.readRSSI import readRSSI
from src.trilateration import trilaterate
from src.kalman import RSSIKalman
import numpy as np

if __name__ == "__main__":
    # Step 1: Load dataset (RSSI → distance, keep top 3 strongest beacons)
    df = readRSSI("data/kaggel/iBeacon_RSSI_Labeled.csv", top_n=3)
    print("Sample cleaned data:")
    print(df.head())

    # Step 2: Define known beacon positions
    """
    A 1, B 2, C 3, D 4 ,E 5, F 6, G 7, H 8, I 9,
   J 10, K 11, L 12, M 13, N 14, O 15, P 16, Q 17, R 18,
   S 19, T 20, U 21, V 22, W 23, X 24, Y 25, Z 26

    """
    beacon_positions = {
        "Beacon1": (6, 9),
        "Beacon2": (10, 4),
        "Beacon3": (14, 4),
        "Beacon4": (18, 4),
        "Beacon5": (10, 7),
        "Beacon6": (14, 7),
        "Beacon7": (18, 7),
        "Beacon8": (10, 10),
        "Beacon9": (4, 15),
        "Beacon10": (10, 15),
        "Beacon11": (14, 15),
        "Beacon12": (18, 15),
        "Beacon13": (23, 15)
    }

    # Step 3: Iterate rows and apply trilateration
    positions = []
    for _, row in df.iterrows():
        # Pick the top 3 valid beacons in this row
        row_beacons = row.dropna()[2:]  # drop Waypoint + Timestamp, keep only beacon cols
        if len(row_beacons) < 3:
            continue  # skip rows with <3 valid beacons

        # Get the 3 closest beacons
        top3 = row_beacons.nsmallest(3)

        # Extract positions + distances
        beacon_ids = top3.index.tolist()
        distances = top3.values.tolist()

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

    print("Trilateration results (first 5):")
    print(positions[:5])

    # Step 4: Smooth with Kalman filter
    kf = RSSIKalman()
    smoothed = []
    for (x, y) in positions:
        kf.predict()
        state = kf.update(x, y)
        smoothed.append((state[0], state[1]))  # X, Y

    print("Smoothed positions (first 5):")
    print(smoothed[:5])

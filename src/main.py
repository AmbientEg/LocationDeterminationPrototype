from src.readRSSI import readRSSI
from src.trilateration import trilaterate
from src.kalman import RSSIKalman
from src.log_normal_propagation import rssi_to_distance
import numpy as np
import os
import pandas as pd
from tabulate import tabulate

if __name__ == "__main__":
    # Step 1: Load dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "kaggle", "iBeacon_RSSI_Labeled.csv")

    df = readRSSI(DATA_PATH, top_n=3)
    print(f"\n📦 Original dataset size: {len(df)} rows")

    print("\n📡 Sample cleaned data:")
    print(df.head())

    # Step 2: Define beacon positions
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

    # Step 3: Trilateration with detailed logging
    records = []

    for _, row in df.iterrows():
        row_beacons = row.drop(["location", "date"], errors="ignore")
        row_beacons = pd.to_numeric(row_beacons, errors="coerce").dropna()

        if len(row_beacons) < 3:
            continue

        top3 = row_beacons.nlargest(3)
        beacon_ids = top3.index.tolist()
        rssi_vals = top3.values.tolist()
        distances = [rssi_to_distance(v) for v in rssi_vals]

        try:
            if any(np.isnan(distances)) or any(np.isinf(distances)) or any(d <= 0 for d in distances):
                continue

            pos = trilaterate(
                beacon_positions[beacon_ids[0]], distances[0],
                beacon_positions[beacon_ids[1]], distances[1],
                beacon_positions[beacon_ids[2]], distances[2]
            )

            if np.isnan(pos[0]) or np.isnan(pos[1]) or abs(pos[0]) > 1000 or abs(pos[1]) > 1000:
                continue

            records.append({
                "Location": row["location"],
                "Beacon 1": beacon_ids[0],
                "RSSI 1": round(rssi_vals[0], 1),
                "Dist 1 (m)": round(distances[0], 2),
                "Beacon 2": beacon_ids[1],
                "RSSI 2": round(rssi_vals[1], 1),
                "Dist 2 (m)": round(distances[1], 2),
                "Beacon 3": beacon_ids[2],
                "RSSI 3": round(rssi_vals[2], 1),
                "Dist 3 (m)": round(distances[2], 2),
                "X": round(pos[0], 3),
                "Y": round(pos[1], 3)
            })

        except Exception as e:
            print(f"⚠️ Trilateration failed on row: {e}")
            continue

    tril_df = pd.DataFrame(records)
    print("\n📍 Trilateration Results:")
    print(tabulate(tril_df.head(1000), headers="keys", tablefmt="fancy_grid"))

    # Step 4: Kalman Filter smoothing
    kf = RSSIKalman()
    smoothed_positions = []
    for _, r in tril_df.iterrows():
        kf.predict()
        state = kf.update(r["X"], r["Y"])
        smoothed_positions.append((float(state[0]), float(state[1])))

    smoothed_df = pd.DataFrame(smoothed_positions, columns=["X_smooth", "Y_smooth"])
    tril_df = pd.concat([tril_df, smoothed_df], axis=1)

    print("\n🧭 Smoothed Positions (Kalman Filter):")
    print(tabulate(tril_df.head(1000), headers="keys", tablefmt="fancy_grid"))

    print(f"\n✅ Total processed rows: {len(tril_df)}")

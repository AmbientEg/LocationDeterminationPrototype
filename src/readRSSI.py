import pandas as pd
import numpy as np
from src.log_normal_propagation import rssi_to_distance

def readRSSI(filepath, top_n=3):
    """
    Read BLE RSSI dataset into a clean DataFrame and keep only top N strongest beacons per row.

    Args:
        filepath (str): Path to the CSV file
        top_n (int): Number of strongest beacons to keep (default = 3)

    Returns:
        DataFrame: ['Waypoint', 'Timestamp', 'BeaconX'..] with NaN for weak/missing signals
    """
    # Define column names
    columns = ["Waypoint", "Timestamp"] + [f"Beacon{i}" for i in range(1, 14)]
    
    # Read CSV
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # Convert RSSI -> distance
    beacon_cols = [f"Beacon{i}" for i in range(1, 14)]
    for col in beacon_cols:
        df[col] = df[col].astype(float)
        df[col] = df[col].apply(lambda r: np.nan if r <= -200 else rssi_to_distance(r))
    
    # For each row: keep only top N strongest beacons, set others to NaN
    def keep_top_n(row):
        strongest = row[beacon_cols].nsmallest(top_n)  # distances → smaller = stronger signal
        mask = row[beacon_cols].index.isin(strongest.index)
        row[beacon_cols] = np.where(mask, row[beacon_cols], np.nan)
        return row
    
    df = df.apply(keep_top_n, axis=1)

    # Convert Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    
    return df

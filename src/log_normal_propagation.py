import numpy as np
import pandas as pd


def rssi_to_distance(rssi, tx_power=-59, n=2.0):
    """
    Convert RSSI (dBm) to distance (meters) using log-normal path loss model.
    
    Args:
        rssi (float): Measured RSSI in dBm
        tx_power (int): RSSI at 1m (default -59)
        n (float): Path-loss exponent (default 2.0)
    
    Returns:
        float: Estimated distance in meters, or NaN if RSSI is invalid
    """
    if pd.isna(rssi) or rssi <= -200:
        return np.nan
    return 10 ** ((tx_power - rssi) / (10 * n))

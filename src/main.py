from src.readRSSI import readRSSI
from src.trilateration import trilaterate
from src.kalman import RSSIKalman
import numpy as np

if __name__ == "__main__":
    # Step 1: Load dataset (already converted RSSI → distance)
    df = readRSSI("data/kaggel/iBeacon_RSSI_Labeled.csv", top_n=3)
    print(df.head())


    # Step 2: log_normal postition 

    # step 3: triletaration 

    # step 4: kalman

    #print balye



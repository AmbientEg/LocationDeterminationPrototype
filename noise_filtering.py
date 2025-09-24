from RSSIKalman import RSSIKalman
import numpy as np

def distance_from_rssi(rssi, RSSI_0=-50, d0=1.0, n=2.0):
    """
    Convert RSSI (dBm) into distance (m).
    RSSI_0 = reference power at d0 (1m typically).
    n = path-loss exponent (2 free-space, 2-4 indoors).
    """
    return d0 * 10 ** ((RSSI_0 - rssi) / (10 * n))


rssi_measurements = [-60, -62, -59, -80, -61, -60, -63, -59, -58]

kf = RSSIKalman(process_var=0.5, meas_var=4, init_rssi=rssi_measurements[0])

filtered_rssi = []

for rssi in rssi_measurements:
    est = kf.update(rssi)
    filtered_rssi.append(est)

print("Raw:      ", rssi_measurements)
print("Filtered: ", np.round(filtered_rssi, 2))


raw_distances = [distance_from_rssi(r) for r in rssi_measurements]
filtered_distances = [distance_from_rssi(r) for r in filtered_rssi]

print("Raw Distances:     ", np.round(raw_distances, 2))
print("Filtered Distances:", np.round(filtered_distances, 2))


true_distance = 2.0  # meters (example known ground truth)

mae_raw = np.mean([abs(d - true_distance) for d in raw_distances])
mae_filt = np.mean([abs(d - true_distance) for d in filtered_distances])

print(f"MAE Raw: {mae_raw:.2f} m,  MAE Filtered: {mae_filt:.2f} m")


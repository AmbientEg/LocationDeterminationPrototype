import numpy as np
from filterpy.kalman import KalmanFilter

class RSSIKalman:
    def __init__(self, process_var=1, meas_var=4, init_rssi=-60):
        # 1D filter (RSSI only)
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        
        # Initial state (RSSI value)
        self.kf.x = np.array([[init_rssi]])  
        
        # State transition (RSSI_t+1 ≈ RSSI_t)
        self.kf.F = np.array([[1]])          
        
        # Measurement function
        self.kf.H = np.array([[1]])          
        
        # Initial covariance
        self.kf.P *= 1000                   
        
        # Process noise (tune!)
        self.kf.Q = np.array([[process_var]]) 
        
        # Measurement noise (tune!)
        self.kf.R = np.array([[meas_var]])  

    def update(self, rssi):
        self.kf.predict()
        self.kf.update(rssi)
        return float(self.kf.x)

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class RSSIKalman:
    def __init__(self, dt=1.0, process_var=1.0, meas_var=4.0):
        """
        4-state Kalman Filter for 2D IPS relative positioning with dynamic velocities.
        State vector: [X, Y, Vx, Vy]
        Measurements: [X_meas, Y_meas]
        Velocities Vx, Vy are estimated by the filter.
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Initialize state: [X, Y, Vx, Vy] all zero
        self.kf.x = np.array([0., 0., 0., 0.])

        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Initial uncertainty
        self.kf.P *= 1000.0

        # Measurement noise
        self.kf.R = np.eye(2) * meas_var

        # Process noise for positions and velocities (must be 4x4)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=process_var, block_size=1)

        # Flag to store first measurement as reference
        self.initialized = False
        self.ref_pos = np.array([0., 0.])

    def predict(self):
        """Predict next state (X, Y, Vx, Vy)."""
        self.kf.predict()
        return self.kf.x.copy()

    def update(self, x_meas, y_meas):
        """Update Kalman filter with a new measurement."""
        z = np.array([x_meas, y_meas])

        if not self.initialized:
            # First measurement sets reference (0,0)
            self.ref_pos = z.copy()
            z = np.array([0., 0.])
            self.initialized = True
        else:
            # Convert measurement to relative coordinates
            z = z - self.ref_pos

        self.kf.update(z)
        return self.kf.x.copy()

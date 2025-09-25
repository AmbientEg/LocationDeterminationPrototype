import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class RSSIKalman:
    def __init__(self, dt=1.0, process_var=1.0, meas_var=4.0):
        """
        4-state Kalman Filter for 2D IPS relative positioning.
        
        State vector:
            [X, Y, Vx, Vy]^T (relative to initial reference)
        
        Measurements:
            [X_meas, Y_meas]^T from trilateration, considered relative to first measurement.
        
        Args:
            dt (float): Time step between measurements
            process_var (float): Process noise variance (velocity uncertainty)
            meas_var (float): Measurement noise variance
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Start at origin (0,0) as reference for relative position
        self.kf.x = np.array([0., 0., 0., 0.])

        # State transition matrix (constant velocity)
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
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

        # Process noise for constant-velocity model
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=process_var, block_size=2)

        # Flag to store the first measurement as reference
        self.initialized = False
        self.ref_pos = np.array([0., 0.])

    def predict(self):
        """Predict next state (relative position and velocity)."""
        self.kf.predict()
        return self.kf.x.copy()

    def update(self, x_meas, y_meas):
        """
        Update Kalman filter with a new measurement.

        For the first measurement, sets the reference point.

        Args:
            x_meas (float): measured X position
            y_meas (float): measured Y position

        Returns:
            np.ndarray: updated state [X, Y, Vx, Vy] relative to reference
        """
        z = np.array([x_meas, y_meas])

        if not self.initialized:
            # Set the first measurement as reference (0,0)
            self.ref_pos = z.copy()
            z = np.array([0., 0.])
            self.initialized = True
        else:
            # Convert measurement to relative coordinates
            z = z - self.ref_pos

        self.kf.update(z)
        return self.kf.x.copy()

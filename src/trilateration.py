import numpy as np
from src.log_normal_propagation import rssi_to_distance


def trilaterate(p1, r1, p2, r2, p3, r3):
    """
    Estimate 2D position using trilateration from 3 known points and their distances.

    Args:
        p1, p2, p3: Tuples representing the (x, y) coordinates of the beacons.
        r1, r2, r3: Distances from the unknown point to each beacon.

    Returns:
        (x, y): Estimated position coordinates.
    """
    
    # Unpack beacon coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    
    # Formulate the linear system using the trilateration equations:
    # (x - xi)^2 + (y - yi)^2 = ri^2
    # Subtracting first equation from the other two gives linear equations
    A = np.array([
        [2*(x2 - x1), 2*(y2 - y1)],
        [2*(x3 - x1), 2*(y3 - y1)]
    ])
    b = np.array([
        r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2,
        r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2
    ])

    # Solve the linear system A * [x, y] = b using least squares
    pos = np.linalg.lstsq(A, b, rcond=None)[0]

    # Return coordinates as floats
    return float(pos[0]), float(pos[1])


if __name__ == "__main__":
    # Known beacon positions
    b1 = (0, 0)
    b2 = (5, 0)
    b3 = (2, 4)

    # Example: REAL RSSI values measured from 3 beacons (in dBm)
    rssi1 = -68
    rssi2 = -70
    rssi3 = -65

    # Convert RSSI -> distances
    r1 = rssi_to_distance(rssi1)
    r2 = rssi_to_distance(rssi2)
    r3 = rssi_to_distance(rssi3)

    # Trilateration to estimate position
    estimated_pos = trilaterate(b1, r1, b2, r2, b3, r3)

    print("True position:       (2, 2)" )
    print("Estimated position: ", estimated_pos)

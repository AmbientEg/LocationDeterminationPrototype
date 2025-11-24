from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from models.SensorData import SensorData
from models.Position import Position

from src.trilateration import trilaterate
from src.kalman import RSSIKalman
from src.log_normal_propagation import rssi_to_distance

import numpy as np
from .logger_config import setup_logging

router = APIRouter()
logger = setup_logging()

# ----------------------------------------------------
# Beacon positions map
# ----------------------------------------------------
BEACON_POSITIONS = {
    "5A5AAF9F-39AA-FD41-212D-921F53C042DB": (3,7.3),
    "B9E56D39-880A-8151-FCD2-57200F8F8FC3": (3,3.5),
    "DDDFE23C-4A1B-1DD9-92E1-EC55F61D08BA": (5.5,5.5),
    "39F7713F-5A3A-D9E5-ED80-11B00AC8E99E": (8.6,7.3),
    "B4C16CE8-BF81-265F-692C-09E4EE2FD6CC": (8.6,3.6)
}

# ----------------------------------------------------
# Kalman filter instance (global)
# ----------------------------------------------------
kalman_filter = RSSIKalman()

# ----------------------------------------------------
# WebSocket Connection Manager
# ----------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for conn in self.active_connections:
            await conn.send_text(message)


manager = ConnectionManager()


# ----------------------------------------------------
# WebSocket Endpoint
# ----------------------------------------------------
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # Temporary buffer to store the latest RSSI readings per beacon
    beacon_buffer: dict[str, float] = {}

    try:
        while True:

            # ------------------------
            # Receive single beacon data from client
            # ------------------------
            raw_data = await websocket.receive_json()
            logger.info(f"Received raw data: {raw_data}")

            beacon_id = raw_data.get("uuid")
            rssi = raw_data.get("rssi")

            if beacon_id is None or rssi is None:
                continue

            # Save latest reading in buffer
            beacon_buffer[beacon_id] = float(rssi)

            # ------------------------
            # Prepare valid beacons
            # ------------------------
            valid_beacons = []
            for b_id, b_rssi in beacon_buffer.items():
                if b_id not in BEACON_POSITIONS:
                    continue

                distance = rssi_to_distance(b_rssi)
                if np.isnan(distance) or distance <= 0:
                    continue

                valid_beacons.append({
                    "id": b_id,
                    "rssi": b_rssi,
                    "distance": distance,
                    "pos": BEACON_POSITIONS[b_id]
                })

            if len(valid_beacons) < 3:
                await websocket.send_json({"error": "Waiting for at least 3 beacons"})
                continue

            # ------------------------
            # Select top 3 strongest signals
            # ------------------------
            top3 = sorted(valid_beacons, key=lambda x: x["rssi"], reverse=True)[:3]

            p1, d1 = top3[0]["pos"], top3[0]["distance"]
            p2, d2 = top3[1]["pos"], top3[1]["distance"]
            p3, d3 = top3[2]["pos"], top3[2]["distance"]

            # ------------------------
            # Trilateration
            # ------------------------
            try:
                x, y = trilaterate(p1, d1, p2, d2, p3, d3)
                if np.isnan(x) or np.isnan(y):
                    raise ValueError("Invalid trilateration output")
            except Exception as e:
                logger.error(f"Trilateration error: {e}")
                await websocket.send_json({"error": "Trilateration failed"})
                continue

            # ------------------------
            # Kalman Filter Smoothing
            # ------------------------
            kalman_filter.predict()
            state = kalman_filter.update(x, y)
            kx, ky = state[0], state[1]

            # ------------------------
            # Build response
            # ------------------------
            output = {
                "raw_position": {"x": float(x), "y": float(y)},
                "kalman_position": {"x": float(kx), "y": float(ky)},
                "beacons_used": [
                    {
                        "beacon_id": b["id"],
                        "rssi": b["rssi"],
                        "distance": b["distance"],
                    }
                    for b in top3
                ]
            }
            print(output)

            # response = kalman position x and y only
            response = output["kalman_position"]
            # ------------------------
            # Send output to client
            # ------------------------
            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected.")

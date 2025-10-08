from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from models.SensorData import SensorData
from models.Position import Position

from .logger_config import setup_logging



router = APIRouter()


# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------

logger = setup_logging()

# ----------------------------------------------------
# WebSocket Manager
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
        logger.info(f"Broadcasting message: {message}")
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# ----------------------------------------------------
# WebSocket Endpoint
# ----------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_json()
            logger.info(f"Received raw data: {raw_data}")

            # Validate and parse using Pydantic
            try:
                sensor_data = SensorData(**raw_data)
                logger.info(f"Validated sensor data: {sensor_data}")
            except Exception as e:
                await websocket.send_json({"error": f"Invalid data: {str(e)}"})
                continue

            # Compute (or mock) position
            # For now, simulate (replace with your real positioning logic)
            # kalman_filter = KalmanFilter(dim_x=2, dim_z=2)
            # position = kalman_filter.update([sensor_data.rssi, sensor_data.rssi])
            position = Position(x=4.32, y=2.85)

            # Send structured response
            await websocket.send_json(position.model_dump())

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected.")

import asyncio
import json
import random
import websockets

URL = "ws://127.0.0.1:8000/ws"  # Your WebSocket endpoint

# Generate random sensor data
def generate_sensor_data():
    return {
        "uuid": "E2C56DB5-DFFB-48D2",
        "major": random.randint(1, 5),
        "minor": random.randint(1, 10),
        "rssi": random.randint(-90, -30),
        "accel": [round(random.uniform(-10, 10), 2) for _ in range(3)],
        "gyro": [round(random.uniform(-1, 1), 3) for _ in range(3)],
    }

async def send_data():
    async with websockets.connect(URL) as websocket:
        for i in range(500):  # Send 500 intensive updates
            data = generate_sensor_data()
            await websocket.send(json.dumps(data))
            print(f"Sent: {data}")

            # Optionally receive server position response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                print(f"Received: {response}")
            except asyncio.TimeoutError:
                pass  # No response yet, continue sending

            # Very short delay to simulate high-frequency sensor updates
            await asyncio.sleep(0.01)

asyncio.run(send_data())

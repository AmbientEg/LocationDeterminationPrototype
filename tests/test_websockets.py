import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws" 
    async with websockets.connect(uri) as websocket:
        # Example sensor data (like Flutter app sends)
        sensor_data = {
            "uuid": "E2C56DB5-DFFB-48D2",
            "major": 1,
            "minor": 3,
            "rssi": -71,
            "accel": [0.1, 0.3, 9.6],
            "gyro": [0.01, -0.02, 0.005]
        }

        # Send the sensor data
        await websocket.send(json.dumps(sensor_data))
        print(f"Sent: {sensor_data}")

        # Receive server response (position)
        response = await websocket.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    asyncio.run(test_websocket())

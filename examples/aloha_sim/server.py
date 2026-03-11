# server.py

import asyncio
import websockets

async def handler(websocket):
    print("✅ Client connected!")

    # Example: wait for a message
    message = await websocket.recv()
    print(f"📥 Received: {message}")

    # Example: send back a response
    await websocket.send("Hello, client!")

async def main():
    server = await websockets.serve(handler, "0.0.0.0", 26000)
    print("🌐 WebSocket server running on ws://0.0.0.0:26000")
    await server.wait_closed()

asyncio.run(main())

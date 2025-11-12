# import asyncio
# import websockets
# import json
# import base64

# async def send_audio():
#     uri = "ws://localhost:8000/ws/transcribe"
#     async with websockets.connect(uri) as websocket:
#         with open("uploaded_pdfs/Recording.wav", "rb") as audio_file:
#             chunk_size = 16000  # ~0.5 sec
#             chunk_id = 0

#             while True:
#                 chunk = audio_file.read(chunk_size)
#                 if not chunk:
#                     break

#                 b64 = base64.b64encode(chunk).decode("utf-8")

#                 await websocket.send(json.dumps({
#                     "type": "audio_chunk",
#                     "chunk_id": chunk_id,
#                     "data": b64
#                 }))

#                 print(f"Sent chunk {chunk_id}")
#                 chunk_id += 1

#         # Send ping before closing (optional)
#         await websocket.send(json.dumps({"type": "ping"}))

#         # Keep receiving responses
#         try:
#             while True:
#                 response = await websocket.recv()
#                 print("Server:", response)
#         except websockets.ConnectionClosed:
#             print("Connection closed by server.")

# asyncio.run(send_audio())

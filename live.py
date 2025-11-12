# server.py
import asyncio
import base64
import json
import os
import tempfile
import torch
from typing import Optional
import string
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import whisperx
# from utils.device import get_device_and_compute_type
import soundfile as sf
import numpy as np
import random

app = FastAPI()
print("Loading WhisperX model (global) ... This happens only once")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
GLOBAL_MODEL = whisperx.load_model("large-v2", device=DEVICE, compute_type=COMPUTE_TYPE, language="en")
UPLOAD_DIR = "uploaded_pdfs"
\
# Per-connection processor
class AudioProcessor:
    def __init__(self, websocket: WebSocket, language: str = "en"):
        self.websocket = websocket
        self.language = language
        self.device = DEVICE
        self.compute_type = COMPUTE_TYPE
        self.model = GLOBAL_MODEL
        self._load_model()

    def _load_model(self):
        # load model once per connection (or share globally if desired)
        # warning: loading large models per connection is expensive. Prefer global shared model for many connections.
        # For demo/poC: load per connection
        print(f"Loading whisperx model on device: {self.device}, compute_type: {self.compute_type}")
        self.model = whisperx.load_model("medium.en", device=self.device, compute_type=self.compute_type, language=self.language)

    async def transcribe_bytes(self, wav_bytes: bytes, chunk_id: Optional[int] = None):
        # Write bytes to temp file as WAV, then run transcription
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        #     # print('ankit', tmp.name, tmp)
        #     tmp_path = tmp.name
        #     tmp.write(wav_bytes)
        file_path = os.path.join(UPLOAD_DIR, 'test'.join(random.choices(string.ascii_letters + string.digits, k=6)))
        logging.info(f"Saving file to {file_path}")
        try:
            print('wav_bytes inside transcribe_bytes')
            # run transcription
            result = self.model.transcribe(file_path)
            # result["text"] is final text for this chunk
            text = result.get("text", "")
            response = {
                "type": "final_transcript",
                "chunk_id": chunk_id,
                "text": text,
                "segments": result.get("segments", [])
            }
            await self.websocket.send_text(json.dumps(response))
        except Exception as exc:
            # Send error info back
            await self.websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass

# Simple connection manager (optional)
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, AudioProcessor] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        proc = AudioProcessor(websocket)
        self.active_connections[websocket] = proc
        return proc

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

manager = ConnectionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def root():
    return {"message": "Hello, World!"}

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    proc = await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Expect JSON messages
            msg = json.loads(data)
            mtype = msg.get("type")
            if mtype == "audio_chunk":
                # client should send base64-encoded WAV bytes
                b64 = msg.get("data")
                chunk_id = msg.get("chunk_id")
                if not b64:
                    await websocket.send_text(json.dumps({"type": "error", "message": "no data"}))
                    continue
                try:
                    wav_bytes = base64.b64decode(b64)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": f"base64 decode failed: {e}"}))
                    continue
                # Launch transcription in background so we can continue receiving
                asyncio.create_task(proc.transcribe_bytes(wav_bytes, chunk_id))
            elif mtype == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            else:
                await websocket.send_text(json.dumps({"type":"info","message":"unknown message type"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

from fastapi import UploadFile, File, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import os
from transcribe import transcribe_audio
from main import get_speaker_diarization_json
import logging
from pydantic import BaseModel
from typing import List
from utils import save_upload_file

class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

class DiarizationResponse(BaseModel):
    transcription: List[Segment]
    _id: str
    file: str
    duration: float
    speakers: int

app = FastAPI()
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
origins = [
    "*",  # allow all origins, or list specific domains like "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"],   # allow all headers
)

@app.get('/')
async def root():
    return {"message": "Hello, World!"}

@app.post('/upload-audio')
async def upload_audio(file: UploadFile = File(...)) -> Dict[str, str]:
    if not file.filename or not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are allowed.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logging.info(f"Saving file to {file_path}")
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        transcribed_text: str = await transcribe_audio(file_path)
        logging.info(f"transcribed_text: {transcribed_text}")
    except Exception as e:
        logging.error(f"Error during file upload or transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"transcription": transcribed_text}

@app.post('/speaker-diarization', response_model=DiarizationResponse)
async def diarize_audio(file: UploadFile = File(...)) -> DiarizationResponse:
    if not file.filename or not file.filename.endswith(".wav") and not file.filename.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only WAV and MP3 files are allowed.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logging.info(f"Saving file to {file_path}")
    try:
        # saved_path = f"uploaded_pdfs/{file.filename}"
        await save_upload_file(file, file_path)
        # with open(saved_path, "wb") as f:
        #     f.write(await file.read())
        diarization_text: str = await get_speaker_diarization_json(file_path)
        logging.info(f"diarization_text: {diarization_text}")
    except Exception as e:
        logging.error(f"Error during file upload or transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return diarization_text


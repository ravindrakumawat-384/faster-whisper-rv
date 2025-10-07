from fastapi import UploadFile, File, HTTPException, FastAPI
from typing import Dict
import os
from transcribe import transcribe_audio
import logging

app = FastAPI()
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    except Exception as e:
        logging.error(f"Error during file upload or transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"transcription": transcribed_text}
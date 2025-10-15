from fastapi import UploadFile
import shutil

async def save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination
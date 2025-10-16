from fastapi import UploadFile
import shutil
import hashlib

async def save_upload_file(upload_file: UploadFile, destination: str):
    # Save the uploaded file
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Create a new MD5 object for this file
    hash_md5 = hashlib.md5()
    
    # Compute MD5 of the saved file
    with open(destination, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

from pymongo import MongoClient
import uuid
from dotenv import load_dotenv

import os
load_dotenv(dotenv_path=".envv")
db_url = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
db_name = os.getenv("DATABASE_NAME", 'speaker_identification')

# Connect to MongoDB
client = MongoClient(db_url)

# Create or access database
db = client[db_name]

# Create or access collection
user_collection = db["users"]
diarization_collection = db["diarizations"]

def insert_with_uuid(collection, document):
    """Always insert documents with UUID as _id"""
    document["_id"] = str(uuid.uuid4())
    return collection.insert_one(document)
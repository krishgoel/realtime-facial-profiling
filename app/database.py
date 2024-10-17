"""Database operations for facial recognition system."""

import logging
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import Optional, List, Dict, Any
from .config import FacialRecognitionConfiguration as Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pc = Pinecone(api_key=Config.PINECONE_API_KEY)
client = MongoClient(Config.MONGO_URI)
db = client[Config.MONGO_DB_NAME]
collection = db[Config.MONGO_DB_COLLECTION_NAME]

if Config.INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=Config.INDEX_NAME,
        dimension=Config.FEATURE_VECTOR_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(Config.INDEX_NAME)

def insert_vector(vector: List[float], name: str, analysis: Dict[str, Any]) -> Optional[str]:
    """
    Insert the vector into the Pinecone index and the name into the MongoDB collection.
    
    Args:
        vector (List[float]): The feature vector to insert.
        name (str): The name of the person.
        analysis (Dict[str, Any]): The analysis of the person from Deepface.
        
    Returns:
        Optional[str]: The MongoDB ID of the inserted record or None if insertion failed.
    """
    if not isinstance(vector, list) or len(vector) != Config.FEATURE_VECTOR_DIMENSION:
        logger.error(f"Invalid vector format. Type: {type(vector)}, Length: {len(vector)}")
        return None

    try:
        mongo_record = collection.insert_one({
            'name': name,
            "analysis": analysis
        })
        mongo_id = str(mongo_record.inserted_id)

        index.upsert(vectors=[{"id": mongo_id, "values": vector}])
        logger.info(f"Inserted vector for name: {name} [Mongo ID: {mongo_id}]")
        return mongo_id
    
    except Exception as e:
        logger.error(f"Error inserting vector: {e}")
        return None

def search_vector(vector: List[float]) -> Optional[str]:
    """
    Search for a matching vector in the Pinecone index and return the MongoDB ID if found.
    
    Args:
        vector (List[float]): The feature vector to search.
        
    Returns:
        Optional[str]: The MongoDB ID of the matching record or None if no match found.
    """
    if not isinstance(vector, list) or len(vector) != Config.FEATURE_VECTOR_DIMENSION:
        logger.error("Invalid vector format.")
        return None

    try:
        results = index.query(vector=vector, top_k=1)
        if not results['matches']:
            logger.info("No vectors or records exist.")
            return None
        
        top_match = results['matches'][0]
        if top_match['score'] > Config.FACIAL_SIMILARITY_THRESHOLD:
            match_id = top_match['id']
            logger.info(f"Match found in Pinecone, ID: {match_id}")

            user_record = collection.find_one({'_id': ObjectId(match_id)})
            if user_record:
                logger.info(f"Match found in DB, MongoDB ID: {match_id}, Name: {user_record['name']}")
                return match_id
            else:
                logger.info("No match found in MongoDB.")
                return None
        else:
            logger.info("No suitable match found in Pinecone.")
            return None
    except Exception as e:
        logger.error(f"Error searching vector: {e}")
        return None
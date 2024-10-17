"""Module for feature vector extraction and facial analysis."""

import logging
import numpy as np
import os
from deepface import DeepFace
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_feature_vector(directory: str) -> Optional[np.ndarray]:
    """
    Calculate the average feature vector from all PNG images in the specified directory using DeepFace.

    Args:
    - directory (str): Path to the directory containing PNG images.

    Returns:
    - Optional[np.ndarray]: Average feature vector if successful, None otherwise.
    """
    feature_vectors = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            try:
                output = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet512",
                    enforce_detection=False
                )[0]
                
                if 'embedding' in output:
                    feature_vector = output['embedding']
                    feature_vectors.append(feature_vector)
                    logger.info(f"Feature vector for {img_path} calculated successfully.")
                else:
                    logger.error(f"Feature vector 'embedding' key not found in the output for {img_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {e}")

    if feature_vectors:
        feature_vectors = np.array(feature_vectors)
        average_vector = np.mean(feature_vectors, axis=0)
        return average_vector
    else:
        logger.info("No valid images processed.")
        return None

def analyze_features(directory: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes the feature vector of the first PNG image in a directory using DeepFace to get attributes like age, gender, and race.

    Args:
    - directory (str): Path to the directory containing PNG images.

    Returns:
    - Optional[Dict[str, Any]]: Dictionary containing analysis results if successful, None otherwise.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            try:
                output = DeepFace.analyze(
                    img_path=img_path,
                    actions=['age', 'gender', 'race']
                )[0]
                
                logger.info(f"Analysis Output: {output}")
                
                results = {
                    "age": output["age"],
                    "gender": output["dominant_gender"],
                    "race": output["dominant_race"]
                }
                return results
            except Exception as e:
                logger.error(f"Failed to analyze image {img_path}: {e}")
                return None
            
    logger.info("No PNG files found in the directory.")
    return None

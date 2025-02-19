"""Main module for facial recognition and servo tracking application."""

import os
import cv2
import logging
import shutil
import asyncio
import numpy as np
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Any, Dict

from .config import FacialRecognitionConfiguration as Config
from .utils import save_face_image, extract_ltrb_from_track
from .servo_tracking import open_port, move_servo, setup_and_process_video
from .vector import get_feature_vector, analyze_features
from .database import insert_vector, search_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_track(frame: np.ndarray, track: Any, img_width: int, img_height: int, frame_count: int) -> None:
    """
    Handle a single track, saving face images and processing feature vectors.

    Args:
    - frame (np.ndarray): The current frame.
    - track (Any): The track object.
    - img_width (int): Width of the frame.
    - img_height (int): Height of the frame.
    - frame_count (int): Current frame count.
    """
    logger.info(f"Processing track ID {track.track_id}")
    track_id = track.track_id
    x, y, w, h = extract_ltrb_from_track(track)

    if not hasattr(track, 'track_info'):
        track.track_info = {'images_saved': 0, 'dir_path': f"{Config.IMAGE_SAVE_DIR}/{track_id}"}
        os.makedirs(track.track_info['dir_path'], exist_ok=True)

    track_info = track.track_info

    if track_info['images_saved'] < Config.FACE_IMG_SAVE_LIMIT and frame_count % Config.FRAME_SKIP == 0:
        save_face_image(frame, x, y, w, h, track_info, track_id, img_width, img_height)

    if track_info['images_saved'] == Config.FACE_IMG_SAVE_LIMIT and 'feature_vector' not in track_info:
        await process_feature_vector(track_info, track_id)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

async def process_feature_vector(track_info: Dict[str, Any], track_id: int) -> None:
    """
    Process the feature vector for a track.

    Args:
    - track_info (Dict[str, Any]): Information about the track.
    - track_id (int): ID of the track.
    """
    async def generate_feature_vector():
        return get_feature_vector(track_info['dir_path'])
    
    async def facial_analysis():
        analysis = analyze_features(track_info['dir_path'])
        logger.info(f"Analysis for track ID {track_id}: {analysis}")
        return analysis
    
    feature_vector, analysis = await asyncio.gather(generate_feature_vector(), facial_analysis())

    if feature_vector is not None:
        feature_vector = feature_vector.tolist() if not isinstance(feature_vector, list) else feature_vector
        track_info['feature_vector'] = feature_vector
        logger.info(f"Feature vector for track ID {track_id}: {feature_vector}")
        match = search_vector(feature_vector)
        if match is None:
            name = "Temp"
            insert_vector(feature_vector, name, analysis)
    else:
        logger.warning(f"No feature vector generated for track ID {track_id}")

async def main() -> None:
    """Main function to run the facial recognition and servo tracking application."""
    if os.path.exists(Config.IMAGE_SAVE_DIR):
        shutil.rmtree(Config.IMAGE_SAVE_DIR)
    os.makedirs(Config.IMAGE_SAVE_DIR, exist_ok=True)

    # if not open_port():
    #     logger.error("Failed to open serial port. Exiting.")
    #     return

    try:
        await setup_and_process_video(video_source=2)
    except Exception as e:
        logger.error(f"An error occurred in the main loop: {e}")

if __name__ == "__main__":
    asyncio.run(main())

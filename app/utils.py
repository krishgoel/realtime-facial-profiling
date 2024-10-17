"""Utility functions for image processing and tracking."""

import os
import cv2
import logging
import numpy as np
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_face_image(frame: np.ndarray, x: int, y: int, w: int, h: int, 
                    track_info: Dict[str, Any], track_id: int, 
                    img_width: int, img_height: int) -> None:
    """
    Save a face image from the given frame.

    Args:
    - frame (np.ndarray): The input frame.
    - x, y, w, h (int): Bounding box coordinates and dimensions.
    - track_info (Dict[str, Any]): Information about the current track.
    - track_id (int): ID of the current track.
    - img_width (int): Width of the input frame.
    - img_height (int): Height of the input frame.
    """
    face_img_path = f"{track_info['dir_path']}/face_{track_id}_{track_info['images_saved']}.png"
    margin = 100
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(img_width, x + w + margin)
    y_end = min(img_height, y + h + margin)
    face_img = frame[int(y_start):int(y_end), int(x_start):int(x_end)]
    cv2.imwrite(face_img_path, face_img)

    track_info['images_saved'] += 1
    logger.info(f"Saving image for track ID: {track_id} at {face_img_path}")

def extract_ltrb_from_track(track: Any) -> Tuple[int, int, int, int]:
    """
    Extract left, top, right, bottom coordinates from a track.

    Args:
    - track (Any): The track object.

    Returns:
    - Tuple[int, int, int, int]: x, y, w, h coordinates.
    """
    ltrb = track.to_ltrb(orig=True)
    x = int(ltrb[0])
    y = int(ltrb[1])
    w = int(ltrb[2] - ltrb[0])
    h = int(ltrb[3] - ltrb[1])
    return x, y, w, h

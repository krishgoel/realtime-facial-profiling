import os
from dotenv import load_dotenv

load_dotenv()

class FacialRecognitionConfiguration:
    """Configuration class for the Facial Recognition application."""

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    INDEX_NAME = "facial-profiling"
    MONGO_DB_NAME = "facial_profiling"
    MONGO_DB_COLLECTION_NAME = "users"
    FEATURE_VECTOR_DIMENSION = 512
    FACIAL_SIMILARITY_THRESHOLD = 0.5

    # Servo tracking constants
    SERIAL_PORT = 'COM7'
    BAUDRATE = 1000000
    SCS_MOVING_SPEED = 3000
    SCS_MOVING_ACC = 150
    PAN_MAX = 3100
    PAN_MIN = 1450
    TILT_MAX = 3000
    TILT_MIN = 2250
    PAN_START = 2560
    TILT_START = 2625
    STEP_SIZE = 15
    X_THRESHOLD = 100
    Y_THRESHOLD = 50

    # Recognition constants
    MIN_DETECTION_CONFIDENCE = 0.5
    MAX_AGE = 10
    IMAGE_SAVE_DIR = "recognition"
    FACE_IMG_SAVE_LIMIT = 5
    FRAME_SKIP = 2

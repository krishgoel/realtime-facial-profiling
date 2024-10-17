import cv2
import asyncio
import numpy as np
import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS
from typing import Tuple
from .config import FacialRecognitionConfiguration as Config

def open_port() -> bool:
    """
    Open the serial port for servo communication.

    Returns:
        bool: True if port opened successfully, False otherwise.
    """
    portHandler = PortHandler(Config.SERIAL_PORT)
    if not portHandler.openPort():
        print("Failed to open the port")
        return False
    if not portHandler.setBaudRate(Config.BAUDRATE):
        print("Failed to change the baud rate")
        return False
    return True

def move_servo(pan_pos: int, tilt_pos: int) -> None:
    """
    Move the servo to the specified pan and tilt positions.

    Args:
        pan_pos (int): Pan position.
        tilt_pos (int): Tilt position.
    """
    print(f"Moving servos to Pan: {pan_pos}, Tilt: {tilt_pos}")
    packetHandler = sms_sts(PortHandler(Config.SERIAL_PORT))
    packetHandler.SyncWritePosEx(1, tilt_pos, Config.SCS_MOVING_SPEED, Config.SCS_MOVING_ACC)
    packetHandler.SyncWritePosEx(2, pan_pos, Config.SCS_MOVING_SPEED, Config.SCS_MOVING_ACC)
    packetHandler.groupSyncWrite.txPacket()
    packetHandler.groupSyncWrite.clearParam()

async def process_frame(frame: np.ndarray, face_detection: mp.solutions.face_detection.FaceDetection, 
                        tracker: DeepSort, current_pan: int, current_tilt: int) -> Tuple[np.ndarray, int, int]:
    """
    Process a single frame for face detection and tracking.

    Args:
        frame (np.ndarray): Input frame.
        face_detection (mp.solutions.face_detection.FaceDetection): Face detection model.
        tracker (DeepSort): DeepSort tracker.
        current_pan (int): Current pan position.
        current_tilt (int): Current tilt position.

    Returns:
        Tuple[np.ndarray, int, int]: Processed frame, updated pan position, updated tilt position.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    img_height, img_width = frame.shape[:2]
    first_confirmed = False

    if results.detections:
        bbs = []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * img_width)
            y = int(bboxC.ymin * img_height)
            w = int(bboxC.width * img_width)
            h = int(bboxC.height * img_height)
            bbs.append(([x, y, w, h], detection.score[0], 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        tracks = tracker.update_tracks(bbs, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            if not first_confirmed:
                x, y, w, h = track.to_ltwh()
                cx, cy = int(x + w / 2), int(y + h / 2)
                frame_center_x, frame_center_y = img_width // 2, img_height // 2
                distance_x = frame_center_x - cx
                distance_y = frame_center_y - cy

                cv2.line(frame, (frame_center_x, frame_center_y), (cx, cy), (255, 0, 0), 2)
                distance_label = f"x: {distance_x}, y: {distance_y}"
                cv2.putText(frame, distance_label, ((frame_center_x + cx) // 2, (frame_center_y + cy) // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if abs(distance_x) > Config.X_THRESHOLD or abs(distance_y) > Config.Y_THRESHOLD:
                    pan_step = -np.sign(distance_x) * Config.STEP_SIZE
                    tilt_step = -np.sign(distance_y) * Config.STEP_SIZE
                    current_pan = max(Config.PAN_MIN, min(Config.PAN_MAX, current_pan + pan_step))
                    current_tilt = max(Config.TILT_MIN, min(Config.TILT_MAX, current_tilt + tilt_step))
                    move_servo(current_pan, current_tilt)

                first_confirmed = True
    else:
        print("No faces detected.")

    return frame, current_pan, current_tilt

async def main():
    current_pan = Config.PAN_START
    current_tilt = Config.TILT_START
    move_servo(current_pan, current_tilt)
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE)
    tracker = DeepSort(max_age=Config.MAX_AGE)
    cap = cv2.VideoCapture(2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame, current_pan, current_tilt = await process_frame(frame, face_detection, tracker, current_pan, current_tilt)
        cv2.imshow('Face Tracking with Servo Control', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())

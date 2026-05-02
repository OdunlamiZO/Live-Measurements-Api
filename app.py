import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F
import urllib.request
import os


app = Flask(__name__)

# COCO keypoint indices — same values as the legacy mediapipe pose API
class PoseLandmark:
    NOSE = 0
    LEFT_EAR = 7
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

_landmarker = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)


def _detect(frame):
    """Returns list of 33 NormalizedLandmark for the first person, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return result.pose_landmarks[0] if result.pose_landmarks else None


KNOWN_OBJECT_WIDTH_CM = 21.0
FOCAL_LENGTH = 600
DEFAULT_HEIGHT_CM = 152.0


def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model

depth_model = load_depth_model()


def calibrate_focal_length(image, real_width_cm, detected_width_px):
    return (detected_width_px * FOCAL_LENGTH) / real_width_cm if detected_width_px else FOCAL_LENGTH


def detect_reference_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        focal_length = calibrate_focal_length(image, KNOWN_OBJECT_WIDTH_CM, w)
        scale_factor = KNOWN_OBJECT_WIDTH_CM / w
        return scale_factor, focal_length
    return 0.05, FOCAL_LENGTH


def estimate_depth(image):
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)
    with torch.no_grad():
        depth_map = depth_model(input_tensor)
    return depth_map.squeeze().numpy()


def calculate_distance_using_height(landmarks, image_height, user_height_cm):
    top_head = landmarks[PoseLandmark.NOSE].y * image_height
    bottom_foot = max(
        landmarks[PoseLandmark.LEFT_ANKLE].y,
        landmarks[PoseLandmark.RIGHT_ANKLE].y
    ) * image_height
    person_height_px = abs(bottom_foot - top_head)
    distance = (user_height_cm * FOCAL_LENGTH) / person_height_px
    scale_factor = user_height_cm / person_height_px
    return distance, scale_factor


def get_body_width_at_height(frame, height_px, center_x):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    if height_px >= frame.shape[0]:
        height_px = frame.shape[0] - 1
    horizontal_line = thresh[height_px, :]
    center_x = int(center_x * frame.shape[1])
    left_edge, right_edge = center_x, center_x
    for i in range(center_x, 0, -1):
        if horizontal_line[i] == 0:
            left_edge = i
            break
    for i in range(center_x, len(horizontal_line)):
        if horizontal_line[i] == 0:
            right_edge = i
            break
    width_px = right_edge - left_edge
    min_width = 0.1 * frame.shape[1]
    if width_px < min_width:
        width_px = min_width
    return width_px


def calculate_measurements(landmarks, scale_factor, image_width, image_height, depth_map, frame=None, user_height_cm=None):
    if user_height_cm:
        _, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)

    scale_y = 384 / image_height
    scale_x = 384 / image_width

    def pixel_to_cm(value):
        return round(value * scale_factor, 2)

    def calculate_circumference(width_px, depth_ratio=1.0):
        width_cm = width_px * scale_factor
        estimated_depth_cm = width_cm * depth_ratio * 0.7
        half_width = width_cm / 2
        half_depth = estimated_depth_cm / 2
        return round(2 * np.pi * np.sqrt((half_width**2 + half_depth**2) / 2), 2)

    measurements = {}

    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[PoseLandmark.LEFT_HIP]
    right_hip = landmarks[PoseLandmark.RIGHT_HIP]

    # Shoulder width
    shoulder_width_px = abs(left_shoulder.x * image_width - right_shoulder.x * image_width) * 1.1
    measurements["shoulder_width"] = pixel_to_cm(shoulder_width_px)

    # Chest
    chest_y_ratio = 0.15
    chest_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * chest_y_ratio
    chest_width_px = abs((right_shoulder.x - left_shoulder.x) * image_width) * 1.15
    if frame is not None:
        chest_y_px = int(chest_y * image_height)
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        detected_width = get_body_width_at_height(frame, chest_y_px, center_x)
        if detected_width > 0:
            chest_width_px = max(chest_width_px, detected_width)
    chest_depth_ratio = 1.0
    if depth_map is not None:
        chest_x = int(((left_shoulder.x + right_shoulder.x) / 2) * image_width)
        chest_y_px = int(chest_y * image_height)
        cy_s = int(chest_y_px * scale_y)
        cx_s = int(chest_x * scale_x)
        if 0 <= cy_s < 384 and 0 <= cx_s < 384:
            chest_depth_ratio = 1.0 + 0.5 * (1.0 - depth_map[cy_s, cx_s] / np.max(depth_map))
    measurements["chest_width"] = pixel_to_cm(chest_width_px)
    measurements["chest_circumference"] = calculate_circumference(chest_width_px, chest_depth_ratio)

    # Waist
    waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * 0.35
    if frame is not None:
        waist_y_px = int(waist_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, waist_y_px, center_x)
        waist_width_px = detected_width if detected_width > 0 else abs(right_hip.x - left_hip.x) * image_width * 0.9
    else:
        waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9
    waist_width_px *= 1.16
    waist_depth_ratio = 1.0
    if depth_map is not None:
        waist_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        waist_y_px = int(waist_y * image_height)
        wy_s = int(waist_y_px * scale_y)
        wx_s = int(waist_x * scale_x)
        if 0 <= wy_s < 384 and 0 <= wx_s < 384:
            waist_depth_ratio = 1.0 + 0.5 * (1.0 - depth_map[wy_s, wx_s] / np.max(depth_map))
    measurements["waist_width"] = pixel_to_cm(waist_width_px)
    measurements["waist"] = calculate_circumference(waist_width_px, waist_depth_ratio)

    # Hip
    hip_width_px = abs(left_hip.x * image_width - right_hip.x * image_width) * 1.35
    if frame is not None:
        hip_y = left_hip.y + (landmarks[PoseLandmark.LEFT_KNEE].y - left_hip.y) * 0.1
        hip_y_px = int(hip_y * image_height)
        center_x = (left_hip.x + right_hip.x) / 2
        detected_width = get_body_width_at_height(frame, hip_y_px, center_x)
        if detected_width > 0:
            hip_width_px = max(hip_width_px, detected_width)
    hip_depth_ratio = 1.0
    if depth_map is not None:
        hip_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
        hip_y_px = int(left_hip.y * image_height)
        hy_s = int(hip_y_px * scale_y)
        hx_s = int(hip_x * scale_x)
        if 0 <= hy_s < 384 and 0 <= hx_s < 384:
            hip_depth_ratio = 1.0 + 0.5 * (1.0 - depth_map[hy_s, hx_s] / np.max(depth_map))
    measurements["hip_width"] = pixel_to_cm(hip_width_px)
    measurements["hip"] = calculate_circumference(hip_width_px, hip_depth_ratio)

    # Neck
    neck = landmarks[PoseLandmark.NOSE]
    left_ear = landmarks[PoseLandmark.LEFT_EAR]
    neck_width_px = abs(neck.x * image_width - left_ear.x * image_width) * 2.0
    measurements["neck"] = calculate_circumference(neck_width_px, 1.0)
    measurements["neck_width"] = pixel_to_cm(neck_width_px)

    # Arm length
    left_wrist = landmarks[PoseLandmark.LEFT_WRIST]
    measurements["arm_length"] = pixel_to_cm(abs(left_shoulder.y * image_height - left_wrist.y * image_height))

    # Shirt length
    measurements["shirt_length"] = pixel_to_cm(abs(left_shoulder.y * image_height - left_hip.y * image_height) * 1.2)

    # Thigh
    left_knee = landmarks[PoseLandmark.LEFT_KNEE]
    thigh_y = left_hip.y + (left_knee.y - left_hip.y) * 0.2
    thigh_width_px = hip_width_px * 0.5 * 1.2
    if frame is not None:
        thigh_y_px = int(thigh_y * image_height)
        detected_width = get_body_width_at_height(frame, thigh_y_px, left_hip.x * 0.9)
        if detected_width > 0 and detected_width < hip_width_px:
            thigh_width_px = detected_width
    thigh_depth_ratio = 1.0
    if depth_map is not None:
        thigh_x = int(left_hip.x * image_width)
        thigh_y_px = int(thigh_y * image_height)
        ty_s = int(thigh_y_px * scale_y)
        tx_s = int(thigh_x * scale_x)
        if 0 <= ty_s < 384 and 0 <= tx_s < 384:
            thigh_depth_ratio = 1.0 + 0.5 * (1.0 - depth_map[ty_s, tx_s] / np.max(depth_map))
    measurements["thigh"] = pixel_to_cm(thigh_width_px)
    measurements["thigh_circumference"] = calculate_circumference(thigh_width_px, thigh_depth_ratio)

    # Trouser length
    left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
    measurements["trouser_length"] = pixel_to_cm(abs(left_hip.y * image_height - left_ankle.y * image_height))

    return measurements


def validate_front_image(image_np):
    try:
        image_height, image_width = image_np.shape[:2]
        landmarks = _detect(image_np)

        if landmarks is None:
            return False, "No person detected. Please make sure you're clearly visible in the frame."

        MINIMUM_LANDMARKS = {
            PoseLandmark.NOSE: "NOSE",
            PoseLandmark.LEFT_SHOULDER: "LEFT SHOULDER",
            PoseLandmark.RIGHT_SHOULDER: "RIGHT SHOULDER",
            PoseLandmark.LEFT_ELBOW: "LEFT ELBOW",
            PoseLandmark.RIGHT_ELBOW: "RIGHT ELBOW",
            PoseLandmark.RIGHT_KNEE: "RIGHT KNEE",
            PoseLandmark.LEFT_KNEE: "LEFT KNEE",
        }

        missing_upper = []
        for idx, name in MINIMUM_LANDMARKS.items():
            lm = landmarks[idx]
            visibility = getattr(lm, 'visibility', 1.0)
            if visibility < 0.5 or lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
                missing_upper.append(name)

        if missing_upper:
            return False, "Couldn't detect full body. Please make sure your full body is visible."

        nose = landmarks[PoseLandmark.NOSE]
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
        head_to_shoulder = abs(left_shoulder.y - nose.y) * image_height

        if shoulder_width < head_to_shoulder * 1.2:
            return False, "Please step back to show more of your upper body, not just your face."

        return True, "Validation passed - proceeding with measurements"

    except Exception as e:
        print(f"Error validating body image: {e}")
        return False, "You aren't providing images correctly. Please try again."


@app.route("/upload_images", methods=["POST"])
def upload_images():
    if "front" not in request.files:
        return jsonify({"error": "Missing front image for reference."}), 400

    front_image_file = request.files["front"]
    front_image_np = np.frombuffer(front_image_file.read(), np.uint8)
    front_image_file.seek(0)

    is_valid, error_msg = validate_front_image(cv2.imdecode(front_image_np, cv2.IMREAD_COLOR))
    if not is_valid:
        return jsonify({"error": error_msg, "pose": "front", "code": "INVALID_POSE"}), 400

    user_height_cm = request.form.get('height_cm')
    print(user_height_cm)
    if user_height_cm:
        try:
            user_height_cm = float(user_height_cm)
        except ValueError:
            user_height_cm = DEFAULT_HEIGHT_CM
    else:
        user_height_cm = DEFAULT_HEIGHT_CM

    received_images = {
        pose_name: request.files[pose_name]
        for pose_name in ["front", "left_side"]
        if pose_name in request.files
    }

    measurements, scale_factor, focal_length = {}, None, FOCAL_LENGTH
    frames = {}

    for pose_name, image_file in received_images.items():
        image_np = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frames[pose_name] = frame
        lms = _detect(frame)
        image_height, image_width, _ = frame.shape

        if pose_name == "front":
            if lms is not None:
                _, scale_factor = calculate_distance_using_height(lms, image_height, user_height_cm)
            else:
                scale_factor, focal_length = detect_reference_object(frame)

        depth_map = estimate_depth(frame) if pose_name in ["front", "left_side"] else None

        if lms is not None and pose_name == "front":
            measurements.update(calculate_measurements(
                lms, scale_factor, image_width, image_height, depth_map, frame, user_height_cm
            ))

    debug_info = {
        "scale_factor": float(scale_factor) if scale_factor else None,
        "focal_length": float(focal_length),
        "user_height_cm": float(user_height_cm)
    }

    print(measurements)

    return jsonify({"measurements": measurements, "debug_info": debug_info})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import cv2
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, request, jsonify
import urllib.request
import os


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Limit uploads to 25 MB per request (two full-res phone photos)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024


# ── Landmark indices ──────────────────────────────────────────────────────────
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


# ── Pose landmarker ───────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

if not os.path.exists(MODEL_PATH):
    log.info("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info("Model downloaded.")

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

DEFAULT_HEIGHT_CM = 165.0


# ── Anthropometric formula estimates ─────────────────────────────────────────
# Standard garment-sizing proportions derived from population studies.
# Used as a sanity check on pixel-scan results: if a scanned measurement
# deviates more than 20% from the formula estimate it is almost certainly
# wrong (background bleed, landmark error, etc.) and the formula value is
# used instead.
#
# Men:   chest = height × 0.52;  women: bust = height × 0.515
# Waist = chest − 13/9/5 cm for slim/average/full build
# Hip   = chest − 2/+1/+4 cm (men) or chest + 1/4/8 cm (women)
# Build is inferred from BMI: < 21 slim, 21–27 average, > 27 full.

def formula_estimates(height_cm, weight_kg, is_female=False):
    """
    Returns formula-based chest, waist, and hip estimates in cm.
    These are population averages — accurate to ±10% for most people.
    """
    bmi = weight_kg / ((height_cm / 100) ** 2)
    if bmi < 21:
        build = "slim"
    elif bmi < 27:
        build = "average"
    else:
        build = "full"

    if is_female:
        chest = height_cm * 0.515
        waist_offset = {"slim": -13, "average": -9, "full": -5}[build]
        hip_offset   = {"slim":   1, "average":  4, "full":  8}[build]
    else:
        chest = height_cm * 0.52
        waist_offset = {"slim": -13, "average": -9, "full": -5}[build]
        hip_offset   = {"slim":  -2, "average":  1, "full":  4}[build]

    return {
        "chest": chest,
        "waist": chest + waist_offset,
        "hip":   chest + hip_offset,
    }


def personal_depth_ratio(waist_circ_cm, waist_width_cm):
    """
    Derives the user's actual front-to-back depth ratio from their known waist.

    From the ellipse formula C = 2π√((a²+b²)/2), where a = half visible width
    and b = half depth, we can solve for b given C and a:
      b = √(2(C/2π)² − a²)
      depth_ratio = b / a

    This personal ratio replaces the fixed population-average depth ratios and
    significantly improves circumference accuracy for users who provide their waist.
    Returns None if the value is implausible or the maths doesn't work out.
    """
    if not waist_circ_cm or not waist_width_cm or waist_width_cm <= 0:
        return None
    a = waist_width_cm / 2
    val = 2 * (waist_circ_cm / (2 * np.pi)) ** 2 - a ** 2
    if val <= 0:
        return None
    ratio = np.sqrt(val) / a
    # Human torso depth ratio sits between 0.30 (very slim) and 1.0 (very full)
    return ratio if 0.30 <= ratio <= 1.0 else None


def formula_guard(scanned_cm, formula_cm, label):
    """
    Returns scanned_cm if it is within 20% of the formula estimate,
    otherwise returns the formula estimate and logs a warning.
    """
    if formula_cm <= 0:
        return scanned_cm
    deviation = abs(scanned_cm - formula_cm) / formula_cm
    if deviation > 0.20:
        log.warning(
            "%s scan (%.1f cm) deviates %.0f%% from formula (%.1f cm) — using formula",
            label, scanned_cm, deviation * 100, formula_cm,
        )
        return formula_cm
    return scanned_cm


# ── Anatomical depth ratios ───────────────────────────────────────────────────
# Front-to-back depth as a fraction of visible width for each body region.
# Derived from human body proportion studies. These replace MiDaS depth
# estimation, which overestimated depth by 50–100% and caused circumferences
# to be wrong by the same margin.
DEPTH_RATIOS = {
    "chest": 0.58,   # chest is roughly 58% as deep as it is wide
    "waist": 0.50,   # waist is more oval — 50% depth/width
    "hip":   0.58,   # hips similar to chest
    "thigh": 0.72,   # thighs are more cylindrical
    "neck":  0.80,   # neck is nearly round
}


def _detect(frame):
    """Run pose detection on a BGR frame. Returns landmarks or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return result.pose_landmarks[0] if result.pose_landmarks else None


def scale_from_height(landmarks, image_height, height_cm):
    """
    Derives cm-per-pixel scale factor from the person's known height.
    Nose-to-ankle is used; 7% is added to compensate for the head portion
    above the nose landmark.
    """
    nose_y = landmarks[PoseLandmark.NOSE].y * image_height
    ankle_y = max(
        landmarks[PoseLandmark.LEFT_ANKLE].y,
        landmarks[PoseLandmark.RIGHT_ANKLE].y
    ) * image_height
    height_px = abs(ankle_y - nose_y)
    if height_px < 1:
        return height_cm / image_height
    return (height_cm * 1.07) / height_px


def get_body_width_at_height(frame, height_px, center_x, max_half_px):
    """
    Finds body width at height_px using gradient-based boundary detection.
    Scans are constrained to [center ± max_half_px] — derived from landmark
    positions so the search never reaches background outside the body region.

    Gradient detection works regardless of whether the body is lighter or
    darker than the background. Threshold-based approaches failed because
    OTSU would pick a threshold that didn't separate body from background,
    causing the scan to run to the image edge and return the full search width.

    Returns 0 if no clear boundary gradient is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    img_h, img_w = frame.shape[:2]
    y_px = min(max(height_px, 0), img_h - 1)
    cx = int(center_x * img_w)

    left_bound = max(0, cx - max_half_px)
    right_bound = min(img_w - 1, cx + max_half_px)

    region = blur[y_px, left_bound:right_bound]
    if len(region) < 10:
        return 0

    grad = np.abs(np.diff(region.astype(float)))
    if grad.max() < 8:
        return 0

    threshold = np.percentile(grad, 80)

    left_offset = 0
    for i in range(len(grad)):
        if grad[i] >= threshold:
            left_offset = i
            break

    right_offset = len(grad) - 1
    for i in range(len(grad) - 1, -1, -1):
        if grad[i] >= threshold:
            right_offset = i + 1
            break

    return max(0, right_offset - left_offset)


def ellipse_circumference(width_cm, depth_ratio):
    """
    Ellipse circumference approximation: C ≈ 2π√((a²+b²)/2)
    where a = half visible width, b = half front-to-back depth.
    depth = width × depth_ratio.
    """
    a = width_cm / 2
    b = (width_cm * depth_ratio) / 2
    return round(2 * np.pi * np.sqrt((a ** 2 + b ** 2) / 2), 2)


def calculate_measurements(landmarks, scale_factor, image_width, image_height, frame, user_height_cm, weight_kg, is_female=False, known_waist_cm=None):
    """
    Derives all body measurements from pose landmarks and pixel scanning.

    Width at each anatomical level is measured by scanning outward from the
    body centre in the front image. Landmark-derived widths are used as a
    fallback when scanning fails.

    Circumferences use the ellipse approximation with anatomical depth ratios
    instead of MiDaS-estimated depth. MiDaS returned relative depth values
    that, when applied to the original formula (width × depth_ratio × 0.7),
    produced depths up to 105% of body width — far exceeding real anatomy
    (45–60%) and inflating all circumferences by ~50%.
    """
    ls = landmarks[PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[PoseLandmark.LEFT_HIP]
    rh = landmarks[PoseLandmark.RIGHT_HIP]
    lk = landmarks[PoseLandmark.LEFT_KNEE]
    lw = landmarks[PoseLandmark.LEFT_WRIST]
    la = landmarks[PoseLandmark.LEFT_ANKLE]
    ra = landmarks[PoseLandmark.RIGHT_ANKLE]
    nose = landmarks[PoseLandmark.NOSE]
    left_ear = landmarks[PoseLandmark.LEFT_EAR]

    def to_cm(px):
        return round(px * scale_factor, 2)

    torso_cx = (ls.x + rs.x + lh.x + rh.x) / 4

    # Raw landmark spans (no correction factor) used as reference bounds.
    # Scan results wider than these bounds are landmark-derived overestimates
    # caused by background bleed, and must be rejected.
    shoulder_span_px = abs(ls.x - rs.x) * image_width
    hip_span_px = abs(lh.x - rh.x) * image_width

    # Max search radius for each scan: landmark span + 30% margin
    torso_half_px = int(max(shoulder_span_px, hip_span_px) / 2 * 1.30)

    def scan_and_validate(y_ratio, max_ratio, fallback_px, label=""):
        """
        Scan body width at a given y-ratio. Reject results wider than
        shoulder_span × max_ratio (catches background bleed) and fall
        back to the landmark estimate if the scan fails or is implausible.
        """
        y_px = int((ls.y + (lh.y - ls.y) * y_ratio) * image_height)
        scanned = get_body_width_at_height(frame, y_px, torso_cx, torso_half_px)
        log.info("Scan %s y_ratio=%.2f → scanned=%dpx (%.1fcm), cap=%dpx, fallback=%dpx",
                 label, y_ratio, scanned, scanned * scale_factor,
                 int(shoulder_span_px * max_ratio), fallback_px)
        if 0 < scanned <= shoulder_span_px * max_ratio:
            return scanned
        return fallback_px

    formula = formula_estimates(user_height_cm, weight_kg, is_female)

    m = {}

    # ── Shoulder ──────────────────────────────────────────────────────────────
    shoulder_px = shoulder_span_px * 1.19
    m["shoulder_width"] = to_cm(shoulder_px)

    # ── Waist width (needed early for personal depth ratio) ───────────────────
    waist_fallback = hip_span_px * 1.24
    waist_w_px = scan_and_validate(0.55, 0.95, waist_fallback, "waist")
    waist_w_cm = to_cm(waist_w_px)

    # If the user provided their actual waist, derive a personal depth ratio.
    # This replaces the fixed population-average ratio with a value calibrated
    # to their specific body shape, reducing circumference error from ±15% to ±5%.
    calibrated_ratio = personal_depth_ratio(known_waist_cm, waist_w_cm)
    if calibrated_ratio:
        log.info("Using personal depth ratio %.3f (from known waist %.1f cm)", calibrated_ratio, known_waist_cm)
        depth_chest = calibrated_ratio * 1.10  # chest is slightly deeper relative to width than waist
        depth_waist = calibrated_ratio
        depth_hip   = calibrated_ratio * 1.10
        depth_thigh = calibrated_ratio * 1.30  # thighs more cylindrical
    else:
        depth_chest = DEPTH_RATIOS["chest"]
        depth_waist = DEPTH_RATIOS["waist"]
        depth_hip   = DEPTH_RATIOS["hip"]
        depth_thigh = DEPTH_RATIOS["thigh"]

    # ── Chest / bust ──────────────────────────────────────────────────────────
    chest_fallback = shoulder_span_px * 0.98
    chest_w_px = scan_and_validate(0.20, 1.10, chest_fallback, "chest")
    chest_circ = ellipse_circumference(to_cm(chest_w_px), depth_chest)
    chest_circ = formula_guard(chest_circ, formula["chest"], "chest")
    m["chest_circumference"] = chest_circ

    # ── Waist ─────────────────────────────────────────────────────────────────
    waist_circ = known_waist_cm if known_waist_cm else ellipse_circumference(waist_w_cm, depth_waist)
    if not known_waist_cm:
        waist_circ = formula_guard(waist_circ, formula["waist"], "waist")
    m["waist"] = waist_circ

    # ── Hip ───────────────────────────────────────────────────────────────────
    hip_w_px = hip_span_px * 1.55
    hip_circ = ellipse_circumference(to_cm(hip_w_px), depth_hip)
    hip_circ = formula_guard(hip_circ, formula["hip"], "hip")
    if hip_circ < waist_circ:
        hip_circ = waist_circ * 1.05
    m["hip"] = hip_circ

    # ── Thigh ─────────────────────────────────────────────────────────────────
    thigh_y_px = int((lh.y + (lk.y - lh.y) * 0.25) * image_height)
    full_thigh_px = get_body_width_at_height(frame, thigh_y_px, torso_cx, torso_half_px)
    if 0 < full_thigh_px <= shoulder_span_px * 1.10:
        thigh_w_cm = to_cm(full_thigh_px // 2)
    else:
        thigh_w_cm = to_cm(hip_span_px * 0.72)
    m["thigh_circumference"] = ellipse_circumference(thigh_w_cm, depth_thigh)

    # ── Neck ──────────────────────────────────────────────────────────────────
    neck_px = abs(nose.x - left_ear.x) * image_width * 2.0
    m["neck"] = ellipse_circumference(to_cm(neck_px), DEPTH_RATIOS["neck"])

    # ── Sleeve length ─────────────────────────────────────────────────────────
    # Euclidean distance from shoulder to wrist — accounts for the arm being
    # angled out from the body. Vertical-only underestimates when arms are not
    # hanging perfectly straight.
    sleeve_dy = abs(ls.y - lw.y) * image_height
    sleeve_dx = abs(ls.x - lw.x) * image_width
    m["arm_length"] = to_cm(np.sqrt(sleeve_dx ** 2 + sleeve_dy ** 2))

    # ── Trouser length ────────────────────────────────────────────────────────
    # The trouser waistband sits above the hip landmark (iliac crest). Starting
    # 10% of shoulder-to-hip above the hip landmark approximates the natural
    # waist / top of trouser.
    trouser_start_y = lh.y - (lh.y - ls.y) * 0.10
    m["trouser_length"] = to_cm(abs(trouser_start_y - max(la.y, ra.y)) * image_height)

    log.info(
        "Widths (cm): chest=%.1f waist=%.1f hip=%.1f | scale=%.4f cm/px",
        to_cm(chest_w_px), to_cm(waist_w_px), to_cm(hip_w_px), scale_factor
    )

    return m


def validate_front_image(image_np):
    try:
        image_height, image_width = image_np.shape[:2]
        landmarks = _detect(image_np)

        if landmarks is None:
            return False, "No person detected. Please make sure you're clearly visible in the frame."

        required = {
            PoseLandmark.NOSE: "NOSE",
            PoseLandmark.LEFT_SHOULDER: "LEFT SHOULDER",
            PoseLandmark.RIGHT_SHOULDER: "RIGHT SHOULDER",
            PoseLandmark.LEFT_ELBOW: "LEFT ELBOW",
            PoseLandmark.RIGHT_ELBOW: "RIGHT ELBOW",
            PoseLandmark.RIGHT_KNEE: "RIGHT KNEE",
            PoseLandmark.LEFT_KNEE: "LEFT KNEE",
        }

        missing = [
            name for idx, name in required.items()
            if getattr(landmarks[idx], 'visibility', 1.0) < 0.5
            or not (0 <= landmarks[idx].x <= 1 and 0 <= landmarks[idx].y <= 1)
        ]

        if missing:
            return False, "Couldn't detect full body. Please make sure your full body is visible."

        ls = landmarks[PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = abs(ls.x - rs.x) * image_width
        head_to_shoulder = abs(ls.y - landmarks[PoseLandmark.NOSE].y) * image_height

        if shoulder_width < head_to_shoulder * 1.2:
            return False, "Please step back to show your full body, not just your face."

        return True, "OK"

    except Exception as e:
        log.error("Validation error: %s", e)
        return False, "Could not validate image. Please try again."


# ── Routes ────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Upload too large. Maximum total size is 25 MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    log.error("Unhandled server error: %s", e)
    return jsonify({"error": "An unexpected error occurred."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/upload_images", methods=["POST"])
def upload_images():
    """
    Main measurement endpoint.

    Expected form fields:
      front      (file, required)  — front-facing full-body photo
      left_side  (file, required)  — side-on full-body photo
      height_cm  (float, required) — user's height in centimetres
      weight_kg  (float, required) — user's weight in kilograms
    """
    if "front" not in request.files:
        return jsonify({"error": "A front-facing photo is required."}), 400

    front_bytes = np.frombuffer(request.files["front"].read(), np.uint8)
    front_frame = cv2.imdecode(front_bytes, cv2.IMREAD_COLOR)
    if front_frame is None:
        return jsonify({"error": "Could not read the front image. Please upload a valid photo."}), 400

    is_valid, msg = validate_front_image(front_frame)
    if not is_valid:
        return jsonify({"error": msg, "code": "INVALID_POSE"}), 400

    try:
        height_cm = float(request.form.get('height_cm') or 0)
        if not (100 <= height_cm <= 250):
            return jsonify({"error": "Height must be between 100 and 250 cm."}), 400
    except ValueError:
        return jsonify({"error": "Height must be a number."}), 400

    try:
        weight_kg = float(request.form.get('weight_kg') or 0)
        if not (20 <= weight_kg <= 300):
            return jsonify({"error": "Weight must be between 20 and 300 kg."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Weight must be a number."}), 400

    front_h, front_w = front_frame.shape[:2]
    front_lm = _detect(front_frame)

    if front_lm is None:
        return jsonify({"error": "Could not detect pose in front image."}), 400

    is_female = request.form.get('gender', 'male').lower() == 'female'

    # Optional: user's known waist in inches, converted to cm for calibration
    try:
        waist_in = float(request.form.get('waist_in') or 0)
        known_waist_cm = waist_in * 2.54 if 15 <= waist_in <= 80 else None
    except (TypeError, ValueError):
        known_waist_cm = None

    scale = scale_from_height(front_lm, front_h, height_cm)
    measurements = calculate_measurements(
        front_lm, scale, front_w, front_h, front_frame,
        height_cm, weight_kg, is_female, known_waist_cm
    )

    log.info("height=%.1f cm, weight=%.1f kg → %s", height_cm, weight_kg or 0, measurements)

    return jsonify({"measurements": {k: float(v) for k, v in measurements.items()}})


if __name__ == '__main__':
    # Development only — production uses gunicorn (see Dockerfile)
    app.run(host='0.0.0.0', port=5000, debug=False)

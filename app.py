import cv2
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, request, jsonify
import urllib.request
import os

if os.environ.get("LOG_FORMAT") == "json":
    from pythonjsonlogger import jsonlogger
    handler = logging.StreamHandler()
    handler.setFormatter(jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    ))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

log = logging.getLogger(__name__)

app = Flask(__name__)

# Limit uploads to 25 MB per request (two full-res phone photos)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024


# ── Scan guidelines ───────────────────────────────────────────────────────────
# Returned with validation errors so the client can display them to the user.
SCAN_GUIDELINES = [
    "Wear fitted clothing — no oversized, loose, or flowing garments",
    "Stand against a plain, uncluttered background",
    "Ensure good, even lighting with no harsh shadows",
    "Show your full body from head to toe",
    "Keep your arms slightly away from your body",
    "Hold the camera at waist or chest height",
    "Do not use mirror selfies",
    "Stand straight and face the camera directly (front photo)",
    "Stand sideways with arms at your sides (side photo)",
]


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

SEGMENTER_PATH = "selfie_segmenter.tflite"
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

if not os.path.exists(SEGMENTER_PATH):
    log.info("Downloading selfie segmenter model...")
    urllib.request.urlretrieve(SEGMENTER_URL, SEGMENTER_PATH)
    log.info("Segmenter model downloaded.")

_segmenter = mp_vision.ImageSegmenter.create_from_options(
    mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=SEGMENTER_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_confidence_masks=True,
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



def _formula_score(scanned_cm, formula_cm, label):
    """
    Returns (confidence_penalty, warning_or_None).
    Does NOT replace the scanned value — deviation reduces confidence and adds a warning.
    """
    if formula_cm <= 0:
        return 0.0, None
    deviation = abs(scanned_cm - formula_cm) / formula_cm
    log.info("Formula check %s: scanned=%.1f, formula=%.1f, deviation=%.0f%%",
             label, scanned_cm, formula_cm, deviation * 100)
    if deviation <= 0.10:
        return 0.0, None
    elif deviation <= 0.20:
        return 0.10, f"{label.capitalize()} is {deviation*100:.0f}% outside the expected range"
    elif deviation <= 0.35:
        return 0.20, f"{label.capitalize()} deviates {deviation*100:.0f}% from expected — possible clothing interference"
    else:
        return 0.35, f"{label.capitalize()} deviates {deviation*100:.0f}% from expected — result may be unreliable"


def _mask_quality(mask, image_h, image_w):
    """Returns (confidence_penalty, warning_or_None) based on segmentation mask coverage."""
    if mask is None:
        return 0.15, "Segmentation unavailable — gradient detection used"
    person_ratio = mask.sum() / (image_h * image_w)
    if 0.12 <= person_ratio <= 0.70:
        return 0.0, None
    elif person_ratio < 0.12:
        return 0.10, "Person appears small in frame — step closer for better accuracy"
    else:
        return 0.10, "Background complexity may affect boundary detection"


def _landmark_score(landmarks, indices):
    """Returns (confidence_penalty, warning_or_None) based on landmark visibility."""
    visibilities = [getattr(landmarks[i], 'visibility', 1.0) for i in indices]
    avg = sum(visibilities) / len(visibilities)
    if avg >= 0.75:
        return 0.0, None
    elif avg >= 0.55:
        return 0.10, "Reduced landmark visibility — measurement may be less precise"
    else:
        return 0.25, "Low landmark visibility — measurement confidence is reduced"


def _build_measurement(value, penalties, warnings):
    """Assembles the final measurement dict, clamping confidence to [0, 1]."""
    return {
        "value": round(float(value), 1),
        "confidence": round(max(0.0, min(1.0, 1.0 - sum(penalties))), 2),
        "warnings": [w for w in warnings if w],
    }


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


# ── Image helpers ─────────────────────────────────────────────────────────────

def decode_image(file_storage):
    """Decode an uploaded file into a BGR numpy array. Returns None on failure."""
    buf = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def get_segmentation_mask(frame, threshold=0.5):
    """
    Returns a binary uint8 mask (1=person, 0=background) using MediaPipe
    Selfie Segmentation, or None if segmentation fails.

    Morphological closing fills small holes inside the silhouette (gaps between
    limbs, clothing texture) and opening removes isolated noise pixels outside
    the body. Both improve the accuracy of width measurements taken from the mask.
    """
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _segmenter.segment(mp_img)
        if not result.confidence_masks:
            return None
        confidence = result.confidence_masks[0].numpy_view()
        mask = (confidence > threshold).astype(np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_kernel)
        return mask
    except Exception as e:
        log.warning("Segmentation failed: %s — will fall back to gradient scanning", e)
        return None


def width_from_mask(mask, y_px):
    """
    Returns body width in pixels at row y_px by finding the leftmost and
    rightmost person pixel in the segmentation mask.

    Returns 0 if fewer than 5 body pixels are found (noise rejection).
    """
    y = min(max(y_px, 0), mask.shape[0] - 1)
    person_cols = np.where(mask[y, :] > 0)[0]
    if len(person_cols) < 5:
        return 0
    return int(person_cols[-1] - person_cols[0])


def is_blurry(frame, threshold=80):
    """
    Returns True if the image is too blurry to reliably extract measurements.
    Uses the variance of the Laplacian — a low variance means little edge
    detail, which indicates blur. Threshold of 80 rejects noticeably blurry
    phone photos while accepting typical handheld shots.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def _detect(frame):
    """Run pose detection on a BGR frame. Returns landmarks or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return result.pose_landmarks[0] if result.pose_landmarks else None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_front_image(frame):
    """
    Validates that the front image is suitable for measurement.
    Returns (ok: bool, error_message: str, error_code: str | None).
    """
    try:
        if is_blurry(frame):
            return False, "The front photo is too blurry. Please retake it in good lighting.", "IMAGE_TOO_BLURRY"

        image_height, image_width = frame.shape[:2]
        landmarks = _detect(frame)

        if landmarks is None:
            return False, "No person detected in the front photo. Make sure you are clearly visible.", "POSE_NOT_DETECTED"

        required = {
            PoseLandmark.NOSE: "nose",
            PoseLandmark.LEFT_SHOULDER: "left shoulder",
            PoseLandmark.RIGHT_SHOULDER: "right shoulder",
            PoseLandmark.LEFT_ELBOW: "left elbow",
            PoseLandmark.RIGHT_ELBOW: "right elbow",
            PoseLandmark.LEFT_KNEE: "left knee",
            PoseLandmark.RIGHT_KNEE: "right knee",
            PoseLandmark.LEFT_ANKLE: "left ankle",
            PoseLandmark.RIGHT_ANKLE: "right ankle",
        }

        missing = [
            name for idx, name in required.items()
            if getattr(landmarks[idx], 'visibility', 1.0) < 0.5
            or not (0 <= landmarks[idx].x <= 1 and 0 <= landmarks[idx].y <= 1)
        ]

        if missing:
            return False, "Full body not visible in the front photo. Please step back until your entire body is in frame.", "BODY_NOT_VISIBLE"

        ls = landmarks[PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = abs(ls.x - rs.x) * image_width
        head_to_shoulder = abs(ls.y - landmarks[PoseLandmark.NOSE].y) * image_height

        if shoulder_width < head_to_shoulder * 1.2:
            return False, "You are too close to the camera. Please step back to show your full body.", "TOO_CLOSE"

        return True, "OK", None

    except Exception as e:
        log.error("Front image validation error: %s", e)
        return False, "Could not validate the front photo. Please try again.", "VALIDATION_ERROR"


def validate_side_image(frame):
    """
    Validates that the side image is suitable for depth measurement.
    Returns (ok: bool, error_message: str, error_code: str | None).

    In a side profile, MediaPipe still detects landmarks but shoulder width
    will appear narrow. We check that the body is fully visible and not blurry
    rather than trying to distinguish left-from-right profile orientation.
    """
    try:
        if is_blurry(frame):
            return False, "The side photo is too blurry. Please retake it in good lighting.", "IMAGE_TOO_BLURRY"

        image_height, image_width = frame.shape[:2]
        landmarks = _detect(frame)

        if landmarks is None:
            return False, "No person detected in the side photo. Make sure you are clearly visible from the side.", "POSE_NOT_DETECTED"

        # In a true side profile at least one shoulder, hip, knee, and ankle must be visible.
        side_required = {
            PoseLandmark.LEFT_SHOULDER: "shoulder",
            PoseLandmark.LEFT_HIP: "hip",
            PoseLandmark.LEFT_KNEE: "knee",
            PoseLandmark.LEFT_ANKLE: "ankle",
        }

        missing = [
            name for idx, name in side_required.items()
            if getattr(landmarks[idx], 'visibility', 1.0) < 0.4
        ]

        if len(missing) > 1:
            return False, "Full body not visible in the side photo. Please step back until your entire body is in frame.", "BODY_NOT_VISIBLE"

        return True, "OK", None

    except Exception as e:
        log.error("Side image validation error: %s", e)
        return False, "Could not validate the side photo. Please try again.", "VALIDATION_ERROR"


# ── Measurement helpers ───────────────────────────────────────────────────────

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


def extract_side_depths(side_frame, side_lm, height_cm, side_mask=None):
    """
    Scans the side image at key anatomical levels to measure real front-to-back
    body depth in cm. Returns a dict of {label: depth_cm | None}.

    The horizontal "width" seen in a side-profile image at a given height is the
    body's front-to-back depth. We reuse get_body_width_at_height with a center_x
    derived from the mean of all well-visible landmarks, and a max search radius
    capped at 30 cm (generous upper bound for human body depth half).

    Any result outside [12, 50] cm is discarded — those bounds cover virtually all
    human body depths and filter out scan failures or background bleed.
    """
    side_h, side_w = side_frame.shape[:2]
    scale = scale_from_height(side_lm, side_h, height_cm)

    ls = side_lm[PoseLandmark.LEFT_SHOULDER]
    lh = side_lm[PoseLandmark.LEFT_HIP]
    lk = side_lm[PoseLandmark.LEFT_KNEE]

    # Center x: mean of well-visible landmarks — gives a point in the middle of
    # the body depth-wise regardless of which way the person is facing.
    visible_xs = [
        side_lm[i].x for i in range(33)
        if getattr(side_lm[i], 'visibility', 0) > 0.4 and 0 <= side_lm[i].x <= 1
    ]
    center_x = sum(visible_xs) / len(visible_xs) if visible_xs else 0.5

    # Cap search at 30 cm each side — no human body is deeper than ~60 cm total.
    max_half_px = int(30 / scale)

    def scan_depth(y_norm, label):
        y_px = int(y_norm * side_h)
        if side_mask is not None:
            depth_px = width_from_mask(side_mask, y_px)
            method = "mask"
        else:
            depth_px = get_body_width_at_height(side_frame, y_px, center_x, max_half_px)
            method = "gradient"
        if depth_px <= 0:
            log.warning("Side depth scan returned nothing for %s [%s]", label, method)
            return None
        depth_cm = round(depth_px * scale, 2)
        if not (12 <= depth_cm <= 50):
            log.warning("Side depth %s=%.1f cm outside plausible range [12, 50] — discarding [%s]", label, depth_cm, method)
            return None
        log.info("Side depth %s=%.1f cm (px=%d, scale=%.4f cm/px) [%s]", label, depth_cm, depth_px, scale, method)
        return depth_cm

    return {
        "chest":  scan_depth(ls.y + (lh.y - ls.y) * 0.20, "chest"),
        "waist":  scan_depth(ls.y + (lh.y - ls.y) * 0.55, "waist"),
        "hip":    scan_depth(ls.y + (lh.y - ls.y) * 0.85, "hip"),
        "thigh":  scan_depth(lh.y  + (lk.y - lh.y) * 0.25, "thigh"),
    }


def _resolve_depth_ratio(label, front_width_cm, side_depths, fallback_ratio):
    """
    Picks the best available depth ratio for a circumference calculation.

    Priority:
      1. Actual measured side depth (most accurate)
      2. Population-average fallback

    The side depth is converted to a ratio (depth / front_width) so the existing
    ellipse_circumference interface is unchanged. Ratios outside [0.25, 1.20] are
    rejected — that range covers all realistic human body proportions.
    """
    if side_depths and side_depths.get(label) and front_width_cm > 0:
        ratio = side_depths[label] / front_width_cm
        if 0.25 <= ratio <= 1.20:
            log.info("Depth %s: using side image ratio %.3f (%.1f cm depth / %.1f cm width)",
                     label, ratio, side_depths[label], front_width_cm)
            return ratio
        log.warning("Side depth ratio %.2f for %s out of plausible range — falling back", ratio, label)
    return fallback_ratio


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


def calculate_measurements(landmarks, scale_factor, image_width, image_height, frame, user_height_cm, weight_kg, is_female=False, known_waist_cm=None, side_depths=None, front_mask=None):
    """
    Derives all body measurements from pose landmarks and pixel scanning.

    Returns a dict of structured measurements, each with:
      value       — measurement in cm
      confidence  — 0.0 (unreliable) to 1.0 (highly reliable)
      warnings    — list of human-readable issue descriptions

    Width scanning uses the segmentation mask when available, falling back to
    gradient detection. Depth comes from the side image, falling back to
    population-average anatomical ratios. Formula estimates reduce confidence
    when deviation is high but never silently replace the measured value.
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
    shoulder_span_px = abs(ls.x - rs.x) * image_width
    hip_span_px      = abs(lh.x - rh.x) * image_width
    torso_half_px    = int(max(shoulder_span_px, hip_span_px) / 2 * 1.30)

    # ── Shared quality signals (applied to all scanned measurements) ──────────
    mask_penalty, mask_warn = _mask_quality(front_mask, image_height, image_width)

    def scan_and_validate(y_ratio, max_ratio, fallback_px, label=""):
        """Returns (px, used_fallback). Uses mask when available, else gradient."""
        y_px = int((ls.y + (lh.y - ls.y) * y_ratio) * image_height)
        if front_mask is not None:
            scanned = width_from_mask(front_mask, y_px)
            method = "mask"
        else:
            scanned = get_body_width_at_height(frame, y_px, torso_cx, torso_half_px)
            method = "gradient"
        log.info("Scan %s y_ratio=%.2f → %dpx (%.1fcm) [%s]",
                 label, y_ratio, scanned, scanned * scale_factor, method)
        if 0 < scanned <= shoulder_span_px * max_ratio:
            return scanned, False
        return fallback_px, True

    def circ_measurement(label, front_w_cm, depth_ratio, formula_cm, lm_indices, fallback_used, no_side_depth):
        """Builds a confidence-scored circumference measurement."""
        value = ellipse_circumference(front_w_cm, depth_ratio)
        penalties, warnings = [mask_penalty], [mask_warn]

        if fallback_used:
            penalties.append(0.20)
            warnings.append(f"{label.capitalize()} width scan failed — landmark estimate used")

        if no_side_depth:
            penalties.append(0.10)
            warnings.append(f"Side depth unavailable for {label} — population average used")

        lm_p, lm_w = _landmark_score(landmarks, lm_indices)
        penalties.append(lm_p)
        warnings.append(lm_w)

        f_p, f_w = _formula_score(value, formula_cm, label)
        penalties.append(f_p)
        warnings.append(f_w)

        return _build_measurement(value, penalties, warnings)

    formula = formula_estimates(user_height_cm, weight_kg, is_female)
    m = {}

    # ── Shoulder width ────────────────────────────────────────────────────────
    shoulder_px = shoulder_span_px * 1.19
    lm_p, lm_w = _landmark_score(landmarks, [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER])
    m["shoulder_width"] = _build_measurement(to_cm(shoulder_px), [lm_p], [lm_w])

    # ── Waist width (scanned early — needed for depth resolution) ────────────
    waist_fallback = hip_span_px * 1.24
    waist_w_px, waist_fallback_used = scan_and_validate(0.55, 0.95, waist_fallback, "waist")
    waist_w_cm = to_cm(waist_w_px)

    # ── Chest / bust ──────────────────────────────────────────────────────────
    chest_fallback = shoulder_span_px * 0.98
    chest_w_px, chest_fallback_used = scan_and_validate(0.20, 1.10, chest_fallback, "chest")
    chest_w_cm = to_cm(chest_w_px)
    depth_chest   = _resolve_depth_ratio("chest", chest_w_cm, side_depths, DEPTH_RATIOS["chest"])
    no_side_chest = not (side_depths and side_depths.get("chest"))
    m["chest_circumference"] = circ_measurement(
        "chest", chest_w_cm, depth_chest, formula["chest"],
        [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER],
        chest_fallback_used, no_side_chest,
    )

    # ── Waist ─────────────────────────────────────────────────────────────────
    depth_waist = _resolve_depth_ratio("waist", waist_w_cm, side_depths, DEPTH_RATIOS["waist"])
    if known_waist_cm:
        # User-provided: treat as ground truth
        m["waist"] = _build_measurement(known_waist_cm, [], ["User-provided waist measurement used"])
        m["waist"]["confidence"] = 1.0
    else:
        no_side_waist = not (side_depths and side_depths.get("waist"))
        m["waist"] = circ_measurement(
            "waist", waist_w_cm, depth_waist, formula["waist"],
            [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP],
            waist_fallback_used, no_side_waist,
        )

    # ── Hip ───────────────────────────────────────────────────────────────────
    hip_w_cm    = to_cm(hip_span_px * 1.55)
    depth_hip   = _resolve_depth_ratio("hip", hip_w_cm, side_depths, DEPTH_RATIOS["hip"])
    no_side_hip = not (side_depths and side_depths.get("hip"))
    hip_result  = circ_measurement(
        "hip", hip_w_cm, depth_hip, formula["hip"],
        [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP],
        False, no_side_hip,  # hip uses landmark span directly, not scan_and_validate
    )
    # Anatomical constraint: hip >= waist
    if hip_result["value"] < m["waist"]["value"]:
        hip_result["value"] = round(m["waist"]["value"] * 1.05, 1)
        hip_result["confidence"] = round(max(0.0, hip_result["confidence"] - 0.10), 2)
        hip_result["warnings"].append("Hip adjusted to exceed waist — landmark positions may be inaccurate")
    m["hip"] = hip_result

    # ── Thigh ─────────────────────────────────────────────────────────────────
    thigh_y_px = int((lh.y + (lk.y - lh.y) * 0.25) * image_height)
    if front_mask is not None:
        full_thigh_px = width_from_mask(front_mask, thigh_y_px)
    else:
        full_thigh_px = get_body_width_at_height(frame, thigh_y_px, torso_cx, torso_half_px)
    thigh_fallback_used = not (0 < full_thigh_px <= shoulder_span_px * 1.10)
    thigh_w_cm = to_cm(hip_span_px * 0.72) if thigh_fallback_used else to_cm(full_thigh_px // 2)
    depth_thigh   = _resolve_depth_ratio("thigh", thigh_w_cm, side_depths, DEPTH_RATIOS["thigh"])
    no_side_thigh = not (side_depths and side_depths.get("thigh"))
    m["thigh_circumference"] = circ_measurement(
        "thigh", thigh_w_cm, depth_thigh, 0,  # no population formula for thigh
        [PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE],
        thigh_fallback_used, no_side_thigh,
    )

    # ── Neck ──────────────────────────────────────────────────────────────────
    # Derived from nose-to-ear distance — inherently approximate; base penalty 0.25
    neck_px  = abs(nose.x - left_ear.x) * image_width * 2.0
    neck_val = ellipse_circumference(to_cm(neck_px), DEPTH_RATIOS["neck"])
    lm_p, lm_w = _landmark_score(landmarks, [PoseLandmark.NOSE, PoseLandmark.LEFT_EAR])
    m["neck"] = _build_measurement(neck_val, [0.25, lm_p], [lm_w])

    # ── Arm length ────────────────────────────────────────────────────────────
    # Euclidean shoulder-to-wrist distance accounts for the arm being angled out.
    sleeve_dy = abs(ls.y - lw.y) * image_height
    sleeve_dx = abs(ls.x - lw.x) * image_width
    arm_val   = to_cm(np.sqrt(sleeve_dx ** 2 + sleeve_dy ** 2))
    lm_p, lm_w = _landmark_score(landmarks, [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_WRIST])
    m["arm_length"] = _build_measurement(arm_val, [lm_p], [lm_w])

    # ── Trouser length ────────────────────────────────────────────────────────
    # 10% above hip landmark approximates the natural waistband / top of trouser.
    trouser_start_y = lh.y - (lh.y - ls.y) * 0.10
    trouser_val     = to_cm(abs(trouser_start_y - max(la.y, ra.y)) * image_height)
    lm_p, lm_w = _landmark_score(landmarks, [PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE])
    m["trouser_length"] = _build_measurement(trouser_val, [lm_p], [lm_w])

    log.info(
        "Measurements complete | chest=%.1fcm (conf=%.2f) waist=%.1fcm (conf=%.2f) hip=%.1fcm (conf=%.2f) | scale=%.4f cm/px",
        m["chest_circumference"]["value"], m["chest_circumference"]["confidence"],
        m["waist"]["value"],               m["waist"]["confidence"],
        m["hip"]["value"],                 m["hip"]["confidence"],
        scale_factor,
    )

    return m


# ── Response helpers ──────────────────────────────────────────────────────────

def error_response(message, code=400, error_code=None):
    body = {"error": message}
    if error_code:
        body["code"] = error_code
    return jsonify(body), code


def parse_numeric_field(name, min_val, max_val):
    """
    Parses a numeric form field and validates it falls within [min_val, max_val].
    Returns (value, error_response) — error_response is None on success.
    """
    raw = request.form.get(name)
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        return None, error_response(f"{name.replace('_', ' ').capitalize()} must be a number.", error_code=f"INVALID_{name.upper()}")
    if not (min_val <= value <= max_val):
        return None, error_response(
            f"{name.replace('_', ' ').capitalize()} must be between {min_val} and {max_val}.",
            error_code=f"INVALID_{name.upper()}"
        )
    return value, None


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


@app.route("/guidelines", methods=["GET"])
def guidelines():
    """Returns the scan guidelines the client should display to the user."""
    return jsonify({"guidelines": SCAN_GUIDELINES})


@app.route("/upload_images", methods=["POST"])
def upload_images():
    """
    Main measurement endpoint.

    Expected form fields:
      front      (file, required)  — front-facing full-body photo
      side       (file, required)  — side-on full-body photo
      height_cm  (float, required) — user's height in centimetres
      weight_kg  (float, required) — user's weight in kilograms
      gender     (str, required)   — "male" or "female"
      waist_in   (float, optional) — known waist in inches for depth calibration
    """
    if "front" not in request.files:
        return error_response("A front-facing photo is required.", error_code="MISSING_FRONT_IMAGE")

    if "side" not in request.files:
        return error_response("A side-on photo is required.", error_code="MISSING_SIDE_IMAGE")

    front_frame = decode_image(request.files["front"])
    if front_frame is None:
        return error_response("Could not read the front photo. Please upload a valid image.", error_code="INVALID_FRONT_IMAGE")

    side_frame = decode_image(request.files["side"])
    if side_frame is None:
        return error_response("Could not read the side photo. Please upload a valid image.", error_code="INVALID_SIDE_IMAGE")

    ok, msg, code = validate_front_image(front_frame)
    if not ok:
        return error_response(msg, error_code=code)

    ok, msg, code = validate_side_image(side_frame)
    if not ok:
        return error_response(msg, error_code=code)

    height_cm, err = parse_numeric_field("height_cm", 100, 250)
    if err:
        return err

    weight_kg, err = parse_numeric_field("weight_kg", 20, 300)
    if err:
        return err

    gender = request.form.get('gender', '').strip().lower()
    if gender not in ('male', 'female'):
        return jsonify({'error': 'gender is required and must be male or female'}), 400
    is_female = gender == 'female'

    try:
        waist_in = float(request.form.get('waist_in') or 0)
        known_waist_cm = waist_in * 2.54 if 15 <= waist_in <= 80 else None
    except (TypeError, ValueError):
        known_waist_cm = None

    front_h, front_w = front_frame.shape[:2]
    front_lm = _detect(front_frame)

    if front_lm is None:
        return error_response("Could not detect pose in the front photo.", error_code="POSE_NOT_DETECTED")

    front_mask = get_segmentation_mask(front_frame)
    side_mask  = get_segmentation_mask(side_frame)

    if front_mask is None:
        log.warning("Front segmentation failed — width scanning will use gradient detection")
    if side_mask is None:
        log.warning("Side segmentation failed — depth scanning will use gradient detection")

    side_lm = _detect(side_frame)
    side_depths = extract_side_depths(side_frame, side_lm, height_cm, side_mask) if side_lm else None

    if side_depths is None:
        log.warning("Side image pose detection failed — depth will use population-average ratios")

    scale = scale_from_height(front_lm, front_h, height_cm)
    measurements = calculate_measurements(
        front_lm, scale, front_w, front_h, front_frame,
        height_cm, weight_kg, is_female, known_waist_cm, side_depths, front_mask
    )

    log.info("height=%.1f cm, weight=%.1f kg | scan complete", height_cm, weight_kg)

    return jsonify({
        "measurements": measurements,
        "scan": {
            "front_segmentation": front_mask is not None,
            "side_segmentation":  side_mask is not None,
            "side_depths": {k: v is not None for k, v in (side_depths or {}).items()},
        },
    })


if __name__ == '__main__':
    # Development only — production uses gunicorn (see Dockerfile)
    app.run(host='0.0.0.0', port=5000, debug=False)

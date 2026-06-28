# live-measurements

AI-assisted body measurement service for Mima's Threadlore. Takes front and side full-body photos and returns per-measurement values with confidence scores and warnings for tailor review.

---

## How it works

1. **Pose detection** — MediaPipe PoseLandmarker locates key body landmarks (shoulders, hips, knees, ankles, wrists) in both images.
2. **Segmentation** — MediaPipe Selfie Segmentation extracts a clean body silhouette mask, replacing gradient-based edge detection.
3. **Width measurement** — Body width at each anatomical level is read directly from the segmentation mask (gradient scanning used as fallback).
4. **Depth measurement** — The side image is scanned at the same anatomical levels to get real front-to-back depth, replacing fixed population-average ratios.
5. **Circumference** — Ellipse approximation: `C ≈ 2π√((a²+b²)/2)` where `a = front width / 2`, `b = side depth / 2`.
6. **Confidence scoring** — Every measurement carries a confidence score (0–1) and a warnings list. Formula estimates validate plausibility but never silently replace measured values.
7. **Scale** — Derived from the person's known height (nose-to-ankle landmark distance + 7% head correction).

---

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `GET /guidelines`
Returns the scan guidelines the client should display to the user before capture.

### `POST /upload_images`

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `front` | file | ✅ | Front-facing full-body photo |
| `side` | file | ✅ | Side-on full-body photo |
| `height_cm` | float | ✅ | Person's height in centimetres (100–250) |
| `weight_kg` | float | ✅ | Person's weight in kilograms (20–300) |
| `gender` | string | | `male` or `female` (default: `male`) |
| `waist_in` | float | | Known waist in inches — used as ground truth if provided |

**Response:**

```json
{
  "measurements": {
    "shoulder_width": {
      "value": 16.2,
      "confidence": 0.91,
      "warnings": []
    },
    "chest_circumference": {
      "value": 38.5,
      "confidence": 0.81,
      "warnings": ["Side depth unavailable for chest — population average used"]
    },
    "waist": {
      "value": 32.0,
      "confidence": 0.62,
      "warnings": ["Waist deviates 22% from expected — possible clothing interference"]
    },
    "hip": { "value": 40.1, "confidence": 0.85, "warnings": [] },
    "thigh_circumference": { "value": 22.3, "confidence": 0.75, "warnings": [] },
    "arm_length": { "value": 24.5, "confidence": 0.78, "warnings": [] },
    "trouser_length": { "value": 42.0, "confidence": 0.88, "warnings": [] }
  },
  "scan": {
    "front_segmentation": true,
    "side_segmentation": true,
    "side_depths": {
      "chest": true,
      "waist": true,
      "hip": false,
      "thigh": true
    }
  }
}
```

All values are in **centimetres**. The Java integration layer converts to inches before storing.

**Error response:**

```json
{
  "error": "The side photo is too blurry. Please retake it in good lighting.",
  "code": "IMAGE_TOO_BLURRY"
}
```

**Error codes:** `MISSING_FRONT_IMAGE`, `MISSING_SIDE_IMAGE`, `INVALID_FRONT_IMAGE`, `INVALID_SIDE_IMAGE`, `IMAGE_TOO_BLURRY`, `POSE_NOT_DETECTED`, `BODY_NOT_VISIBLE`, `TOO_CLOSE`, `INVALID_HEIGHT_CM`, `INVALID_WEIGHT_KG`, `VALIDATION_ERROR`.

---

## Confidence scoring

Each measurement's confidence starts at 1.0 and is reduced by:

| Signal | Penalty |
|---|---|
| Segmentation unavailable (gradient fallback) | −0.15 |
| Width scan fell back to landmark estimate | −0.20 |
| Side depth unavailable (population average used) | −0.10 |
| Low landmark visibility | −0.10 to −0.25 |
| Formula deviation 10–20% | −0.10 |
| Formula deviation 20–35% | −0.20 |
| Formula deviation >35% | −0.35 |

**Routing thresholds** (applied by the Java layer at payment confirmation):

| Confidence | Outcome |
|---|---|
| ≥ 0.75 | Auto-approved, order confirmed |
| 0.50–0.74 | Tailor review queue |
| < 0.50 | Tailor review + live session recommended |

---

## Scan guidelines

Shown to the customer before capture:

- Wear fitted clothing — no oversized, loose, or flowing garments
- Stand against a plain, uncluttered background
- Ensure good, even lighting with no harsh shadows
- Show your full body from head to toe
- Keep your arms slightly away from your body
- Hold the camera at waist or chest height
- Do not use mirror selfies
- Stand straight and face the camera directly (front photo)
- Stand sideways with arms at your sides (side photo)

---

## Models

| File | Source | Purpose |
|---|---|---|
| `pose_landmarker_heavy.task` | MediaPipe | Body landmark detection |
| `selfie_segmenter.tflite` | MediaPipe | Body silhouette segmentation |

Both are downloaded automatically on first run if not present.

---

## Calibration

The admin dashboard (`/scans`) shows a **Calibration Drift** table summarising the average delta between AI-predicted and tailor-corrected measurements across all historical adjustments.

When a measurement consistently shows significant drift, update the corresponding value in `app.py`:

- **Circumference over/under-prediction** → adjust `DEPTH_RATIOS[<measurement>]`
- **Formula deviation warnings firing too often** → adjust the `0.20` threshold in `_formula_score`

Re-deploy the Flask service after any change. The correction logs in `scan_corrections` are the source of truth — no raw images are retained.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

In production, Gunicorn is used (see `Dockerfile`). Structured JSON logging is enabled automatically when `LOG_FORMAT=json` is set (configured in `docker-compose.yml`).

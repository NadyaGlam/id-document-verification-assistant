# ID Document Verification Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red?logo=opencv)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-purple)
![pytest](https://img.shields.io/badge/tested%20with-pytest-yellow?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

> **Disclaimer**: This project is a technical prototype for document image analysis and verification-readiness assessment. It is **not** intended for legal identity verification or official decision-making.

---

## Overview

A production-style Computer Vision backend that accepts uploaded ID document images and returns a structured JSON analysis covering image quality, document detection, optional OCR, and a risk-based recommendation.

Built to demonstrate a realistic CV pipeline — from raw image bytes to structured, explainable output — using Python, FastAPI, and OpenCV.

---

## Why This Project Matters

Identity document verification is a core challenge in fintech, regtech, and digital onboarding. Building this project demonstrates:

- **Computer Vision pipeline design** — blur detection, edge/contour analysis, face detection, glare flagging
- **Production API architecture** — versioned endpoints, Pydantic v2 schemas, typed responses, Swagger docs
- **Explainable ML-style output** — risk scores with human-readable reason lists, not just a binary pass/fail
- **Extensibility mindset** — OCR falls back gracefully; document type classifier is a named placeholder ready for a real model
- **Software engineering practices** — clean modular services, type hints throughout, pytest test suite, Docker support

---

## Features

| Feature | Detail |
|---|---|
| Image quality analysis | Blur (Laplacian variance), brightness (mean), contrast (std dev) |
| Document detection | Edge → contour → quadrilateral approximation (OpenCV) |
| Face detection | Haar cascade frontal face classifier |
| Glare detection | Pixel saturation ratio threshold |
| Text region detection | Morphological dilation + contour analysis |
| OCR | Optional — EasyOCR or Tesseract, graceful fallback if neither installed |
| Risk scoring | Rule-based scoring (0–100) with reasons list |
| Recommendation | `accepted` / `manual_review` / `retake_photo` |
| Swagger UI | Auto-generated at `/docs` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Image processing | OpenCV 4.8, NumPy |
| Data validation | Pydantic v2 |
| Configuration | pydantic-settings, python-dotenv |
| OCR (optional) | EasyOCR or pytesseract |
| Testing | pytest, httpx |
| Containerisation | Docker |

---

## Project Structure

```
id-document-verification-assistant/
├── app/
│   ├── api/
│   │   └── routes.py          # POST /analyze-document
│   ├── core/
│   │   └── config.py          # Settings (env-driven thresholds)
│   ├── schemas/
│   │   └── document_analysis.py   # Pydantic response models
│   ├── services/
│   │   ├── image_quality.py   # Blur / brightness / contrast
│   │   ├── document_detector.py   # Contour, face, text, glare
│   │   ├── ocr_service.py     # EasyOCR / Tesseract fallback
│   │   └── risk_scoring.py    # Rule-based scoring engine
│   ├── utils/
│   │   └── image_loader.py    # Upload validation + OpenCV decode
│   ├── main.py                # FastAPI app factory
│   └── logger.py              # Structured console logger
├── tests/
│   ├── test_image_quality.py
│   └── test_risk_scoring.py
├── sample_images/             # Drop test images here
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── Dockerfile
```

---

## Architecture

```
HTTP POST /api/v1/analyze-document
         │
         ▼
  ┌─────────────┐
  │ image_loader│  validate ext → decode bytes → OpenCV ndarray
  └──────┬──────┘
         │
   ┌─────┼──────────────┐
   ▼     ▼              ▼
┌──────┐ ┌──────────┐ ┌─────┐
│Image │ │Document  │ │ OCR │
│Qual. │ │Detector  │ │     │
└──┬───┘ └────┬─────┘ └──┬──┘
   │          │           │
   └────┬─────┘           │
        ▼                 │
  ┌───────────┐           │
  │   Risk    │◄──────────┘
  │  Scoring  │
  └─────┬─────┘
        ▼
  JSON response (Pydantic)
```

---

## API Endpoints

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0" }
```

### `POST /api/v1/analyze-document`

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | JPG or PNG image, max 10 MB |

**Example (curl)**

```bash
curl -X POST http://localhost:8000/api/v1/analyze-document \
  -F "file=@sample_images/id_card.jpg"
```

**Example response**

```json
{
  "document_type": "id_card",
  "image_quality": {
    "overall": "good",
    "blur_score": 312.47,
    "brightness_score": 131.2,
    "contrast_score": 58.9,
    "is_blurry": false,
    "is_too_dark": false,
    "is_too_bright": false,
    "has_low_contrast": false
  },
  "document_checks": {
    "document_detected": true,
    "face_detected": true,
    "text_detected": true,
    "is_cropped": false,
    "has_glare_warning": false
  },
  "ocr": {
    "text_extracted": true,
    "sample_text": "JOHN DOE 01 JAN 1990 123456789",
    "confidence": 0.914
  },
  "risk_assessment": {
    "risk_level": "low",
    "risk_score": 0,
    "reasons": []
  },
  "recommendation": "accepted"
}
```

**Recommendation values**

| Value | Meaning |
|---|---|
| `accepted` | Risk score ≤ 30 — image is usable |
| `manual_review` | Risk score 31–60 — flag for human review |
| `retake_photo` | Risk score > 60 — ask the user to re-photograph |

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/id-document-verification-assistant.git
cd id-document-verification-assistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to adjust thresholds
```

### 4. (Optional) Enable OCR

Uncomment the OCR lines in `requirements.txt`, then:

```bash
# EasyOCR
pip install easyocr

# OR Tesseract (also requires the system binary)
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
pip install pytesseract
```

---

## Running Locally

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Running with Docker

```bash
docker build -t id-doc-verify .
docker run -p 8000:8000 id-doc-verify
```

---

## Running Tests

```bash
pytest -v
```

```
tests/test_image_quality.py::TestBlurScore::test_uniform_image_scores_near_zero  PASSED
tests/test_image_quality.py::TestBrightnessScore::test_dark_image                PASSED
tests/test_risk_scoring.py::TestComputeRisk::test_low_risk_on_good_inputs        PASSED
tests/test_risk_scoring.py::TestComputeRisk::test_high_risk_on_poor_inputs       PASSED
...
```

---

## Configuration

All thresholds are environment-driven via `.env`:

| Variable | Default | Description |
|---|---|---|
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance below this → blurry |
| `BRIGHTNESS_MIN` | `50.0` | Mean pixel intensity below this → too dark |
| `BRIGHTNESS_MAX` | `200.0` | Mean pixel intensity above this → overexposed |
| `CONTRAST_MIN` | `30.0` | Std dev below this → low contrast |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max upload size in MB |

---

## Future Improvements

The project is deliberately structured for easy extension:

| Improvement | Approach |
|---|---|
| **Train a document type classifier** | Fine-tune MobileNetV3 or EfficientNet-Lite on ID/passport/licence images |
| **YOLO field detection** | Detect MRZ zones, expiry dates, photo zones with YOLOv8 |
| **Face liveness / anti-spoofing** | Add a depth or texture-based spoof detector |
| **Face matching** | Compare the photo zone face to a selfie using ArcFace or FaceNet |
| **Fraud signal detection** | Flag font inconsistencies, unusual edge artefacts, copy-paste traces |
| **Async processing** | Offload heavy CV work to Celery + Redis; return a job ID |
| **MLflow experiment tracking** | Track threshold tuning, model evaluation, dataset versions |
| **Production deployment** | AWS Lambda + ECR or ECS Fargate; add CloudWatch logging |
| **Monitoring** | Prometheus metrics endpoint, Grafana dashboard for latency / error rate |

---

## License

MIT

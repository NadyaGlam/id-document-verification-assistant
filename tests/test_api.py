"""
Integration tests for the FastAPI endpoints.
Uses FastAPI's TestClient (synchronous) with synthetic OpenCV images.
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jpeg_bytes(width: int = 200, height: int = 150) -> bytes:
    """Create a realistic-looking synthetic image encoded as JPEG."""
    rng = np.random.default_rng(seed=0)
    image = rng.integers(40, 220, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", image)
    return buf.tobytes()


def _png_bytes(width: int = 200, height: int = 150) -> bytes:
    rng = np.random.default_rng(seed=1)
    image = rng.integers(40, 220, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", image)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_is_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_version_present(self):
        r = client.get("/health")
        assert "version" in r.json()


# ---------------------------------------------------------------------------
# /api/v1/analyze-document — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeDocumentHappyPath:
    def test_jpeg_returns_200(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_png_returns_200(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_response_top_level_keys(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        data = r.json()
        for key in (
            "document_type", "image_quality", "document_checks",
            "ocr", "risk_assessment", "recommendation",
        ):
            assert key in data, f"Missing key: {key}"

    def test_image_quality_fields(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        q = r.json()["image_quality"]
        assert q["overall"] in ("good", "medium", "poor")
        assert isinstance(q["blur_score"], float)
        assert isinstance(q["brightness_score"], float)
        assert isinstance(q["is_blurry"], bool)
        assert isinstance(q["is_too_dark"], bool)
        assert isinstance(q["is_too_bright"], bool)
        assert isinstance(q["has_low_contrast"], bool)

    def test_document_checks_fields(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        c = r.json()["document_checks"]
        for field in ("document_detected", "face_detected", "text_detected",
                      "is_cropped", "has_glare_warning"):
            assert isinstance(c[field], bool), f"{field} should be bool"

    def test_risk_score_range(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        score = r.json()["risk_assessment"]["risk_score"]
        assert isinstance(score, int)
        assert 0 <= score <= 100

    def test_risk_level_valid(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.json()["risk_assessment"]["risk_level"] in ("low", "medium", "high")

    def test_recommendation_valid(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.json()["recommendation"] in ("accepted", "manual_review", "retake_photo")

    def test_ocr_fields_present(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        ocr = r.json()["ocr"]
        assert "text_extracted" in ocr
        assert "sample_text" in ocr
        assert "confidence" in ocr

    def test_document_type_valid(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("id.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert r.json()["document_type"] in (
            "unknown", "id_card", "passport", "driver_license"
        )


# ---------------------------------------------------------------------------
# /api/v1/analyze-document — error cases
# ---------------------------------------------------------------------------

class TestAnalyzeDocumentErrors:
    def test_unsupported_file_type_returns_415(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("doc.pdf", b"%PDF fake", "application/pdf")},
        )
        assert r.status_code == 415

    def test_txt_file_returns_415(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("doc.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 415

    def test_corrupted_jpeg_returns_422(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("broken.jpg", b"\xff\xd8 not valid jpeg bytes", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_missing_file_returns_422(self):
        r = client.post("/api/v1/analyze-document")
        assert r.status_code == 422

    def test_error_response_has_detail_field(self):
        r = client.post(
            "/api/v1/analyze-document",
            files={"file": ("doc.pdf", b"fake", "application/pdf")},
        )
        assert "detail" in r.json()

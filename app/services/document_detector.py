from dataclasses import dataclass

import cv2
import numpy as np

from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentCheckResult:
    document_detected: bool
    face_detected: bool
    text_detected: bool
    is_cropped: bool
    has_glare_warning: bool
    document_type: str


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()


def detect_document_rect(image: np.ndarray) -> tuple[bool, bool]:
    """
    Detect a rectangular document shape via edge + contour analysis.
    Returns (document_detected, is_cropped).
    """
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, False

    h, w = image.shape[:2]
    image_area = h * w

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < image_area * 0.1:
        return False, False

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    # A document should approximate a quadrilateral
    if not (4 <= len(approx) <= 6):
        return False, False

    x, y, rect_w, rect_h = cv2.boundingRect(largest)
    margin = 0.02
    is_cropped = (
        x < w * margin
        or y < h * margin
        or (x + rect_w) > w * (1 - margin)
        or (y + rect_h) > h * (1 - margin)
    )

    return True, is_cropped


def detect_face(image: np.ndarray) -> bool:
    """Detect frontal faces using OpenCV's built-in Haar cascade."""
    gray = _to_gray(image)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces) > 0
    except Exception as exc:
        logger.warning(f"Face detection failed: {exc}")
        return False


def detect_text_regions(image: np.ndarray) -> bool:
    """
    Rough morphological text-region detector.
    Dilates thresholded image horizontally to merge character blobs into word blobs.
    """
    gray = _to_gray(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]

    text_blobs = [
        c for c in contours
        if 100 < cv2.contourArea(c) < image_area * 0.3
    ]
    return len(text_blobs) >= 2


def detect_glare(image: np.ndarray) -> bool:
    """Flag images where >5% of pixels are near-saturated white (potential glare)."""
    gray = _to_gray(image)
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = np.sum(bright_mask > 0) / gray.size
    return glare_ratio > 0.05


def classify_document_type(
    document_detected: bool,
    face_detected: bool,
    text_detected: bool,
    image: np.ndarray,
) -> str:
    """
    Heuristic document type classifier based on aspect ratio and face/text presence.
    Falls back to aspect-ratio classification even when contour detection fails,
    as long as face or text signals confirm a document is present.
    Replace with a trained classifier for production use.
    """
    h, w = image.shape[:2]
    aspect = w / h

    has_document_signals = document_detected or face_detected or text_detected
    if not has_document_signals:
        return "unknown"

    if face_detected:
        # Passport: portrait orientation
        if aspect < 0.85:
            return "passport"
        # Driver's license: wide landscape
        if aspect >= 1.55:
            return "driver_license"
        # ID card: standard landscape
        if 1.3 <= aspect < 1.55:
            return "id_card"

    # Text-only document (no face): classify by aspect ratio only
    if text_detected:
        if aspect < 0.85:
            return "passport"
        if 1.3 <= aspect <= 1.7:
            return "id_card"

    return "unknown"


def analyze_document(image: np.ndarray) -> DocumentCheckResult:
    document_detected, is_cropped = detect_document_rect(image)
    face_detected = detect_face(image)
    text_detected = detect_text_regions(image)
    has_glare_warning = detect_glare(image)
    document_type = classify_document_type(
        document_detected, face_detected, text_detected, image
    )

    return DocumentCheckResult(
        document_detected=document_detected,
        face_detected=face_detected,
        text_detected=text_detected,
        is_cropped=is_cropped,
        has_glare_warning=has_glare_warning,
        document_type=document_type,
    )

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.logger import get_logger

logger = get_logger(__name__)

_SAMPLE_TEXT_LIMIT = 100  # chars returned in the response
_LOG_PREVIEW_LIMIT = 50   # chars logged (avoid logging sensitive data in full)


@dataclass
class OCRResult:
    text_extracted: bool
    sample_text: str
    confidence: Optional[float]


def _patch_ssl() -> None:
    """Fix SSL certificate verification on macOS when system certs are missing."""
    import os
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass


def _try_easyocr(image: np.ndarray) -> Optional[OCRResult]:
    try:
        _patch_ssl()
        import easyocr  # type: ignore

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(image)

        if not results:
            return OCRResult(text_extracted=False, sample_text="", confidence=None)

        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]

        combined = " ".join(texts)
        preview = combined[:_LOG_PREVIEW_LIMIT] + ("..." if len(combined) > _LOG_PREVIEW_LIMIT else "")
        logger.info(f"EasyOCR result preview: {preview}")

        return OCRResult(
            text_extracted=True,
            sample_text=combined[:_SAMPLE_TEXT_LIMIT],
            confidence=round(sum(confidences) / len(confidences), 3),
        )
    except ImportError:
        return None
    except Exception as exc:
        logger.warning(f"EasyOCR failed: {exc}")
        return None


def _try_tesseract(image: np.ndarray) -> Optional[OCRResult]:
    try:
        import pytesseract  # type: ignore
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        text = pytesseract.image_to_string(gray).strip()

        if not text:
            return OCRResult(text_extracted=False, sample_text="", confidence=None)

        preview = text[:_LOG_PREVIEW_LIMIT] + ("..." if len(text) > _LOG_PREVIEW_LIMIT else "")
        logger.info(f"Tesseract result preview: {preview}")

        return OCRResult(
            text_extracted=True,
            sample_text=text[:_SAMPLE_TEXT_LIMIT],
            confidence=None,  # basic pytesseract mode does not expose per-result confidence
        )
    except ImportError:
        return None
    except Exception as exc:
        logger.warning(f"Tesseract failed: {exc}")
        return None


def run_ocr(image: np.ndarray) -> OCRResult:
    """
    Attempt OCR using EasyOCR, then Tesseract.
    Falls back gracefully if neither library is installed.
    Install easyocr or pytesseract to enable this feature.
    """
    result = _try_easyocr(image)
    if result is not None:
        return result

    result = _try_tesseract(image)
    if result is not None:
        return result

    logger.info(
        "No OCR library available. "
        "Install 'easyocr' or 'pytesseract' and uncomment in requirements.txt to enable OCR."
    )
    return OCRResult(text_extracted=False, sample_text="", confidence=None)

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings


@dataclass
class QualityResult:
    blur_score: float
    brightness_score: float
    contrast_score: float
    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool
    has_low_contrast: bool
    overall: str


def compute_blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher value means sharper image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness_score(gray: np.ndarray) -> float:
    """Mean pixel intensity of the grayscale image (0–255)."""
    return float(np.mean(gray))


def compute_contrast_score(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities — higher means more contrast."""
    return float(np.std(gray))


def classify_overall_quality(
    is_blurry: bool,
    is_too_dark: bool,
    is_too_bright: bool,
    has_low_contrast: bool,
) -> str:
    issue_count = sum([is_blurry, is_too_dark, is_too_bright, has_low_contrast])
    if issue_count == 0:
        return "good"
    if issue_count == 1:
        return "medium"
    return "poor"


def analyze_quality(image: np.ndarray) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    blur_score = compute_blur_score(gray)
    brightness_score = compute_brightness_score(gray)
    contrast_score = compute_contrast_score(gray)

    is_blurry = blur_score < settings.blur_threshold
    is_too_dark = brightness_score < settings.brightness_min
    is_too_bright = brightness_score > settings.brightness_max
    has_low_contrast = contrast_score < settings.contrast_min

    overall = classify_overall_quality(is_blurry, is_too_dark, is_too_bright, has_low_contrast)

    return QualityResult(
        blur_score=round(blur_score, 2),
        brightness_score=round(brightness_score, 2),
        contrast_score=round(contrast_score, 2),
        is_blurry=is_blurry,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        has_low_contrast=has_low_contrast,
        overall=overall,
    )

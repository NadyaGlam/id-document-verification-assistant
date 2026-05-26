import numpy as np
import pytest

from app.services.image_quality import (
    analyze_quality,
    classify_overall_quality,
    compute_blur_score,
    compute_brightness_score,
    compute_contrast_score,
)


def gray(value: int, size: tuple = (100, 100)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


def noisy(size: tuple = (100, 100)) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size, dtype=np.uint8)


def bgr(value: int, size: tuple = (100, 100, 3)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


class TestBlurScore:
    def test_uniform_image_scores_near_zero(self):
        score = compute_blur_score(gray(128))
        assert score < 5.0

    def test_noisy_image_scores_high(self):
        score = compute_blur_score(noisy())
        assert score > 1000.0

    def test_score_is_non_negative(self):
        assert compute_blur_score(gray(0)) >= 0.0


class TestBrightnessScore:
    def test_dark_image(self):
        assert compute_brightness_score(gray(10)) < 50.0

    def test_bright_image(self):
        assert compute_brightness_score(gray(230)) > 200.0

    def test_midrange_image(self):
        score = compute_brightness_score(gray(128))
        assert 120.0 < score < 135.0


class TestContrastScore:
    def test_uniform_has_zero_contrast(self):
        assert compute_contrast_score(gray(100)) == pytest.approx(0.0)

    def test_noisy_has_high_contrast(self):
        assert compute_contrast_score(noisy()) > 50.0


class TestClassifyOverallQuality:
    def test_no_issues_returns_good(self):
        assert classify_overall_quality(False, False, False, False) == "good"

    def test_one_issue_returns_medium(self):
        assert classify_overall_quality(True, False, False, False) == "medium"
        assert classify_overall_quality(False, True, False, False) == "medium"

    def test_two_issues_returns_poor(self):
        assert classify_overall_quality(True, True, False, False) == "poor"

    def test_all_issues_returns_poor(self):
        assert classify_overall_quality(True, True, True, True) == "poor"


class TestAnalyzeQuality:
    def test_dark_bgr_image_flags_too_dark(self):
        result = analyze_quality(bgr(10))
        assert result.is_too_dark is True
        assert result.overall in ("medium", "poor")

    def test_bright_bgr_image_flags_too_bright(self):
        result = analyze_quality(bgr(240))
        assert result.is_too_bright is True

    def test_uniform_image_flags_blurry_and_low_contrast(self):
        result = analyze_quality(bgr(128))
        assert result.is_blurry is True
        assert result.has_low_contrast is True
        assert result.overall == "poor"

    def test_result_fields_are_floats(self):
        result = analyze_quality(bgr(128))
        assert isinstance(result.blur_score, float)
        assert isinstance(result.brightness_score, float)
        assert isinstance(result.contrast_score, float)

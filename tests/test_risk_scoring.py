import pytest

from app.services.document_detector import DocumentCheckResult
from app.services.image_quality import QualityResult
from app.services.risk_scoring import compute_risk


def good_quality() -> QualityResult:
    return QualityResult(
        blur_score=300.0,
        brightness_score=128.0,
        contrast_score=65.0,
        is_blurry=False,
        is_too_dark=False,
        is_too_bright=False,
        has_low_contrast=False,
        overall="good",
    )


def poor_quality() -> QualityResult:
    return QualityResult(
        blur_score=20.0,
        brightness_score=18.0,
        contrast_score=5.0,
        is_blurry=True,
        is_too_dark=True,
        is_too_bright=False,
        has_low_contrast=True,
        overall="poor",
    )


def good_checks() -> DocumentCheckResult:
    return DocumentCheckResult(
        document_detected=True,
        face_detected=True,
        text_detected=True,
        is_cropped=False,
        has_glare_warning=False,
        document_type="id_card",
    )


def poor_checks() -> DocumentCheckResult:
    return DocumentCheckResult(
        document_detected=False,
        face_detected=False,
        text_detected=False,
        is_cropped=True,
        has_glare_warning=True,
        document_type="unknown",
    )


class TestComputeRisk:
    def test_low_risk_on_good_inputs(self):
        result = compute_risk(good_quality(), good_checks())
        assert result.risk_level == "low"
        assert result.risk_score <= 30
        assert result.recommendation == "accepted"
        assert result.reasons == []

    def test_high_risk_on_poor_inputs(self):
        result = compute_risk(poor_quality(), poor_checks())
        assert result.risk_level == "high"
        assert result.risk_score > 60
        assert result.recommendation == "retake_photo"
        assert len(result.reasons) >= 4

    def test_blurry_adds_twenty_points(self):
        q = good_quality()
        q.is_blurry = True
        result = compute_risk(q, good_checks())
        assert result.risk_score == 20
        assert any("blurry" in r.lower() for r in result.reasons)

    def test_no_document_adds_thirty_points(self):
        c = good_checks()
        c.document_detected = False
        result = compute_risk(good_quality(), c)
        assert result.risk_score == 30
        assert any("document" in r.lower() for r in result.reasons)

    def test_risk_score_capped_at_100(self):
        result = compute_risk(poor_quality(), poor_checks())
        assert result.risk_score <= 100

    def test_medium_risk_gives_manual_review(self):
        # blurry (+20) + too dark (+15) = 35 → medium
        q = good_quality()
        q.is_blurry = True
        q.is_too_dark = True
        result = compute_risk(q, good_checks())
        assert result.risk_level == "medium"
        assert result.recommendation == "manual_review"

    def test_glare_adds_ten_points(self):
        c = good_checks()
        c.has_glare_warning = True
        result = compute_risk(good_quality(), c)
        assert result.risk_score == 10
        assert any("glare" in r.lower() for r in result.reasons)

    def test_cropped_adds_fifteen_points(self):
        c = good_checks()
        c.is_cropped = True
        result = compute_risk(good_quality(), c)
        assert result.risk_score == 15
        assert any("cropped" in r.lower() for r in result.reasons)

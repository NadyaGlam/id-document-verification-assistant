from dataclasses import dataclass, field

from app.services.document_detector import DocumentCheckResult
from app.services.image_quality import QualityResult


@dataclass
class RiskResult:
    risk_score: int
    risk_level: str
    reasons: list[str] = field(default_factory=list)
    recommendation: str = "accepted"


def compute_risk(quality: QualityResult, checks: DocumentCheckResult) -> RiskResult:
    score = 0
    reasons: list[str] = []

    if not checks.document_detected:
        score += 30
        reasons.append("No document-like object detected in the image")

    if quality.is_blurry:
        score += 20
        reasons.append("Image is blurry — low sharpness detected")

    if not checks.text_detected:
        score += 20
        reasons.append("No text regions detected in the document")

    if quality.is_too_dark:
        score += 15
        reasons.append("Image is too dark — insufficient lighting")

    if quality.is_too_bright:
        score += 15
        reasons.append("Image is overexposed — too much ambient light")

    if checks.is_cropped:
        score += 15
        reasons.append("Document appears cropped or partially outside the frame")

    if quality.has_low_contrast:
        score += 10
        reasons.append("Low contrast — text and features may be hard to distinguish")

    if checks.has_glare_warning:
        score += 10
        reasons.append("Glare or reflection detected on the document surface")

    score = min(score, 100)

    if score <= 30:
        risk_level = "low"
        recommendation = "accepted"
    elif score <= 60:
        risk_level = "medium"
        recommendation = "manual_review"
    else:
        risk_level = "high"
        recommendation = "retake_photo"

    return RiskResult(
        risk_score=score,
        risk_level=risk_level,
        reasons=reasons,
        recommendation=recommendation,
    )

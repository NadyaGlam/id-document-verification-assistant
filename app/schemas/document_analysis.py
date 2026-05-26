from typing import Optional
from pydantic import BaseModel
from enum import Enum


class DocumentType(str, Enum):
    unknown = "unknown"
    id_card = "id_card"
    passport = "passport"
    driver_license = "driver_license"


class OverallQuality(str, Enum):
    good = "good"
    medium = "medium"
    poor = "poor"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Recommendation(str, Enum):
    accepted = "accepted"
    manual_review = "manual_review"
    retake_photo = "retake_photo"


class ImageQualitySchema(BaseModel):
    overall: OverallQuality
    blur_score: float
    brightness_score: float
    contrast_score: float
    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool
    has_low_contrast: bool


class DocumentChecksSchema(BaseModel):
    document_detected: bool
    face_detected: bool
    text_detected: bool
    is_cropped: bool
    has_glare_warning: bool


class OCRSchema(BaseModel):
    text_extracted: bool
    sample_text: str
    confidence: Optional[float] = None


class RiskAssessmentSchema(BaseModel):
    risk_level: RiskLevel
    risk_score: int
    reasons: list[str]


class DocumentAnalysisResponse(BaseModel):
    document_type: DocumentType
    image_quality: ImageQualitySchema
    document_checks: DocumentChecksSchema
    ocr: OCRSchema
    risk_assessment: RiskAssessmentSchema
    recommendation: Recommendation

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_type": "id_card",
                "image_quality": {
                    "overall": "good",
                    "blur_score": 312.5,
                    "brightness_score": 128.0,
                    "contrast_score": 62.3,
                    "is_blurry": False,
                    "is_too_dark": False,
                    "is_too_bright": False,
                    "has_low_contrast": False,
                },
                "document_checks": {
                    "document_detected": True,
                    "face_detected": True,
                    "text_detected": True,
                    "is_cropped": False,
                    "has_glare_warning": False,
                },
                "ocr": {
                    "text_extracted": True,
                    "sample_text": "JOHN DOE 01 JAN 1990",
                    "confidence": 0.91,
                },
                "risk_assessment": {
                    "risk_level": "low",
                    "risk_score": 0,
                    "reasons": [],
                },
                "recommendation": "accepted",
            }
        }
    }

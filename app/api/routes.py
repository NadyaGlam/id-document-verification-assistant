from fastapi import APIRouter, File, UploadFile

from app.logger import get_logger
from app.schemas.document_analysis import (
    DocumentAnalysisResponse,
    DocumentChecksSchema,
    DocumentType,
    ImageQualitySchema,
    OCRSchema,
    OverallQuality,
    Recommendation,
    RiskAssessmentSchema,
    RiskLevel,
)
from app.services.document_detector import analyze_document
from app.services.image_quality import analyze_quality
from app.services.ocr_service import run_ocr
from app.services.risk_scoring import compute_risk
from app.utils.image_loader import load_image_from_upload

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/analyze-document",
    response_model=DocumentAnalysisResponse,
    summary="Analyze an ID document image",
    description=(
        "Upload a JPG or PNG image of an ID document. "
        "Returns image quality metrics, document detection results, "
        "optional OCR text, and a risk-based recommendation. "
        "This endpoint is for portfolio/educational purposes only — "
        "it does NOT perform legal identity verification."
    ),
    tags=["Document Analysis"],
)
async def analyze_document_endpoint(
    file: UploadFile = File(..., description="ID document image (JPG or PNG, max 10 MB)"),
) -> DocumentAnalysisResponse:
    logger.info(
        f"Upload received: '{file.filename}' | content_type={file.content_type}"
    )

    image = await load_image_from_upload(file)

    quality = analyze_quality(image)
    checks = analyze_document(image)
    ocr_result = run_ocr(image)
    risk = compute_risk(quality, checks)

    logger.info(
        f"Analysis complete: '{file.filename}' | "
        f"quality={quality.overall} | risk={risk.risk_level} ({risk.risk_score}) | "
        f"recommendation={risk.recommendation}"
    )

    return DocumentAnalysisResponse(
        document_type=DocumentType(checks.document_type),
        image_quality=ImageQualitySchema(
            overall=OverallQuality(quality.overall),
            blur_score=quality.blur_score,
            brightness_score=quality.brightness_score,
            contrast_score=quality.contrast_score,
            is_blurry=quality.is_blurry,
            is_too_dark=quality.is_too_dark,
            is_too_bright=quality.is_too_bright,
            has_low_contrast=quality.has_low_contrast,
        ),
        document_checks=DocumentChecksSchema(
            document_detected=checks.document_detected,
            face_detected=checks.face_detected,
            text_detected=checks.text_detected,
            is_cropped=checks.is_cropped,
            has_glare_warning=checks.has_glare_warning,
        ),
        ocr=OCRSchema(
            text_extracted=ocr_result.text_extracted,
            sample_text=ocr_result.sample_text,
            confidence=ocr_result.confidence,
        ),
        risk_assessment=RiskAssessmentSchema(
            risk_level=RiskLevel(risk.risk_level),
            risk_score=risk.risk_score,
            reasons=risk.reasons,
        ),
        recommendation=Recommendation(risk.recommendation),
    )

import numpy as np
import cv2
from fastapi import UploadFile, HTTPException

from app.core.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def _validate_file_type(filename: str) -> None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '.{ext}'. "
                f"Allowed types: {settings.allowed_extensions}"
            ),
        )


async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    _validate_file_type(file.filename or "")

    contents = await file.read()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size: {settings.max_upload_size_mb} MB",
        )

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. The file may be corrupted or is not a valid image.",
        )

    logger.info(
        f"Image loaded: '{file.filename}' | "
        f"size={len(contents):,} bytes | shape={image.shape}"
    )
    return image

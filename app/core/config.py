from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ID Document Verification Assistant"
    app_version: str = "1.0.0"
    debug: bool = False

    max_upload_size_mb: int = 10
    allowed_extensions: list[str] = ["jpg", "jpeg", "png"]

    # Image quality thresholds
    blur_threshold: float = 100.0
    brightness_min: float = 50.0
    brightness_max: float = 200.0
    contrast_min: float = 30.0

    model_config = {"env_file": ".env"}


settings = Settings()

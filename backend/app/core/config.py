"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central configuration for MediSense AI backend."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="MediSense AI", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    api_prefix: str = Field(default="/api", alias="API_PREFIX")

    database_url: str = Field(
        default="postgresql+psycopg2://medisense:CHANGE_ME@localhost:5432/medisense_ai",
        alias="DATABASE_URL",
    )

    jwt_secret_key: str = Field(
        default="CHANGE_ME_TO_A_LONG_RANDOM_SECRET",
        alias="JWT_SECRET_KEY",
    )
    access_token_expire_minutes: int = Field(
        default=60,
        alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        alias="REFRESH_TOKEN_EXPIRE_DAYS",
    )
    algorithm: str = Field(default="HS256", alias="ALGORITHM")

    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )

    max_upload_size_mb: int = Field(default=10, alias="MAX_UPLOAD_SIZE_MB")
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")

    xray_model_dir: str = Field(default="models/xray", alias="XRAY_MODEL_DIR")
    xray_default_model: str = Field(
        default="chest_resnet18",
        alias="XRAY_DEFAULT_MODEL",
    )
    xray_chest_checkpoint: str = Field(default="", alias="XRAY_CHEST_CHECKPOINT")

    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    llm_fallback: str = Field(default="gemini", alias="LLM_FALLBACK")
    llm_offline: str = Field(default="ollama", alias="LLM_OFFLINE")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")

    provider_data_source: str = Field(default="", alias="PROVIDER_DATA_SOURCE")
    provider_api_base_url: str = Field(default="", alias="PROVIDER_API_BASE_URL")
    provider_api_key: str = Field(default="", alias="PROVIDER_API_KEY")

    facility_data_source: str = Field(default="", alias="FACILITY_DATA_SOURCE")
    facility_api_base_url: str = Field(default="", alias="FACILITY_API_BASE_URL")
    facility_api_key: str = Field(default="", alias="FACILITY_API_KEY")

    medical_shop_data_source: str = Field(
        default="",
        alias="MEDICAL_SHOP_DATA_SOURCE",
    )
    medical_shop_api_base_url: str = Field(
        default="",
        alias="MEDICAL_SHOP_API_BASE_URL",
    )
    medical_shop_api_key: str = Field(
        default="",
        alias="MEDICAL_SHOP_API_KEY",
    )

    provider_page_size: int = Field(default=10, alias="PROVIDER_PAGE_SIZE")
    facility_page_size: int = Field(default=10, alias="FACILITY_PAGE_SIZE")
    medical_shop_page_size: int = Field(default=10, alias="MEDICAL_SHOP_PAGE_SIZE")

    local_radius_km: int = Field(default=25, alias="LOCAL_RADIUS_KM")
    expanded_radius_km: int = Field(default=50, alias="EXPANDED_RADIUS_KM")

    map_default_zoom: int = Field(default=13, alias="MAP_DEFAULT_ZOOM")

    geoapify_api_key: str = Field(default="", alias="GEOAPIFY_API_KEY")
    geoapify_base_url: str = Field(
        default="https://api.geoapify.com/v2/places",
        alias="GEOAPIFY_BASE_URL",
    )
    geoapify_facility_category: str = Field(
        default="healthcare.hospital",
        alias="GEOAPIFY_FACILITY_CATEGORY",
    )
    geoapify_medical_shop_category: str = Field(
        default="healthcare.pharmacy",
        alias="GEOAPIFY_MEDICAL_SHOP_CATEGORY",
    )
    geoapify_default_radius_km: int = Field(default=25, alias="GEOAPIFY_DEFAULT_RADIUS_KM")
    geoapify_expanded_radius_km: int = Field(default=50, alias="GEOAPIFY_EXPANDED_RADIUS_KM")
    geoapify_page_size: int = Field(default=10, alias="GEOAPIFY_PAGE_SIZE")
    geoapify_timeout_seconds: int = Field(default=10, alias="GEOAPIFY_TIMEOUT_SECONDS")

    # Version strings for audit logging (updated as modules mature)
    rules_version: str = "triage-rules-v0.1"
    nlp_version: str = "symptom-nlp-v0.1"
    ocr_version: str = "report-ocr-v0.1"
    llm_version: str = "llm-orchestrator-v0.1"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _strip_origins(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path

    @property
    def xray_model_path(self) -> Path:
        path = Path(self.xray_model_dir)
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()

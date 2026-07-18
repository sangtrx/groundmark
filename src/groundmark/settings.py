from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROUNDMARK_", env_file=".env", extra="ignore", populate_by_name=True
    )

    data_dir: Path = Path("./.groundmark")
    cache_dir: Path | None = None
    database_path: Path | None = None
    max_upload_mb: int = Field(default=100, ge=1, le=2048)
    max_pages: int = Field(default=1000, ge=1, le=10000)
    api_key: str | None = None
    cors_origins: str = ""
    worker_poll_seconds: float = Field(default=0.5, ge=0.05, le=60)
    job_lease_seconds: int = Field(default=900, ge=30, le=86400)
    result_ttl_hours: int = Field(default=168, ge=1, le=8760)
    log_json: bool = True

    # OpenAI Responses API. The provider is enabled only when openai_api_key is set.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GROUNDMARK_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("GROUNDMARK_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )
    openai_model: str = "gpt-5.6-luna"
    openai_accurate_model: str = "gpt-5.6-terra"
    openai_vision_model: str | None = None
    openai_timeout_seconds: float = Field(default=180, ge=5, le=900)
    openai_max_retries: int = Field(default=3, ge=0, le=10)
    openai_max_input_chars: int = Field(default=400_000, ge=10_000, le=4_000_000)
    openai_max_output_tokens: int = Field(default=12_000, ge=256, le=128_000)
    openai_store: bool = False
    openai_fallback_on_error: bool = True
    openai_default_for_extract: bool = True
    openai_default_for_classify: bool = True
    openai_default_for_schema: bool = True
    openai_default_for_section: bool = False
    openai_max_repair_regions: int = Field(default=20, ge=1, le=500)

    # MarkItDown can use the same OpenAI client for image descriptions/plugins.
    markitdown_enable_plugins: bool = False
    markitdown_llm_model: str | None = None

    # Deprecated v0.3 compatibility settings. OpenAI settings above take precedence.
    vision_base_url: str | None = None
    vision_api_key: SecretStr | None = None
    vision_model: str | None = None
    repair_endpoint: str | None = None
    repair_api_key: SecretStr | None = None
    repair_model: str | None = None

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or self.data_dir / "cache"

    @property
    def resolved_database_path(self) -> Path:
        return self.database_path or self.data_dir / "groundmark.sqlite3"

    @property
    def resolved_openai_api_key(self) -> str | None:
        secret = self.openai_api_key or self.vision_api_key or self.repair_api_key
        if secret:
            value = secret.get_secret_value().strip()
            if value:
                return value
        # Accept the official SDK convention without forcing users to duplicate secrets.
        return os.getenv("OPENAI_API_KEY") or None

    @property
    def resolved_openai_base_url(self) -> str:
        return (
            os.getenv("OPENAI_BASE_URL")
            or self.openai_base_url
            or self.vision_base_url
            or self.repair_endpoint
            or "https://api.openai.com/v1"
        )

    @property
    def resolved_openai_vision_model(self) -> str:
        return (
            self.openai_vision_model
            or self.markitdown_llm_model
            or self.vision_model
            or self.repair_model
            or self.openai_accurate_model
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_database_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

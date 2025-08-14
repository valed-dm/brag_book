from enum import Enum
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from src.core.base_paths import BasePaths


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BasePaths, BaseSettings):
    """
    Application settings with environment variable loading.
    Inherits path-related properties from BasePaths.
    """

    model_config = SettingsConfigDict(
        env_file=BasePaths.PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core Application Settings ---
    APP_NAME: str = Field(
        ...,
        validation_alias="APP_NAME",
        description="Name of the application.",
        examples=["My App"],
    )

    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        validation_alias="ENVIRONMENT",
        description="Runtime environment.",
    )

    LOG_LEVEL: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Application log level.",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )

    OPENAI_API_KEY: str = Field(
        ...,
        validation_alias="OPENAI_API_KEY",
        description="OpenAPI Key",
    )

    # --- AI Provider Selection ---
    AI_PROVIDER: str = Field(
        ...,
        validation_alias="AI_PROVIDER",
        description="Select your AI provider: 'ollama' or 'openai'",
        examples=["ollama", "openai"],
    )

    # --- AI Model Settings ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    GENERATOR_MODEL_NAME: str = "gpt-3.5-turbo"
    ROUTER_MODEL_NAME: str = "gpt-3.5-turbo"

    # --- Local Ollama ---
    OLLAMA_MODEL_NAME: str = "llama3"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- Translator ---
    TRANSLATION_PROVIDER: str = "ollama"
    TRANSLATOR_MODEL_MAPPING: Dict[str, str] = {
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    }

    CHROMA_DISTANCE_THRESHOLD: float = 0.5  # Default L2 threshold

    # --- Scraping & Chunking Settings ---
    REQUEST_HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    CHUNK_SIZE: int = 1000  # Target size in characters
    CHUNK_OVERLAP: int = 200  # Overlap between consecutive chunks


settings = Settings()

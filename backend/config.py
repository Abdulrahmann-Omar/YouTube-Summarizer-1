"""
Configuration management for the YouTube Summarizer application.
Uses pydantic-settings for type-safe environment variable handling.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google Gemini API
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    
    # spaCy Configuration
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    
    # Summarization Settings
    max_summary_length: int = Field(default=1000, env="MAX_SUMMARY_LENGTH")
    default_fraction: float = Field(default=0.3, env="DEFAULT_FRACTION")
    min_fraction: float = Field(default=0.1, env="MIN_FRACTION")
    max_fraction: float = Field(default=0.8, env="MAX_FRACTION")
    
    # Server Configuration
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        env="CORS_ORIGINS"
    )
    
    # Application Settings
    debug_mode: bool = Field(default=True, env="DEBUG_MODE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    
    # Cache Settings
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins string into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


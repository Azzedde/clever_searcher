"""Configuration management for Clever Searcher"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Suppress TensorFlow warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # LLM Provider Selection
    llm_provider: str = "openai"  # "openai" or "ollama"
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    
    # OpenAI specific settings
    openai_base_url_openai: str = "https://api.openai.com/v1"
    model_planner_openai: str = "gpt-4o"
    model_summary_openai: str = "gpt-4o"
    
    # Ollama specific settings
    openai_base_url_ollama: str = "http://localhost:11434/v1"
    model_planner_ollama: str = "llama3.2:3b"
    model_summary_ollama: str = "llama3.2:3b"
    
    # Search APIs
    tavily_api_key: Optional[str] = None
    default_search_engine: str = "tavily"  # duckduckgo, tavily
    
    # Database
    database_url: str = "sqlite:///clever_searcher.db"
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Search Configuration
    max_pages_per_run: int = 50
    max_pages_per_domain: int = 10
    request_timeout: int = 30
    request_delay: float = 1.0
    
    # Content Processing
    min_content_length: int = 200
    max_content_length: int = 50000
    similarity_threshold: float = 0.85
    
    # Personalization
    personalization_threshold: float = 0.3
    feedback_learning_rate: float = 0.1
    
    # Output
    output_dir: Path = Path("output")
    digest_format: str = "markdown"  # markdown, html, json
    
    # Scheduling
    default_revisit_hours: int = 24
    scheduler_timezone: str = "UTC"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
    
    @property
    def openai_base_url(self) -> str:
        """Get the appropriate OpenAI base URL based on the selected provider"""
        if self.llm_provider == "openai":
            return self.openai_base_url_openai
        else:  # ollama
            return self.openai_base_url_ollama
    
    @property
    def model_planner(self) -> str:
        """Get the appropriate planner model based on the selected provider"""
        if self.llm_provider == "openai":
            return self.model_planner_openai
        else:  # ollama
            return self.model_planner_ollama
    
    @property
    def model_summary(self) -> str:
        """Get the appropriate summary model based on the selected provider"""
        if self.llm_provider == "openai":
            return self.model_summary_openai
        else:  # ollama
            return self.model_summary_ollama


# Global settings instance
settings = Settings()


def get_data_dir() -> Path:
    """Get the data directory for storing databases and files"""
    data_dir = Path.home() / ".clever_searcher"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_database_path() -> str:
    """Get the full database path"""
    if settings.database_url.startswith("sqlite:///"):
        # Convert relative path to absolute in data directory
        db_name = settings.database_url.replace("sqlite:///", "")
        if not os.path.isabs(db_name):
            db_path = get_data_dir() / db_name
            return f"sqlite:///{db_path}"
    return settings.database_url


def get_output_dir() -> Path:
    """Get the output directory for digests and reports"""
    if settings.output_dir.is_absolute():
        output_dir = settings.output_dir
    else:
        output_dir = get_data_dir() / settings.output_dir
    
    output_dir.mkdir(exist_ok=True)
    return output_dir
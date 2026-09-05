"""Application configuration management using Pydantic Settings.

Reads configuration parameters from environment variables and an optional
`.env` file, providing typed and validated settings across all application modules.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration class for the Multi-Agent Analyst application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration
    PROJECT_NAME: str = "Multi-Agent Code & Document Analyst API"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Ollama Service Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # LLM & Embedding Model Identifiers
    SUPERVISOR_MODEL: str = "llama3.1:8b"
    CODE_ANALYST_MODEL: str = "qwen2.5-coder:7b"
    DOC_PARSER_MODEL: str = "llama3.1:8b"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Data Persistence Paths
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHECKPOINT_DB_PATH: str = "./data/checkpoints.db"
    CHROMA_COLLECTION_NAME: str = "system_docs"

    # Document Chunking Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    def ensure_directories(self) -> None:
        """Ensures that all persistent data directories exist on the local filesystem."""
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(self.CHECKPOINT_DB_PATH).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
settings.ensure_directories()

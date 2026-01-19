"""Configuration management for the Weather RAG application."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class OpenAIConfig:
    """OpenAI configuration settings."""
    api_key: str
    model: str
    embedding_model: str


@dataclass
class WeatherConfig:
    """OpenWeatherMap configuration settings."""
    api_key: str
    base_url: str = "https://api.openweathermap.org/data/2.5"


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration settings."""
    host: str
    port: int
    collection_name: str = "weather_documents"


@dataclass
class LangSmithConfig:
    """LangSmith configuration settings."""
    api_key: str
    tracing_enabled: bool
    project_name: str


@dataclass
class AppConfig:
    """Main application configuration."""
    openai: OpenAIConfig
    weather: WeatherConfig
    qdrant: QdrantConfig
    langsmith: LangSmithConfig
    docs_path: str


def load_config() -> AppConfig:
    """Load configuration from environment variables.
    
    Returns:
        AppConfig: Application configuration object.
        
    Raises:
        ValueError: If required environment variables are missing.
    """
    # Validate required environment variables
    required_vars = ["OPENAI_API_KEY", "OPENWEATHERMAP_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return AppConfig(
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ),
        weather=WeatherConfig(
            api_key=os.getenv("OPENWEATHERMAP_API_KEY", ""),
            base_url=os.getenv("OPENWEATHERMAP_BASE_URL", "https://api.openweathermap.org/data/2.5"),
        ),
        qdrant=QdrantConfig(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION", "weather_documents"),
        ),
        langsmith=LangSmithConfig(
            api_key=os.getenv("LANGCHAIN_API_KEY", ""),
            tracing_enabled=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            project_name=os.getenv("LANGCHAIN_PROJECT", "weather-rag-pipeline"),
        ),
        docs_path=os.getenv("DOCS_PATH", "docs"),
    )


# Global configuration instance
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the application configuration singleton.
    
    Returns:
        AppConfig: Application configuration object.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config

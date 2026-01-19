"""Pytest configuration and shared fixtures."""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_openai_api_key():
    """Provide a mock OpenAI API key."""
    return "sk-test-key-12345"


@pytest.fixture
def mock_weather_api_key():
    """Provide a mock weather API key."""
    return "test-weather-api-key"


@pytest.fixture
def mock_langsmith_api_key():
    """Provide a mock LangSmith API key."""
    return "test-langsmith-key"


@pytest.fixture
def mock_environment(mock_openai_api_key, mock_weather_api_key, mock_langsmith_api_key):
    """Set up mock environment variables."""
    original_env = os.environ.copy()
    
    os.environ["OPENAI_API_KEY"] = mock_openai_api_key
    os.environ["OPENWEATHERMAP_API_KEY"] = mock_weather_api_key
    os.environ["LANGCHAIN_API_KEY"] = mock_langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["QDRANT_HOST"] = "localhost"
    os.environ["QDRANT_PORT"] = "6333"
    
    yield
    
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_weather_response():
    """Provide a sample weather API response."""
    return {
        "name": "London",
        "sys": {"country": "GB"},
        "main": {
            "temp": 15.5,
            "feels_like": 14.0,
            "humidity": 75,
            "pressure": 1013,
        },
        "weather": [{"description": "partly cloudy"}],
        "wind": {"speed": 5.2},
        "visibility": 10000,
        "clouds": {"all": 40},
    }


@pytest.fixture
def sample_forecast_response():
    """Provide a sample forecast API response."""
    return {
        "city": {"name": "London", "country": "GB"},
        "list": [
            {
                "dt_txt": "2024-01-15 12:00:00",
                "main": {
                    "temp": 16.0,
                    "feels_like": 15.0,
                    "humidity": 70,
                    "pressure": 1012,
                },
                "weather": [{"description": "cloudy"}],
                "wind": {"speed": 4.0},
            },
            {
                "dt_txt": "2024-01-15 15:00:00",
                "main": {
                    "temp": 17.0,
                    "feels_like": 16.0,
                    "humidity": 65,
                    "pressure": 1011,
                },
                "weather": [{"description": "sunny"}],
                "wind": {"speed": 3.5},
            },
        ],
    }


@pytest.fixture
def sample_embedding():
    """Provide a sample embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def sample_documents():
    """Provide sample document data."""
    return [
        {
            "content": "Weather forecasting uses numerical weather prediction models.",
            "metadata": {"source": "forecast.pdf", "page": 1},
        },
        {
            "content": "Climate normals are 30-year averages of weather data.",
            "metadata": {"source": "climate.pdf", "page": 2},
        },
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = Mock()
    mock.invoke.return_value = Mock(content="Test response")
    return mock


@pytest.fixture
def mock_chat_openai():
    """Create a mock ChatOpenAI instance."""
    mock = Mock()
    mock.invoke.return_value = Mock(content="Test response")
    return mock

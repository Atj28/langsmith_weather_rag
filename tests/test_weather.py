"""Tests for the weather service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from app.services.weather import (
    WeatherService,
    WeatherData,
    ForecastData,
    WeatherServiceError,
)


class TestWeatherData:
    """Tests for WeatherData model."""
    
    def test_weather_data_creation(self):
        """Test creating a WeatherData instance."""
        data = WeatherData(
            city="London",
            country="GB",
            temperature=15.5,
            feels_like=14.0,
            humidity=75,
            description="Partly cloudy",
            wind_speed=5.2,
            pressure=1013,
            visibility=10000,
            clouds=40,
        )
        
        assert data.city == "London"
        assert data.country == "GB"
        assert data.temperature == 15.5
        assert data.humidity == 75
    
    def test_weather_data_summary(self):
        """Test the summary generation."""
        data = WeatherData(
            city="Paris",
            country="FR",
            temperature=20.0,
            feels_like=19.0,
            humidity=60,
            description="Clear sky",
            wind_speed=3.0,
            pressure=1015,
            visibility=10000,
            clouds=10,
        )
        
        summary = data.to_summary()
        
        assert "Paris" in summary
        assert "FR" in summary
        assert "20.0°C" in summary
        assert "Clear sky" in summary
        assert "60%" in summary


class TestForecastData:
    """Tests for ForecastData model."""
    
    def test_forecast_data_creation(self):
        """Test creating a ForecastData instance."""
        forecasts = [
            {"datetime": "2024-01-15 12:00:00", "temperature": 18.0, "description": "Sunny"},
            {"datetime": "2024-01-15 15:00:00", "temperature": 20.0, "description": "Cloudy"},
        ]
        
        data = ForecastData(
            city="Berlin",
            country="DE",
            forecasts=forecasts,
        )
        
        assert data.city == "Berlin"
        assert len(data.forecasts) == 2
    
    def test_forecast_summary(self):
        """Test the forecast summary generation."""
        forecasts = [
            {"datetime": "2024-01-15 12:00:00", "temperature": 18.0, "description": "Sunny"},
        ]
        
        data = ForecastData(
            city="Tokyo",
            country="JP",
            forecasts=forecasts,
        )
        
        summary = data.to_summary()
        
        assert "Tokyo" in summary
        assert "JP" in summary


class TestWeatherService:
    """Tests for WeatherService."""
    
    @pytest.fixture
    def weather_service(self):
        """Create a weather service instance for testing."""
        return WeatherService(api_key="test_api_key")
    
    @pytest.fixture
    def mock_weather_response(self):
        """Mock weather API response."""
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
    def mock_forecast_response(self):
        """Mock forecast API response."""
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
            ],
        }
    
    def test_get_current_weather_success(self, weather_service, mock_weather_response):
        """Test successful weather fetch."""
        with patch.object(weather_service, '_make_request', return_value=mock_weather_response):
            result = weather_service.get_current_weather("London")
            
            assert isinstance(result, WeatherData)
            assert result.city == "London"
            assert result.country == "GB"
            assert result.temperature == 15.5
    
    def test_get_current_weather_city_not_found(self, weather_service):
        """Test weather fetch with invalid city."""
        with patch.object(weather_service, '_make_request') as mock_request:
            mock_request.side_effect = WeatherServiceError("City not found")
            
            with pytest.raises(WeatherServiceError) as exc_info:
                weather_service.get_current_weather("InvalidCity123")
            
            assert "City not found" in str(exc_info.value)
    
    def test_get_forecast_success(self, weather_service, mock_forecast_response):
        """Test successful forecast fetch."""
        with patch.object(weather_service, '_make_request', return_value=mock_forecast_response):
            result = weather_service.get_forecast("London")
            
            assert isinstance(result, ForecastData)
            assert result.city == "London"
            assert len(result.forecasts) > 0
    
    def test_api_key_error(self):
        """Test handling of invalid API key."""
        service = WeatherService(api_key="invalid_key")
        
        with patch.object(service._client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Invalid API key"
            mock_get.return_value = mock_response
            
            with pytest.raises(WeatherServiceError) as exc_info:
                service.get_current_weather("London")
            
            assert "Invalid API key" in str(exc_info.value)
    
    def test_timeout_error(self, weather_service):
        """Test handling of timeout errors."""
        with patch.object(weather_service._client, 'get') as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Connection timed out")
            
            with pytest.raises(WeatherServiceError) as exc_info:
                weather_service.get_current_weather("London")
            
            assert "timed out" in str(exc_info.value)
    
    def test_network_error(self, weather_service):
        """Test handling of network errors."""
        with patch.object(weather_service._client, 'get') as mock_get:
            mock_get.side_effect = httpx.RequestError("Network error")
            
            with pytest.raises(WeatherServiceError) as exc_info:
                weather_service.get_current_weather("London")
            
            assert "Network error" in str(exc_info.value)
    
    def test_context_manager(self):
        """Test using service as context manager."""
        with WeatherService(api_key="test_key") as service:
            assert service is not None
    
    def test_get_weather_by_coordinates(self, weather_service, mock_weather_response):
        """Test fetching weather by coordinates."""
        with patch.object(weather_service, '_make_request', return_value=mock_weather_response):
            result = weather_service.get_weather_by_coordinates(51.5074, -0.1278)
            
            assert isinstance(result, WeatherData)
            assert result.city == "London"


class TestWeatherServiceIntegration:
    """Integration tests for weather service (require network)."""
    
    @pytest.mark.skip(reason="Requires valid API key and network access")
    def test_real_api_call(self):
        """Test with real API call (skip in CI)."""
        import os
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        
        if not api_key:
            pytest.skip("No API key available")
        
        service = WeatherService(api_key=api_key)
        result = service.get_current_weather("London")
        
        assert result.city == "London"
        assert result.temperature is not None

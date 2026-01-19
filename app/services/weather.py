"""OpenWeatherMap API integration service."""

import httpx
from dataclasses import dataclass
from typing import Any


@dataclass
class WeatherData:
    """Weather data response model."""
    city: str
    country: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int
    visibility: int
    clouds: int
    
    def to_summary(self) -> str:
        """Generate a human-readable weather summary."""
        return (
            f"Weather in {self.city}, {self.country}:\n"
            f"- Temperature: {self.temperature}°C (feels like {self.feels_like}°C)\n"
            f"- Conditions: {self.description}\n"
            f"- Humidity: {self.humidity}%\n"
            f"- Wind Speed: {self.wind_speed} m/s\n"
            f"- Pressure: {self.pressure} hPa\n"
            f"- Visibility: {self.visibility / 1000:.1f} km\n"
            f"- Cloud Cover: {self.clouds}%"
        )


@dataclass
class ForecastData:
    """Forecast data response model."""
    city: str
    country: str
    forecasts: list[dict[str, Any]]
    
    def to_summary(self) -> str:
        """Generate a human-readable forecast summary."""
        summary = f"Weather Forecast for {self.city}, {self.country}:\n\n"
        for forecast in self.forecasts[:8]:  # Next 24 hours (3-hour intervals)
            summary += (
                f"📅 {forecast['datetime']}:\n"
                f"   Temperature: {forecast['temperature']}°C, "
                f"{forecast['description']}\n"
            )
        return summary


class WeatherServiceError(Exception):
    """Custom exception for weather service errors."""
    pass


class WeatherService:
    """Service for fetching weather data from OpenWeatherMap API."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openweathermap.org/data/2.5"):
        """Initialize the weather service.
        
        Args:
            api_key: OpenWeatherMap API key.
            base_url: Base URL for the API.
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client = httpx.Client(timeout=30.0)
    
    def _make_request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make a request to the OpenWeatherMap API.
        
        Args:
            endpoint: API endpoint to call.
            params: Query parameters.
            
        Returns:
            JSON response from the API.
            
        Raises:
            WeatherServiceError: If the API request fails.
        """
        params["appid"] = self.api_key
        params["units"] = "metric"  # Use Celsius
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self._client.get(url, params=params)
            
            if response.status_code == 401:
                raise WeatherServiceError("Invalid API key. Please check your OpenWeatherMap API key.")
            elif response.status_code == 404:
                raise WeatherServiceError(f"City not found. Please check the city name.")
            elif response.status_code != 200:
                raise WeatherServiceError(f"API request failed with status {response.status_code}: {response.text}")
            
            return response.json()
            
        except httpx.TimeoutException:
            raise WeatherServiceError("Request timed out. Please try again.")
        except httpx.RequestError as e:
            raise WeatherServiceError(f"Network error: {str(e)}")
    
    def get_current_weather(self, city: str) -> WeatherData:
        """Get current weather for a city.
        
        Args:
            city: City name (can include country code, e.g., "London,UK").
            
        Returns:
            WeatherData object with current conditions.
            
        Raises:
            WeatherServiceError: If the request fails.
        """
        data = self._make_request("weather", {"q": city})
        
        return WeatherData(
            city=data["name"],
            country=data["sys"]["country"],
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            humidity=data["main"]["humidity"],
            description=data["weather"][0]["description"].capitalize(),
            wind_speed=data["wind"]["speed"],
            pressure=data["main"]["pressure"],
            visibility=data.get("visibility", 10000),
            clouds=data["clouds"]["all"],
        )
    
    def get_forecast(self, city: str, days: int = 5) -> ForecastData:
        """Get weather forecast for a city.
        
        Args:
            city: City name (can include country code).
            days: Number of days to forecast (max 5 for free tier).
            
        Returns:
            ForecastData object with forecast information.
            
        Raises:
            WeatherServiceError: If the request fails.
        """
        # OpenWeatherMap free tier provides 5-day forecast with 3-hour intervals
        cnt = min(days * 8, 40)  # Max 40 data points (5 days)
        data = self._make_request("forecast", {"q": city, "cnt": cnt})
        
        forecasts = []
        for item in data["list"]:
            forecasts.append({
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "description": item["weather"][0]["description"].capitalize(),
                "wind_speed": item["wind"]["speed"],
                "pressure": item["main"]["pressure"],
            })
        
        return ForecastData(
            city=data["city"]["name"],
            country=data["city"]["country"],
            forecasts=forecasts,
        )
    
    def get_weather_by_coordinates(self, lat: float, lon: float) -> WeatherData:
        """Get current weather by geographic coordinates.
        
        Args:
            lat: Latitude.
            lon: Longitude.
            
        Returns:
            WeatherData object with current conditions.
            
        Raises:
            WeatherServiceError: If the request fails.
        """
        data = self._make_request("weather", {"lat": lat, "lon": lon})
        
        return WeatherData(
            city=data["name"],
            country=data["sys"]["country"],
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            humidity=data["main"]["humidity"],
            description=data["weather"][0]["description"].capitalize(),
            wind_speed=data["wind"]["speed"],
            pressure=data["main"]["pressure"],
            visibility=data.get("visibility", 10000),
            clouds=data["clouds"]["all"],
        )
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

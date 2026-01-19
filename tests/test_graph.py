"""Tests for the LangGraph agent pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Literal

from app.agents.nodes import AgentNodes, create_router_condition
from app.agents.graph import (
    AgentState,
    create_agent_graph,
    create_agent_with_services,
    run_agent_sync,
)


class TestAgentNodes:
    """Tests for individual agent nodes."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = Mock()
        mock.invoke.return_value = Mock(content="weather")
        return mock
    
    @pytest.fixture
    def mock_weather_service(self):
        """Create a mock weather service."""
        mock = Mock()
        weather_data = Mock()
        weather_data.to_summary.return_value = "Weather in London: 15°C, Cloudy"
        mock.get_current_weather.return_value = weather_data
        
        forecast_data = Mock()
        forecast_data.to_summary.return_value = "Forecast: 16°C tomorrow"
        mock.get_forecast.return_value = forecast_data
        
        return mock
    
    @pytest.fixture
    def mock_rag_retriever(self):
        """Create a mock RAG retriever."""
        mock = Mock()
        result = Mock()
        result.context = "Document content about weather forecasting..."
        result.num_results = 3
        result.get_sources.return_value = ["doc1.pdf", "doc2.pdf"]
        mock.retrieve.return_value = result
        return mock
    
    @pytest.fixture
    def agent_nodes(self, mock_llm, mock_weather_service, mock_rag_retriever):
        """Create agent nodes with mocked dependencies."""
        return AgentNodes(
            llm=mock_llm,
            weather_service=mock_weather_service,
            rag_retriever=mock_rag_retriever,
        )
    
    def test_router_node_weather(self, agent_nodes, mock_llm):
        """Test router correctly identifies weather queries."""
        mock_llm.invoke.return_value = Mock(content="weather")
        
        state = {"query": "What's the weather in New York?"}
        result = agent_nodes.router_node(state)
        
        assert result["query_type"] == "weather"
    
    def test_router_node_document(self, agent_nodes, mock_llm):
        """Test router correctly identifies document queries."""
        mock_llm.invoke.return_value = Mock(content="document")
        
        state = {"query": "How do weather forecasts work?"}
        result = agent_nodes.router_node(state)
        
        assert result["query_type"] == "document"
    
    def test_router_node_normalizes_response(self, agent_nodes, mock_llm):
        """Test router normalizes LLM response."""
        mock_llm.invoke.return_value = Mock(content="WEATHER\n")
        
        state = {"query": "Temperature in Paris?"}
        result = agent_nodes.router_node(state)
        
        assert result["query_type"] == "weather"
    
    def test_weather_node_success(self, agent_nodes, mock_llm, mock_weather_service):
        """Test weather node fetches data successfully."""
        mock_llm.invoke.return_value = Mock(content="London")
        
        state = {"query": "What's the weather in London?", "query_type": "weather"}
        result = agent_nodes.weather_node(state)
        
        assert "Weather in London" in result["context"]
        assert result["source"] == "weather_api"
        assert result["error"] is None
    
    def test_weather_node_with_forecast(self, agent_nodes, mock_llm, mock_weather_service):
        """Test weather node includes forecast when requested."""
        mock_llm.invoke.return_value = Mock(content="London")
        
        state = {"query": "Will it rain tomorrow in London?", "query_type": "weather"}
        result = agent_nodes.weather_node(state)
        
        mock_weather_service.get_forecast.assert_called_once()
        assert "Forecast" in result["context"]
    
    def test_weather_node_error(self, agent_nodes, mock_llm, mock_weather_service):
        """Test weather node handles errors gracefully."""
        from app.services.weather import WeatherServiceError
        
        mock_llm.invoke.return_value = Mock(content="InvalidCity")
        mock_weather_service.get_current_weather.side_effect = WeatherServiceError("City not found")
        
        state = {"query": "Weather in InvalidCity", "query_type": "weather"}
        result = agent_nodes.weather_node(state)
        
        assert result["error"] is not None
        assert "City not found" in result["context"]
    
    def test_rag_node_success(self, agent_nodes, mock_rag_retriever):
        """Test RAG node retrieves documents successfully."""
        state = {"query": "What is climate normal?", "query_type": "document"}
        result = agent_nodes.rag_node(state)
        
        assert result["source"] == "documents"
        assert result["num_results"] == 3
        assert len(result["sources"]) == 2
        assert result["error"] is None
    
    def test_rag_node_no_results(self, agent_nodes, mock_rag_retriever):
        """Test RAG node handles no results."""
        mock_result = Mock()
        mock_result.context = ""
        mock_result.num_results = 0
        mock_result.get_sources.return_value = []
        mock_rag_retriever.retrieve.return_value = mock_result
        
        state = {"query": "Obscure topic", "query_type": "document"}
        result = agent_nodes.rag_node(state)
        
        assert "No relevant documents" in result["context"]
    
    def test_rag_node_error(self, agent_nodes, mock_rag_retriever):
        """Test RAG node handles errors gracefully."""
        mock_rag_retriever.retrieve.side_effect = Exception("Database error")
        
        state = {"query": "Test query", "query_type": "document"}
        result = agent_nodes.rag_node(state)
        
        assert result["error"] is not None
        assert "Error retrieving documents" in result["context"]
    
    def test_response_node_weather(self, agent_nodes, mock_llm):
        """Test response node generates weather response."""
        mock_llm.invoke.return_value = Mock(content="The weather in London is pleasant with 15°C.")
        
        state = {
            "query": "What's the weather?",
            "query_type": "weather",
            "context": "Weather in London: 15°C",
            "error": None,
        }
        result = agent_nodes.response_node(state)
        
        assert "response" in result
        assert len(result["response"]) > 0
    
    def test_response_node_document(self, agent_nodes, mock_llm):
        """Test response node generates document-based response."""
        mock_llm.invoke.return_value = Mock(content="Based on the documents, weather forecasting uses...")
        
        state = {
            "query": "How do forecasts work?",
            "query_type": "document",
            "context": "Document content...",
            "error": None,
        }
        result = agent_nodes.response_node(state)
        
        assert "response" in result
    
    def test_response_node_with_error(self, agent_nodes):
        """Test response node handles errors in state."""
        state = {
            "query": "Test",
            "query_type": "weather",
            "context": "",
            "error": "City not found",
        }
        result = agent_nodes.response_node(state)
        
        assert "error" in result["response"].lower()


class TestRouterCondition:
    """Tests for the router condition function."""
    
    def test_router_condition_weather(self):
        """Test router condition returns weather."""
        state = {"query_type": "weather"}
        result = create_router_condition(state)
        assert result == "weather"
    
    def test_router_condition_document(self):
        """Test router condition returns document."""
        state = {"query_type": "document"}
        result = create_router_condition(state)
        assert result == "document"
    
    def test_router_condition_default(self):
        """Test router condition defaults to document."""
        state = {}
        result = create_router_condition(state)
        assert result == "document"


class TestAgentGraph:
    """Tests for the agent graph creation and execution."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        mock = Mock()
        mock.openai.api_key = "test-key"
        mock.openai.model = "gpt-4o"
        mock.openai.embedding_model = "text-embedding-3-small"
        mock.weather.api_key = "weather-key"
        mock.weather.base_url = "https://api.openweathermap.org/data/2.5"
        mock.qdrant.host = "localhost"
        mock.qdrant.port = 6333
        mock.qdrant.collection_name = "test_collection"
        return mock
    
    def test_create_agent_with_services(self):
        """Test creating agent with pre-initialized services."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="document")
        
        mock_weather = Mock()
        mock_rag = Mock()
        mock_rag.retrieve.return_value = Mock(
            context="Test context",
            num_results=1,
            get_sources=lambda: ["test.pdf"]
        )
        
        graph = create_agent_with_services(
            llm=mock_llm,
            weather_service=mock_weather,
            rag_retriever=mock_rag,
        )
        
        assert graph is not None
    
    def test_run_agent_sync(self):
        """Test synchronous agent execution."""
        mock_llm = Mock()
        # Router returns document
        mock_llm.invoke.side_effect = [
            Mock(content="document"),  # Router
            Mock(content="Based on the documents..."),  # Response
        ]
        
        mock_weather = Mock()
        mock_rag = Mock()
        mock_rag.retrieve.return_value = Mock(
            context="Document content",
            num_results=1,
            get_sources=lambda: ["test.pdf"]
        )
        
        graph = create_agent_with_services(
            llm=mock_llm,
            weather_service=mock_weather,
            rag_retriever=mock_rag,
        )
        
        result = run_agent_sync(graph, "What is weather forecasting?")
        
        assert "response" in result
        assert result["query_type"] == "document"


class TestAgentState:
    """Tests for the AgentState schema."""
    
    def test_agent_state_creation(self):
        """Test creating an agent state."""
        state: AgentState = {
            "query": "Test query",
            "query_type": "weather",
            "context": "Test context",
            "response": "Test response",
            "error": None,
            "source": "weather_api",
            "sources": [],
            "num_results": 0,
            "messages": [],
        }
        
        assert state["query"] == "Test query"
        assert state["query_type"] == "weather"
    
    def test_agent_state_partial(self):
        """Test creating a partial agent state."""
        state: AgentState = {
            "query": "Test query",
        }
        
        assert state["query"] == "Test query"
        assert state.get("query_type") is None

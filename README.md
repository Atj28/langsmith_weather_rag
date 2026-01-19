<img width="1508" height="825" alt="Screenshot 2026-01-19 at 1 02 39 PM" src="https://github.com/user-attachments/assets/1eee72f2-3996-4325-aeb7-283feda1d0fd" /># Weather RAG Pipeline

An AI-powered pipeline that combines real-time weather data with document-based question answering using LangGraph, LangChain, and LangSmith.

## Features

- **Intelligent Query Routing**: Automatically routes queries to either weather API or document RAG based on intent
- **Real-time Weather Data**: Fetches current weather and forecasts from OpenWeatherMap API
- **Document RAG**: Retrieval-Augmented Generation for answering questions from PDF documents
- **Vector Database**: Uses Qdrant for efficient similarity search
- **LLM Evaluation**: Integrated LangSmith for response quality evaluation
- **Modern UI**: Streamlit-based chat interface

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  LangGraph      │────▶│   LangSmith     │
│   (Chat)        │     │  Agent Pipeline │     │   (Evaluation)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                      ▼
            ┌───────────────┐      ┌───────────────┐
            │ Weather Node  │      │   RAG Node    │
            │ (OpenWeather) │      │ (Qdrant+PDF)  │
            └───────────────┘      └───────────────┘
```

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key
- OpenWeatherMap API key
- LangSmith API key (optional, for evaluation)

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd weather_rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key

# Optional - LangSmith for evaluation
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=weather-rag-pipeline

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Model Configuration (optional)
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 5. Start Qdrant

```bash
docker-compose up -d
```

### 6. Ingest Documents

Place your PDF files in the `docs/` directory, then run:

```bash
python -m app.main --ingest --docs docs
```

## Usage

### Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

The UI will be available at `http://localhost:8501`

### Command Line

**Single Query:**
```bash
python -m app.main --query "What's the weather in New York?"
```

**Interactive Mode:**
```bash
python -m app.main
```

## Example Queries

**Weather Queries:**
- "What's the weather in London?"
- "Will it rain tomorrow in Paris?"
- "Temperature forecast for Tokyo"

**Document Queries:**
- "What is climate normal?"
- "How do weather forecasts work?"
- "Explain numerical weather prediction"

## Project Structure

```
weather_rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration management
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py         # LangGraph pipeline
│   │   └── nodes.py         # Agent nodes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── weather.py       # OpenWeatherMap API
│   │   ├── embeddings.py    # OpenAI embeddings
│   │   └── vector_store.py  # Qdrant operations
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   └── retriever.py
│   └── evaluation/
│       ├── __init__.py
│       └── langsmith.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_weather.py
│   ├── test_rag.py
│   ├── test_vector_store.py
│   └── test_graph.py
├── docs/                    # PDF documents for RAG
├── streamlit_app.py         # Streamlit UI
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_weather.py -v
```

## LangSmith Evaluation

The application automatically logs traces to LangSmith when configured. Evaluation metrics include:

- **Relevance**: How well the response addresses the query
- **Coherence**: Response clarity and structure
- **Faithfulness**: (For RAG) How well the response reflects source documents

### Viewing Results

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project ("weather-rag-pipeline")
3. View traces, latency, and evaluation scores

## API Reference

### Weather Service

```python
from app.services.weather import WeatherService

service = WeatherService(api_key="your_key")
weather = service.get_current_weather("London")
forecast = service.get_forecast("London", days=5)
```

### RAG Retriever

```python
from app.rag.retriever import RAGRetriever

retriever = RAGRetriever(embedding_service, vector_store)
retriever.ingest_documents("docs/")
result = retriever.retrieve("What is climate?", top_k=5)
```

### Agent Pipeline

```python
from app.agents.graph import create_agent_graph, run_agent_sync
from app.config import load_config

config = load_config()
graph = create_agent_graph(config)
result = run_agent_sync(graph, "What's the weather in NYC?")
```
### Added few photos
<img width="1508" height="825" alt="Screenshot 2026-01-19 at 1 02 39 PM" src="https://github.com/user-attachments/assets/a98514f7-7ceb-4fd7-80e4-ce1aa1dea93c" />


## Troubleshooting

### Qdrant Connection Error
```
Error: Failed to connect to Qdrant
```
Solution: Ensure Qdrant is running with `docker-compose up -d`

### OpenAI API Error
```
Error: Invalid API key
```
Solution: Check your `OPENAI_API_KEY` in `.env`

### No Documents Found
```
Warning: No relevant documents found
```
Solution: Run document ingestion first with `--ingest`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request


## Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [LangSmith](https://smith.langchain.com/) - LLM evaluation
- [Qdrant](https://qdrant.tech/) - Vector database
- [OpenWeatherMap](https://openweathermap.org/) - Weather API
- [Streamlit](https://streamlit.io/) - UI framework

"""Streamlit UI for the Weather RAG application."""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st
from langchain_openai import ChatOpenAI

from app.config import load_config, AppConfig
from app.agents.graph import create_agent_graph, run_agent_sync
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import RAGRetriever
from app.evaluation.langsmith import setup_langsmith_tracing, LangSmithEvaluator


@st.cache_resource
def get_agent_graph(_config: AppConfig):
    """Create and cache the agent graph.
    
    Uses @st.cache_resource to avoid storing in session_state,
    which prevents 'response' key conflicts with Streamlit's state management.
    
    Args:
        _config: Application configuration (underscore prefix prevents hashing).
        
    Returns:
        Compiled LangGraph agent.
    """
    return create_agent_graph(_config)


# Page configuration
st.set_page_config(
    page_title="Weather RAG Assistant",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #0ea5e9;
        --primary-dark: #0284c7;
        --secondary: #f97316;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: #334155;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Space Grotesk', sans-serif;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        border: 1px solid #3b82f6;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        margin-right: 2rem;
    }
    
    .query-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .weather-badge {
        background: linear-gradient(90deg, #f97316, #fb923c);
        color: white;
    }
    
    .document-badge {
        background: linear-gradient(90deg, #8b5cf6, #a78bfa);
        color: white;
    }
    
    .sidebar-section {
        background: rgba(30, 41, 59, 0.5);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0ea5e9;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #f1f5f9;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        color: white;
        border: none;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.3);
    }
    
    .source-citation {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #64748b;
        padding: 0.5rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_processed_input" not in st.session_state:
        st.session_state.last_processed_input = None
    if "config" not in st.session_state:
        # Try to load config automatically from environment
        try:
            st.session_state.config = load_config()
        except ValueError:
            st.session_state.config = None
    if "documents_ingested" not in st.session_state:
        st.session_state.documents_ingested = False
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = None
        # Initialize LangSmith evaluator if config is available
        if st.session_state.config and st.session_state.config.langsmith.api_key:
            try:
                st.session_state.evaluator = setup_langsmith_tracing(
                    api_key=st.session_state.config.langsmith.api_key,
                    project_name=st.session_state.config.langsmith.project_name
                )
            except Exception:
                pass


def render_sidebar():
    """Render the sidebar with configuration and actions."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Configuration Status
        if st.session_state.config:
            st.success("✅ Configuration loaded from environment")
        else:
            st.warning("⚠️ Configuration not found")
            st.info("""
            Please set the following environment variables:
            - `OPENAI_API_KEY`
            - `OPENWEATHERMAP_API_KEY`
            - `LANGCHAIN_API_KEY` (optional)
            
            Or create a `.env` file in the project root.
            """)
        
        # Document Management
        st.markdown("### 📄 Documents")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Ingest Docs", disabled=not st.session_state.config):
                with st.spinner("Ingesting documents..."):
                    try:
                        config = st.session_state.config
                        embedding_service = EmbeddingService(
                            api_key=config.openai.api_key,
                            model=config.openai.embedding_model,
                        )
                        vector_store = VectorStoreService(
                            host=config.qdrant.host,
                            port=config.qdrant.port,
                            collection_name=config.qdrant.collection_name,
                            embedding_dimension=embedding_service.embedding_dimension,
                        )
                        document_loader = DocumentLoader()
                        rag_retriever = RAGRetriever(
                            embedding_service=embedding_service,
                            vector_store=vector_store,
                            document_loader=document_loader,
                        )
                        
                        result = rag_retriever.ingest_documents("docs", recreate_collection=True)
                        st.session_state.documents_ingested = True
                        st.success(f"✅ Ingested {result.get('documents_ingested', 0)} chunks!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            status = "✅" if st.session_state.documents_ingested else "❌"
            st.markdown(f"**Status:** {status}")
        
        # System Status
        st.markdown("### 📊 Status")
        
        status_items = [
            ("OpenAI", bool(os.getenv("OPENAI_API_KEY"))),
            ("Weather API", bool(os.getenv("OPENWEATHERMAP_API_KEY"))),
            ("LangSmith", bool(os.getenv("LANGCHAIN_API_KEY"))),
            ("Docs Loaded", st.session_state.documents_ingested),
        ]
        
        for name, is_active in status_items:
            icon = "🟢" if is_active else "🔴"
            st.markdown(f"{icon} {name}")
        
        # Clear chat
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def render_chat_message(role: str, content: str, query_type: str | None = None, sources: list | None = None):
    """Render a chat message with styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        badge_class = "weather-badge" if query_type == "weather" else "document-badge"
        badge_text = "🌤️ Weather" if query_type == "weather" else "📄 Document"
        
        sources_html = ""
        if sources:
            sources_text = ", ".join([Path(s).name for s in sources])
            sources_html = f'<div class="source-citation">Sources: {sources_text}</div>'
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <span class="query-type-badge {badge_class}">{badge_text}</span><br>
            {content}
            {sources_html}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌤️ Weather RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask about weather conditions or query documents about weather forecasting</p>', unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    if not st.session_state.config:
        st.error("""
        ## ⚠️ Configuration Required
        
        The application requires API keys to be set via environment variables.
        
        **Required:**
        - `OPENAI_API_KEY` - For LLM and embeddings
        - `OPENWEATHERMAP_API_KEY` - For weather data
        
        **Optional:**
        - `LANGCHAIN_API_KEY` - For LangSmith evaluation
        
        **Setup Options:**
        
        1. **Create a `.env` file** in the project root:
        ```env
        OPENAI_API_KEY=your_key_here
        OPENWEATHERMAP_API_KEY=your_key_here
        LANGCHAIN_API_KEY=your_key_here
        ```
        
        2. **Or set environment variables** before running:
        ```bash
        export OPENAI_API_KEY=your_key_here
        export OPENWEATHERMAP_API_KEY=your_key_here
        streamlit run streamlit_app.py
        ```
        """)
        
        # Show example queries
        st.markdown("### 💡 Example Queries (once configured)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Weather Queries:**
            - What's the weather in New York?
            - Will it rain tomorrow in London?
            - Temperature in Tokyo
            """)
        
        with col2:
            st.markdown("""
            **Document Queries:**
            - What is climate normal?
            - How do weather forecasts work?
            - Explain weather prediction models
            """)
        return
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            render_chat_message(
                role=message["role"],
                content=message["content"],
                query_type=message.get("query_type"),
                sources=message.get("sources"),
            )
    
    # Input area
    st.markdown("---")
    
    # Use form to handle input properly and prevent infinite loops
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask about weather or documents...",
                label_visibility="collapsed",
            )
        
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)
    
    # Process input only if form was submitted and input is not empty
    # Also check that we haven't already processed this exact input
    if submitted and user_input and user_input.strip():
        # Check if this input was already processed to prevent duplicate processing
        if user_input != st.session_state.get("last_processed_input"):
            st.session_state.last_processed_input = user_input
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Get the cached agent graph (not stored in session_state to avoid key conflicts)
                    agent_graph = get_agent_graph(st.session_state.config)
                    
                    result = run_agent_sync(agent_graph, user_input)
                    
                    # Extract all values before deleting result
                    agent_response = result.get("response", "Sorry, I couldn't generate a response.")
                    query_type = result.get("query_type", "document")
                    sources = result.get("sources", [])
                    context = result.get("context", "")
                    
                    # Clear the result dict reference to avoid any state tracking issues
                    del result
                    
                    # Evaluate response if LangSmith is configured
                    if st.session_state.evaluator and st.session_state.evaluator.is_enabled:
                        try:
                            llm = ChatOpenAI(
                                api_key=st.session_state.config.openai.api_key,
                                model=st.session_state.config.openai.model,
                            )
                            scores = st.session_state.evaluator.evaluate_response(
                                llm=llm,
                                query=user_input,
                                response=agent_response,
                                context=context,
                                query_type=query_type,
                            )
                            # Store scores in message for display
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": agent_response,
                                "query_type": query_type,
                                "sources": sources,
                                "scores": scores,
                            })
                        except Exception:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": agent_response,
                                "query_type": query_type,
                                "sources": sources,
                            })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": agent_response,
                            "query_type": query_type,
                            "sources": sources,
                        })
                    
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "query_type": "error",
                    })
            
            # Rerun to display the new messages
            st.rerun()
    
    # Show evaluation scores if available
    if st.session_state.messages and st.session_state.messages[-1].get("scores"):
        scores = st.session_state.messages[-1]["scores"]
        with st.expander("📊 Response Evaluation (LangSmith)"):
            cols = st.columns(len(scores))
            for i, (metric, score) in enumerate(scores.items()):
                with cols[i]:
                    st.metric(metric.title(), f"{score:.2f}")


if __name__ == "__main__":
    main()

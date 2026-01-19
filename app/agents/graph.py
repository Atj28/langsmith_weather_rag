"""LangGraph pipeline implementation."""

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from app.services.weather import WeatherService
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.rag.retriever import RAGRetriever
from app.rag.document_loader import DocumentLoader
from app.agents.nodes import AgentNodes, create_router_condition
from app.config import AppConfig


class AgentState(TypedDict, total=False):
    """State schema for the LangGraph agent."""
    query: str
    query_type: Literal["weather", "document", "greeting"]
    context: str
    response: str
    error: str | None
    source: str
    sources: list[str]
    num_results: int
    messages: list[dict]


def create_agent_graph(config: AppConfig) -> StateGraph:
    """Create the LangGraph agent pipeline.
    
    Args:
        config: Application configuration.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Initialize services
    llm = ChatOpenAI(
        api_key=config.openai.api_key,
        model=config.openai.model,
        temperature=0,
    )
    
    weather_service = WeatherService(
        api_key=config.weather.api_key,
        base_url=config.weather.base_url,
    )
    
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
    
    # Initialize nodes
    nodes = AgentNodes(
        llm=llm,
        weather_service=weather_service,
        rag_retriever=rag_retriever,
    )
    
    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("router", nodes.router_node)
    graph.add_node("weather", nodes.weather_node)
    graph.add_node("rag", nodes.rag_node)
    graph.add_node("greeting", nodes.greeting_node)
    graph.add_node("generate_response", nodes.response_node)  # Renamed to avoid conflict with state key
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        create_router_condition,
        {
            "weather": "weather",
            "document": "rag",
            "greeting": "greeting",
        }
    )
    
    # Add edges to generate_response node (except greeting which goes directly to END)
    graph.add_edge("weather", "generate_response")
    graph.add_edge("rag", "generate_response")
    graph.add_edge("greeting", END)  # Greetings skip the response node
    
    # Add edge to end
    graph.add_edge("generate_response", END)
    
    return graph.compile()


def create_agent_with_services(
    llm: ChatOpenAI,
    weather_service: WeatherService,
    rag_retriever: RAGRetriever,
) -> StateGraph:
    """Create agent graph with pre-initialized services.
    
    Useful for testing and when services are already initialized.
    
    Args:
        llm: Language model instance.
        weather_service: Weather service instance.
        rag_retriever: RAG retriever instance.
        
    Returns:
        Compiled StateGraph.
    """
    nodes = AgentNodes(
        llm=llm,
        weather_service=weather_service,
        rag_retriever=rag_retriever,
    )
    
    graph = StateGraph(AgentState)
    
    graph.add_node("router", nodes.router_node)
    graph.add_node("weather", nodes.weather_node)
    graph.add_node("rag", nodes.rag_node)
    graph.add_node("greeting", nodes.greeting_node)
    graph.add_node("generate_response", nodes.response_node)  # Renamed to avoid conflict
    
    graph.set_entry_point("router")
    
    graph.add_conditional_edges(
        "router",
        create_router_condition,
        {
            "weather": "weather",
            "document": "rag",
            "greeting": "greeting",
        }
    )
    
    graph.add_edge("weather", "generate_response")
    graph.add_edge("rag", "generate_response")
    graph.add_edge("greeting", END)
    graph.add_edge("generate_response", END)
    
    return graph.compile()


async def run_agent(graph: StateGraph, query: str) -> dict:
    """Run the agent with a query.
    
    Args:
        graph: Compiled LangGraph.
        query: User query.
        
    Returns:
        Final state after execution.
    """
    initial_state = AgentState(
        query=query,
        messages=[],
    )
    
    result = await graph.ainvoke(initial_state)
    return result


def run_agent_sync(graph: StateGraph, query: str) -> dict:
    """Run the agent synchronously.
    
    Args:
        graph: Compiled LangGraph.
        query: User query.
        
    Returns:
        Final state after execution.
    """
    initial_state = AgentState(
        query=query,
        messages=[],
    )
    
    result = graph.invoke(initial_state)
    return result

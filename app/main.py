"""Main entry point for the Weather RAG application."""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import load_config, AppConfig
from app.agents.graph import create_agent_graph, run_agent_sync
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import RAGRetriever
from app.evaluation.langsmith import setup_langsmith_tracing


def initialize_services(config: AppConfig) -> dict:
    """Initialize all application services.
    
    Args:
        config: Application configuration.
        
    Returns:
        Dictionary of initialized services.
    """
    # Initialize embedding service
    embedding_service = EmbeddingService(
        api_key=config.openai.api_key,
        model=config.openai.embedding_model,
    )
    
    # Initialize vector store
    vector_store = VectorStoreService(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name,
        embedding_dimension=embedding_service.embedding_dimension,
    )
    
    # Initialize document loader
    document_loader = DocumentLoader()
    
    # Initialize RAG retriever
    rag_retriever = RAGRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        document_loader=document_loader,
    )
    
    return {
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "document_loader": document_loader,
        "rag_retriever": rag_retriever,
    }


def ingest_documents(config: AppConfig, docs_path: str | None = None) -> dict:
    """Ingest documents into the vector store.
    
    Args:
        config: Application configuration.
        docs_path: Path to documents directory.
        
    Returns:
        Ingestion statistics.
    """
    docs_path = docs_path or config.docs_path
    
    services = initialize_services(config)
    rag_retriever = services["rag_retriever"]
    
    # Ingest documents
    result = rag_retriever.ingest_documents(docs_path, recreate_collection=True)
    
    return result


def run_query(config: AppConfig, query: str) -> dict:
    """Run a query through the agent pipeline.
    
    Args:
        config: Application configuration.
        query: User query.
        
    Returns:
        Agent response.
    """
    # Setup LangSmith tracing
    setup_langsmith_tracing(
        api_key=config.langsmith.api_key,
        project_name=config.langsmith.project_name,
    )
    
    # Create and run agent
    graph = create_agent_graph(config)
    result = run_agent_sync(graph, query)
    
    return result


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Weather RAG Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents")
    parser.add_argument("--docs", type=str, default="docs", help="Documents directory")
    parser.add_argument("--query", type=str, help="Query to run")
    
    args = parser.parse_args()
    
    try:
        config = load_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease create a .env file with the required environment variables.")
        print("See .env.example for the required variables.")
        sys.exit(1)
    
    if args.ingest:
        print(f"Ingesting documents from {args.docs}...")
        result = ingest_documents(config, args.docs)
        print(f"Ingestion result: {result}")
    
    elif args.query:
        print(f"Running query: {args.query}")
        result = run_query(config, args.query)
        print(f"\nQuery Type: {result.get('query_type')}")
        print(f"Source: {result.get('source')}")
        print(f"\nResponse:\n{result.get('response')}")
    
    else:
        # Interactive mode
        print("Weather RAG Pipeline - Interactive Mode")
        print("=" * 50)
        print("Type 'quit' to exit, 'ingest' to load documents")
        print()
        
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            if query.lower() == "quit":
                print("Goodbye!")
                break
            
            if query.lower() == "ingest":
                print(f"Ingesting documents from {args.docs}...")
                result = ingest_documents(config, args.docs)
                print(f"Result: {result}")
                continue
            
            result = run_query(config, query)
            print(f"\n[{result.get('query_type', 'unknown').upper()}] {result.get('response')}\n")


if __name__ == "__main__":
    main()

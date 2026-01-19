"""RAG retriever module for document retrieval and context building."""

from dataclasses import dataclass
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.rag.document_loader import DocumentLoader


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    documents: list[dict]
    context: str
    num_results: int
    
    def get_sources(self) -> list[str]:
        """Get unique source files from results."""
        sources = set()
        for doc in self.documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            sources.add(source)
        return list(sources)


class RAGRetrieverError(Exception):
    """Custom exception for RAG retriever errors."""
    pass


class RAGRetriever:
    """Retriever for RAG-based document retrieval."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        document_loader: DocumentLoader | None = None,
    ):
        """Initialize the RAG retriever.
        
        Args:
            embedding_service: Service for generating embeddings.
            vector_store: Service for vector storage.
            document_loader: Optional document loader for ingestion.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.document_loader = document_loader or DocumentLoader()
    
    def ingest_documents(self, source_path: str, recreate_collection: bool = False) -> dict:
        """Ingest documents from a file or directory into the vector store.
        
        Args:
            source_path: Path to PDF file or directory.
            recreate_collection: If True, recreate the collection.
            
        Returns:
            Dictionary with ingestion statistics.
        """
        # Create collection if needed
        self.vector_store.create_collection(recreate=recreate_collection)
        
        # Load and process documents
        processed_docs = self.document_loader.process_documents(source_path)
        
        if not processed_docs:
            return {
                "status": "no_documents",
                "documents_ingested": 0,
            }
        
        # Generate embeddings
        texts = [doc.content for doc in processed_docs]
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Prepare metadata
        metadatas = []
        for doc in processed_docs:
            metadatas.append({
                "source": doc.source,
                "page": doc.page,
                **doc.metadata,
            })
        
        # Store in vector database
        count = self.vector_store.upsert_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        return {
            "status": "success",
            "documents_ingested": count,
            "source_path": source_path,
        }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = 0.7,
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score.
            
        Returns:
            RetrievalResult with retrieved documents and context.
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        )
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("metadata", {}).get("source", "unknown")
            page = result.get("metadata", {}).get("page", "?")
            text = result.get("text", "")
            context_parts.append(
                f"[Document {i}] (Source: {source}, Page: {page})\n{text}"
            )
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        
        return RetrievalResult(
            query=query,
            documents=results,
            context=context,
            num_results=len(results),
        )
    
    def retrieve_with_sources(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> RetrievalResult:
        """Retrieve documents with optional source filtering.
        
        Args:
            query: Search query.
            top_k: Maximum number of results.
            source_filter: Optional source file to filter by.
            
        Returns:
            RetrievalResult with retrieved documents.
        """
        query_embedding = self.embedding_service.embed_text(query)
        
        filter_conditions = None
        if source_filter:
            filter_conditions = {"source": source_filter}
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            filter_conditions=filter_conditions,
        )
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("metadata", {}).get("source", "unknown")
            page = result.get("metadata", {}).get("page", "?")
            text = result.get("text", "")
            score = result.get("score", 0)
            context_parts.append(
                f"[Document {i}] (Source: {source}, Page: {page}, Relevance: {score:.2f})\n{text}"
            )
        
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        
        return RetrievalResult(
            query=query,
            documents=results,
            context=context,
            num_results=len(results),
        )
    
    def get_stats(self) -> dict:
        """Get retriever statistics.
        
        Returns:
            Dictionary with retriever stats.
        """
        try:
            collection_info = self.vector_store.get_collection_info()
            return {
                "collection_name": collection_info.get("name"),
                "document_count": collection_info.get("points_count", 0),
                "status": collection_info.get("status"),
            }
        except Exception:
            return {
                "collection_name": self.vector_store.collection_name,
                "document_count": 0,
                "status": "not_initialized",
            }

"""Embedding generation service using OpenAI."""

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class EmbeddingServiceError(Exception):
    """Custom exception for embedding service errors."""
    pass


class EmbeddingService:
    """Service for generating text embeddings using OpenAI."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        """Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key.
            model: Embedding model to use.
        """
        self.model = model
        self._embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
        )
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        # text-embedding-3-small: 1536 dimensions
        # text-embedding-3-large: 3072 dimensions
        # text-embedding-ada-002: 1536 dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the embedding.
            
        Raises:
            EmbeddingServiceError: If embedding generation fails.
        """
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to generate embedding: {str(e)}")
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embeddings.
            
        Raises:
            EmbeddingServiceError: If embedding generation fails.
        """
        if not texts:
            return []
        
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to generate embeddings: {str(e)}")
    
    def embed_documents(self, documents: list[Document]) -> list[tuple[Document, list[float]]]:
        """Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects.
            
        Returns:
            List of tuples containing (document, embedding).
            
        Raises:
            EmbeddingServiceError: If embedding generation fails.
        """
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_texts(texts)
        
        return list(zip(documents, embeddings))
    
    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """Get the underlying LangChain embeddings object.
        
        Returns:
            OpenAIEmbeddings instance for use with LangChain components.
        """
        return self._embeddings

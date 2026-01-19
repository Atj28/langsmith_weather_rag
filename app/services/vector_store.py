"""Qdrant vector database service."""

import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct


class VectorStoreError(Exception):
    """Custom exception for vector store errors."""
    pass


class VectorStoreService:
    """Service for managing vector storage with Qdrant."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "weather_documents",
        embedding_dimension: int = 1536,
    ):
        """Initialize the vector store service.
        
        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            collection_name: Name of the collection to use.
            embedding_dimension: Dimension of embeddings.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        
        try:
            self._client = QdrantClient(host=host, port=port)
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Qdrant: {str(e)}")
    
    def create_collection(self, recreate: bool = False) -> bool:
        """Create the vector collection.
        
        Args:
            recreate: If True, delete and recreate the collection.
            
        Returns:
            True if collection was created, False if it already existed.
        """
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                if recreate:
                    self._client.delete_collection(self.collection_name)
                else:
                    return False
            
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            return True
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection: {str(e)}")
    
    def collection_exists(self) -> bool:
        """Check if the collection exists.
        
        Returns:
            True if collection exists, False otherwise.
        """
        try:
            collections = self._client.get_collections().collections
            return self.collection_name in [c.name for c in collections]
        except Exception as e:
            raise VectorStoreError(f"Failed to check collection: {str(e)}")
    
    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection information.
        """
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            raise VectorStoreError(f"Failed to get collection info: {str(e)}")
    
    def upsert_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """Insert or update documents in the collection.
        
        Args:
            texts: List of document texts.
            embeddings: List of embeddings corresponding to texts.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
            
        Returns:
            Number of documents upserted.
            
        Raises:
            VectorStoreError: If upsert fails.
        """
        if len(texts) != len(embeddings):
            raise VectorStoreError("Number of texts and embeddings must match")
        
        if not texts:
            return 0
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Use empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Create points
        points = []
        for i, (text, embedding, metadata, doc_id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            payload = {
                "text": text,
                **metadata,
            }
            points.append(PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload,
            ))
        
        try:
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
            
            return len(points)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert documents: {str(e)}")
    
    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score (0-1 for cosine).
            filter_conditions: Optional filter conditions.
            
        Returns:
            List of search results with text, metadata, and score.
        """
        try:
            # Build filter if conditions provided
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
                query_filter = models.Filter(must=must_conditions)
            
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            
            return [
                {
                    "id": str(result.id),
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "score": result.score,
                }
                for result in results
            ]
            
        except Exception as e:
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID.
        
        Args:
            ids: List of document IDs to delete.
            
        Returns:
            Number of documents deleted.
        """
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids,
                ),
            )
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {str(e)}")
    
    def clear_collection(self) -> bool:
        """Delete all documents in the collection.
        
        Returns:
            True if successful.
        """
        try:
            self._client.delete_collection(self.collection_name)
            self.create_collection()
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to clear collection: {str(e)}")
    
    def count_documents(self) -> int:
        """Get the number of documents in the collection.
        
        Returns:
            Number of documents.
        """
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            raise VectorStoreError(f"Failed to count documents: {str(e)}")

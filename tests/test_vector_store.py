"""Tests for the vector store service."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid

from app.services.vector_store import (
    VectorStoreService,
    VectorStoreError,
)


class TestVectorStoreService:
    """Tests for VectorStoreService."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        with patch('app.services.vector_store.QdrantClient') as mock:
            client_instance = Mock()
            mock.return_value = client_instance
            
            # Setup default behaviors
            collection_mock = Mock()
            collection_mock.name = "weather_documents"
            client_instance.get_collections.return_value = Mock(collections=[collection_mock])
            
            yield client_instance
    
    @pytest.fixture
    def vector_store(self, mock_qdrant_client):
        """Create a vector store instance with mocked client."""
        return VectorStoreService(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            embedding_dimension=1536,
        )
    
    def test_init_success(self, mock_qdrant_client):
        """Test successful initialization."""
        store = VectorStoreService(
            host="localhost",
            port=6333,
            collection_name="test",
            embedding_dimension=1536,
        )
        
        assert store.host == "localhost"
        assert store.port == 6333
        assert store.collection_name == "test"
        assert store.embedding_dimension == 1536
    
    def test_init_connection_error(self):
        """Test initialization with connection error."""
        with patch('app.services.vector_store.QdrantClient') as mock:
            mock.side_effect = Exception("Connection refused")
            
            with pytest.raises(VectorStoreError) as exc_info:
                VectorStoreService()
            
            assert "Failed to connect" in str(exc_info.value)
    
    def test_create_collection_new(self, vector_store, mock_qdrant_client):
        """Test creating a new collection."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        result = vector_store.create_collection()
        
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
    
    def test_create_collection_exists(self, vector_store, mock_qdrant_client):
        """Test creating collection when it already exists."""
        collection_mock = Mock()
        collection_mock.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = Mock(collections=[collection_mock])
        
        result = vector_store.create_collection(recreate=False)
        
        assert result is False
        mock_qdrant_client.create_collection.assert_not_called()
    
    def test_create_collection_recreate(self, vector_store, mock_qdrant_client):
        """Test recreating an existing collection."""
        collection_mock = Mock()
        collection_mock.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = Mock(collections=[collection_mock])
        
        result = vector_store.create_collection(recreate=True)
        
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()
    
    def test_collection_exists_true(self, vector_store, mock_qdrant_client):
        """Test checking if collection exists (true case)."""
        collection_mock = Mock()
        collection_mock.name = "test_collection"
        mock_qdrant_client.get_collections.return_value = Mock(collections=[collection_mock])
        
        result = vector_store.collection_exists()
        
        assert result is True
    
    def test_collection_exists_false(self, vector_store, mock_qdrant_client):
        """Test checking if collection exists (false case)."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        result = vector_store.collection_exists()
        
        assert result is False
    
    def test_get_collection_info(self, vector_store, mock_qdrant_client):
        """Test getting collection information."""
        info_mock = Mock()
        info_mock.vectors_count = 100
        info_mock.points_count = 100
        info_mock.status = Mock(value="green")
        mock_qdrant_client.get_collection.return_value = info_mock
        
        result = vector_store.get_collection_info()
        
        assert result["name"] == "test_collection"
        assert result["vectors_count"] == 100
        assert result["points_count"] == 100
        assert result["status"] == "green"
    
    def test_upsert_documents_success(self, vector_store, mock_qdrant_client):
        """Test upserting documents."""
        texts = ["doc1", "doc2"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadatas = [{"source": "test1.pdf"}, {"source": "test2.pdf"}]
        
        result = vector_store.upsert_documents(texts, embeddings, metadatas)
        
        assert result == 2
        mock_qdrant_client.upsert.assert_called_once()
    
    def test_upsert_documents_with_ids(self, vector_store, mock_qdrant_client):
        """Test upserting documents with custom IDs."""
        texts = ["doc1"]
        embeddings = [[0.1] * 1536]
        ids = ["custom-id-1"]
        
        result = vector_store.upsert_documents(texts, embeddings, ids=ids)
        
        assert result == 1
    
    def test_upsert_documents_mismatched_lengths(self, vector_store):
        """Test upserting with mismatched text and embedding counts."""
        texts = ["doc1", "doc2"]
        embeddings = [[0.1] * 1536]  # Only one embedding
        
        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.upsert_documents(texts, embeddings)
        
        assert "must match" in str(exc_info.value)
    
    def test_upsert_documents_empty(self, vector_store):
        """Test upserting empty document list."""
        result = vector_store.upsert_documents([], [])
        
        assert result == 0
    
    def test_search_success(self, vector_store, mock_qdrant_client):
        """Test searching for similar documents."""
        search_result = Mock()
        search_result.id = "test-id"
        search_result.payload = {"text": "Test content", "source": "test.pdf"}
        search_result.score = 0.95
        mock_qdrant_client.search.return_value = [search_result]
        
        query_embedding = [0.1] * 1536
        results = vector_store.search(query_embedding, limit=5)
        
        assert len(results) == 1
        assert results[0]["id"] == "test-id"
        assert results[0]["text"] == "Test content"
        assert results[0]["score"] == 0.95
    
    def test_search_with_filter(self, vector_store, mock_qdrant_client):
        """Test searching with filter conditions."""
        mock_qdrant_client.search.return_value = []
        
        query_embedding = [0.1] * 1536
        filter_conditions = {"source": "specific.pdf"}
        
        vector_store.search(query_embedding, filter_conditions=filter_conditions)
        
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert call_kwargs.get("query_filter") is not None
    
    def test_search_with_threshold(self, vector_store, mock_qdrant_client):
        """Test searching with score threshold."""
        mock_qdrant_client.search.return_value = []
        
        query_embedding = [0.1] * 1536
        vector_store.search(query_embedding, score_threshold=0.8)
        
        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert call_kwargs.get("score_threshold") == 0.8
    
    def test_delete_documents(self, vector_store, mock_qdrant_client):
        """Test deleting documents by ID."""
        ids = ["id1", "id2"]
        
        result = vector_store.delete_documents(ids)
        
        assert result == 2
        mock_qdrant_client.delete.assert_called_once()
    
    def test_clear_collection(self, vector_store, mock_qdrant_client):
        """Test clearing all documents from collection."""
        mock_qdrant_client.get_collections.return_value = Mock(collections=[])
        
        result = vector_store.clear_collection()
        
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()
    
    def test_count_documents(self, vector_store, mock_qdrant_client):
        """Test counting documents in collection."""
        info_mock = Mock()
        info_mock.points_count = 50
        mock_qdrant_client.get_collection.return_value = info_mock
        
        result = vector_store.count_documents()
        
        assert result == 50
    
    def test_count_documents_empty(self, vector_store, mock_qdrant_client):
        """Test counting documents in empty collection."""
        info_mock = Mock()
        info_mock.points_count = None
        mock_qdrant_client.get_collection.return_value = info_mock
        
        result = vector_store.count_documents()
        
        assert result == 0


class TestVectorStoreServiceIntegration:
    """Integration tests for vector store (require Qdrant)."""
    
    @pytest.mark.skip(reason="Requires running Qdrant instance")
    def test_real_operations(self):
        """Test with real Qdrant instance."""
        store = VectorStoreService(
            host="localhost",
            port=6333,
            collection_name="test_integration",
            embedding_dimension=1536,
        )
        
        # Create collection
        store.create_collection(recreate=True)
        
        # Insert documents
        texts = ["Test document 1", "Test document 2"]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        store.upsert_documents(texts, embeddings)
        
        # Search
        results = store.search([0.1] * 1536, limit=2)
        
        assert len(results) > 0
        
        # Cleanup
        store.clear_collection()

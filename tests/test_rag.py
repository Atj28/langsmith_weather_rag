"""Tests for the RAG components."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.rag.document_loader import (
    DocumentLoader,
    ProcessedDocument,
    DocumentLoaderError,
)
from app.rag.retriever import (
    RAGRetriever,
    RetrievalResult,
    RAGRetrieverError,
)


class TestDocumentLoader:
    """Tests for DocumentLoader."""
    
    @pytest.fixture
    def document_loader(self):
        """Create a document loader instance."""
        return DocumentLoader(chunk_size=500, chunk_overlap=50)
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        loader = DocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100
    
    def test_load_pdf_file_not_found(self, document_loader):
        """Test loading a non-existent PDF."""
        with pytest.raises(DocumentLoaderError) as exc_info:
            document_loader.load_pdf("/nonexistent/path.pdf")
        
        assert "File not found" in str(exc_info.value)
    
    def test_load_pdf_wrong_extension(self, document_loader):
        """Test loading a non-PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentLoaderError) as exc_info:
                document_loader.load_pdf(temp_path)
            
            assert "not a PDF" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_load_directory_not_found(self, document_loader):
        """Test loading from non-existent directory."""
        with pytest.raises(DocumentLoaderError) as exc_info:
            document_loader.load_directory("/nonexistent/directory")
        
        assert "Directory not found" in str(exc_info.value)
    
    def test_load_directory_not_a_directory(self, document_loader):
        """Test loading from a file path instead of directory."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentLoaderError) as exc_info:
                document_loader.load_directory(temp_path)
            
            assert "not a directory" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
    def test_split_documents_empty_list(self, document_loader):
        """Test splitting an empty document list."""
        result = document_loader.split_documents([])
        assert result == []
    
    def test_split_documents_adds_chunk_index(self, document_loader):
        """Test that split_documents adds chunk_index to metadata."""
        from langchain.schema import Document
        
        docs = [
            Document(page_content="A" * 1000, metadata={"source": "test.pdf"}),
        ]
        
        chunks = document_loader.split_documents(docs)
        
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata
    
    def test_process_documents_with_pdf_directory(self, document_loader):
        """Test processing documents from the docs directory."""
        docs_path = Path("docs")
        
        if not docs_path.exists():
            pytest.skip("docs directory not available")
        
        with patch.object(document_loader, 'load_directory') as mock_load:
            from langchain.schema import Document
            mock_load.return_value = [
                Document(page_content="Test content", metadata={"source": "test.pdf", "page": 0})
            ]
            
            with patch.object(document_loader, 'split_documents') as mock_split:
                mock_split.return_value = [
                    Document(page_content="Test content", metadata={"source": "test.pdf", "page": 0, "chunk_index": 0})
                ]
                
                result = document_loader.process_documents(str(docs_path))
                
                assert len(result) > 0
                assert isinstance(result[0], ProcessedDocument)


class TestProcessedDocument:
    """Tests for ProcessedDocument model."""
    
    def test_processed_document_creation(self):
        """Test creating a ProcessedDocument."""
        doc = ProcessedDocument(
            content="Test content",
            metadata={"key": "value"},
            source="test.pdf",
            page=1,
        )
        
        assert doc.content == "Test content"
        assert doc.source == "test.pdf"
        assert doc.page == 1
        assert doc.metadata["key"] == "value"


class TestRetrievalResult:
    """Tests for RetrievalResult model."""
    
    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            query="test query",
            documents=[
                {"text": "content", "metadata": {"source": "doc1.pdf"}},
                {"text": "more content", "metadata": {"source": "doc2.pdf"}},
            ],
            context="Combined context",
            num_results=2,
        )
        
        assert result.query == "test query"
        assert result.num_results == 2
        assert len(result.documents) == 2
    
    def test_get_sources(self):
        """Test getting unique sources from results."""
        result = RetrievalResult(
            query="test",
            documents=[
                {"text": "content", "metadata": {"source": "doc1.pdf"}},
                {"text": "more content", "metadata": {"source": "doc1.pdf"}},
                {"text": "other content", "metadata": {"source": "doc2.pdf"}},
            ],
            context="context",
            num_results=3,
        )
        
        sources = result.get_sources()
        
        assert len(sources) == 2
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources


class TestRAGRetriever:
    """Tests for RAGRetriever."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        mock = Mock()
        mock.embed_text.return_value = [0.1] * 1536
        mock.embed_texts.return_value = [[0.1] * 1536]
        mock.embedding_dimension = 1536
        return mock
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock = Mock()
        mock.create_collection.return_value = True
        mock.upsert_documents.return_value = 1
        mock.search.return_value = [
            {
                "id": "1",
                "text": "Test document content",
                "metadata": {"source": "test.pdf", "page": 0},
                "score": 0.9,
            }
        ]
        mock.get_collection_info.return_value = {
            "name": "test_collection",
            "points_count": 10,
            "status": "green",
        }
        return mock
    
    @pytest.fixture
    def mock_document_loader(self):
        """Create a mock document loader."""
        mock = Mock()
        mock.process_documents.return_value = [
            ProcessedDocument(
                content="Test content",
                metadata={"source": "test.pdf", "page": 0},
                source="test.pdf",
                page=0,
            )
        ]
        return mock
    
    @pytest.fixture
    def rag_retriever(self, mock_embedding_service, mock_vector_store, mock_document_loader):
        """Create a RAG retriever with mocked dependencies."""
        return RAGRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            document_loader=mock_document_loader,
        )
    
    def test_ingest_documents(self, rag_retriever, mock_vector_store, mock_embedding_service):
        """Test document ingestion."""
        result = rag_retriever.ingest_documents("/test/path")
        
        assert result["status"] == "success"
        assert result["documents_ingested"] == 1
        mock_vector_store.create_collection.assert_called_once()
        mock_embedding_service.embed_texts.assert_called_once()
    
    def test_ingest_documents_empty(self, rag_retriever, mock_document_loader):
        """Test ingestion with no documents."""
        mock_document_loader.process_documents.return_value = []
        
        result = rag_retriever.ingest_documents("/test/path")
        
        assert result["status"] == "no_documents"
        assert result["documents_ingested"] == 0
    
    def test_retrieve(self, rag_retriever, mock_embedding_service, mock_vector_store):
        """Test document retrieval."""
        result = rag_retriever.retrieve("test query", top_k=5)
        
        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert result.num_results == 1
        mock_embedding_service.embed_text.assert_called_with("test query")
        mock_vector_store.search.assert_called_once()
    
    def test_retrieve_no_results(self, rag_retriever, mock_vector_store):
        """Test retrieval with no matching documents."""
        mock_vector_store.search.return_value = []
        
        result = rag_retriever.retrieve("obscure query")
        
        assert result.num_results == 0
        assert "No relevant documents" in result.context
    
    def test_retrieve_with_sources(self, rag_retriever):
        """Test retrieval with source filtering."""
        result = rag_retriever.retrieve_with_sources(
            "test query",
            top_k=3,
            source_filter="specific.pdf"
        )
        
        assert isinstance(result, RetrievalResult)
    
    def test_get_stats(self, rag_retriever, mock_vector_store):
        """Test getting retriever statistics."""
        stats = rag_retriever.get_stats()
        
        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 10
        assert stats["status"] == "green"
    
    def test_get_stats_not_initialized(self, rag_retriever, mock_vector_store):
        """Test getting stats when collection doesn't exist."""
        mock_vector_store.get_collection_info.side_effect = Exception("Not found")
        
        stats = rag_retriever.get_stats()
        
        assert stats["status"] == "not_initialized"
        assert stats["document_count"] == 0

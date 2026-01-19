"""PDF document loading and processing module."""

import os
from pathlib import Path
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


@dataclass
class ProcessedDocument:
    """Represents a processed document chunk."""
    content: str
    metadata: dict
    source: str
    page: int


class DocumentLoaderError(Exception):
    """Custom exception for document loading errors."""
    pass


class DocumentLoader:
    """Service for loading and processing PDF documents."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the document loader.
        
        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def load_pdf(self, file_path: str) -> list[Document]:
        """Load a single PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of Document objects from the PDF.
            
        Raises:
            DocumentLoaderError: If the file cannot be loaded.
        """
        if not os.path.exists(file_path):
            raise DocumentLoaderError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise DocumentLoaderError(f"File is not a PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            raise DocumentLoaderError(f"Failed to load PDF {file_path}: {str(e)}")
    
    def load_directory(self, directory_path: str) -> list[Document]:
        """Load all PDF files from a directory.
        
        Args:
            directory_path: Path to the directory containing PDF files.
            
        Returns:
            List of Document objects from all PDFs.
            
        Raises:
            DocumentLoaderError: If the directory cannot be accessed.
        """
        if not os.path.exists(directory_path):
            raise DocumentLoaderError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise DocumentLoaderError(f"Path is not a directory: {directory_path}")
        
        all_documents = []
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            raise DocumentLoaderError(f"No PDF files found in {directory_path}")
        
        for pdf_path in pdf_files:
            try:
                documents = self.load_pdf(str(pdf_path))
                all_documents.extend(documents)
            except DocumentLoaderError as e:
                # Log warning but continue with other files
                print(f"Warning: Skipping {pdf_path}: {str(e)}")
        
        return all_documents
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split.
            
        Returns:
            List of chunked Document objects.
        """
        if not documents:
            return []
        
        chunks = self._text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def process_documents(self, source_path: str) -> list[ProcessedDocument]:
        """Load and process documents from a file or directory.
        
        Args:
            source_path: Path to a PDF file or directory of PDFs.
            
        Returns:
            List of ProcessedDocument objects.
        """
        # Load documents
        if os.path.isdir(source_path):
            documents = self.load_directory(source_path)
        else:
            documents = self.load_pdf(source_path)
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Convert to ProcessedDocument objects
        processed = []
        for chunk in chunks:
            processed.append(ProcessedDocument(
                content=chunk.page_content,
                metadata=chunk.metadata,
                source=chunk.metadata.get("source", "unknown"),
                page=chunk.metadata.get("page", 0),
            ))
        
        return processed
    
    def get_document_stats(self, source_path: str) -> dict:
        """Get statistics about documents in a path.
        
        Args:
            source_path: Path to a PDF file or directory.
            
        Returns:
            Dictionary with document statistics.
        """
        if os.path.isdir(source_path):
            pdf_files = list(Path(source_path).glob("*.pdf"))
            documents = self.load_directory(source_path)
        else:
            pdf_files = [Path(source_path)]
            documents = self.load_pdf(source_path)
        
        chunks = self.split_documents(documents)
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        chunk_chars = sum(len(chunk.page_content) for chunk in chunks)
        
        return {
            "num_files": len(pdf_files),
            "num_pages": len(documents),
            "num_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": chunk_chars // len(chunks) if chunks else 0,
            "files": [str(f) for f in pdf_files],
        }

"""Vector store and document ingestion service.

Handles document loading (PDFs, Markdown, text), splitting text into overlapping
semantic chunks, generating vector embeddings via Ollama, and persisting
them into a local ChromaDB collection.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings


class VectorStoreService:
    """Manages document chunking, embedding generation, and ChromaDB persistence."""

    def __init__(self) -> None:
        """Initializes Ollama embeddings and the persistent Chroma store."""
        settings.ensure_directories()
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        self.vector_db = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def ingest_pdf(self, file_path: str) -> int:
        """Loads a PDF document, splits it into chunks, and persists it into ChromaDB.

        Args:
            file_path: Local filesystem path to the PDF file.

        Returns:
            The number of chunks indexed.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If PDF parsing or vector indexing fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        if chunks:
            self.vector_db.add_documents(chunks)
        return len(chunks)

    def ingest_text_file(self, file_path: str) -> int:
        """Loads a plain text or Markdown document, splits and persists it.

        Args:
            file_path: Local filesystem path to the text file.

        Returns:
            The number of chunks indexed.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        if chunks:
            self.vector_db.add_documents(chunks)
        return len(chunks)

    def ingest_raw_text(self, text: str, source: str = "manual_input") -> int:
        """Chunks and indexes raw in-memory string content.

        Args:
            text: Raw string content to index.
            source: Source identifier metadata.

        Returns:
            The number of chunks indexed.
        """
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.text_splitter.split_documents([doc])
        if chunks:
            self.vector_db.add_documents(chunks)
        return len(chunks)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Queries the vector database for relevant documentation excerpts.

        Args:
            query: Natural language query string.
            k: Maximum number of closest matches to retrieve.

        Returns:
            A list of matching `Document` instances.
        """
        return self.vector_db.similarity_search(query, k=k)


# Singleton instance accessor
_vector_service_instance: VectorStoreService | None = None


def get_vector_store_service() -> VectorStoreService:
    """Returns or creates the singleton VectorStoreService instance."""
    global _vector_service_instance
    if _vector_service_instance is None:
        _vector_service_instance = VectorStoreService()
    return _vector_service_instance

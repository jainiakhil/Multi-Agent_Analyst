"""Unit tests for document ingestion and text splitting logic."""

import os
import tempfile
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def test_recursive_character_text_splitter():
    """Validates that document text is correctly divided into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "The Multi-Agent Analyst system orchestrates multiple specialized AI agents. "
        "Each agent has specific capabilities such as static code analysis and document retrieval. "
        "The supervisor agent routes tasks dynamically based on user prompts."
    )
    docs = [Document(page_content=sample_text, metadata={"source": "test_doc.txt"})]
    chunks = splitter.split_documents(docs)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.page_content) <= 120
        assert chunk.metadata["source"] == "test_doc.txt"


def test_text_document_loading():
    """Validates loading and splitting from a local plain text file."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Line 1: System Spec\nLine 2: Architectural Details\nLine 3: Implementation Requirements.")
        temp_path = f.name

    try:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(temp_path, encoding="utf-8")
        docs = loader.load()
        assert len(docs) == 1
        assert "Architectural Details" in docs[0].page_content
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

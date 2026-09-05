"""Custom tool implementations for Python AST parsing and Vector Document Search.

Provides agents with capabilities to structurally inspect source code files
and retrieve contextual documentation from local ChromaDB embeddings.
"""

import ast
import os
from langchain_core.tools import tool
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings
from app.services.vector_store import get_vector_store_service

# Initialize local embeddings and Chroma vector store instance
embeddings = OllamaEmbeddings(
    model=settings.EMBEDDING_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
)

vector_db = Chroma(
    collection_name=settings.CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=settings.CHROMA_PERSIST_DIR,
)


@tool
def parse_python_ast(file_path: str) -> str:
    """Parses a local Python source file and returns class definitions, function signatures, and docstrings.

    Args:
        file_path: Relative or absolute path to the Python source code file.

    Returns:
        A formatted string detailing the discovered classes, methods, functions, and arguments.
    """
    clean_path = file_path.strip().strip("'\"")
    if not os.path.exists(clean_path):
        return f"Error: File '{clean_path}' does not exist on the filesystem."

    try:
        with open(clean_path, "r", encoding="utf-8", errors="replace") as f:
            code_content = f.read()

        tree = ast.parse(code_content, filename=clean_path)
        classes: list[str] = []
        functions: list[str] = []
        module_docstring = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                method_str = f" (methods: {', '.join(methods)})" if methods else ""
                classes.append(f"{node.name}{method_str}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only include top-level functions (not methods already grouped under classes)
                is_method = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and node in parent.body:
                        is_method = True
                        break
                if not is_method:
                    args = [a.arg for a in node.args.args]
                    functions.append(f"{node.name}({', '.join(args)})")

        return (
            f"AST Analysis for {clean_path}:\n"
            f"- Module Docstring: {module_docstring if module_docstring else 'None'}\n"
            f"- Classes Identified: {classes if classes else 'None'}\n"
            f"- Top-level Functions Identified: {functions if functions else 'None'}\n"
        )
    except SyntaxError as syn_err:
        return f"Syntax Error parsing {clean_path} at line {syn_err.lineno}: {syn_err.msg}"
    except Exception as exc:
        return f"Failed to generate AST for {clean_path}: {str(exc)}"


@tool
def query_documentation(query: str) -> str:
    """Performs vector similarity search over indexed PDF specs and technical documents.

    Args:
        query: Natural language query string describing the desired specification context.

    Returns:
        Concatenated relevant excerpts from indexed documents with source metadata.
    """
    try:
        service = get_vector_store_service()
        results = service.similarity_search(query, k=3)
        if not results:
            return "No relevant documentation matches found in the vector store."

        context_blocks = [
            f"--- Excerpt {idx + 1} (Source: {doc.metadata.get('source', 'Unknown')}) ---\n{doc.page_content}"
            for idx, doc in enumerate(results)
        ]
        return "\n\n".join(context_blocks)
    except Exception as exc:
        return f"Error querying vector documentation store: {str(exc)}"

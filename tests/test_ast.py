"""Unit tests for the Python AST parser tool."""

import os
import tempfile
import pytest
from app.graph.tools import parse_python_ast


def test_parse_python_ast_valid_file():
    """Verifies that classes, methods, and functions are properly extracted from a valid Python file."""
    sample_path = os.path.join("tests", "sample_code.py")
    result = parse_python_ast.invoke({"file_path": sample_path})

    assert "AST Analysis for" in result
    assert "DataPipeline" in result
    assert "compute_metrics(accuracy, loss)" in result
    assert "Module Docstring" in result


def test_parse_python_ast_nonexistent_file():
    """Verifies that an appropriate error message is returned when file does not exist."""
    fake_path = "non_existent_file_xyz_123.py"
    result = parse_python_ast.invoke({"file_path": fake_path})

    assert "Error:" in result
    assert "does not exist" in result


def test_parse_python_ast_syntax_error():
    """Verifies that a syntax error is gracefully caught and reported."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("def broken_func(:\n    pass")
        temp_path = f.name

    try:
        result = parse_python_ast.invoke({"file_path": temp_path})
        assert "Syntax Error" in result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

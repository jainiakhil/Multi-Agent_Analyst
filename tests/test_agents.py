"""Unit and mock tests for Supervisor and Worker Agent nodes."""

from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.state import AgentState, RouterOutput
from app.graph.agents import supervisor_node, code_analyst_node, doc_parser_node


def test_supervisor_node_routes_to_worker():
    """Validates that the supervisor node invokes router_chain and returns the next_node."""
    mock_decision = RouterOutput(
        next_node="code_analyst",
        reasoning="User provided a file path for code inspection.",
    )
    with patch("app.graph.agents.router_chain") as mock_router:
        mock_router.invoke.return_value = mock_decision
        state = AgentState(
            messages=[HumanMessage(content="Audit app/main.py")],
            next_node="",
            target_file="app/main.py",
            report=None,
        )
        result = supervisor_node(state)
        assert result == {"next_node": "code_analyst"}


def test_supervisor_node_fallback_on_error():
    """Validates that the supervisor node falls back to FINISH if the LLM invocation fails."""
    with patch("app.graph.agents.router_chain") as mock_router:
        mock_router.invoke.side_effect = RuntimeError("Ollama offline")
        state = AgentState(
            messages=[HumanMessage(content="Audit app/main.py")],
            next_node="",
            target_file="app/main.py",
            report=None,
        )
        result = supervisor_node(state)
        assert result == {"next_node": "FINISH"}


def test_code_analyst_node_with_tool_call():
    """Validates that code_analyst_node properly executes AST tool call when requested by model."""
    mock_call_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "parse_python_ast",
            "args": {"file_path": "tests/sample_code.py"},
            "id": "call_123",
        }],
    )
    mock_final_response = AIMessage(content="The code contains class DataPipeline and function compute_metrics.")

    with patch("app.graph.agents.code_agent_with_tools") as mock_code:
        mock_code.invoke.side_effect = [mock_call_response, mock_final_response]
        state = AgentState(
            messages=[HumanMessage(content="Check tests/sample_code.py")],
            next_node="code_analyst",
            target_file="tests/sample_code.py",
            report=None,
        )
        result = code_analyst_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "[Code Analyst]:" in result["messages"][0].content
        assert "DataPipeline" in result["messages"][0].content


def test_doc_parser_node_with_tool_call():
    """Validates that doc_parser_node properly executes documentation query tool call."""
    mock_call_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "query_documentation",
            "args": {"query": "API endpoints"},
            "id": "call_456",
        }],
    )
    mock_final_response = AIMessage(content="Based on the specifications, the API uses FastAPI.")

    with patch("app.graph.agents.doc_agent_with_tools") as mock_doc:
        mock_doc.invoke.side_effect = [mock_call_response, mock_final_response]
        state = AgentState(
            messages=[HumanMessage(content="What API framework is used?")],
            next_node="doc_parser",
            target_file=None,
            report=None,
        )
        result = doc_parser_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "[Doc Parser]:" in result["messages"][0].content
        assert "FastAPI" in result["messages"][0].content

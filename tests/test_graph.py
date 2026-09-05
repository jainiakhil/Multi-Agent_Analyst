"""Unit tests for LangGraph state definitions, router output, and workflow topology."""

import pytest
from langgraph.graph import END
from app.graph.state import RouterOutput, AnalysisReport, AgentState
from app.graph.workflow import route_next, build_workflow


def test_router_output_schema():
    """Validates the RouterOutput Pydantic model serialization and constraints."""
    valid_route = RouterOutput(
        next_node="code_analyst",
        reasoning="User asked for Python code review.",
    )
    assert valid_route.next_node == "code_analyst"
    assert "code review" in valid_route.reasoning

    with pytest.raises(ValueError):
        RouterOutput(next_node="invalid_route", reasoning="Test")


def test_analysis_report_schema():
    """Validates AnalysisReport default values and schema structure."""
    report = AnalysisReport(
        summary="Audit completed successfully.",
        critical_issues=["Unbounded loop in worker node"],
        recommendations=["Refactor to generator"],
    )
    assert report.summary == "Audit completed successfully."
    assert len(report.critical_issues) == 1
    assert len(report.recommendations) == 1


def test_route_next_logic():
    """Validates conditional routing logic based on supervisor next_node decision."""
    state_finish = AgentState(
        messages=[],
        next_node="FINISH",
        target_file=None,
        report=None,
    )
    assert route_next(state_finish) == END

    state_code = AgentState(
        messages=[],
        next_node="code_analyst",
        target_file="app/main.py",
        report=None,
    )
    assert route_next(state_code) == "code_analyst"

    state_doc = AgentState(
        messages=[],
        next_node="doc_parser",
        target_file=None,
        report=None,
    )
    assert route_next(state_doc) == "doc_parser"


def test_workflow_structure():
    """Verifies that the StateGraph has all required nodes registered."""
    wf = build_workflow()
    nodes = wf.nodes
    assert "supervisor" in nodes
    assert "code_analyst" in nodes
    assert "doc_parser" in nodes

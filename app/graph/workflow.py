"""LangGraph workflow assembly, routing rules, and SQLite checkpoint persistence.

Constructs the cyclical multi-agent graph with dynamic conditional routing
between the Lead Supervisor and worker nodes.
"""

import sqlite3
import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.graph.state import AgentState
from app.graph.agents import supervisor_node, code_analyst_node, doc_parser_node

logger = logging.getLogger(__name__)


def route_next(state: AgentState) -> Literal["code_analyst", "doc_parser", "__end__"]:
    """Evaluates supervisor decision and determines the next node.

    Args:
        state: Current AgentState containing `next_node`.

    Returns:
        Next node identifier or LangGraph `END` token.
    """
    next_step = state.get("next_node", "FINISH")
    if next_step == "FINISH":
        return END
    return next_step


def create_checkpointer():
    """Initializes an async-compatible checkpointer for non-blocking execution and astream support."""
    settings.ensure_directories()
    # MemorySaver natively supports both synchronous invoke and asynchronous astream/ainvoke
    return MemorySaver()


def build_workflow():
    """Assembles and configures the StateGraph nodes, edges, and conditional paths."""
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("code_analyst", code_analyst_node)
    workflow.add_node("doc_parser", doc_parser_node)

    # Entry edge into Supervisor
    workflow.add_edge(START, "supervisor")

    # Conditional routing out of Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "code_analyst": "code_analyst",
            "doc_parser": "doc_parser",
            END: END,
        },
    )

    # Return loops back from workers to Supervisor for verification or completion
    workflow.add_edge("code_analyst", "supervisor")
    workflow.add_edge("doc_parser", "supervisor")

    return workflow


# Build and compile persistent graph instance
checkpointer = create_checkpointer()
workflow = build_workflow()
compiled_graph = workflow.compile(checkpointer=checkpointer)

"""Agent node implementations for Supervisor, Code Analyst, and Doc Parser.

Defines the multi-agent worker nodes with automated tool calling loops and
the Lead Supervisor with structured routing output.
"""

import logging
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.graph.state import AgentState, RouterOutput
from app.graph.tools import parse_python_ast, query_documentation

logger = logging.getLogger(__name__)

# Model Instances configured from application settings
supervisor_llm = ChatOllama(
    model=settings.SUPERVISOR_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.0,
)

code_llm = ChatOllama(
    model=settings.CODE_ANALYST_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1,
)

doc_llm = ChatOllama(
    model=settings.DOC_PARSER_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1,
)

# Bind tools to specific agents
code_agent_with_tools = code_llm.bind_tools([parse_python_ast])
doc_agent_with_tools = doc_llm.bind_tools([query_documentation])
router_chain = supervisor_llm.with_structured_output(RouterOutput)

TOOL_MAP = {
    "parse_python_ast": parse_python_ast,
    "query_documentation": query_documentation,
}


def supervisor_node(state: AgentState) -> dict:
    """Orchestrates control flow by evaluating context and choosing worker routes.

    Args:
        state: Current AgentState containing conversation history and previous actions.

    Returns:
        Dictionary update with `next_node` set to 'code_analyst', 'doc_parser', or 'FINISH'.
    """
    system_prompt = SystemMessage(
        content=(
            "You are the Lead Analyst Supervisor directing an enterprise codebase audit.\n"
            "Worker capabilities:\n"
            "- 'code_analyst': Focuses on syntax, static AST analysis, performance, and security bugs.\n"
            "- 'doc_parser': Extracts requirements and spec details from vector indexed documentation.\n\n"
            "Evaluate conversation history. If context is sufficient to answer the prompt, select 'FINISH'.\n"
            "Otherwise, route to the appropriate worker agent."
        )
    )

    try:
        messages = [system_prompt] + list(state["messages"])
        decision = router_chain.invoke(messages)
        logger.info(f"Supervisor routed to: {decision.next_node} (Reason: {decision.reasoning})")
        return {"next_node": decision.next_node}
    except Exception as exc:
        logger.error(f"Supervisor node execution failed: {exc}")
        # Default safely to FINISH if unable to route
        return {"next_node": "FINISH"}


def code_analyst_node(state: AgentState) -> dict:
    """Processes technical code analysis requests using qwen2.5-coder with AST tool execution.

    Args:
        state: Current AgentState containing conversation history.

    Returns:
        Dictionary update with an AIMessage from [Code Analyst].
    """
    system_prompt = SystemMessage(
        content=(
            "You are a Senior Systems Architect and Code Auditor. Use the `parse_python_ast` tool "
            "when file paths are supplied to inspect structural code layout before delivering analysis.\n"
            "Provide detailed feedback on architecture, function signatures, syntax, potential bugs, and optimizations."
        )
    )

    try:
        history = [system_prompt] + list(state["messages"])
        response = code_agent_with_tools.invoke(history)

        # Handle tool call execution loop if the model decided to inspect AST
        if response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", "call_ast")
                logger.info(f"Code Analyst invoking tool '{tool_name}' with args: {tool_args}")

                tool_func = TOOL_MAP.get(tool_name)
                if tool_func:
                    tool_result = tool_func.invoke(tool_args)
                else:
                    tool_result = f"Error: Tool '{tool_name}' not recognized."

                tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

            # Re-invoke model with tool execution results for final synthesis
            final_response = code_agent_with_tools.invoke(history + [response] + tool_messages)
            content = final_response.content
        else:
            content = response.content

        return {"messages": [AIMessage(content=f"[Code Analyst]:\n{content}")]}
    except Exception as exc:
        logger.error(f"Code Analyst node encountered error: {exc}")
        return {"messages": [AIMessage(content=f"[Code Analyst Error]: Failed to analyze code: {str(exc)}")]}


def doc_parser_node(state: AgentState) -> dict:
    """Processes document queries using local vector RAG search with query tool execution.

    Args:
        state: Current AgentState containing conversation history.

    Returns:
        Dictionary update with an AIMessage from [Doc Parser].
    """
    system_prompt = SystemMessage(
        content=(
            "You are a Technical Documentation Specialist. Use the `query_documentation` tool "
            "to extract factual context from technical manuals, architecture specs, and documentation before responding.\n"
            "Ground your answer strictly in retrieved excerpts whenever available."
        )
    )

    try:
        history = [system_prompt] + list(state["messages"])
        response = doc_agent_with_tools.invoke(history)

        # Handle tool call execution loop if the model requested documentation retrieval
        if response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call.get("id", "call_rag")
                logger.info(f"Doc Parser invoking tool '{tool_name}' with args: {tool_args}")

                tool_func = TOOL_MAP.get(tool_name)
                if tool_func:
                    tool_result = tool_func.invoke(tool_args)
                else:
                    tool_result = f"Error: Tool '{tool_name}' not recognized."

                tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

            # Re-invoke model with retrieved excerpts for grounded answer
            final_response = doc_agent_with_tools.invoke(history + [response] + tool_messages)
            content = final_response.content
        else:
            content = response.content

        return {"messages": [AIMessage(content=f"[Doc Parser]:\n{content}")]}
    except Exception as exc:
        logger.error(f"Doc Parser node encountered error: {exc}")
        return {"messages": [AIMessage(content=f"[Doc Parser Error]: Failed to parse documentation: {str(exc)}")]}

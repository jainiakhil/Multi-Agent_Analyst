"""Agent node implementations for Supervisor, Code Analyst, and Doc Parser.

Defines the multi-agent worker nodes with automated tool calling loops,
dynamic LLM model resolution, and the Lead Supervisor with structured routing output.
"""

import logging
from typing import Optional
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.core.llm_factory import llm_factory
from app.graph.state import AgentState, RouterOutput
from app.graph.tools import parse_python_ast, query_documentation

logger = logging.getLogger(__name__)

# Baseline default model instances resolved via LLMFactory
supervisor_llm = llm_factory.create_chat_model(
    model_name=settings.SUPERVISOR_MODEL,
    provider=settings.LLM_PROVIDER,
    temperature=0.0,
)

code_llm = llm_factory.create_chat_model(
    model_name=settings.CODE_ANALYST_MODEL,
    provider=settings.LLM_PROVIDER,
    temperature=0.1,
)

doc_llm = llm_factory.create_chat_model(
    model_name=settings.DOC_PARSER_MODEL,
    provider=settings.LLM_PROVIDER,
    temperature=0.1,
)

# Bind tools to default agents
code_agent_with_tools = code_llm.bind_tools([parse_python_ast])
doc_agent_with_tools = doc_llm.bind_tools([query_documentation])
router_chain = supervisor_llm.with_structured_output(RouterOutput)

TOOL_MAP = {
    "parse_python_ast": parse_python_ast,
    "query_documentation": query_documentation,
}


def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    """Orchestrates control flow by evaluating context and choosing worker routes.

    Args:
        state: Current AgentState containing conversation history and previous actions.
        config: Optional LangGraph runtime execution configuration with model overrides.

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

    configurable = (config or {}).get("configurable", {})
    custom_model = configurable.get("supervisor_model")
    custom_provider = configurable.get("provider")

    # Resolve active router chain (custom override or default)
    if custom_model or custom_provider:
        active_llm = llm_factory.create_chat_model(
            model_name=custom_model,
            provider=custom_provider,
            temperature=0.0,
        )
        active_router = active_llm.with_structured_output(RouterOutput)
    else:
        active_router = router_chain

    try:
        messages = [system_prompt] + list(state["messages"])
        decision = active_router.invoke(messages)
        logger.info(f"Supervisor routed to: {decision.next_node} (Reason: {decision.reasoning})")
        return {"next_node": decision.next_node}
    except Exception as exc:
        logger.error(f"Supervisor node execution failed: {exc}")
        # Default safely to FINISH if unable to route
        return {"next_node": "FINISH"}


def code_analyst_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    """Processes technical code analysis requests using AST tool execution.

    Args:
        state: Current AgentState containing conversation history.
        config: Optional LangGraph runtime execution configuration with model overrides.

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

    configurable = (config or {}).get("configurable", {})
    custom_model = configurable.get("code_analyst_model") or configurable.get("code_model")
    custom_provider = configurable.get("provider")

    if custom_model or custom_provider:
        active_llm = llm_factory.create_chat_model(
            model_name=custom_model,
            provider=custom_provider,
            temperature=0.1,
        )
        active_agent = active_llm.bind_tools([parse_python_ast])
    else:
        active_agent = code_agent_with_tools

    try:
        history = [system_prompt] + list(state["messages"])
        response = active_agent.invoke(history)

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
            final_response = active_agent.invoke(history + [response] + tool_messages)
            content = final_response.content
        else:
            content = response.content

        return {"messages": [AIMessage(content=f"[Code Analyst]:\n{content}")]}
    except Exception as exc:
        logger.error(f"Code Analyst node encountered error: {exc}")
        return {"messages": [AIMessage(content=f"[Code Analyst Error]: Failed to analyze code: {str(exc)}")]}


def doc_parser_node(state: AgentState, config: Optional[RunnableConfig] = None) -> dict:
    """Processes document queries using local vector RAG search with query tool execution.

    Args:
        state: Current AgentState containing conversation history.
        config: Optional LangGraph runtime execution configuration with model overrides.

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

    configurable = (config or {}).get("configurable", {})
    custom_model = configurable.get("doc_parser_model") or configurable.get("doc_model")
    custom_provider = configurable.get("provider")

    if custom_model or custom_provider:
        active_llm = llm_factory.create_chat_model(
            model_name=custom_model,
            provider=custom_provider,
            temperature=0.1,
        )
        active_agent = active_llm.bind_tools([query_documentation])
    else:
        active_agent = doc_agent_with_tools

    try:
        history = [system_prompt] + list(state["messages"])
        response = active_agent.invoke(history)

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
            final_response = active_agent.invoke(history + [response] + tool_messages)
            content = final_response.content
        else:
            content = response.content

        return {"messages": [AIMessage(content=f"[Doc Parser]:\n{content}")]}
    except Exception as exc:
        logger.error(f"Doc Parser node encountered error: {exc}")
        return {"messages": [AIMessage(content=f"[Doc Parser Error]: Failed to parse documentation: {str(exc)}")]}

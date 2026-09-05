"""LLM Model Factory and Dynamic Model Registry.

Provides multi-provider instantiation (Ollama, OpenAI-compatible APIs),
dynamic model resolution, and runtime model switching capabilities.
"""

import logging
from typing import Optional, List, Dict
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMFactory:
    """Manages chat model instantiation, active model configuration, and provider routing."""

    def __init__(self) -> None:
        self.active_provider: str = settings.LLM_PROVIDER
        self.active_supervisor_model: str = settings.SUPERVISOR_MODEL
        self.active_code_model: str = settings.CODE_ANALYST_MODEL
        self.active_doc_model: str = settings.DOC_PARSER_MODEL
        self.active_embedding_model: str = settings.EMBEDDING_MODEL

    def get_active_models(self) -> Dict[str, str]:
        """Returns the currently configured active models and provider."""
        return {
            "provider": self.active_provider,
            "supervisor_model": self.active_supervisor_model,
            "code_analyst_model": self.active_code_model,
            "doc_parser_model": self.active_doc_model,
            "embedding_model": self.active_embedding_model,
        }

    def switch_models(
        self,
        supervisor_model: Optional[str] = None,
        code_analyst_model: Optional[str] = None,
        doc_parser_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, str]:
        """Switches active models and/or provider dynamically at runtime."""
        if supervisor_model:
            self.active_supervisor_model = supervisor_model
        if code_analyst_model:
            self.active_code_model = code_analyst_model
        if doc_parser_model:
            self.active_doc_model = doc_parser_model
        if embedding_model:
            self.active_embedding_model = embedding_model
        if provider:
            self.active_provider = provider

        logger.info(f"Updated active models: {self.get_active_models()}")
        return self.get_active_models()

    def create_chat_model(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.0,
    ) -> BaseChatModel:
        """Instantiates a BaseChatModel instance based on provider and model name.

        Args:
            model_name: Name of the model to load (e.g. 'llama3.1:8b', 'qwen2.5-coder:7b').
            provider: Provider name ('ollama' or 'openai_compatible').
            temperature: Sampling temperature for output generation.

        Returns:
            An instantiated LangChain BaseChatModel.
        """
        target_provider = provider or self.active_provider

        if target_provider == "openai_compatible":
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=model_name or "gpt-4o",
                    base_url=settings.OPENAI_API_BASE,
                    api_key=settings.OPENAI_API_KEY or "none",
                    temperature=temperature,
                )
            except ImportError:
                logger.warning(
                    "langchain_openai package not found. Falling back to ChatOllama."
                )

        # Default to Ollama
        return ChatOllama(
            model=model_name or self.active_supervisor_model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    def get_available_ollama_models(self) -> List[str]:
        """Queries local Ollama tags API to discover installed local models.

        Returns:
            List of model names available locally, or an empty list if unreachable.
        """
        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])
                    return [m.get("name") for m in models if "name" in m]
        except Exception as exc:
            logger.debug(f"Unable to query local Ollama tags ({exc})")
        return []


# Global singleton factory instance
llm_factory = LLMFactory()

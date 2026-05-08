# llm_wrapper.py
# ─────────────────────────────────────────────────────────────────────
# Single entry point for all Ollama LLM calls in this system.
#
# Centralising here means students only need to read one file to
# understand how the system communicates with the language model.
# Every agentic path (Tutor, Practice, Doc-to-Concept) calls call_llm()
# rather than constructing OllamaLLM instances directly, so the
# invocation pattern is consistent and observable in one place.
# ─────────────────────────────────────────────────────────────────────
import logging
import os

from langchain_ollama import OllamaLLM

from config import LLMConfig

logger = logging.getLogger(__name__)


def call_llm(prompt: str, config: LLMConfig, task_name: str) -> str:
    """
    Invoke the Ollama LLM with the given prompt and configuration.

    This is the single call site for all LLM interactions in the system.
    Using one wrapper ensures that logging, error handling, and config
    application are consistent across all three agentic paths.

    Args:
        prompt:    The fully-rendered prompt string (XML-delimited per Req 9.2).
        config:    LLMConfig dataclass specifying model, temperature, max_tokens.
        task_name: Human-readable label logged at DEBUG level for observability.
                   Examples: "router", "tutor_generate", "mcq_generation",
                   "formula_extract", "example_gen", "chunk_classify".

    Returns:
        The raw text response from the LLM as a string.

    Raises:
        KeyError:  If OLLAMA_BASE_URL is not set in the environment.
        Exception: Propagates any network or model errors from Ollama.
    """
    # Log all parameters at DEBUG so students can observe them at runtime.
    # Run with LOG_LEVEL=DEBUG (or set logging.basicConfig level) to see these.
    logger.debug(
        "LLM call | task=%s model=%s temperature=%s max_tokens=%s",
        task_name,
        config.model,
        config.temperature,
        config.max_tokens,
    )

    # Read base URL from environment — never hardcode service addresses.
    # OLLAMA_BASE_URL is typically http://host.docker.internal:11434 when
    # Ollama runs on the host machine outside the Docker Compose network.
    base_url = os.environ["OLLAMA_BASE_URL"]

    llm = OllamaLLM(
        base_url=base_url,
        model=config.model,
        temperature=config.temperature,
        # num_predict maps to Ollama's token limit parameter
        num_predict=config.max_tokens,
    )

    response: str = llm.invoke(prompt)
    return response

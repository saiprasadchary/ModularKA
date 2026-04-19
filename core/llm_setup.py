# Filename: core/llm_setup.py

import os
import logging
import urllib.request
from urllib.error import URLError, HTTPError

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from utils import get_logger

# Streamlit is optional (for scripts); handle gracefully
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None

# Initialize logger
logger = get_logger(__name__)

# Ensure environment variables are loaded
load_dotenv()

# In-memory cache for LLM clients keyed by (provider, model, temperature, kind)
_LLM_CACHE = {}


# ---------------------------------------------------------------------------
# Provider selection helpers
# ---------------------------------------------------------------------------

def _get_base_provider_from_env() -> str:
    """
    Decide provider from USE_OLLAMA env when no Streamlit session is present.
    USE_OLLAMA=1 -> 'Ollama', otherwise 'Groq'.
    """
    use_ollama = os.getenv("USE_OLLAMA", "0")
    return "Ollama" if str(use_ollama).strip() == "1" else "Groq"


def _get_active_provider() -> str:
    """
    Determine the active provider, preferring Streamlit session_state if available.
    Falls back to USE_OLLAMA env when outside Streamlit.
    """
    if st is not None:
        try:
            if "llm_provider" in st.session_state:
                provider = st.session_state["llm_provider"]
                if provider in ("Groq", "Ollama"):
                    return provider
        except Exception:
            # If session_state access fails, ignore and fall back to env
            pass

    return _get_base_provider_from_env()


def _get_groq_api_key() -> str:
    """Retrieves the Groq API key, raising an error if not found."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    return api_key


def _check_ollama_healthy(base_url: str) -> None:
    """
    Fast health check for Ollama by calling /api/tags with a short timeout.
    Raises RuntimeError with a clear message if unreachable.
    """
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Ollama health check failed with status code {resp.status}."
                )
    except Exception as e:
        general_model = os.getenv("OLLAMA_GENERAL_MODEL", "llama3:8b")
        coder_model = os.getenv("OLLAMA_CODER_MODEL", "codellama:7b")
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. "
            f"Start Ollama with `ollama serve` and pull models:\n"
            f"  ollama pull {general_model}\n"
            f"  ollama pull {coder_model}\n"
            f"Underlying error: {e}"
        )


def _get_model_name(provider: str, *, general: bool) -> str:
    """
    Resolve model name from env for the given provider + usage.
    Falls back to sensible defaults if env is missing.
    """
    if provider == "Groq":
        env_var = "GROQ_GENERAL_MODEL" if general else "GROQ_CODE_MODEL"
    else:
        env_var = "OLLAMA_GENERAL_MODEL" if general else "OLLAMA_CODER_MODEL"

    model = os.getenv(env_var)
    if model:
        return model

    if provider == "Groq":
        return (
            "meta-llama/llama-4-maverick-17b-128e-instruct"
            if general
            else "qwen-2.5-coder-32b"
        )
    else:
        return "llama3:8b" if general else "codellama:7b"


def _get_cached_or_new_llm(
    *, provider: str, model_name: str, temperature: float, kind: str
):
    """
    Either return a cached LLM or create a new one.
    `kind` is a label ('general' or 'code') for logging.
    """
    key = (provider, model_name, float(temperature), kind)

    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    if provider == "Groq":
        api_key = _get_groq_api_key()
        llm_client = ChatGroq(
            temperature=temperature,
            groq_api_key=api_key,
            model_name=model_name,
        )
    elif provider == "Ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        _check_ollama_healthy(base_url)
        llm_client = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )
    else:
        raise RuntimeError(f"Unknown LLM provider: {provider}")

    logger.info(
        f"Initialized {kind} LLM: provider={provider}, model={model_name}"
    )
    _LLM_CACHE[key] = llm_client
    return llm_client


# ---------------------------------------------------------------------------
# Public LLM getters
# ---------------------------------------------------------------------------

def get_llm():
    """
    Initializes (or retrieves cached) LLM for general tasks (Q&A, Summarization).
    Provider is chosen dynamically based on Streamlit session_state['llm_provider']
    when available, otherwise USE_OLLAMA env.
    """
    provider = _get_active_provider()
    model_name = _get_model_name(provider, general=True)
    temperature = 0.1  # factual, consistent outputs
    return _get_cached_or_new_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        kind="general",
    )


def get_codellm():
    """
    Initializes (or retrieves cached) LLM for code generation tasks.
    Uses the same provider selection logic as get_llm().
    """
    provider = _get_active_provider()
    model_name = _get_model_name(provider, general=False)
    temperature = 0.2  # slightly higher for code creativity, still low
    return _get_cached_or_new_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        kind="code",
    )


# ---------------------------------------------------------------------------
# Direct test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("--- Testing LLM Initialization ---")
    try:
        general_llm_instance = get_llm()
        logger.info(
            f"Successfully initialized General LLM: {type(general_llm_instance)}"
        )

        logger.info("-" * 20)

        coding_llm_instance = get_codellm()
        logger.info(
            f"Successfully initialized Coding LLM: {type(coding_llm_instance)}"
        )

    except Exception as e:
        logger.error(f"Configuration / LLM Error: {e}")
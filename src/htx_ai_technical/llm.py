"""Model-agnostic chat model factory.

`get_llm` is the only place the code reaches a model. It goes through LangChain's
`init_chat_model`, so switching provider is a config change. The API key comes only from
the `API_KEY` environment variable, whatever the provider, never from config. A non-default provider's
LangChain package is installed on first use (from an allowlist), unless
`llm.auto_install` is false. Every call through the model is logged (see `llm_logging`).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import ConfigError, LLMConfig
from .llm_logging import LLMCallLogger

API_KEY_ENV = "API_KEY"

# provider -> (pip package, whether it needs an API key). Mirrors init_chat_model's
# supported providers. Keyless providers authenticate another way: a local server
# (ollama) or cloud credentials (Vertex AI via gcloud, Bedrock via AWS).
PROVIDER_PACKAGES: dict[str, tuple[str, bool]] = {
    "openai": ("langchain-openai", True),
    "anthropic": ("langchain-anthropic", True),
    "azure_openai": ("langchain-openai", True),
    "azure_ai": ("langchain-azure-ai", True),
    "google_vertexai": ("langchain-google-vertexai", False),
    "google_genai": ("langchain-google-genai", True),
    "anthropic_bedrock": ("langchain-aws", False),
    "bedrock": ("langchain-aws", False),
    "bedrock_converse": ("langchain-aws", False),
    "bedrock_mantle_openai": ("langchain-aws[openai]", False),
    "bedrock_mantle_anthropic": ("langchain-aws[anthropic]", False),
    "cohere": ("langchain-cohere", True),
    "fireworks": ("langchain-fireworks", True),
    "together": ("langchain-together", True),
    "mistralai": ("langchain-mistralai", True),
    "huggingface": ("langchain-huggingface", True),
    "groq": ("langchain-groq", True),
    "ollama": ("langchain-ollama", False),
    "google_anthropic_vertex": ("langchain-google-vertexai", False),
    "deepseek": ("langchain-deepseek", True),
    "ibm": ("langchain-ibm", True),
    "nvidia": ("langchain-nvidia-ai-endpoints", True),
    "xai": ("langchain-xai", True),
    "openrouter": ("langchain-openrouter", True),
    "perplexity": ("langchain-perplexity", True),
    "upstage": ("langchain-upstage", True),
    "baseten": ("langchain-baseten", True),
    "litellm": ("langchain-litellm", True),
    "meta": ("langchain-meta", True),
    "langsmith": ("langchain-openai", True),
}


def _install(package: str) -> None:
    """Install the provider package, holding langchain-core at its installed (locked) version.

    Without the pin, the resolver may upgrade langchain-core for the provider's latest release;
    the next `uv run` then syncs it back to the lockfile, breaking the provider's import.
    """
    packages = [package, f"langchain-core=={importlib.metadata.version('langchain-core')}"]
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--python", sys.executable, *packages]
        if uv
        else [sys.executable, "-m", "pip", "install", *packages]
    )
    print(f"[llm] Installing {package} for this provider: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    importlib.invalidate_caches()


def _api_key(provider: str, needs_key: bool) -> str | None:
    """The key from API_KEY, or None for keyless providers. Fails early if it's required."""
    if not needs_key:
        return None
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise ConfigError(f"Missing API key: set the {API_KEY_ENV} environment variable (provider {provider!r}).")
    return key


def get_llm(llm_config: LLMConfig) -> BaseChatModel:
    """Build the chat model from the `llm` section of config."""
    if llm_config.provider not in PROVIDER_PACKAGES:
        valid = ", ".join(sorted(PROVIDER_PACKAGES))
        raise ConfigError(f"Unknown provider {llm_config.provider!r} in llm.provider. Valid providers: {valid}.")
    package, needs_key = PROVIDER_PACKAGES[llm_config.provider]

    call_logger = LLMCallLogger(llm_config.input_cost_per_million, llm_config.output_cost_per_million)
    kwargs = {"temperature": llm_config.temperature, "callbacks": [call_logger]}
    if api_key := _api_key(llm_config.provider, needs_key):
        kwargs["api_key"] = api_key

    def build() -> BaseChatModel:
        return init_chat_model(llm_config.model, model_provider=llm_config.provider, **kwargs)

    try:
        return build()
    except ImportError:
        if not llm_config.auto_install:
            raise ConfigError(
                f"Provider {llm_config.provider!r} needs the {package} package. Install it with "
                f"`uv pip install {package}`, or set llm.auto_install: true in config.yaml."
            ) from None
        _install(package)
        return build()

"""Load prompt templates from `prompts/*.yaml`."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompts(name: str) -> dict[str, Any]:
    """Read `prompts/<name>.yaml` as a dict."""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return yaml.safe_load(path.read_text())


def chat_prompt(name: str) -> ChatPromptTemplate:
    """A system + human ChatPromptTemplate from the `system` and `human` keys of a prompt file."""
    spec = load_prompts(name)
    return ChatPromptTemplate.from_messages([("system", spec["system"]), ("human", spec["human"])])

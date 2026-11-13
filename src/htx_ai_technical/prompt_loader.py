"""Load prompt templates from `prompts/*.yaml`."""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent / "prompts"


def chat_prompt(name: str) -> ChatPromptTemplate:
    """A system + human ChatPromptTemplate from the `system` and `human` keys of `prompts/<name>.yaml`."""
    spec = yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text())
    return ChatPromptTemplate.from_messages([("system", spec["system"]), ("human", spec["human"])])

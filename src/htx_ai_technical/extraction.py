"""Generic grounded extraction: locate block -> LLM structured output -> answers with provenance.

A *field group* is a Pydantic class (see `fields.py`): its fields are what the LLM
extracts, their descriptions are the definitions it reads, and its `anchor` and `block_type`
class variables locate the source block. Any part can define its own groups
and reuse `extract`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from .prompt_loader import chat_prompt
from .retrieval import locate, render_block
from .schemas import Block, ParsedDocument, Provenance

@dataclass
class GroupResult:
    """One field group, its source block and the LLM output for it, or why there is none."""

    group: type[BaseModel]
    block: Block | None
    output: BaseModel | None = None
    error: str | None = None


def locate_block(doc: ParsedDocument, group: type[BaseModel]) -> Block | None:
    """The first block matching the group's anchor, or None."""
    hits = locate(doc, group.anchor, group.block_type)
    return hits[0] if hits else None


def definitions(group: type[BaseModel]) -> str:
    """The group's field descriptions as a bullet list for the prompt."""
    return "\n".join(f"- {name}: {info.description}" for name, info in group.model_fields.items())


def extract(doc: ParsedDocument, groups: list[type[BaseModel]], llm: BaseChatModel) -> list[GroupResult]:
    """Run one LLM call per field group, in parallel. Failures are recorded, not raised."""
    results = [GroupResult(g, locate_block(doc, g)) for g in groups]

    def run(result: GroupResult) -> BaseModel:
        chain = chat_prompt("extraction") | llm.with_structured_output(result.group)
        return chain.invoke(
            {
                "fy": doc.fiscal_year,
                "definitions": definitions(result.group),
                "context": render_block(result.block),
            },
            config={"metadata": {"field_group": result.group.__name__}},  # shown in the LLM call log
        )

    located = [r for r in results if r.block]
    outputs = RunnableLambda(run).batch(located, return_exceptions=True)
    for result, output in zip(located, outputs):
        if isinstance(output, Exception):
            result.error = f"{type(output).__name__}: {output}"
        else:
            result.output = output
    for result in results:
        if not result.block:
            result.error = "No source block found (no anchor match)."
    return results


def field_answers(result: GroupResult) -> dict[str, tuple[Any, Provenance]]:
    """`{field: (value, provenance)}` for every field of a group: the LLM's answer, the block it
    read and the evidence it quoted."""
    out: dict[str, tuple[Any, Provenance]] = {}
    for name in result.group.model_fields:
        answer = getattr(result.output, name, None) if result.output else None
        prov = Provenance()
        if result.block:
            prov.block_id, prov.block_title, prov.pages = result.block.id, result.block.title, result.block.pages
        if answer is None:
            prov.error = result.error or "No output for this field."
            out[name] = (None, prov)
            continue
        prov.row = getattr(answer, "row_label", None)  # only a TableValue has row and column labels
        prov.column = getattr(answer, "column_label", None)
        prov.evidence = answer.evidence
        out[name] = (answer.value, prov)
    return out

"""Part 2: find, normalise and classify dates.

Two steps:
1. extraction (`prompts/date_extraction.yaml`): a tool-calling agent reads the located blocks,
   finds each date and normalises it with `normalize_date` from the local MCP server
   (`mcp_server.py`).
2. classification (`prompts/date_classification.yaml`): an LLM call with no tools gets the
   normalised dates with their context sentences and the reference date, and reasons about
   each to choose Expired, Upcoming or Ongoing.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRequest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel

from .extraction import definitions
from .prompt_loader import chat_prompt
from .retrieval import render_block
from .schemas import (
    Block,
    Classifications,
    DateStatus,
    NormalisedDate,
    NormalisedDates,
    Provenance,
    ToolCallRecord,
)

# Model calls allowed in one extraction agent run. A normal run takes 2 (call the tool for every
# date, then answer), so this leaves room for retries. Without the cap, a model that keeps calling
# tools would loop until `create_agent`'s recursion limit (9,999 graph steps). The last call
# must give the answer (see `AnswerOnLastCall`).
MAX_MODEL_CALLS = 6
LAST_CALL_NOTE = (
    "This is your last call and you can't call tools any more. Give your answer now, using the "
    "tool results you already have."
)

# The local MCP server (`mcp_server.py`), started as a subprocess that talks over stdio.
MCP_SERVERS = {
    "dates": {"command": sys.executable, "args": ["-m", "htx_ai_technical.mcp_server"], "transport": "stdio"},
}


@dataclass
class DateResults:
    """What Part 2 found: step 1's normalised dates and tool calls, step 2's statuses, and the
    reason if step 1 gave no answer."""

    normalized: list[NormalisedDate] = field(default_factory=list)
    dates: list[DateStatus] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None


class AnswerOnLastCall(ModelCallLimitMiddleware):
    """`ModelCallLimitMiddleware` whose last allowed call must give the structured answer: the
    tools are taken away, the answer schema is forced, and the model is told it's the last call."""

    def __init__(self, run_limit: int, note: str) -> None:
        super().__init__(run_limit=run_limit, exit_behavior="end")
        self.note = note

    def last_call(self, request: ModelRequest) -> ModelRequest:
        """`request` as is, or made to answer if it is the last allowed call."""
        if request.state.get("run_model_call_count", 0) < self.run_limit - 1:
            return request
        return request.override(
            tools=[],
            # A ToolStrategy binds the answer schema as the only tool, with tool_choice="any".
            response_format=ToolStrategy(request.response_format.schema),
            messages=[*request.messages, HumanMessage(self.note)],
        )

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self.last_call(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self.last_call(request))


def tool_trace(messages: list[BaseMessage], tools: list[BaseTool]) -> list[ToolCallRecord]:
    """The calls to `tools` in an agent's messages, with their outputs. Skips the structured-output
    tool call the agent may use to return its answer."""
    names = {t.name for t in tools}
    calls = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    return [
        ToolCallRecord(tool=m.name, args=calls[m.tool_call_id]["args"], output=m.text)
        for m in messages
        if isinstance(m, ToolMessage) and m.name in names and m.tool_call_id in calls
    ]


async def extract_dates(
    llm: BaseChatModel, tools: list[BaseTool], fiscal_year: int, definitions: str, context: str
) -> tuple[NormalisedDates | None, list[ToolCallRecord]]:
    """Step 1: run the tool-calling agent on `prompts/date_extraction.yaml`. Returns its answer
    (None if even its last allowed call gave no valid answer) and the tool calls it made."""
    agent = create_agent(
        llm,
        tools=tools,
        response_format=NormalisedDates,
        middleware=[AnswerOnLastCall(MAX_MODEL_CALLS, note=LAST_CALL_NOTE)],
    )
    messages = chat_prompt("date_extraction").format_messages(fy=fiscal_year, definitions=definitions, context=context)
    result = await agent.ainvoke(
        {"messages": messages},
        config={"metadata": {"step": "date_extraction"}},  # shown in the LLM call log
    )
    return result.get("structured_response"), tool_trace(result["messages"], tools)


async def find_and_classify_async(
    llm: BaseChatModel, blocks: dict[type[BaseModel], Block], fiscal_year: int, reference_date: date
) -> DateResults:
    """Both steps, for the field groups in `blocks` (`{group: its located block}`)."""
    results = DateResults()
    async with MultiServerMCPClient(MCP_SERVERS).session("dates") as session:
        normalised, results.tool_calls = await extract_dates(
            llm,
            await load_mcp_tools(session),
            fiscal_year,
            definitions="\n".join(definitions(g) for g in blocks),
            context="\n\n".join(render_block(b) for b in blocks.values()),
        )
    if normalised is None:
        results.error = (
            f"The date extraction agent gave no valid answer within {MAX_MODEL_CALLS} model calls, "
            "including the last one, which had to answer. Its tool calls are in tool_calls."
        )
        return results
    results.normalized = normalised.dates
    if not results.normalized:
        return results

    chain = chat_prompt("date_classification") | llm.with_structured_output(Classifications)
    classified = await chain.ainvoke(
        {
            "reference_date": reference_date.isoformat(),
            "dates": "\n".join(
                f'- id: {d.id} | date: {d.normalized_date} | context: "{d.original_text}"' for d in results.normalized
            ),
        },
        config={"metadata": {"step": "date_classification"}},  # shown in the LLM call log
    )

    by_id = {c.id: c for c in classified.dates}
    results.dates = [
        DateStatus(**d.model_dump(), status=by_id[d.id].status, reasoning=by_id[d.id].reasoning)
        for d in results.normalized
        if d.id in by_id
    ]
    return results


def find_and_classify(
    llm: BaseChatModel, blocks: dict[type[BaseModel], Block], fiscal_year: int, reference_date: date
) -> DateResults:
    """Synchronous wrapper for the CLI: the MCP client is async, so this runs it with `asyncio.run`."""
    return asyncio.run(find_and_classify_async(llm, blocks, fiscal_year, reference_date))


def date_provenance(blocks: dict[type[BaseModel], Block | None], dates: list[NormalisedDate]) -> dict[str, Provenance]:
    """`{field: provenance}` for every Part 2 field: the block it was looked for in, the sentence the
    LLM quoted, or why there is no answer."""
    by_id = {d.id: d for d in dates}
    out = {}
    for group, block in blocks.items():
        for name in group.model_fields:
            prov = Provenance()
            if block:
                prov.block_id, prov.block_title, prov.pages = block.id, block.title, block.pages
            if name in by_id:
                prov.evidence = by_id[name].original_text
            else:
                prov.error = "No date returned for this field." if block else "No source block found (no anchor match)."
            out[name] = prov
    return out

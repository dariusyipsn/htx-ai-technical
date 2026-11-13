"""Part 3: a LangGraph supervisor over a Revenue Agent and an Expenditure Agent.

    START -> supervisor -+-> revenue ------+
               ^         +-> expenditure --+
               +---------+-----------------+
                         +-> synthesize -> END

- supervisor (`prompts/supervisor.yaml`): a structured-output call (`Route`) that reasons about
  what the query still needs, then picks the next agent and writes its task, or finishes.
- revenue / expenditure (`prompts/<name>_agent.yaml`): tool-calling agents (`create_agent`)
  with the same tools (`retrieval` lookups and `sum_values`) and different prompts. Each returns
  an `AgentReport`, and sees only its task, not the query or other reports.
- synthesize (`prompts/synthesis.yaml`): writes the final answer from the reports and the
  supervisor's reason for finishing, which tells an off-topic query apart from one the
  document can't answer.

Every decision and agent run is appended to `steps`, which is the trace.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from . import retrieval
from .dates import tool_trace
from .prompt_loader import chat_prompt
from .schemas import AgentName, AgentReport, ParsedDocument, Route, SupervisorDecision, WorkerRun

logger = logging.getLogger(__name__)

MAX_AGENT_CALLS = 4  # then the supervisor must finish
# Model calls per agent run (search, read, sum, answer, with room to spare). An agent that hits it
# fails with ModelCallLimitExceededError, which is recorded under its report's gaps.
MAX_AGENT_MODEL_CALLS = 8


class State(TypedDict):
    query: str
    fiscal_year: int
    steps: Annotated[list[SupervisorDecision | WorkerRun], operator.add]
    final_answer: str


def document_tools(doc: ParsedDocument) -> list[BaseTool]:
    """Read-only lookups over the parsed document, shared by both agents."""

    @tool
    def list_blocks() -> str:
        """List every section and table in the document: id, type, heading path and pages."""
        return retrieval.list_blocks(doc)

    @tool
    def search(keywords: str) -> str:
        """Find the sections and tables that best match the keywords (e.g. 'Future Energy Fund'
        or 'Corporate Income Tax'). Returns each block's id, title, pages and best-matching line."""
        return retrieval.search(doc, keywords)

    @tool
    def read_blocks(block_ids: list[str]) -> str:
        """Read the full text of one or more sections, or tables as markdown, by id (e.g.
        ['s023', 't026']). Ask for every block you need in one call."""
        return retrieval.read_blocks(doc, block_ids)

    return [list_blocks, search, read_blocks]


@tool
def sum_values(values: list[float]) -> str:
    """Add up a list of numbers exactly. Use it for every total; never add in your head."""
    return f"{round(sum(values), 6):g}"


def render_reports(steps: list[SupervisorDecision | WorkerRun]) -> str:
    """Every agent report so far, for the supervisor and synthesis prompts."""
    runs = [s for s in steps if isinstance(s, WorkerRun)]
    if not runs:
        return "(none yet)"
    parts = []
    for i, run in enumerate(runs, 1):
        figures = "\n".join(
            f"  - {f.label}: {f.value_bn:g} bn ({', '.join(f.block_ids)}; p. {', '.join(map(str, f.pages))})"
            for f in run.report.figures
        )
        parts.append(
            f"Report {i}, from {run.agent}\nTask: {run.task}\nAnswer: {run.report.answer}\n"
            f"Figures:\n{figures or '  (none)'}\nGaps: {run.report.gaps or '(none)'}"
        )
    return "\n\n".join(parts)


def next_node(state: State) -> str:
    """Where to go after the supervisor: the agent it chose, or synthesis."""
    decision = state["steps"][-1]
    return "synthesize" if decision.next == "finish" else decision.next


def build_graph(llm: BaseChatModel, doc: ParsedDocument) -> CompiledStateGraph:
    tools = [*document_tools(doc), sum_values]
    route = chat_prompt("supervisor") | llm.with_structured_output(Route)

    def supervisor(state: State) -> dict:
        calls_left = MAX_AGENT_CALLS - sum(isinstance(s, WorkerRun) for s in state["steps"])
        if calls_left <= 0:
            decision = SupervisorDecision(
                reasoning=f"Reached the limit of {MAX_AGENT_CALLS} agent calls.", next="finish", task=""
            )
        else:
            out = route.invoke(
                {
                    "fy": state["fiscal_year"],
                    "calls_left": calls_left,
                    "query": state["query"],
                    "reports": render_reports(state["steps"]),
                },
                config={"metadata": {"node": "supervisor"}},  # shown in the LLM call log
            )
            decision = SupervisorDecision(**out.model_dump())
        logger.info("supervisor -> %s: %s", decision.next, decision.reasoning)
        return {"steps": [decision]}

    def worker(name: AgentName):
        agent = create_agent(
            llm,
            tools=tools,
            response_format=AgentReport,
            middleware=[ModelCallLimitMiddleware(run_limit=MAX_AGENT_MODEL_CALLS, exit_behavior="error")],
        )

        def run(state: State) -> dict:
            task = state["steps"][-1].task
            try:
                result = agent.invoke(
                    {"messages": chat_prompt(f"{name}_agent").format_messages(fy=state["fiscal_year"], task=task)},
                    config={"metadata": {"node": name}},
                )
                report, calls = result["structured_response"], tool_trace(result["messages"], tools)
            except Exception as exc:  # recorded in the trace for the supervisor to work around
                report, calls = AgentReport(answer="", figures=[], gaps=f"Agent failed: {type(exc).__name__}: {exc}"), []
            return {"steps": [WorkerRun(agent=name, task=task, tool_calls=calls, report=report)]}

        return run

    synthesis = chat_prompt("synthesis") | llm

    def synthesize(state: State) -> dict:
        answer = synthesis.invoke(
            {
                "fy": state["fiscal_year"],
                "query": state["query"],
                "finish_reason": state["steps"][-1].reasoning,
                "reports": render_reports(state["steps"]),
            },
            config={"metadata": {"node": "synthesize"}},
        )
        return {"final_answer": answer.text}

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("revenue", worker("revenue"))
    graph.add_node("expenditure", worker("expenditure"))
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", next_node, ["revenue", "expenditure", "synthesize"])
    graph.add_edge("revenue", "supervisor")
    graph.add_edge("expenditure", "supervisor")
    graph.add_edge("synthesize", END)
    return graph.compile()

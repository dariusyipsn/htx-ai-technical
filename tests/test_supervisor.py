from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from htx_ai_technical.schemas import AgentReport, Figure, ParsedDocument, SupervisorDecision, WorkerRun
from htx_ai_technical.supervisor import (
    MAX_AGENT_CALLS,
    MAX_AGENT_MODEL_CALLS,
    attempted_report,
    build_graph,
    next_node,
    render_reports,
    sum_values,
)


def decision(next_: str) -> SupervisorDecision:
    return SupervisorDecision(reasoning="r", next=next_, task="" if next_ == "finish" else "t")


def run(agent: str = "expenditure") -> WorkerRun:
    figure = Figure(label="Future Energy Fund top-up, Estimated FY2024", value_bn=5.0, block_ids=["t026"], pages=[20])
    return WorkerRun(agent=agent, task="Find the FEF top-up", report=AgentReport(answer="$5.0bn (p. 20)", figures=[figure], gaps=""))


def test_sum_values_is_exact():
    assert sum_values.invoke({"values": [6.0, 5.0, 2.0, 2.0, 2.0, 1.8, 1.0, 0.5, 0.05, 0.002]}) == "20.352"


def test_next_node_follows_the_last_decision():
    assert next_node({"steps": [decision("revenue")]}) == "revenue"
    assert next_node({"steps": [decision("revenue"), run(), decision("finish")]}) == "synthesize"


def test_render_reports():
    assert render_reports([decision("revenue")]) == "(none yet)"
    assert render_reports([decision("expenditure"), run()]) == (
        "Report 1, from expenditure\nTask: Find the FEF top-up\nAnswer: $5.0bn (p. 20)\n"
        "Figures:\n  - Future Energy Fund top-up, Estimated FY2024: 5 bn (t026; p. 20)\nGaps: (none)"
    )


class FakeModel(GenericFakeChatModel):
    """Answers plain calls from its scripted messages; fails if asked to route or call tools."""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def fail(_):
            raise AssertionError("the supervisor must not call the LLM")

        return RunnableLambda(fail)


def test_supervisor_finishes_without_an_llm_call_at_the_agent_call_limit():
    llm = FakeModel(messages=iter([AIMessage("final answer")]))
    graph = build_graph(llm, ParsedDocument(source="fake.pdf", fiscal_year=2030, blocks=[]))
    steps = [run() for _ in range(MAX_AGENT_CALLS)]

    state = graph.invoke({"query": "q", "fiscal_year": 2030, "steps": steps})

    assert state["steps"][-1] == SupervisorDecision(
        reasoning=f"Reached the limit of {MAX_AGENT_CALLS} agent calls.", next="finish", task=""
    )
    assert state["final_answer"] == "final answer"


def test_attempted_report_keeps_the_answer_and_gaps_but_not_the_figures():
    invalid = {"answer": "$5.0bn (p. 20)", "figures": [{"label": "FEF top-up", "value_bn": "five"}], "gaps": "No FY2025 figure."}
    messages = [
        HumanMessage("Find the FEF top-up"),
        AIMessage("", tool_calls=[{"name": "AgentReport", "args": invalid, "id": "a"}]),
        ToolMessage("Error: value_bn is not a number", name="AgentReport", tool_call_id="a"),
        AIMessage(f"Model call limits exceeded: run limit ({MAX_AGENT_MODEL_CALLS}/{MAX_AGENT_MODEL_CALLS})"),
    ]
    report = attempted_report(messages)
    assert (report.answer, report.figures) == ("$5.0bn (p. 20)", [])
    assert report.gaps.endswith("have no figures to cite. No FY2025 figure.")


def test_attempted_report_without_an_attempt():
    assert attempted_report([HumanMessage("Find the FEF top-up")]).answer == ""

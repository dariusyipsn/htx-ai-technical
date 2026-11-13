import pytest

from evals.part1_eval import matches
from evals.part3_eval import score_case
from htx_ai_technical.schemas import AgentReport, Figure, Part3Result, SupervisorDecision, WorkerRun


@pytest.mark.parametrize(
    "expected, actual, passed",
    [
        (28.03, 28.034, True),
        (28.03, 28.04, False),
        (-1.2, -1.2, True),
        (1.72, None, False),
        (None, None, True),
        (None, 1.72, False),
        (2024, 2024, True),
        (2024, 2025, False),
        (["Corporate Income Tax", "Betting Taxes"], ["betting taxes", "Corporate  Income Tax"], True),
        (["Corporate Income Tax", "Betting Taxes"], ["Corporate Income Tax"], False),
        (["Corporate Income Tax"], ["Corporate Income Tax", "Other Taxes"], False),
    ],
)
def test_matches(expected, actual, passed):
    assert matches(expected, actual) is passed


def test_part3_score_case():
    figure = Figure(label="Total top-ups", value_bn=20.352, block_ids=["t026"], pages=[20])
    report = AgentReport(answer="", figures=[figure], gaps="")
    result = Part3Result(
        source_pdf="fake.pdf",
        fiscal_year=2024,
        query="q",
        steps=[SupervisorDecision(reasoning="", next="expenditure", task="t"), WorkerRun(agent="expenditure", task="t", report=report)],
        final_answer="",
    )

    scores = score_case({"agents": ["expenditure"], "figures": [20.36, 6.0]}, result)

    assert [(s.field, s.actual, s.passed) for s in scores] == [
        ("agents", ["expenditure"], True),
        ("figure 20.36", 20.352, True),
        ("figure 6", 20.352, False),
    ]

import asyncio
import itertools

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from htx_ai_technical.dates import MAX_MODEL_CALLS, date_provenance, extract_dates, tool_trace
from htx_ai_technical.extraction import locate_block
from htx_ai_technical.fields import DistributionDate, EstateDutyDate
from htx_ai_technical.schemas import Block, NormalisedDate, ParsedDocument

COVER = Block(id="s0", type="section", title="(cover)", pages=[1], text="")
DOC = ParsedDocument(
    source="fake.pdf",
    fiscal_year=2030,
    blocks=[
        COVER,
        Block(id="t1", type="table", title="Assets Taxes", pages=[8], text=""),
        Block(id="s2", type="section", title="Asset Taxes", section_path=["GLOSSARY"], pages=[37], text=""),
    ],
)


def test_part2_anchors():
    assert locate_block(DOC, DistributionDate) == COVER
    assert locate_block(DOC, EstateDutyDate) == DOC.blocks[2]  # a section, so the table is skipped


def test_date_provenance():
    budget_day = NormalisedDate(
        id="distribution_date",
        original_text="Distributed on Budget Day: 16 February 2024",
        normalized_date="2024-02-16",
    )
    prov = date_provenance({DistributionDate: COVER, EstateDutyDate: None}, [budget_day])
    assert (prov["distribution_date"].block_id, prov["distribution_date"].pages) == ("s0", [1])
    assert (prov["distribution_date"].evidence, prov["distribution_date"].error) == (budget_day.original_text, None)
    assert prov["estate_duty_date"].error == "No source block found (no anchor match)."

    prov = date_provenance({DistributionDate: COVER}, [])
    assert prov["distribution_date"].error == "No date returned for this field."


@tool
def normalize_date(date_text: str) -> str:
    """Stand-in for the MCP tool."""
    return date_text


def test_tool_trace_keeps_only_given_tools():
    messages = [
        HumanMessage("find the dates"),
        AIMessage("", tool_calls=[{"name": "normalize_date", "args": {"date_text": "16 February 2024"}, "id": "a"}]),
        ToolMessage("2024-02-16", name="normalize_date", tool_call_id="a"),
        AIMessage("", tool_calls=[{"name": "NormalisedDates", "args": {"dates": []}, "id": "b"}]),
        ToolMessage("Returning structured response", name="NormalisedDates", tool_call_id="b"),
    ]
    [call] = tool_trace(messages, [normalize_date])
    assert (call.tool, call.args, call.output) == ("normalize_date", {"date_text": "16 February 2024"}, "2024-02-16")


class LoopingModel(GenericFakeChatModel):
    """Calls a tool on every turn and never answers."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_extract_dates_stops_a_model_that_keeps_calling_tools():
    @tool
    def normalize_date(date_text: str) -> str:
        """Normalise a date."""
        return "2024-02-16"

    calls = itertools.count()
    llm = LoopingModel(
        messages=(
            AIMessage("", tool_calls=[{"name": "normalize_date", "args": {"date_text": "16 Feb 2024"}, "id": f"c{i}"}])
            for i in calls
        )
    )
    answer, trace = asyncio.run(extract_dates(llm, [normalize_date], 2030, definitions="", context=""))
    assert answer is None
    assert len(trace) == MAX_MODEL_CALLS  # every call is kept for the output's tool_calls
    assert next(calls) == MAX_MODEL_CALLS

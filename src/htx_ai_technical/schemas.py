"""Pydantic models shared by every part: the parsed document, LLM output schemas, results."""

from __future__ import annotations

from functools import cached_property
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Parsed document
# ---------------------------------------------------------------------------


class Block(BaseModel):
    """A heading-delimited section or a table, with the pages it spans."""

    id: str
    type: Literal["section", "table"]
    title: str
    section_path: list[str] = Field(default_factory=list, description="Ancestor headings, outermost first.")
    pages: list[int] = Field(default_factory=list)
    text: str = Field(description="Section paragraphs, or the table rendered as markdown.")
    grid: list[list[str]] | None = Field(default=None, description="Table cells, row-major. None for sections.")
    header_rows: int = Field(default=0, description="How many leading grid rows are column headers.")

    def describe(self) -> str:
        path = " > ".join([*self.section_path, self.title])
        pages = ", ".join(map(str, self.pages))
        return f"[{self.id}] {self.type.upper()}: {path} (pages {pages})"


class ParsedDocument(BaseModel):
    source: str
    fiscal_year: int | None
    blocks: list[Block]

    @cached_property
    def blocks_by_id(self) -> dict[str, Block]:
        """Blocks keyed by id, built on first use. Not a field, so it isn't saved in the parse cache."""
        return {b.id: b for b in self.blocks}


# ---------------------------------------------------------------------------
# LLM output building blocks (reused by any field-group schema)
# ---------------------------------------------------------------------------


class TableValue(BaseModel):
    """A number read from one table cell. The row/column labels show where it was read."""

    value: float | None = Field(description="The number in the requested unit, or null if not stated.")
    row_label: str | None = Field(description="Exact row label of the cell you read, as printed in the table.")
    column_label: str | None = Field(
        description="Exact column header of the cell you read (include the parent header, e.g. 'FY2024 Budget Estimated')."
    )
    evidence: str | None = Field(description="The cell text exactly as printed, e.g. '(1.2)' or '20,352'.")


class TextList(BaseModel):
    """A list of items named in running text, each backed by a verbatim quote."""

    value: list[str] = Field(description="Item names exactly as written in the text.")
    evidence: list[str] = Field(description="Short verbatim quotes from the text that mention the items.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    block_id: str | None = None
    block_title: str | None = None
    pages: list[int] = Field(default_factory=list)
    row: str | None = None
    column: str | None = None
    evidence: str | list[str] | None = None
    error: str | None = None


class Part1Result(BaseModel):
    source_pdf: str
    fiscal_year: int
    corporate_income_tax_bn: float | None = None
    corporate_income_tax_yoy_pct: float | None = None
    total_topups_bn: float | None = None
    operating_revenue_taxes: list[str] = Field(default_factory=list)
    latest_actual_fiscal_position_bn: float | None = None
    provenance: dict[str, Provenance] = Field(default_factory=dict)


class NormalisedDate(BaseModel):
    """Step 1 output: a date found in the excerpts, normalised with the MCP tool."""

    id: str = Field(description="The date's name, exactly as listed under 'Dates to find'.")
    original_text: str = Field(description="The full sentence or line containing the date, copied verbatim.")
    normalized_date: str = Field(description="The normalisation tool's output for this date, copied exactly.")


class NormalisedDates(BaseModel):
    dates: list[NormalisedDate] = Field(description="One item per date found and normalised.")


class Classification(BaseModel):
    """Step 2 output: a normalised date classified by the LLM. `reasoning` comes before `status`
    so the model reasons before it answers."""

    id: str = Field(description="The date's id, exactly as given.")
    reasoning: str = Field(
        description="One or two sentences: the subject, what the date marks for it (event, start or "
        "end of a period) and how it compares with the reference date."
    )
    status: Literal["Expired", "Upcoming", "Ongoing"]


class Classifications(BaseModel):
    dates: list[Classification] = Field(description="One item per date given.")


class DateStatus(BaseModel):
    id: str
    original_text: str
    normalized_date: str
    status: Literal["Expired", "Upcoming", "Ongoing"]
    reasoning: str


class ToolCallRecord(BaseModel):
    """One MCP tool call made by the LLM, for the trace."""

    tool: str
    args: dict
    output: str


class Part2Result(BaseModel):
    source_pdf: str
    fiscal_year: int
    reference_date: str
    normalized_dates: list[str] = Field(default_factory=list)
    dates: list[DateStatus] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    provenance: dict[str, Provenance] = Field(default_factory=dict)
    error: str | None = None  # the extraction agent gave no answer


# ---------------------------------------------------------------------------
# Part 3: multi-agent supervisor
# ---------------------------------------------------------------------------

AgentName = Literal["revenue", "expenditure"]


class Route(BaseModel):
    """The supervisor's decision. `reasoning` comes before `next` so it reasons before it routes."""

    reasoning: str = Field(
        description="Two or three sentences: what the query still needs, given the reports so far, "
        "and which agent (if any) can supply it."
    )
    next: Literal["revenue", "expenditure", "finish"] = Field(
        description="The agent to call next, or 'finish' when the reports cover the query, the reports "
        "say the document doesn't have what's left, or the query is off-topic."
    )
    task: str = Field(
        description="When calling an agent: a self-contained instruction saying exactly what to find, "
        "for which fiscal year(s). The agent sees only this, not the query or other reports. Empty when finishing."
    )


class Figure(BaseModel):
    """A number an agent read (or summed) from the document, with where it came from."""

    label: str = Field(description="What the number is, e.g. 'Corporate Income Tax, Estimated FY2024'.")
    value_bn: float = Field(description="The value in $ billions ($ million figures divided by 1,000).")
    block_ids: list[str] = Field(description="Id(s) of the block(s) it was read from, e.g. ['t026'].")
    pages: list[int] = Field(description="Page(s) of those blocks.")


class AgentReport(BaseModel):
    answer: str = Field(description="A concise answer to the task, citing pages like (p. 16).")
    figures: list[Figure] = Field(description="Every number the answer relies on.")
    gaps: str = Field(description="What the task asked for that the document doesn't say. Empty if none.")


class SupervisorDecision(BaseModel):
    kind: Literal["decision"] = "decision"
    reasoning: str
    next: Literal["revenue", "expenditure", "finish"]
    task: str


class WorkerRun(BaseModel):
    kind: Literal["worker"] = "worker"
    agent: AgentName
    task: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    report: AgentReport


class Part3Result(BaseModel):
    source_pdf: str
    fiscal_year: int
    query: str
    steps: list[SupervisorDecision | WorkerRun] = Field(default_factory=list)
    final_answer: str

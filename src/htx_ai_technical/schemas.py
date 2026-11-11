"""Pydantic models shared by every part: the parsed document, LLM output schemas, results."""

from __future__ import annotations

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

"""Field groups: what to extract, and where in the document to find it.

Each group is one LLM call over one source block. Its fields' descriptions are the field
definitions the LLM sees. The class variables say how to find the source block and are
never sent to the LLM:
- `anchor`: title pattern; `*` matches any text, `{fy}` is the document's fiscal year and
  `{fy-1}` the year before it (see `retrieval.anchor_matcher`).
- `block_type`: whether the source is a table or a section.

Descriptions say "the fiscal year" rather than a year: the prompt states the document's
fiscal year, read from its cover.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from .retrieval import BlockType
from .schemas import TableValue, TextList


class CorporateIncomeTax(BaseModel):
    """Corporate Income Tax for the fiscal year, from the 'Budget for FY<year>' summary table."""

    anchor: ClassVar[str] = "Budget for FY{fy}"
    block_type: ClassVar[BlockType] = "table"

    corporate_income_tax_bn: TableValue = Field(
        description="Corporate Income Tax revenue for the fiscal year, in $ billions. The fiscal "
        "year's only figures are its Estimated column (e.g. 'Estimated FY2024' in the FY2024 "
        "document); do not use the Revised figure for the previous year or any Actual figure."
    )
    corporate_income_tax_yoy_pct: TableValue = Field(
        description="Year-on-year % change of Corporate Income Tax: the table's '% change' of the "
        "fiscal year's Estimated figure over the previous year's Revised figure, exactly as "
        "printed ('(3.2)' is -3.2). Do not compute it."
    )


class TopUps(BaseModel):
    """Top-ups for the fiscal year."""

    anchor: ClassVar[str] = "Top-ups * in FY{fy}"
    block_type: ClassVar[BlockType] = "table"

    total_topups_bn: TableValue = Field(
        description="Total Top-ups in the fiscal year, in $ billions: the Total row of the "
        "'Top-ups ... in FY<year>' table (the funds it names vary by year). If the table is in "
        "$ million, divide by 1000 ('20,352' is 20.352)."
    )


class OperatingRevenueTaxes(BaseModel):
    """Revenue lines named in the Operating Revenue update for the previous fiscal year."""

    anchor: ClassVar[str] = "Operating Revenue"
    block_type: ClassVar[BlockType] = "section"

    operating_revenue_taxes: TextList = Field(
        description="Every revenue line (tax, duty, levy or premium category) that the Operating "
        "Revenue section names as a driver of the change in Operating Revenue, e.g. 'Corporate "
        "Income Tax' or 'Vehicle Quota Premiums'. Use the names exactly as written and list each "
        "once. Exclude sub-components mentioned only as part of another line, e.g. 'Foreign "
        "Worker Levy' (part of 'Other Taxes') or 'Casino Taxes' (part of 'Betting Taxes')."
    )


class FiscalPosition(BaseModel):
    """The latest Actual Overall Fiscal Position, from the two-year fiscal position table."""

    anchor: ClassVar[str] = "Fiscal Position in FY{fy-2} and FY{fy-1}"
    block_type: ClassVar[BlockType] = "table"

    latest_actual_fiscal_position_bn: TableValue = Field(
        description="The OVERALL FISCAL POSITION in the most recent column labelled 'Actual' (not "
        "Estimated or Revised), in $ billions. The column label you return must include its "
        "fiscal year, e.g. 'Actual FY2022'."
    )


PART1_GROUPS: list[type[BaseModel]] = [CorporateIncomeTax, TopUps, OperatingRevenueTaxes, FiscalPosition]

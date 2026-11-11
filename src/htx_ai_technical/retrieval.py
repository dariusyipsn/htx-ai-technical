"""Find blocks in a parsed document by their titles."""

from __future__ import annotations

import re
from typing import Literal

from .schemas import Block, ParsedDocument

BlockType = Literal["section", "table"]

_PLACEHOLDER = re.compile(r"\{fy([+-]\d+)?\}")


def anchor_matcher(pattern: str, fiscal_year: int):
    """Compile an anchor pattern into a predicate on titles (case-insensitive, matched anywhere).

    `*` matches any text. `{fy}` is the document's fiscal year and `{fy-2}` that year minus
    two, so for FY2024 "Fiscal Position in FY{fy-2} and FY{fy-1}" matches "... FY2022 and FY2023".
    """
    pattern = _PLACEHOLDER.sub(lambda m: str(fiscal_year + int(m.group(1) or 0)), pattern)
    regex = ".*?".join(r"\s+".join(map(re.escape, part.split())) for part in pattern.split("*"))
    compiled = re.compile(regex, re.IGNORECASE)
    return lambda title: compiled.search(" ".join(title.split())) is not None


def locate(doc: ParsedDocument, pattern: str, block_type: BlockType) -> list[Block]:
    """Blocks of `block_type` whose title matches the anchor `pattern` for the document's fiscal
    year (see `anchor_matcher`), in document order."""
    title_ok = anchor_matcher(pattern, doc.fiscal_year)
    return [b for b in doc.blocks if b.type == block_type and title_ok(b.title)]


def render_block(block: Block) -> str:
    return f"{block.describe()}\n{block.text}"

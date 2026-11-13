"""Find blocks in a parsed document by their titles."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from .schemas import Block, ParsedDocument

BlockType = Literal["section", "table"]

# The fiscal-year placeholder in an anchor pattern: "{fy}", "{fy-1}", "{fy+1}", ...
_PLACEHOLDER = re.compile(r"\{fy([+-]\d+)?\}")


def anchor_matcher(pattern: str, fiscal_year: int) -> Callable[[str], bool]:
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
    """A block as the LLM reads it: its description line, then its text."""
    return f"{block.describe()}\n{block.text}"


# ---------------------------------------------------------------------------
# Lookups for the Part 3 agents (wrapped as tools in `supervisor.py`)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")  # a keyword, in lowercased text
_SENTENCE_OR_LINE = re.compile(r"\n|(?<=\.)\s+")  # where to split a block's text into sentences or table rows
SEARCH_RESULTS = 5  # how many blocks `search` returns
TITLE_WEIGHT = 3  # a keyword in a block's title counts as this many in its text
SNIPPET_CHARS = 300  # how much of the best-matching sentence or row `search` shows


def list_blocks(doc: ParsedDocument) -> str:
    """The document's table of contents: one line per block with its id, type, title and pages."""
    return "\n".join(b.describe() for b in doc.blocks)


def read_blocks(doc: ParsedDocument, block_ids: list[str]) -> str:
    """The full text of each block (a table as markdown), headed by its description, in the order
    asked. An unknown id gets a note in its place, so the other blocks are still returned."""
    if not block_ids:
        raise ValueError("Give at least one block id.")
    return "\n\n".join(
        render_block(block) if (block := doc.blocks_by_id.get(i)) else f"[{i}] No block with this id. Use list_blocks to see the ids."
        for i in block_ids
    )


def search(doc: ParsedDocument, keywords: str) -> str:
    """The `SEARCH_RESULTS` blocks that best match `keywords`, each with its best-matching sentence
    or table row. Blocks are ranked by how many of the keywords they contain, then by how often
    (a keyword in the title counts `TITLE_WEIGHT` times)."""
    terms = set(_WORD.findall(keywords.lower()))
    if not terms:
        raise ValueError("Give at least one keyword.")

    def score(title: str, text: str) -> tuple[int, int]:
        """(distinct keywords found, weighted number of occurrences)."""
        title_words, text_words = _WORD.findall(title.lower()), _WORD.findall(text.lower())
        found = terms & {*title_words, *text_words}
        return len(found), sum(TITLE_WEIGHT * (w in terms) for w in title_words) + sum(w in terms for w in text_words)

    ranked = sorted(doc.blocks, key=lambda b: score(b.title, b.text), reverse=True)
    lines = []
    for b in ranked[:SEARCH_RESULTS]:
        if score(b.title, b.text)[0] == 0:
            break
        best = max(_SENTENCE_OR_LINE.split(b.text), key=lambda part: score("", part))
        lines.append(f"{b.describe()}\n    {best[:SNIPPET_CHARS]}")
    return "\n".join(lines) or f"No block mentions {keywords!r}."

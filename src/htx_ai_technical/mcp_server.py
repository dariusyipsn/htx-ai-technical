"""Local MCP server with the Part 2 date-normalisation tool, run over stdio:

    python -m htx_ai_technical.mcp_server

The tool is deterministic: parsing happens here, in code. The LLM finds the dates, decides when
to call the tool and what to pass, and reports its output.
"""

from __future__ import annotations

from datetime import datetime

from dateutil import parser
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dates", log_level="WARNING")  # its INFO logs would interleave with the CLI's output

# Two different defaults: a component the text leaves out comes back different in each parse.
_DEFAULTS = (datetime(1, 1, 1), datetime(2, 2, 2))


@mcp.tool()
def normalize_date(date_text: str) -> str:
    """Convert a date written in text (e.g. '16 February 2024') to ISO 8601 (YYYY-MM-DD).
    Pass only the date itself, not the sentence around it."""
    first, second = (parser.parse(date_text, dayfirst=True, default=d) for d in _DEFAULTS)
    if first != second:
        raise ValueError(f"{date_text!r} is not a full date: it needs a day, a month and a year.")
    return first.date().isoformat()


if __name__ == "__main__":
    mcp.run()

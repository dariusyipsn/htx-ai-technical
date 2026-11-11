"""Entry points. Each part has a `run_partN()` for notebooks and a CLI choice."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ConfigError, load_config
from .extraction import extract, field_answers
from .fields import PART1_GROUPS
from .llm import get_llm
from .parsing import load_document
from .schemas import ParsedDocument, Part1Result

@dataclass
class Context:
    """What every part starts from: the parsed document and its fiscal year."""

    doc: ParsedDocument
    fiscal_year: int


def prepare(config: Config, pdf_path: Path) -> Context:
    """Load the (cached) parsed document at `pdf_path`. The fiscal year is detected from the PDF."""
    doc = load_document(pdf_path, config.cache_dir)
    if doc.fiscal_year is None:
        raise ConfigError(f"Could not detect the fiscal year of {pdf_path.name} from its first page.")
    return Context(doc, doc.fiscal_year)


def run_part1(config: Config, pdf_path: Path) -> Part1Result:
    """Extract the Part 1 fields from `pdf_path` and write them, with provenance, to `outputs/part1_fy<year>.json`."""
    llm = get_llm(config.llm)  # fail fast on key/provider errors, before parsing
    ctx = prepare(config, pdf_path)

    values, provenance = {}, {}
    for group_result in extract(ctx.doc, PART1_GROUPS, llm, ctx.fiscal_year).values():
        for name, (value, prov) in field_answers(group_result).items():
            values[name], provenance[name] = value, prov
    if values.get("operating_revenue_taxes") is None:
        values["operating_revenue_taxes"] = []

    result = Part1Result(source_pdf=ctx.doc.source, fiscal_year=ctx.fiscal_year, **values, provenance=provenance)
    out = config.output_dir / f"part1_fy{ctx.fiscal_year}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_part1(result: Part1Result, config: Config) -> None:
    print(f"\n{result.source_pdf} (FY{result.fiscal_year})\n")
    for name, prov in result.provenance.items():
        value = getattr(result, name)
        where = f"{prov.block_title} p.{','.join(map(str, prov.pages))}" if prov.block_title else "-"
        print(f"{name}: {value}\n     source:   {where}\n     evidence: {prov.evidence}")
        if prov.error:
            print(f"     error:    {prov.error}")
    print(f"\nSaved to {config.output_dir / f'part1_fy{result.fiscal_year}.json'}")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="htx-ai-technical")
    parser.add_argument("part", choices=["part1"], help="which part to run")
    parser.add_argument("--pdf", required=True, type=Path, help="path to the PDF to extract from")

    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("htx_ai_technical").setLevel(logging.INFO)
    try:
        config = load_config()
        if args.part == "part1":
            result = run_part1(config, args.pdf)
            _print_part1(result, config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0

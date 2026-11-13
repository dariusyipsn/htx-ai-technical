"""Entry points: a `run_partN()` for each part, and the CLI that calls them."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

from .config import Config, ConfigError, load_config
from .dates import DateResults, date_provenance, find_and_classify
from .extraction import extract, field_answers, locate_block
from .fields import PART1_GROUPS, PART2_GROUPS
from .llm import get_llm
from .parsing import load_document
from pydantic import BaseModel

from .schemas import ParsedDocument, Part1Result, Part2Result, Part3Result, Provenance, WorkerRun
from .supervisor import build_graph


def prepare(config: Config, pdf_path: Path) -> ParsedDocument:
    """What every part starts from: the (cached) parsed document at `pdf_path`, with the fiscal
    year detected from its cover."""
    doc = load_document(pdf_path, config.cache_dir)
    if doc.fiscal_year is None:
        raise ConfigError(f"Could not detect the fiscal year of {pdf_path.name} from its first page.")
    return doc


def save(config: Config, filename: str, result: BaseModel) -> None:
    """Write `result` as JSON to `<output_dir>/<filename>`."""
    out = config.output_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2))


def run_part1(config: Config, pdf_path: Path) -> Part1Result:
    """Extract the Part 1 fields from `pdf_path` and write them, with provenance, to `outputs/part1_fy<year>.json`."""
    llm = get_llm(config.llm)  # fail fast on key/provider errors, before parsing
    doc = prepare(config, pdf_path)

    values, provenance = {}, {}
    for group_result in extract(doc, PART1_GROUPS, llm):
        for name, (value, prov) in field_answers(group_result).items():
            values[name], provenance[name] = value, prov
    if values["operating_revenue_taxes"] is None:  # the one field that is a list, not nullable
        values["operating_revenue_taxes"] = []

    result = Part1Result(source_pdf=doc.source, fiscal_year=doc.fiscal_year, **values, provenance=provenance)
    save(config, part1_output_name(result), result)
    return result


def run_part2(config: Config, pdf_path: Path, reference_date: date) -> Part2Result:
    """Find the distribution and estate duty dates in `pdf_path`, normalise them and classify them
    against `reference_date` with the local MCP tools, and write `outputs/part2_<pdf-name>_ref<reference-date>.json`."""
    llm = get_llm(config.llm)
    doc = prepare(config, pdf_path)

    blocks = {g: locate_block(doc, g) for g in PART2_GROUPS}
    found = {g: b for g, b in blocks.items() if b}
    results = find_and_classify(llm, found, doc.fiscal_year, reference_date) if found else DateResults()

    result = Part2Result(
        source_pdf=doc.source,
        fiscal_year=doc.fiscal_year,
        reference_date=reference_date.isoformat(),
        normalized_dates=[d.normalized_date for d in results.normalized],
        dates=results.dates,
        tool_calls=results.tool_calls,
        provenance=date_provenance(blocks, results.normalized),
        error=results.error,
    )
    save(config, part2_output_name(result), result)
    return result


def run_part3(config: Config, pdf_path: Path, query: str) -> Part3Result:
    """Answer `query` about `pdf_path` with the supervisor and its Revenue and Expenditure agents,
    and write the answer and trace to `outputs/part3_fy<year>_<query-slug>.json`."""
    llm = get_llm(config.llm)
    doc = prepare(config, pdf_path)

    state = build_graph(llm, doc).invoke({"query": query, "fiscal_year": doc.fiscal_year, "steps": []})
    result = Part3Result(
        source_pdf=doc.source,
        fiscal_year=doc.fiscal_year,
        query=query,
        steps=state["steps"],
        final_answer=state["final_answer"],
    )
    save(config, part3_output_name(result), result)
    return result


def part1_output_name(result: Part1Result) -> str:
    return f"part1_fy{result.fiscal_year}.json"


def part2_output_name(result: Part2Result) -> str:
    """One file per PDF and reference date, so runs with different reference dates don't overwrite each other."""
    return f"part2_{Path(result.source_pdf).stem}_ref{result.reference_date}.json"


def part3_output_name(result: Part3Result) -> str:
    """One file per fiscal year and query, named after the query's first 8 words."""
    slug = "-".join(re.findall(r"[a-z0-9]+", result.query.lower())[:8])
    return f"part3_fy{result.fiscal_year}_{slug}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _source(prov: Provenance) -> str:
    """A provenance's block title and pages, e.g. "Table 2.1 Budget for FY2024 p.16"."""
    return f"{prov.block_title} p.{','.join(map(str, prov.pages))}" if prov.block_title else "-"


def _print_part1(result: Part1Result, config: Config) -> None:
    print(f"\n{result.source_pdf} (FY{result.fiscal_year})\n")
    for name, prov in result.provenance.items():
        print(f"{name}: {getattr(result, name)}\n     source:   {_source(prov)}\n     evidence: {prov.evidence}")
        if prov.error:
            print(f"     error:    {prov.error}")
    print(f"\nSaved to {config.output_dir / part1_output_name(result)}")


def _print_part2(result: Part2Result, config: Config) -> None:
    print(f"\n{result.source_pdf} (FY{result.fiscal_year}), reference date {result.reference_date}\n")
    print("Found:")
    for name, prov in result.provenance.items():
        print(f"  {name}: {prov.evidence}\n     source: {_source(prov)}")
        if prov.error:
            print(f"     error:  {prov.error}")
    print("\nTool calls:")
    for call in result.tool_calls:
        print(f"  {call.tool}({call.args}) -> {' '.join(call.output.split())}")
    if result.error:
        print(f"\nError: {result.error}")
    print(f"\nNormalized dates: {result.normalized_dates}\n")
    print(json.dumps([d.model_dump() for d in result.dates], indent=2))
    print(f"\nSaved to {config.output_dir / part2_output_name(result)}")


def _print_part3(result: Part3Result, config: Config) -> None:
    print(f"\n{result.source_pdf} (FY{result.fiscal_year})\nQuery: {result.query}\n")
    for step in result.steps:
        if isinstance(step, WorkerRun):
            print(f"  [{step.agent} agent]")
            for call in step.tool_calls:
                output = " ".join(call.output.split())
                print(f"    {call.tool}({call.args}) -> {output[:120]}{'...' if len(output) > 120 else ''}")
            print(f"    report: {step.report.answer}")
            for f in step.report.figures:
                print(f"      - {f.describe()}")
            if step.report.gaps:
                print(f"    gaps: {step.report.gaps}")
        else:
            print(f"[supervisor -> {step.next}] {step.reasoning}")
            if step.task:
                print(f"    task: {step.task}")
        print()
    print(f"Final answer:\n{result.final_answer}")
    print(f"\nSaved to {config.output_dir / part3_output_name(result)}")


def cli() -> int:
    """Run the part named on the command line and print its result. Returns the exit code."""
    parser = argparse.ArgumentParser(prog="htx-ai-technical")
    parser.add_argument("part", choices=["part1", "part2", "part3"], help="which part to run")
    parser.add_argument("--pdf", required=True, type=Path, help="path to the PDF to extract from")
    parser.add_argument(
        "--reference-date", type=date.fromisoformat, help="date to classify Part 2 dates against (YYYY-MM-DD); required for part2"
    )
    parser.add_argument("--query", help="question for the Part 3 supervisor; required for part3")

    args = parser.parse_args()
    if args.part == "part2" and args.reference_date is None:
        parser.error("--reference-date is required for part2")
    if args.part == "part3" and not args.query:
        parser.error("--query is required for part3")
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("htx_ai_technical").setLevel(logging.INFO)  # shows the LLM call logs
    try:
        config = load_config()
        if args.part == "part1":
            result = run_part1(config, args.pdf)
            _print_part1(result, config)
        elif args.part == "part2":
            result = run_part2(config, args.pdf, args.reference_date)
            _print_part2(result, config)
        else:
            result = run_part3(config, args.pdf, args.query)
            _print_part3(result, config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0

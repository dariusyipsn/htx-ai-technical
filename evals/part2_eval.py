"""Part 2 evals: run `run_part2` on every case in `evals/part2_cases.json` and score it.

`normalized_dates` passes when it holds the same dates in any order; `statuses` (normalised
date -> status) must match exactly. Results are printed and written to
`<output_dir>/evals/part2_eval.json`.

    uv run python -m evals.part2_eval
"""

from __future__ import annotations

import json
import logging
from datetime import date

from evals.part1_eval import CaseScore, FieldScore, matches, print_report, save_report
from htx_ai_technical.config import Config, load_config
from htx_ai_technical.main import run_part2
from htx_ai_technical.schemas import Part2Result

CASES_FILE = "evals/part2_cases.json"  # relative to the repo root


def actual_fields(result: Part2Result) -> dict:
    """The parts of a result that the cases score."""
    return {
        "fiscal_year": result.fiscal_year,
        "normalized_dates": result.normalized_dates,
        "statuses": {d.normalized_date: d.status for d in result.dates},
    }


def run_part2_eval(config: Config) -> list[CaseScore]:
    """Score every case. A case whose run raises is recorded with its error, not raised."""
    cases = json.loads(config.resolve(CASES_FILE).read_text())["cases"]
    scores = []
    for case in cases:
        score = CaseScore(pdf_path=case["pdf_path"])
        try:
            result = run_part2(config, config.resolve(case["pdf_path"]), date.fromisoformat(case["reference_date"]))
            actual = actual_fields(result)
        except Exception as exc:
            score.error = f"{type(exc).__name__}: {exc}"
            actual = {}
        score.fields = [
            FieldScore(field=f, expected=v, actual=actual.get(f), passed=f in actual and matches(v, actual[f]))
            for f, v in case["expected"].items()
        ]
        scores.append(score)
    save_report(config, "part2_eval.json", scores)
    return scores


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("htx_ai_technical").setLevel(logging.INFO)
    print_report(run_part2_eval(load_config()))

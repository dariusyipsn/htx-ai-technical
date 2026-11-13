"""Part 1 evals: run `run_part1` on every case in `evals/part1_cases.json` and score each field.

Numbers pass within `TOLERANCE`; the taxes list passes when it names the same taxes in any
order (case-insensitive). Results are printed and written to `<output_dir>/evals/part1_eval.json`.

    uv run python -m evals.part1_eval
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from htx_ai_technical.config import Config, load_config
from htx_ai_technical.main import run_part1

CASES_FILE = "evals/part1_cases.json"  # relative to the repo root
TOLERANCE = 0.005  # a number passes within this of the expected value: half a unit in the 2nd decimal place


class FieldScore(BaseModel):
    """One expected value compared with what the run returned."""

    field: str
    expected: Any
    actual: Any
    passed: bool


class CaseScore(BaseModel):
    """One case's field scores. `pdf_path` names the case in the report."""

    pdf_path: str
    fields: list[FieldScore] = []
    error: str | None = None  # the run itself failed; every field counts as failed

    @property
    def passed(self) -> int:
        return sum(f.passed for f in self.fields)


def matches(expected: Any, actual: Any) -> bool:
    """Lists match as sets of names, numbers within `TOLERANCE`, anything else (e.g. null) exactly."""
    if isinstance(expected, list):
        return isinstance(actual, list) and {_key(x) for x in expected} == {_key(x) for x in actual}
    if isinstance(expected, (int, float)) and actual is not None:
        return abs(expected - actual) <= TOLERANCE
    return expected == actual


def _key(text: str) -> str:
    """A name with case and extra spaces ignored."""
    return " ".join(text.split()).casefold()


def run_part1_eval(config: Config) -> list[CaseScore]:
    """Score every case. A case whose run raises is recorded with its error, not raised."""
    cases = json.loads(config.resolve(CASES_FILE).read_text())["cases"]
    scores = []
    for case in cases:
        score = CaseScore(pdf_path=case["pdf_path"])
        try:
            actual = run_part1(config, config.resolve(case["pdf_path"])).model_dump()
        except Exception as exc:
            score.error = f"{type(exc).__name__}: {exc}"
            actual = {}
        score.fields = [
            FieldScore(field=f, expected=v, actual=actual.get(f), passed=f in actual and matches(v, actual[f]))
            for f, v in case["expected"].items()
        ]
        scores.append(score)
    save_report(config, "part1_eval.json", scores)
    return scores


def save_report(config: Config, filename: str, scores: list[CaseScore]) -> None:
    """Write the scores to `<output_dir>/evals/<filename>`."""
    out = config.output_dir / "evals" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s.model_dump() for s in scores], indent=2))


def print_report(scores: list[CaseScore]) -> None:
    for score in scores:
        print(f"\n{score.pdf_path}: {score.passed}/{len(score.fields)}")
        if score.error:
            print(f"  error: {score.error}")
        for f in score.fields:
            print(f"  {'PASS' if f.passed else 'FAIL'} {f.field}: expected {f.expected!r}, got {f.actual!r}")
    passed, total = sum(s.passed for s in scores), sum(len(s.fields) for s in scores)
    print(f"\nTotal: {passed}/{total} fields passed")


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("htx_ai_technical").setLevel(logging.INFO)
    print_report(run_part1_eval(load_config()))

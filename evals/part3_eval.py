"""Part 3 evals: run `run_part3` on every case in `evals/part3_cases.json` and score it.

Per case, `agents` (the agents the supervisor called) must match in any order, and each expected
figure ($bn) must be within ±0.05 of a figure in some agent report. The final answer's wording
isn't scored. Results are printed and written to `<output_dir>/evals/part3_eval.json`.

    uv run python -m evals.part3_eval
"""

from __future__ import annotations

import json
import logging

from evals.part1_eval import CaseScore, FieldScore, _print_report, matches
from htx_ai_technical.config import Config, load_config
from htx_ai_technical.main import run_part3
from htx_ai_technical.schemas import Part3Result, WorkerRun

CASES_FILE = "evals/part3_cases.json"
FIGURE_TOLERANCE = 0.05


def score_case(expected: dict, result: Part3Result) -> list[FieldScore]:
    runs = [s for s in result.steps if isinstance(s, WorkerRun)]
    agents = sorted({r.agent for r in runs})
    figures = [f.value_bn for r in runs for f in r.report.figures]
    scores = [FieldScore(field="agents", expected=expected["agents"], actual=agents, passed=matches(expected["agents"], agents))]
    for value in expected["figures"]:
        closest = min(figures, key=lambda f: abs(f - value), default=None)
        passed = closest is not None and abs(closest - value) <= FIGURE_TOLERANCE
        scores.append(FieldScore(field=f"figure {value:g}", expected=value, actual=closest, passed=passed))
    return scores


def run_part3_eval(config: Config) -> list[CaseScore]:
    """Score every case. A case whose run raises is recorded with its error, not raised."""
    cases = json.loads(config.resolve(CASES_FILE).read_text())["cases"]
    scores = []
    for case in cases:
        score = CaseScore(pdf_path=f"{case['pdf_path']} | {case['query']}")
        try:
            result = run_part3(config, config.resolve(case["pdf_path"]), case["query"])
        except Exception as exc:
            score.error = f"{type(exc).__name__}: {exc}"
            score.fields = [FieldScore(field="agents", expected=case["expected"]["agents"], actual=None, passed=False)] + [
                FieldScore(field=f"figure {v:g}", expected=v, actual=None, passed=False) for v in case["expected"]["figures"]
            ]
        else:
            score.fields = score_case(case["expected"], result)
        scores.append(score)

    out = config.output_dir / "evals" / "part3_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s.model_dump() for s in scores], indent=2))
    return scores


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("htx_ai_technical").setLevel(logging.INFO)
    _print_report(run_part3_eval(load_config()))

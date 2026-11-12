# Part 1 Plan — Document Extraction & Prompt Engineering

## Goal
- Extract 5 fields from a Singapore MOF "Analysis of Revenue and Expenditure" PDF, for any year (`data/` has FY2024–FY2026). Nothing may depend on hardcoded page numbers or years.
- `run_part1()` in `src/htx_ai_technical/main.py` must be callable from a notebook or the CLI.
- Parts 2 and 3 reuse the same parsed document, LLM factory, extractor and retrieval functions.
- Whoever receives the repo only edits `config.yaml`, and passes the PDF on the command line (`--pdf`).

| Field | FY2024 source | Expected |
|---|---|---|
| Corporate Income Tax, FY2024 ($bn) | Table 2.1 | 28.03 |
| Corporate Income Tax YoY % | Table 2.1 | -1.2 |
| Total top-ups FY2024 ($bn) | Table 2.4 | 20.352 |
| Taxes in "Operating Revenue" section | §1.2 | 7 named revenue lines (CIT, Other Taxes, Vehicle Quota Premiums, PIT, Assets Taxes, Betting Taxes, GST) |
| Latest actual fiscal position ($bn) | Table 1.1, Actual FY2022 | 1.72 |

## Design decisions
- **LangChain for Parts 1–2, LangGraph for Part 3.**
  - Part 1 is a linear pipeline, and an LCEL chain (`prompt | llm.with_structured_output(Schema)`) covers it.
  - Part 3 needs routing and shared state, which is what LangGraph is for. LangGraph builds on LangChain's components, so everything carries over.
- **Model-agnostic code:** `init_chat_model(model, model_provider=provider)` is the only way the code reaches a model. The API key comes only from the `API_KEY` environment variable, for every provider, never from config. No provider package ships with the project; the configured provider's package is installed on first use.
  - For any other provider, `get_llm` looks up its package in an allowlist (`PROVIDER_PACKAGES`), installs it on first use (`uv pip install` or `pip`) and retries.
  - Setting `llm.auto_install: false` disables this.
  - `pyproject.toml` is never modified.
- **Parsing by structure, using Docling:** the PDF is parsed into heading-delimited section blocks and table blocks, each recording its pages.
  - Docling's layout model and TableFormer recover the MOF tables, which have almost no ruled lines, as real grids.
  - Sections and tables that span pages stay whole.
  - Page headers and footers are dropped.
  - The cost (heavy install, roughly a minute per PDF) is paid once, because the result is cached.
  - Fallback: pymupdf4llm. Rejected alternatives: pdfplumber (needs per-layout tuning), MarkItDown (loses structure and page information).
- **LLM call logging:** `get_llm` attaches a LangChain callback handler (`LLMCallLogger`) to the model, so every call logs provider, model, latency, input/output/cached/reasoning tokens, cost and finish reason. Cost uses `input_cost_per_million` and `output_cost_per_million` from `config.yaml`.
- **Evals:** `evals/part1_cases.json` holds a PDF path and expected answers per case. `evals/part1_eval.py` runs Part 1 on each and scores every field: numbers within ±0.005, the taxes list in any order (ignoring case).
- **Units:** all money values in $ billions, which matches the field wording and the headline tables. $million tables are converted.
- **The taxes list** contains named revenue lines only, without sub-components such as Foreign Worker Levy.

## Pipeline
```
PDF ─▶ parse (Docling) + cache ─▶ blocks ─▶ locate ─▶ extract (LLM) ─▶ Part1Result + outputs/part1_fy<year>.json
```
1. **Parse (`parsing.py`):**
   - The output is `Block{id, type: section|table, title, section_path, pages, text, grid}`.
   - A table on the next page with no title and the same column count is merged into the previous table.
   - `fiscal_year` is read from the first page ("Financial Year 2024"). It can't be overridden.
   - The result is cached to `data/processed/<pdf-name>.json` and reused whenever that file exists; delete it to re-parse.
2. **Locate (`retrieval.py`):** an anchor pattern is matched against block titles; the first match in document order wins. `*` matches any text, `{fy}` is the document's fiscal year, `{fy-1}` the year before it.
   - Anchors:
     - `Budget for FY{fy}`
     - `Top-ups * in FY{fy}`
     - `Operating Revenue`
     - `Fiscal Position in FY{fy-2} and FY{fy-1}`
   - Anchors only, no fallback: each anchor must match in every edition. If a later edition rewords a title, its anchor in `fields.py` is updated.
3. **Extract (`extraction.py`):** one LCEL call per field group (4 in total, run with `.batch()`), with structured output.
   - Table fields return `{value, row_label, column_label, evidence}`.
   - Text fields return `{value, evidence[]}`.
   - The prompt (`prompts/extraction.yaml`) includes:
     - a role;
     - field definitions, taken from the Pydantic field descriptions in `fields.py` (year-free; the prompt states the fiscal year);
     - rules: "(x)" means negative, convert to $bn, null instead of guessing, no computing;
     - grounding: the LLM names the row and column it read and quotes its evidence;
     - temperature 0.
   - Each call is tagged with its `field_group` in the call metadata, so the LLM call log shows which group it was.
4. **Provenance (`extraction.py`, `field_answers`):**
   - Each field's answer is output as the LLM returned it. It is not re-checked against the document.
   - Alongside it goes `provenance[field] = {block_id, block_title, pages, row, column, evidence, error}`, so a person can review where the value came from.
   - A failure (no block found, LLM call failed) is recorded in `error`, not raised.

## Reuse in later parts
- **Part 2:** reuses `load_document` and `extract` with a date schema covering the cover date and the estate-duty sentence.
- **Part 3:** block-lookup functions are added to `retrieval.py`, taking plain arguments and returning text, so they can be wrapped as `@tool`.
  - The Revenue and Expenditure agents share these tools and have different prompts.
  - The Revenue Agent answers open-ended revenue questions; it isn't a re-run of Part 1.

## Configuration (`config.yaml`, the only file a user edits)
```yaml
llm:
  provider: google_genai        # google_genai | anthropic | openai | ...
  model: gemini-2.5-flash
  temperature: 0
  auto_install: true            # install the provider's langchain package on first use
  input_cost_per_million: 0.30  # $ per 1M input tokens for this model (logged cost only)
  output_cost_per_million: 2.50 # $ per 1M output tokens, including thinking tokens
paths:
  cache_dir: data/processed
  output_dir: outputs
```
- `cache_dir` and `output_dir` resolve against the repo root, so notebooks in subfolders still work.
- The PDF is not in the config. It is passed to `run_part1(config, pdf_path)`, from the required `--pdf` argument in the CLI, and used as given (relative to the working directory).
- `input_cost_per_million` and `output_cost_per_million` are required; they set the logged cost.
- Errors name what to fix: a missing key, a missing PDF, or an unknown provider (the error lists the valid ones).

## Files (under `src/htx_ai_technical/` unless noted)
- `main.py`:
  - `run_part1(config, pdf_path) -> Part1Result`; everything comes from the `Config` passed in (loaded from `config.yaml` at the repo root), except the PDF, which is passed separately. The CLI takes it from the required `--pdf` argument; a notebook passes it directly;
  - `cli()` with a `part` positional whose choices are `part1`, later `part2` and `part3`.
- `__init__.py`: `main()` delegates to `main.cli()`.
- `config.py`, `llm.py`, `llm_logging.py`, `prompt_loader.py`, `prompts/extraction.yaml` (prompts only).
- `fields.py`: one Pydantic class per field group, with field descriptions and `anchor` / `block_type` class variables.
- `parsing.py`, `retrieval.py`, `schemas.py`, `extraction.py`.
- Root files:
  - `tests/test_retrieval.py`, `tests/test_extraction.py`, `tests/test_parsing.py`, `tests/test_llm_logging.py`, `tests/test_evals.py`: unit tests of individual functions;
  - `evals/part1_cases.json`, `evals/part1_eval.py`: Part 1 evals against expected answers;
  - `config.yaml`;
  - `pyproject.toml`: deps `langchain`, `langchain-core`, `docling`, `pydantic`, `pyyaml`; dev `pytest`, with `pythonpath = ["."]` so tests can import `evals/`; `langgraph` arrives in Part 3;
  - `.gitignore`: `data/processed/`, `outputs/`.
- Root docs:
  - `PLAN.md`: a copy of this plan.
  - `README.md`, with these sections:
    - **Getting started:** prerequisites (Python 3.12, uv, internet on first run), `uv sync`, export the provider's API key variable, choose the provider in `config.yaml`.
    - **How to run:** CLI (`--pdf` required) and notebook usage for Part 1, with placeholders for Parts 2 and 3; where outputs and cache go.
    - **Project structure.**
    - **Architecture:** a pipeline diagram and what each module does.
    - **Design decisions:** LangChain vs LangGraph, parsing choice and rejected alternatives, model-agnostic setup with auto-install, LLM call logging, units, anchors, evidence and provenance.
    - **Configuration reference.**
    - **Assumptions.**
    - **Testing:** unit tests and evals.
    - **Troubleshooting:** first-run Docling download, `uv sync` removing auto-installed packages.

## Implementation order
1. Write `PLAN.md` (this plan) and `README.md` at the repo root.
2. Set up config, the LLM factory and dependencies.
3. Parsing: dump the blocks for all 3 PDFs and check them.
4. Retrieval, schemas, extraction and tests.
5. Wire up `main.py` and the CLI.
6. LLM call logging.
7. Evals.
8. Fill in the README's run and output details.

## Verification
1. Block dumps for all 3 PDFs:
   - Tables 1.1, 2.1 and 2.4 come out with correct grids and headers.
   - §1.2 is a single block covering pages 5–6.
   - Every anchor resolves in every year.
2. `uv run pytest` passes.
3. `uv run htx-ai-technical part1 --pdf data/fy2024_analysis_of_revenue_and_expenditure.pdf` gives the expected FY2024 values, each with evidence and no `error`, and logs one line per LLM call.
4. FY2025 and FY2026 expected answers filled into `evals/part1_cases.json`, and `uv run python -m evals.part1_eval` passes every field.
5. Notebook: `run_part1()` returns an object that matches the JSON output.
6. Fresh-clone hand-off:
   - only `API_KEY` is exported, and the run succeeds;
   - switching to `anthropic` auto-installs the package once and runs;
   - an unknown provider gives a clear error with no install;
   - `auto_install: false` prints the install command;
   - an unset key variable gives a clear error naming it.

---

# Part 3 Plan — Multi-agent supervisor

## Goal
- Answer "What are the key government revenue streams, and how will the Budget for the Future Energy Fund be supported?" (and other queries) with a supervisor that delegates to a Revenue Agent and an Expenditure Agent.
- `run_part3(config, pdf_path, query)` is callable from a notebook or the CLI (`part3 --pdf … --query …`), and returns the final answer with a full trace of the supervisor's decisions.

## Design decisions
- **Hand-built LangGraph `StateGraph`**, not `langgraph-supervisor`. The supervisor is a structured-output call, `Route{reasoning, next, task}`, with `reasoning` first (as in Part 2's `Classification`), so every routing decision is explained in the trace.
- **Workers are `create_agent` tool loops** returning `AgentReport{answer, figures[Figure{label, value_bn, block_ids, pages}], gaps}`. A worker sees only the supervisor's task. A worker failure is recorded under `gaps`, not raised.
- **Tools:** `list_blocks`, `search`, `read_block` (plain functions in `retrieval.py`, wrapped as local `@tool`s, shared by both agents) and `sum_values` (Expenditure Agent only: arithmetic in code). Local tools, not MCP, keep `run_part3` synchronous and notebook-friendly.
- **Keyword search**, not embeddings: about 60 blocks and 15k tokens per document.
- **Separate synthesis node** writes the final answer from the reports only, citing pages and stating gaps.
- **Loop guard:** `MAX_AGENT_CALLS = 4`, then the supervisor finishes without an LLM call.

## Graph
```
START → supervisor ─┬─▶ revenue ─────┐
          ▲         ├─▶ expenditure ─┤
          └─────────┴────────────────┘
                    └─▶ synthesize → END
```
State: `query`, `fiscal_year`, `steps` (appended `SupervisorDecision` / `WorkerRun`), `final_answer`. `steps` is the trace, saved to `outputs/part3_fy<year>_<query-slug>.json` and printed by the CLI.

## Files
- `supervisor.py` (graph, nodes, tools), `retrieval.py` (lookups), `schemas.py` (Part 3 models), `main.py` (`run_part3`, CLI).
- `prompts/supervisor.yaml`, `revenue_agent.yaml`, `expenditure_agent.yaml`, `synthesis.yaml`.
- `evals/part3_cases.json`, `evals/part3_eval.py`: demo queries, scored on routing (set of agents) and key figures (±0.05 $bn).
- `tests/test_retrieval.py`, `tests/test_supervisor.py`, `tests/test_evals.py`.
- `pyproject.toml`: `langgraph` as a direct dependency.

## Assumptions
- "How the FEF is supported" = what the Budget allocates to it, as what and for what (FY2024: $5.0bn initial injection, part of $20.4bn top-ups), plus the revenue that finances the Budget overall. The document earmarks no revenue for the fund, and the answer must say so.
- "Key revenue streams" = the largest Operating Revenue lines, plus NIRC.
- One PDF per run; money in $bn.

## Verification
1. `uv run pytest` passes.
2. The brief's query on FY2024 routes to both agents and the answer cites the $5.0bn FEF injection (p. 18/20) and the main revenue lines.
3. `uv run python -m evals.part3_eval` passes routing and figures for every demo query.

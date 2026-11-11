# HTX AI Engineering Take-Home

This project extracts structured data from Singapore Ministry of Finance (MOF) *Analysis of Revenue and Expenditure* PDFs, then reasons over that data with LLMs. It has three parts:

| Part | What it does | Status |
|---|---|---|
| 1 | Document extraction and prompt engineering (LangChain) | Done |
| 2 | Tool calling (date normalisation over local MCP) and date reasoning | Planned |
| 3 | Multi-agent supervisor with Revenue and Expenditure agents (LangGraph) | Planned |

Each part has a single entry point, `run_partN()`, in `src/htx_ai_technical/main.py`. You can call it from a Jupyter notebook or from the CLI.

---

## Getting started

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An API key for an LLM provider. The default is Google Gemini, which has a free tier.
- Internet access on the first run. Docling downloads its layout models then, and the configured LLM provider's package (including the default, Gemini) is installed then.

### Setup
1. **Install dependencies.**
   ```bash
   uv sync
   ```
2. **Set your API key** in the `API_KEY` environment variable. The same variable is used whatever the provider, and keys are never read from `config.yaml`, so they can't be committed by accident.
   ```bash
   export API_KEY="YOUR_KEY_HERE"
   ```
   In a notebook, set it before calling `run_part1()`: `os.environ["API_KEY"] = "..."`.
3. **Edit `config.yaml`** if needed. It's the only file you need to change, and the defaults work with Gemini:
   ```yaml
   llm:
     provider: google_genai
     model: gemini-2.5-flash
   ```
4. **(Optional) Use a different provider.** Change `provider` and `model` (e.g. `anthropic` with a Claude model) and put that provider's key in `API_KEY`. Keyless providers (`ollama`, `google_vertexai`, `bedrock_converse`) ignore `API_KEY`. The provider's LangChain package installs itself on the first run (see [Model-agnostic LLM setup](#model-agnostic-llm-setup)).

---

## How to run

### Part 1 — Document extraction
CLI (`--pdf` is required):
```bash
uv run htx-ai-technical part1 --pdf data/fy2024_analysis_of_revenue_and_expenditure.pdf
```
Notebook:
```python
from pathlib import Path
from htx_ai_technical.config import load_config
from htx_ai_technical.main import run_part1

config = load_config()
result = run_part1(config, Path("../data/fy2024_analysis_of_revenue_and_expenditure.pdf"))
result.model_dump()
```
The PDF path is used as given, so a relative path is relative to where you run from: `data/...` from the repo root, `../data/...` from a notebook in `notebooks/`.
**Outputs:**
- `outputs/part1_fy<year>.json` holds the extracted fields and their provenance.
- `data/processed/<pdf-name>.json` is the cached parsed document. It's reused by every part whenever it exists; delete it to re-parse (e.g. after replacing the PDF or changing `parsing.py`).

### Part 2 — Tool calling and reasoning
*Coming soon:* `uv run htx-ai-technical part2` / `run_part2()`

### Part 3 — Multi-agent supervisor
*Coming soon:* `uv run htx-ai-technical part3` / `run_part3()`

---

## Project structure
```
config.yaml                     # the only file you edit
data/                           # MOF PDFs; data/processed/ holds the parse cache
outputs/                        # part1_fy<year>.json; evals/part1_eval.json
evals/
  part1_cases.json              # PDF paths + expected answers
  part1_eval.py                 # runs Part 1 on every case and scores it
src/htx_ai_technical/
  main.py                       # run_part1(), prepare(), CLI
  config.py                     # config.yaml -> typed settings, repo-root path resolution
  llm.py                        # get_llm(): init_chat_model + provider auto-install
  llm_logging.py                # LLMCallLogger: per-call latency, tokens and cost
  parsing.py                    # Docling -> section/table blocks, cover block, cache
  retrieval.py                  # locate blocks by anchor pattern
  fields.py                     # field groups: Pydantic classes with field descriptions + anchor
  extraction.py                 # field groups -> LCEL chains -> answers + provenance
  schemas.py                    # Block, ParsedDocument, TableValue/TextList, Part1Result
  prompt_loader.py              # loads YAML prompts
  prompts/extraction.yaml       # system/human prompt only
tests/                          # unit tests, no API key or PDF needed
```

---

## Architecture

### Part 1 pipeline
```
PDF ─▶ parse (Docling) + cache ─▶ blocks ─▶ locate ─▶ extract (LLM) ─▶ Part1Result + JSON
```

| Stage | Module | What it does |
|---|---|---|
| Parse | `parsing.py` | Converts the PDF into **blocks** (see below). Each block records the pages it spans. Detects the fiscal year from the cover. Caches the result keyed by the PDF's hash. |
| Locate | `retrieval.py` | Finds the block for each field group by matching an **anchor pattern** (e.g. `Budget for FY{fy}`) against block titles, taking the first match in document order. Only anchors are used; there is no fallback. |
| Extract | `extraction.py` | One LCEL chain (`prompt \| llm.with_structured_output(Schema)`) per field group, run in parallel with `.batch()`. Each LLM call is logged with its latency, tokens and cost. |

The two kinds of block:
- **Section block:** a heading plus its paragraphs.
- **Table block:** a table as a grid, with its title.

### Fields extracted
| Field | Located via (FY = target year) | FY2024 value |
|---|---|---|
| `corporate_income_tax_bn` | table `Budget for FY{fy}` | 28.03 |
| `corporate_income_tax_yoy_pct` | table `Budget for FY{fy}` | -1.2 |
| `total_topups_bn` | table `Top-ups * in FY{fy}` | 20.352 |
| `operating_revenue_taxes` | first section titled `Operating Revenue` (§1.2) | 7 revenue lines |
| `latest_actual_fiscal_position_bn` | table `Fiscal Position in FY{fy-2} and FY{fy-1}`, Actual column | 1.72 |

In anchor patterns, `*` matches any text, `{fy}` is the document's fiscal year (read from its cover), and `{fy-1}` the year before it. Matching ignores case and extra spaces.

Every field also carries **provenance**: the source block, its pages, the row and column read, and the quoted evidence. Answers are not re-checked against the document: the evidence is there for a person to review.

### How later parts reuse Part 1
- **Part 2** reuses the parsed document and the generic extractor, with a date schema. `extraction.py` knows nothing about Part 1: a *field group* is a Pydantic class in `fields.py`, whose field descriptions are the definitions the LLM reads and whose `anchor` and `block_type` class variables locate its source block. Part 2 adds its own group classes (e.g. a date schema) and calls `extract(doc, PART2_GROUPS, llm, fy)` → `field_answers(...)`. Each `run_partN(config, pdf_path)` takes the loaded config and the PDF, and `main.prepare(config, pdf_path)` gives every part the same cached document and fiscal year.
- **Part 3** gives the Revenue and Expenditure agents block-lookup tools over the same parsed document (to be added to `retrieval.py` then). Each agent answers open-ended questions with its own prompt and focus.

---

## Design decisions

### LangChain vs LangGraph
- **Parts 1–2 use LangChain.** Extraction is a linear pipeline, and an LCEL chain with structured output covers it. A graph would add complexity with no benefit.
- **Part 3 uses LangGraph.** A supervisor needs routing, loops and shared state, which is what LangGraph is for. LangGraph builds on LangChain's components (chat models, tools, messages), so the Part 1 components carry over unchanged.

### Parsing: Docling, by document structure
The PDF is parsed into its logical structure, not page by page.

**Why this matters:**
- A section or table that crosses a page break stays whole.
- The two sections both titled "Operating Revenue" (§1.2 and §2.2) are told apart by their parent heading.

**Why Docling:**
- Its layout model labels headings, paragraphs, tables, captions, and page headers and footers.
- TableFormer rebuilds tables as real grids. The MOF tables have almost no ruled lines, and line-based extractors tend to flatten them.
- Every element records its page, so citations stay accurate.

**Cost:** a heavy install and roughly a minute per PDF on CPU. The cache means this happens once per PDF.

**Cover page:** Docling labels the cover's "Distributed on Budget Day: …" line as a page footer, and page headers and footers are otherwise dropped. So page 1 gets its own `(cover)` block holding all of its text, footers included. The fiscal year is read from it, and Part 2's date extraction relies on it.

**Alternatives considered:**

| Library | Why not chosen |
|---|---|
| pymupdf4llm | Fast, but works page by page and detects tables poorly on these layouts. Kept as a fallback. |
| pdfplumber | Table extraction needs tuning for each layout. |
| MarkItDown | Loses page boundaries and structure. |

### Generalising across years
- No page numbers or years are hardcoded. Anchors are written relative to the fiscal year read from the cover (`{fy}`, `{fy-1}`), so they pick the current year's table even when an earlier table covers the previous year (FY2026 has a `Top-ups ... in FY2025` table before `Top-ups ... in FY2026`).
- The fiscal year is always read from the first page ("Financial Year 2024"); it can't be overridden. It's used in the prompt's field definitions, e.g. which "Estimated FY2024" column to read.
- Anchors are written to match every edition's title wording. There is no fallback, so if a later edition rewords a title, its anchor in `fields.py` must be updated.

### Prompt engineering
- A clear role, plus field definitions written as Pydantic field descriptions in `fields.py`. They name no year ("the fiscal year"); the prompt states the document's fiscal year. The descriptions are listed in the prompt and also travel in the structured-output schema.
- Explicit rules:
  - "(1.2)" means negative.
  - Convert all money to $ billions.
  - Return null rather than guess.
  - Never compute a value the document doesn't state.
- **Grounding:** for table fields, the LLM must name the row and column it read; for every field, it quotes its evidence.
- Pydantic structured output and temperature 0.

### Model-agnostic LLM setup
- All model access goes through LangChain's `init_chat_model(model, model_provider=provider)`, and no provider SDK is imported directly.
- No provider package ships with the project, not even the default's. The configured provider's package (e.g. `langchain-google-genai`) is installed on the first run.
- The API key comes only from the `API_KEY` environment variable, for every provider. `get_llm` reads it, fails early if it's missing, and passes it to `init_chat_model` as `api_key`. Keyless providers (`ollama`, `google_vertexai`, `bedrock_converse`) skip this.
- When `config.yaml` selects another provider, its package is looked up in an allowlist and installed on the first run (via `uv pip`, or `pip` if uv isn't available). `pyproject.toml` is never modified.
- To disable runtime installs, set `llm.auto_install: false`. The required install command is then printed instead.

### LLM call logging
- `get_llm` attaches `LLMCallLogger` (`llm_logging.py`), a LangChain callback handler, to the model. Every call through that model is logged, including inside chains and `with_structured_output`, with no extra code at the call site.
- Each call logs one INFO line: provider, model, call metadata (extraction passes `field_group`), latency, input/output/cached/reasoning tokens, cost and finish reason. A failed call logs a WARNING with its latency and error.
- Cost is input tokens × `input_cost_per_million` plus output tokens × `output_cost_per_million`, from `config.yaml`. Cached-input discounts are not applied.

---

## Configuration reference
```yaml
llm:
  provider: google_genai        # google_genai | anthropic | openai | ... (any init_chat_model provider)
  model: gemini-2.5-flash
  temperature: 0
  auto_install: true            # install the provider's langchain package on first use
  input_cost_per_million: 0.30  # $ per 1M input tokens for this model (logged cost only)
  output_cost_per_million: 2.50 # $ per 1M output tokens, including thinking tokens
paths:
  cache_dir: data/processed
  output_dir: outputs
```
- `input_cost_per_million` and `output_cost_per_million` are required. Update them when you change model, since they set the logged cost (see [LLM call logging](#llm-call-logging)).
- `cache_dir` and `output_dir` resolve against the repository root, so notebooks in subfolders work too.
- The PDF is not in the config: pass it with `--pdf` or as `run_part1(config, pdf_path)`.
- The CLI shows the LLM call logs. In a notebook, turn them on with `logging.basicConfig(); logging.getLogger("htx_ai_technical").setLevel(logging.INFO)`.

---

## Assumptions
- **"Corporate Income Tax in 2024"** means the *Estimated FY2024* figure in Table 2.1. **YoY %** is the change versus *Revised FY2023*, as stated in that table.
- **"Latest Actual Fiscal Position"** is the most recent *Actual* column in Table 1.1, which is FY2022 in the FY2024 document.
- **"Taxes mentioned in Operating Revenue"** refers to the named revenue lines in the update section (§1.2). Sub-components such as Foreign Worker Levy or Casino Taxes are excluded.
- **"Total top-ups in 2024"** is the Total row of the FY2024 top-ups table, converted to $ billions.
- All documents follow the same MOF template, so the same anchors find the source blocks in every year.

---

## Testing
```bash
uv run pytest
```
The tests need no API key and no PDF:
- `tests/test_retrieval.py`: anchor pattern matching, and `locate` returning matches in document order, filtered by block type.
- `tests/test_extraction.py`: `locate_block` taking the first anchor match (or none), the prompt's field definitions, and `field_answers` building provenance for answered and failed groups.
- `tests/test_parsing.py`: heading levels, fiscal-year detection, and table-to-markdown rendering.
- `tests/test_llm_logging.py`: the logged tokens, cost and metadata, and the warning for a failed call, using LangChain's fake chat model.
- `tests/test_evals.py`: the eval's comparison rules (number tolerance, taxes list in any order).

### Evals
Expected answers per PDF live in `evals/part1_cases.json`, with each `pdf_path` relative to the repo root. The eval runs Part 1 on every case (this needs `API_KEY`):
```bash
uv run python evals/part1_eval.py
```
Numbers pass within ±0.005, and the taxes list passes when it names the same taxes in any order (ignoring case). The report is printed and saved to `outputs/evals/part1_eval.json`. A `null` expected value means the field should come back null. A case whose run fails is recorded with its error, and its fields count as failed. Each case also writes its normal `outputs/part1_fy<year>.json`.

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| First run is slow | Docling downloads its models and parses the PDF. Later runs use the cache in `data/processed/`. |
| `Missing API key: set the API_KEY environment variable` | `export API_KEY="..."` with your provider's key. |
| Ollama run fails with a connection error | Start the server (`ollama serve`) and pull the model (`ollama pull <model>`). |
| `the following arguments are required: --pdf` | Pass the PDF: `uv run htx-ai-technical part1 --pdf <path>`. |
| `Unknown provider` | Use one of the providers listed in the error message. |
| A provider package disappears after `uv sync` | `uv sync` removes packages that aren't in the lockfile. The package reinstalls automatically on the next run. |
| A field is null | Check its `error` in the output JSON's provenance: the LLM call failed or returned no value. |

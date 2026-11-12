# HTX AI Engineering Take-Home

This project extracts structured data from Singapore Ministry of Finance (MOF) *Analysis of Revenue and Expenditure* PDFs, then reasons over that data with LLMs. It has three parts:

| Part | What it does | Status |
|---|---|---|
| 1 | Document extraction and prompt engineering (LangChain) | Done |
| 2 | Tool calling (date normalisation over local MCP) and date reasoning | Done |
| 3 | Multi-agent supervisor with Revenue and Expenditure agents (LangGraph) | Done |

Each part has a single entry point, `run_partN()`, in `src/htx_ai_technical/main.py`. Parts 1 and 3 can be run from a Jupyter notebook or from the CLI; Part 2 runs from the CLI only.

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
CLI (`--pdf` and `--reference-date` are required):
```bash
uv run htx-ai-technical part2 --pdf data/fy2024_analysis_of_revenue_and_expenditure.pdf --reference-date 2024-01-01
```
`run_part2` starts the local MCP server itself (as a subprocess over stdio), so there's nothing else to run. It's CLI only: it starts its own event loop with `asyncio.run`, which fails in Jupyter because a notebook already has one running.
**Outputs:** `outputs/part2_fy<year>.json` holds the classified dates, the list of normalised dates, every tool call the LLM made, and the provenance of each date. The CLI prints the same trace: dates found → tool calls → normalised dates → statuses.

Example (FY2024):
```json
[
  {"id": "distribution_date", "original_text": "Distributed on Budget Day: 16 February 2024", "normalized_date": "2024-02-16", "status": "Upcoming", "reasoning": "..."},
  {"id": "estate_duty_date", "original_text": "Estate Duty does not apply to a person who dies after 15 February 2008.", "normalized_date": "2008-02-15", "status": "Expired", "reasoning": "..."}
]
```

### Part 3 — Multi-agent supervisor
CLI (`--pdf` and `--query` are required):
```bash
uv run htx-ai-technical part3 --pdf data/fy2024_analysis_of_revenue_and_expenditure.pdf \
  --query "What are the key government revenue streams, and how will the Budget for the Future Energy Fund be supported?"
```
Notebook:
```python
from pathlib import Path
from htx_ai_technical.config import load_config
from htx_ai_technical.main import run_part3

result = run_part3(load_config(), Path("data/fy2024_analysis_of_revenue_and_expenditure.pdf"), "What do the FY2024 fund top-ups add up to?")
print(result.final_answer)
```
**Outputs:** `outputs/part3_fy<year>_<query-slug>.json` holds the final answer and the full trace (`steps`): every supervisor decision (its reasoning, the agent chosen and the task written for it) and every agent run (its task, tool calls and report with figures and pages). The CLI prints the same trace in order: decision → tool calls → report → … → final answer.

**Demo queries** (all in `evals/part3_cases.json`, run together with `uv run python -m evals.part3_eval`):

| Query | PDF | Agents expected | What it shows |
|---|---|---|---|
| What are the key government revenue streams, and how will the Budget for the Future Energy Fund be supported? | FY2024 | revenue, expenditure | the brief's query: two parts, one per agent, joined up in synthesis |
| Which revenue streams are expected to grow the most in FY2024, and why? | FY2024 | revenue | routing to one agent only |
| What do the FY2024 top-ups to endowment and trust funds add up to, and which fund receives the most? | FY2024 | expenditure | `sum_values`, checked against the table's Total row (20,352) |
| Is FY2024 Operating Revenue enough to cover Total Expenditure, and what closes the gap? | FY2024 | revenue, expenditure | collaboration on one question: deficit, then NIRC and top-ups |
| How much has the Government put into the Future Energy Fund in each year shown, and in total? | FY2026 | expenditure | a sum across years ($5.0bn in FY2024 + $5.0bn in FY2025), with no FY2026 top-up as a gap |
| Who won the 2022 FIFA World Cup? | FY2024 | none | the supervisor finishes without calling an agent |

---

## Project structure
```
config.yaml                     # the only file you edit
data/                           # MOF PDFs; data/processed/ holds the parse cache
outputs/                        # part<N>_fy<year>.json; evals/part<N>_eval.json
evals/
  part1_cases.json              # PDF paths + expected answers
  part1_eval.py                 # runs Part 1 on every case and scores it
  part2_cases.json, part2_eval.py  # the same for Part 2
  part3_cases.json, part3_eval.py  # Part 3 demo queries: expected routing and key figures
src/htx_ai_technical/
  main.py                       # run_part1(), run_part2(), run_part3(), prepare(), CLI
  config.py                     # config.yaml -> typed settings, repo-root path resolution
  llm.py                        # get_llm(): init_chat_model + provider auto-install
  llm_logging.py                # LLMCallLogger: per-call latency, tokens and cost
  parsing.py                    # Docling -> section/table blocks, cover block, cache
  retrieval.py                  # locate blocks by anchor pattern; list/search/read lookups for Part 3
  fields.py                     # field groups: Pydantic classes with field descriptions + anchor
  extraction.py                 # field groups -> LCEL chains -> answers + provenance
  schemas.py                    # Block, ParsedDocument, TableValue/TextList, Part1Result, DateStatus, Part2Result, Route, AgentReport, Part3Result
  mcp_server.py                 # local MCP server: normalize_date
  dates.py                      # Part 2 agent: find, normalise and classify dates with the MCP tools
  supervisor.py                 # Part 3 LangGraph: supervisor, Revenue/Expenditure agents, synthesis
  prompt_loader.py              # loads YAML prompts
  prompts/                      # extraction.yaml (Part 1); date_*.yaml (Part 2); supervisor, *_agent, synthesis.yaml (Part 3)
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

### Part 2 pipeline
```
blocks ─▶ locate (anchors) ─▶ step 1: find dates + normalize_date ─▶ step 2: classify (LLM reasoning) ─▶ Part2Result + JSON
                                                    └─ local MCP server (stdio)
```
1. **Locate** (`fields.py`, reusing `extraction.locate_block`). Each Part 2 field group's anchor finds its source block. The groups' field descriptions say what date to find.

   | Field | Located via | FY2024 text |
   |---|---|---|
   | `distribution_date` | the `(cover)` block | "Distributed on Budget Day: 16 February 2024" |
   | `estate_duty_date` | section `Asset* Taxes` (glossary; "Assets Taxes" up to FY2025, "Asset Taxes" in FY2026) | "Estate Duty does not apply to a person who dies after 15 February 2008." |
2. **Extract and normalise** (`dates.py`, `prompts/date_extraction.yaml`). An agent (`langchain.agents.create_agent`) given only the normalisation tool, with `NormalisedDates` as its structured output. The human prompt carries the fiscal year, the field definitions and the two located blocks. The LLM finds each date, quotes its sentence verbatim, and works out that it should call the tool on the date as printed. It reports the tool's output, never formatting a date itself. Its result is the list of normalised dates.
3. **Classify** (`dates.py`, `prompts/date_classification.yaml`). An LCEL chain with no tools (`prompt | llm.with_structured_output(Classifications)`). The human prompt carries the reference date (`--reference-date`; the brief uses `2024-01-01`) and step 2's output: each normalised date with its context sentence. The system prompt gives the brief's status definitions and walks the LLM through the reasoning: the subject of the sentence, whether the date marks an event or the start or end of a period, and how it compares with the reference date. The schema puts `reasoning` before `status`, so the model reasons before it answers.

Step 1's prompt lists steps but doesn't name the tool or say how to normalise: that logic lives in the tool, and the agent has to work out that it should call it. Step 2 is the reasoning part, so it's left to the LLM. Every tool call is recorded in `tool_calls` for the trace, and provenance records each field's source block and quoted sentence, or why it has no answer.

### MCP server (`mcp_server.py`)
A [FastMCP](https://github.com/modelcontextprotocol/python-sdk) server over stdio, loaded into LangChain with `langchain-mcp-adapters`.

| Tool | Input | Output |
|---|---|---|
| `normalize_date` | `date_text`, e.g. "16 February 2024" | `"2024-02-16"`. Parsed with `dateutil`, day first. Raises an error unless the text has a day, a month and a year. |

The tool is deterministic: date parsing is done in code, which is reliable where LLMs occasionally slip. The LLM finds the dates and decides when to call the tool and with what input. A tool error goes back to the LLM as a message, so it can retry with a cleaner input. The tool is a plain function as well, so it is unit-tested without MCP.

### Part 3 pipeline
```mermaid
graph TD;
	__start__([start]) --> supervisor;
	supervisor -.-> revenue;
	supervisor -.-> expenditure;
	supervisor -.-> synthesize;
	revenue --> supervisor;
	expenditure --> supervisor;
	synthesize --> __end__([end]);
```
A hand-built LangGraph `StateGraph` (`supervisor.py`; the diagram is `build_graph(...).get_graph().draw_mermaid()`). The state is the query, the fiscal year, a list of `steps` that every node appends to, and the final answer.

1. **Supervisor** (`prompts/supervisor.yaml`). An LCEL call with structured output, `Route{reasoning, next, task}`. It sees the query and every report so far, lists the parts of the query, and either picks the agent for a part that isn't covered yet, writing a self-contained task for it, or finishes. `reasoning` comes before `next` in the schema, so the model reasons before it routes, and each decision's reasoning is in the trace. After `MAX_AGENT_CALLS` (4) agent runs the supervisor finishes without asking the LLM, so a confused run can't loop.
2. **Revenue Agent / Expenditure Agent** (`prompts/revenue_agent.yaml`, `prompts/expenditure_agent.yaml`). Each is a `create_agent` tool loop with `AgentReport{answer, figures[], gaps}` as structured output. Each figure is in $ billions with its block ids and pages. An agent sees only the supervisor's task, not the query or the other agent's reports, so its context stays small and focused. The prompts differ in scope (revenue: Operating Revenue, taxes, NIRC; expenditure: spending, Special Transfers, fund top-ups, fiscal position). An agent that fails is recorded in the trace with the error under `gaps`, and the supervisor carries on.
3. **Synthesize** (`prompts/synthesis.yaml`). One LLM call writes the final answer from the reports only. Every number must come from a report's figures, with its page, and the answer must not claim a link the reports don't support.

**Tools:**

| Tool | Agents | What it does |
|---|---|---|
| `list_blocks()` | both | The document's table of contents: block id, type, heading path, pages. |
| `search(keywords)` | both | The 5 blocks that contain the most of the keywords (title matches count 3×), each with its best-matching sentence or table row. |
| `read_block(block_id)` | both | A section's text, or a table as markdown. |
| `sum_values(values)` | expenditure | Adds numbers in code. The prompt requires it for every total and asks the agent to check the sum against the table's Total row. |

The document tools wrap plain functions in `retrieval.py` over the cached parsed document. They are local `@tool`s, not MCP: Part 2 already demonstrates MCP, and local tools keep `run_part3` synchronous, so it works in a notebook.

### How later parts reuse Part 1
- **Part 2** reuses the parsed document and the field-group mechanism. A *field group* is a Pydantic class in `fields.py`, whose field descriptions are the definitions the LLM reads and whose `anchor` and `block_type` class variables locate its source block. Part 2 adds `DistributionDate` and `EstateDutyDate`, locates their blocks with `locate_block`, and puts the blocks and `definitions(...)` into its tool-calling prompt instead of calling `extract`. Each `run_partN(config, pdf_path)` takes the loaded config and the PDF, and `main.prepare(config, pdf_path)` gives every part the same cached document and fiscal year.
- **Part 3** gives the Revenue and Expenditure agents lookup tools (`list_blocks`, `search`, `read_block` in `retrieval.py`) over the same parsed document, reuses Part 2's `tool_trace` for the trace, and reports figures with block ids and pages, the same provenance as Parts 1–2.

---

## Design decisions

### LangChain vs LangGraph
- **Parts 1–2 use LangChain.** Extraction is a linear pipeline, and an LCEL chain with structured output covers it. A graph would add complexity with no benefit. Part 2's tool-calling step uses LangChain's prebuilt `create_agent` (a tool loop plus structured output).
- **Part 3 uses LangGraph.** A supervisor needs routing, loops and shared state, which is what LangGraph is for. LangGraph builds on LangChain's components (chat models, tools, messages), so the Part 1 components carry over unchanged.

### Part 3: a hand-built supervisor
- **Hand-built `StateGraph`, not the `langgraph-supervisor` prebuilt.** The prebuilt routes through handoff tool calls, which record *that* the supervisor handed off but not *why*. Here routing is a structured-output call with the reasoning written first, which gives the brief's "clear trace of the supervisor's decision-making" directly. It is about 100 lines.
- **Workers see a task, not the conversation.** The supervisor writes each agent's task. That keeps each agent's context small and makes the supervisor do the decomposition, which shows in the trace. The cost: the supervisor has to pass on anything an agent needs from earlier reports, and its prompt says so.
- **Structured reports.** Agents return figures with block ids and pages, not just prose. The supervisor decides whether a part is covered from them, synthesis cites them, and the evals score them.
- **Arithmetic in a tool.** The brief asks the Expenditure Agent to sum "to the correct figure". LLMs slip at adding, so `sum_values` does it in code, as Part 2 does with dates.
- **Keyword search, not embeddings.** A document is about 60 blocks (about 15k tokens) with descriptive titles, so keyword search over titles and text finds the right block, and the agents read whole blocks. A vector store would add an embedding model and a second API key for no gain at this size.
- **Separate synthesis node.** The final answer is a separate call from routing, so the routing call stays small and the answer prompt can carry its own rules (cite pages, no new numbers, state gaps).

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

**Cover page:** Docling labels the cover's "Distributed on Budget Day: …" line as a page footer, and page headers and footers are otherwise dropped. So page 1 gets its own `(cover)` block holding all of its text, footers included. The fiscal year is read from it, and Part 2 finds the distribution date in it.

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

### Dependencies
| Package | Why |
|---|---|
| `docling` | PDF parsing by layout and table structure (see [Parsing](#parsing-docling-by-document-structure)). |
| `langchain`, `langchain-core` | Model-agnostic chat models, LCEL chains, structured output, and `create_agent` for tool calling. |
| `langgraph` | The Part 3 supervisor graph (`StateGraph`, conditional routing). `create_agent` also runs on it. |
| `mcp` | FastMCP, the official Python SDK for the Part 2 local MCP server. |
| `langchain-mcp-adapters` | Loads MCP tools as LangChain tools, so the agents call them like any other tool. |
| `python-dateutil` | Parses dates written in many formats inside the `normalize_date` tool. |
| `pydantic` | Schemas for parsed blocks, LLM structured output and results. |
| `pyyaml` | Prompts and config live in YAML, separate from code. |

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
- **Date statuses** follow the brief: *Expired* if the date has already passed, *Upcoming* if it is in the future, *Ongoing* if it refers to a period that is currently active. The LLM judges what each date marks from its sentence: a one-off event, or the start or end of a period.
- **The estate duty date is Expired.** "Estate Duty does not apply to a person who dies after 15 February 2008" marks the end of Estate Duty, and 2008-02-15 is before 2024-01-01. The prompt rules out the other reading, the start of an exemption that is still in force (Ongoing), by stating that "X does not apply after <date>" is the end of X.
- **"How will the Budget for the Future Energy Fund be supported"** is read as: how much the Budget allocates to the fund, as what, and for what (FY2024: an initial $5.0 billion injection, part of the $20.4 billion of top-ups to endowment and trust funds, to invest in critical infrastructure for the energy transition); and the revenue that finances the Budget overall (Operating Revenue and NIRC). The document doesn't earmark any revenue stream for the fund, and the synthesis prompt makes the answer say so rather than invent a link.
- **"Key government revenue streams"** are the largest Operating Revenue lines, plus NIRC. NIRC isn't Operating Revenue, but it is the largest single support for the Budget, so the Revenue Agent covers it.
- **Part 3 answers from one PDF** at a time, for its fiscal year and the years its tables show. Money is reported in $ billions.
- **Budget Day is Upcoming:** a one-off event on 2024-02-16, after 2024-01-01. The brief fixes the reference date at 2024-01-01 for every edition (the eval cases use it), so the FY2025 and FY2026 distribution dates are Upcoming too.

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
- `tests/test_evals.py`: the eval's comparison rules (number tolerance, taxes list in any order) and Part 3 case scoring.
- `tests/test_mcp_server.py`: `normalize_date` on full, partial and invalid dates.
- `tests/test_dates.py`: the Part 2 anchors, provenance for found, missing and unlocated dates, and the tool-call trace (which skips the agent's structured-output call).
- `tests/test_retrieval.py` also covers the Part 3 lookups: `list_blocks`, `read_block` (and its error for an unknown id) and `search` ranking.
- `tests/test_supervisor.py`: `sum_values`, routing after each decision, how reports are rendered for the prompts, and the agent-call limit (the supervisor finishes without an LLM call), using a fake chat model.

### Evals
Expected answers per PDF live in `evals/part1_cases.json`, with each `pdf_path` relative to the repo root. The eval runs Part 1 on every case (this needs `API_KEY`):
```bash
uv run python -m evals.part1_eval
```
Numbers pass within ±0.005, and the taxes list passes when it names the same taxes in any order (ignoring case). The report is printed and saved to `outputs/evals/part1_eval.json`. A `null` expected value means the field should come back null. A case whose run fails is recorded with its error, and its fields count as failed. Each case also writes its normal `outputs/part1_fy<year>.json`.

Part 2 works the same way, with cases in `evals/part2_cases.json`:
```bash
uv run python -m evals.part2_eval
```
It checks the normalised dates (any order) and the status of each date, and saves the report to `outputs/evals/part2_eval.json`.

Part 3's cases in `evals/part3_cases.json` are the demo queries:
```bash
uv run python -m evals.part3_eval
```
For each query it checks the routing (the set of agents the supervisor called) and that each key figure is within ±0.05 of a figure in some agent report. The final answer's wording isn't scored; an LLM judge could be added for that. The report is saved to `outputs/evals/part3_eval.json`, and each run writes its own `outputs/part3_fy<year>_<slug>.json` trace.

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
| A Part 3 answer misses something | Read the trace in `outputs/part3_fy<year>_<slug>.json`: the supervisor's reasoning, the task it wrote, and the agent's tool calls and `gaps` show where it went wrong. |
| A Part 2 date is missing from `dates` | Check `tool_calls` in `outputs/part2_fy<year>.json`: `normalize_date` may have rejected the text twice, or the LLM found no date in the block (see provenance). |

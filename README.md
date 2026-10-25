# Battery Test Investigator

Investigate battery-test measurements, compare selected discharge records, and produce a reproducible engineering report with source identity and explicit interpretation limits.

Example target question: “How has this battery's measured capacity changed, were the test conditions comparable, and what evidence supports the conclusion?”

The intended user is a battery test or validation engineer. The first application uses NASA B0005 laboratory measurements; it does not connect to live equipment.

## Status

The local inspection and automated reference comparison are verified on NASA B0005: 616 recorded operations, including 168 discharge operations. The current suite passes 186 test cases on Python 3.12.14 with the optional model dependencies installed. One command produces CSV/JSON inspection artifacts, plots, a selected-discharge comparison, a Markdown report, and an artifact-hash manifest. Identical source data and selections reuse the same investigation folder.

Programmatic investigation helpers preserve incomplete requests for clarification, validate explicit discharge selections, and return comparisons with the source hash, original operation indices, discharge indices, and selection revision. Changed sources and inconsistent operation mappings are rejected before returning comparison evidence. A separate bounded LangChain/local-Ollama connection check exposes the comparison through a strict tool schema; it has completed live B0005 comparisons for discharges 1/168 and 1/100.

The first-to-last reported-capacity change is -0.531408 Ah (-28.624% relative to the first discharge). This does not establish its physical cause, comparable conditions, or an end-of-life label. See the [reference artifacts and verification](examples/README.md).

A metadata-based protocol lookup returns applicable original passages with document hashes, archive provenance, and line locators from a registered three-document subset. Missing or invalid sources are labelled; wrong-experiment documents are not substituted. A separate LangGraph entry point now connects this lookup and the comparison through live LangChain tool calls, then generates structured protocol claims with exact quotes and evidence IDs.

After the initial comparison and protocol retrieval, the graph asks the model to choose one follow-up: inspect the selected whole-record traces or stop with partial findings. Python supplies the unchanged selection, verifies returned trace identity against the comparison, and prevents additional follow-up calls. The read-only trace tool and separate selected-pair plotting command describe the full recordings without guessing load boundaries. Citation checks prove source identity, applicability, and literal quote matching, not semantic support; claims still require review. This is an initial metadata-retrieval-plus-generation workflow, not an evaluated complete investigation agent.

Embedding-retrieval comparisons, broader decision/citation evaluation, durable resume, a demonstration interface, and optional Jira/Confluence publishing remain planned. The numerical report command still uses Python calculations and a fixed template without model calls. No external publication is implemented.

The data source is NASA's Battery Data Set. The documentation audit found nine distinct short readmes describing materially different experimental conditions and endpoints. The eventual retrieval component will be evaluated against supplying this compact document collection directly.

## Run the local automation

Use Python 3.12 and the `.venv` environment. To create it on a fresh checkout, run the setup commands first. Environment folders, raw downloads, and investigation outputs are excluded from Git. An existing `.venv-py315` can be retained separately; it is not the model-integration environment.

From PowerShell in the project folder:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_investigation
```

Open the `report.md` path printed by the command. The default compares discharge 1 with the final discharge. To compare discharge 1 with 100:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_investigation --early 1 --late 100
```

Protocol verification is unresolved at this stage; equal ambient-temperature metadata and similar median currents are insufficient to establish comparability. The PNG retains the full capacity trend and first source discharge's traces; it is not a selected-pair overlay.

The default source is `data/raw/nasa/package-1/B0005.mat`. The workflow retains each selected discharge's original operation position as well as its discharge-only index. For B0005, discharge 1 is original operation 2 and discharge 168 is original operation 614.

## Setup and verification

For a fresh checkout with Python 3.12 installed, run these commands in PowerShell from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/fetch_sample.py
.\.venv\Scripts\python.exe -m pytest -q
```

The fetch command reads only the first experiment package (approximately 57 MB) from NASA's larger archive, retains it locally, and extracts B0005 plus the corresponding README. Downloads and scratch outputs are ignored by Git. The fetch script records source checksums in `data/manifests/`.

In VS Code, open the project folder, select `.venv\Scripts\python.exe` through **Python: Select Interpreter**, then use the **Run battery investigation** debug configuration. It runs the investigation as a module from the project root.

`requirements-lock.txt` retains the numerical/test package pins. The numerical baseline was previously checked on Python 3.12.14 and 3.15.0; model integration is verified only on 3.12.14. `requirements-dev.txt` defines numerical/development dependency ranges. The optional-layer tests contain 94 cases; without that layer they are skipped and the other 92 cases run.

## Selected whole-record trace inspection

Inspect an explicit pair independently of the model:

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_selected_traces --early 1 --late 168
.\.venv\Scripts\python.exe -m scripts.inspect_selected_traces --early 1 --late 100
```

The command defaults to the known B0005 source and requires both discharge indices. It checks source hash, bounds and original operation mapping, then writes `trace-inspection.json` and `selected-traces.png` under `outputs/trace-inspections/`. The source/selection/revision and analysis version determine the evidence ID and output folder. Repeated inputs reuse that folder; changed pairs or revisions get different identities. The JSON retains the plot hash. Plotting failure on a rerun removes the earlier generated summary rather than leaving it to appear newly successful; a partial plot may remain. These are inspection artifacts, not investigation-completion markers.

Evidence includes signed-current, voltage and temperature sample minima/medians/maxima; first/last samples; the minimum-voltage sample; and the largest positive/negative adjacent current changes with their before/after samples and sampling intervals. Sample positions are one-based within each source discharge. Source time is retained, while plots align only each record's first sample to elapsed time zero. Every selected sample is plotted without resampling, phase matching, or truncating at a presumed load boundary.

These are unweighted sample statistics over the entire recording, not time-weighted or active-discharge statistics. Current sign is retained as recorded. Current changes are descriptive events, not load-state or cutoff classifiers; ties use the first occurrence. There is no current integration, protocol pass/fail tolerance, physical-cause inference, or comparability verdict. The final sample must not automatically be treated as a discharge cutoff.

`make_trace_tool()` exposes the compact, selection-bound analysis through the same strict battery/early/late schema as the comparison tool. It accepts no file/output paths and writes no artifacts; full arrays remain in Python. The graph can execute this adapter once after the model selects trace inspection. Source hash, both index types, and revision must match the comparison. The original inspection plot remains three panels; the selected-pair overlay is separate and is written only by the standalone inspection command.

## Protocol evidence lookup

The initial catalog in `data/manifests/protocol-sources.json` registers three original NASA README files: B0005/B0006/B0007/B0018, B0033/B0034/B0036, and B0049-B0052. They document different protocols and termination criteria. Registration of other batteries' documentation does not imply their numerical measurements have been validated.

Prepare the registered source text, then inspect B0005's evidence:

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_protocols
.\.venv\Scripts\python.exe -m scripts.protocol_evidence --battery-id B0005
```

The preparation command reads exact registered text members from NASA's nested experiment packages and verifies their pinned byte hashes, sizes, and listed battery IDs before saving under ignored `data/raw/nasa/protocols/`. A fresh preparation reads the three nested ZIP packages, approximately 160 MB combined; no numerical files are extracted. Existing sources are verified rather than overwritten. An optional `--document-cache <directory>` uses existing package/member text paths without network fallback. Cache-based preparation was verified locally; the archive-reading branch was tested with constructed ZIPs, not a new live download.

The lookup selects documents by explicit battery applicability before reading their contents. It returns the original Data Description and discharge-field sections with one-based inclusive line ranges, stable evidence IDs, source hashes, catalog identity, archive provenance, and explicit issues. It cross-checks registered battery IDs against each README's Files section. The status is `available`, `partial`, or `unavailable`; these describe evidence availability, not an investigation's completion.

No model, embeddings, vector database, or additional dependencies are needed for this lookup. It is a deterministic retrieval baseline, not completed RAG. Protocol descriptions do not prove actual measured conditions or pair comparability. Documented experiment endpoints and the comparison's percentage denominator must not be conflated. The numerical report and live connection check remain unchanged and do not yet consume this evidence.

## Optional local model connection check

Install the additional pinned dependencies into `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-agent-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements-agent.txt` defines the additional allowed dependency ranges. The lock includes the unchanged numerical/test pins. This layer supports both the connection check below and the first cited graph workflow.

Run Ollama locally with an already-installed tool-capable model. Pass its exact local tag explicitly; the following tag was verified on the development machine and is not downloaded by the script:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_model_connection --model qwen3.8:27b-q4_K_M
.\.venv\Scripts\python.exe -m scripts.check_model_connection --model qwen3.8:27b-q4_K_M --early 1 --late 100
```

This demonstration fixes the battery to B0005 and defaults to the explicit test pair 1/168; it does not resolve an ambiguous natural-language selection. It connects only to `127.0.0.1:11434`, refuses missing or cloud-backed model metadata, and does not download models or fall back to paid APIs. Hosted tracing and environment proxy settings are disabled for these calls. Local inference uses the machine's compute resources; cloud hosting is not provided.

The model receives the question and tool schema, requests one comparison, receives compact Python-calculated evidence (not raw arrays), and produces a short answer. Code requires the arguments to match the explicit selection. The budget is two model responses and one comparison, with 384 output tokens per response and a 60-second HTTP inactivity timeout; there is no automatic retry. The timeout is not a total run-time deadline.

A unique JSON trace under `outputs/model-connection/` records the source/selection, tool arguments/result, generated answer, model digest, settings, package versions, timing, and token counts. Private reasoning is not recorded. A failed check exits with an error and writes no success trace. Completion establishes a tool round trip, not general answer accuracy or a completed investigation; the live examples were checked against the numerical references separately.

## First cited graph workflow

Prepare the numerical source and protocol documents as above, install the optional model layer, and start the local Ollama service with an already-installed tool-capable model. Then supply an explicit selection:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_agent_investigation --model qwen3.8:27b-q4_K_M --battery-id B0005 --early 1 --late 168
.\.venv\Scripts\python.exe -m scripts.run_agent_investigation --model qwen3.8:27b-q4_K_M --battery-id B0005 --early 1 --late 100
```

This entry point does not infer selections from natural-language wording. Missing selection fields return `needs_clarification` before reading battery sources or connecting to the model. A new invocation needs the complete explicit request; it does not resume an earlier run. Only B0005 numerical investigation is currently exposed. `--question` changes the investigation question; `--protocol-catalog` supplies trusted local catalog configuration for evaluation and is never model-controlled.

The graph checks the request, asks the model for the comparison and protocol tools, validates the entire requested batch before execution, and runs those tools. The model may choose their order or request both together; these initial evidence requirements are fixed by code. Tool arguments must match the requested selection. Tools and call IDs cannot be repeated, and raw arrays are not sent to the model.

The next model call sees the question and both results. Its strict structured decision permits only `inspect_discharge_traces` or `stop_with_partial`, with a short evidence-based reason. A conditional edge executes the selected check or moves directly to findings. Python supplies the original arguments; the model cannot revise the pair or provide paths. Trace evidence must match the comparison's complete selection, source hash, original operation mapping and revision before it reaches generation. Missing protocol evidence does not prohibit useful trace inspection. This is one bounded follow-up, not unrestricted planning or a measured decision-quality score.

The budget is at most four model attempts (including the follow-up decision and final findings) and three tool attempts, with no automatic retries. A batched initial request normally uses three model attempts; separate initial requests can use four. Failed attempts consume budget; local preflight failure counts as a model attempt even if no inference starts. One attempt is reserved for final findings. The local model uses 8192 context tokens, a maximum 1024 output tokens per response, thinking disabled, temperature zero, and a 60-second HTTP inactivity timeout rather than a total runtime deadline. The graph adds a 12-step recursion backstop. Hosted tracing and environment proxies are disabled; there is no model download or cloud fallback.

Generated protocol findings include a short claim, a retrieved evidence ID, and a verbatim supporting quote. Unknown IDs, inapplicable citations, invented quotes, malformed findings, and truncated responses are rejected. Findings must also reference exactly the accepted trace evidence ID, or `null` if inspection did not run. Semantic claim support still requires review: a real quote can be misinterpreted. Numerical outputs come directly from Python, not model-written arithmetic. Comparability stays `not_established` and physical cause `not_determined`; executing descriptive trace inspection does not establish either.

With available documentation, executed trace inspection yields `partial` / `comparability_assessment_pending`; stopping without inspection yields `partial` / `selected_trace_check_pending`. Missing or incomplete documentation yields `partial` / `protocol_evidence_incomplete`, whether traces ran or not. Missing evidence must not produce substitute protocol claims. Invalid selections, model/tool failures or bad decisions/findings yield `failed`; exhausted budgets yield `partial` without an invented final answer. No run in this graph is marked complete: active-load assessment, citation semantics and the remaining investigation criteria are unresolved.

Each invocation writes a unique record under `outputs/agent-investigations/` with the request, calculated evidence, retrieved passages, follow-up decision/reason, trace evidence if executed, findings if accepted, stopping reason, attempt counts, concise tool trace, timing/usage, source/model identities, prompt/schema hashes, package versions, and an input/configuration fingerprint. Raw model messages and private reasoning are excluded. Failed and clarification runs are retained separately; failed runs exit with code 2. These JSON records are evidence logs, not disk-backed LangGraph checkpoints or a generated engineering report. The existing numerical report remains a separate command.

## Outputs and calculation

Each investigation writes the following files under `outputs/investigations/`:

| File | Contents |
| --- | --- |
| `report.md` | Selected measurements, calculation, limitations, and evidence links |
| `comparison.json` | Selected records and unrounded comparison results |
| `B0005-discharge-summary.csv` | One row per discharge |
| `B0005-inspection.json` | Source hash, operation counts, and first/last record metadata |
| `B0005-inspection.png` | Capacity trend and first source discharge's voltage/temperature plots |
| `complete.json` | Investigation identity and artifact hashes, written after successful output generation |

The source hash, battery ID, selected indices, and workflow version determine the investigation folder. Identical inputs reuse it. A failed rerun removes any previous completion marker; partial files may remain. The CLI supports one run at a time.

```text
capacity change = later supplied capacity - earlier supplied capacity
percentage change = capacity change / earlier supplied capacity * 100
```

Capacity is read from NASA's records rather than independently integrated from current measurements. The denominator is the earlier selected capacity, not the nominal rating. `recording_duration_s` covers the entire source record and is not a segmented active-discharge duration.

## Source and interpretation

[NASA's official repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) provides the archive. Dataset attribution: B. Saha and K. Goebel (2007), “Battery Data Set,” NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

The records are laboratory experiments. Capacity trends do not establish physical degradation mechanisms. Test endpoints differ by experiment; some runs ended because control software crashed. Recorded discharge capacity is retained as supplied by NASA until its calculation is independently investigated.

# Battery Test Investigator

Investigate battery-test measurements, compare selected discharge records, and produce a reproducible engineering report with source identity and explicit interpretation limits.

Example target question: “How has this battery's measured capacity changed, were the test conditions comparable, and what evidence supports the conclusion?”

The intended user is a battery test or validation engineer. The first application uses NASA B0005 laboratory measurements; it does not connect to live equipment.

## Status

The local inspection and automated reference comparison are verified on NASA B0005: 616 recorded operations, including 168 discharge operations. The current suite passes 71 test cases on Python 3.12.14 with the optional model dependencies installed. One command produces CSV/JSON inspection artifacts, plots, a selected-discharge comparison, a Markdown report, and an artifact-hash manifest. Identical source data and selections reuse the same investigation folder.

Programmatic investigation helpers preserve incomplete requests for clarification, validate explicit discharge selections, and return comparisons with the source hash, original operation indices, discharge indices, and selection revision. Changed sources and inconsistent operation mappings are rejected before returning comparison evidence. A separate bounded LangChain/local-Ollama connection check exposes the comparison through a strict tool schema; it has completed live B0005 comparisons for discharges 1/168 and 1/100.

The first-to-last reported-capacity change is -0.531408 Ah (-28.624% relative to the first discharge). This does not establish its physical cause, comparable conditions, or an end-of-life label. See the [reference artifacts and verification](examples/README.md).

Document retrieval with citations, LangGraph investigations, evidence-dependent agent decisions, a demonstration interface, and optional Jira/Confluence publishing are planned. The numerical report command still uses Python calculations and a fixed template without model calls. The separate model connection check is not yet a complete investigation agent. Neither command publishes external updates.

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

`requirements-lock.txt` retains the numerical/test package pins. The numerical baseline was previously checked on Python 3.12.14 and 3.15.0; model integration is verified only on 3.12.14. `requirements-dev.txt` defines numerical/development dependency ranges. Without the optional layer below, the 20 model-layer cases are skipped and the other 51 cases run.

## Optional local model connection check

Install the additional pinned dependencies into `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-agent-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements-agent.txt` defines the additional allowed dependency ranges. The lock includes the unchanged numerical/test pins. LangGraph is installed with this layer but no investigation graph has been implemented.

Run Ollama locally with an already-installed tool-capable model. Pass its exact local tag explicitly; the following tag was verified on the development machine and is not downloaded by the script:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_model_connection --model qwen3.8:27b-q4_K_M
.\.venv\Scripts\python.exe -m scripts.check_model_connection --model qwen3.8:27b-q4_K_M --early 1 --late 100
```

This demonstration fixes the battery to B0005 and defaults to the explicit test pair 1/168; it does not resolve an ambiguous natural-language selection. It connects only to `127.0.0.1:11434`, refuses missing or cloud-backed model metadata, and does not download models or fall back to paid APIs. Hosted tracing and environment proxy settings are disabled for these calls. Local inference uses the machine's compute resources; cloud hosting is not provided.

The model receives the question and tool schema, requests one comparison, receives compact Python-calculated evidence (not raw arrays), and produces a short answer. Code requires the arguments to match the explicit selection. The budget is two model responses and one comparison, with 384 output tokens per response and a 60-second HTTP inactivity timeout; there is no automatic retry. The timeout is not a total run-time deadline.

A unique JSON trace under `outputs/model-connection/` records the source/selection, tool arguments/result, generated answer, model digest, settings, package versions, timing, and token counts. Private reasoning is not recorded. A failed check exits with an error and writes no success trace. Completion establishes a tool round trip, not general answer accuracy or a completed investigation; the live examples were checked against the numerical references separately.

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

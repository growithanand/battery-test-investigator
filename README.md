# Battery Test Investigator

Investigate battery-test measurements, compare selected discharge records, and produce a reproducible engineering report with source identity and explicit interpretation limits.

Example target question: “How has this battery's measured capacity changed, were the test conditions comparable, and what evidence supports the conclusion?”

The intended user is a battery test or validation engineer. The first application uses NASA B0005 laboratory measurements; it does not connect to live equipment.

## Status

The local inspection and automated reference comparison are verified on NASA B0005: 616 recorded operations, including 168 discharge operations. The current suite passes 51 test cases on Python 3.12.14. One command produces CSV/JSON inspection artifacts, plots, a selected-discharge comparison, a Markdown report, and an artifact-hash manifest. Identical source data and selections reuse the same investigation folder.

Programmatic investigation helpers preserve incomplete requests for clarification, validate explicit discharge selections, and return comparisons with the source hash, original operation indices, discharge indices, and selection revision. Changed sources and inconsistent operation mappings are rejected before returning comparison evidence. These helpers are not yet connected to a model or investigation graph.

The first-to-last reported-capacity change is -0.531408 Ah (-28.624% relative to the first discharge). This does not establish its physical cause, comparable conditions, or an end-of-life label. See the [reference artifacts and verification](examples/README.md).

LangChain tool use, document retrieval with citations, LangGraph investigations, a demonstration interface, and optional Jira/Confluence publishing are planned. The current implementation uses Python calculations and a fixed report template. It makes no model calls or external updates.

The data source is NASA's Battery Data Set. The documentation audit found nine distinct short readmes describing materially different experimental conditions and endpoints. The eventual retrieval component will be evaluated against supplying this compact document collection directly.

## Run the local automation

The commands below use the Python 3.15 environment `.venv-py315`. To create it on a fresh checkout, run the setup commands first. Environment folders, raw downloads, and investigation outputs are excluded from Git.

From PowerShell in the project folder:

```powershell
.\.venv-py315\Scripts\python.exe -m scripts.run_investigation
```

Open the `report.md` path printed by the command. The default compares discharge 1 with the final discharge. To compare discharge 1 with 100:

```powershell
.\.venv-py315\Scripts\python.exe -m scripts.run_investigation --early 1 --late 100
```

Protocol verification is unresolved at this stage; equal ambient-temperature metadata and similar median currents are insufficient to establish comparability. The PNG retains the full capacity trend and first source discharge's traces; it is not a selected-pair overlay.

The default source is `data/raw/nasa/package-1/B0005.mat`. The workflow retains each selected discharge's original operation position as well as its discharge-only index. For B0005, discharge 1 is original operation 2 and discharge 168 is original operation 614.

## Setup and verification

For a fresh checkout with Python 3.15 installed, run these commands in PowerShell from the project folder. The pinned dependencies were verified on Windows with Python 3.12.14 and 3.15.0.

```powershell
py -3.15 -m venv .venv-py315
.\.venv-py315\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-py315\Scripts\python.exe scripts/fetch_sample.py
.\.venv-py315\Scripts\python.exe -m pytest -q
```

The fetch command reads only the first experiment package (approximately 57 MB) from NASA's larger archive, retains it locally, and extracts B0005 plus the corresponding README. Downloads and scratch outputs are ignored by Git. The fetch script records source checksums in `data/manifests/`.

In VS Code, open the project folder, select `.venv-py315\Scripts\python.exe` through **Python: Select Interpreter**, then use the **Run battery investigation** debug configuration. It runs the investigation as a module from the project root.

`requirements-lock.txt` records the exact package versions verified on Windows with Python 3.12.14 and 3.15.0. A standalone Python 3.15 run passed all 12 tests and matched all 168 reference discharge-summary rows and the saved capacity comparison. `requirements-dev.txt` defines the allowed dependency ranges for future updates.

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

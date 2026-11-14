# Battery Test Investigator

A battery test engineer comparing two discharge records needs to know how much capacity changed, which test conditions were documented, and what the measurements can support. This project does the comparison, checks the relevant experiment notes, and saves a report with source references and open questions.

In the NASA B0005 records, reported capacity fell from 1.856487 Ah on discharge 1 to 1.325079 Ah on discharge 168. That's a change of **−0.531408 Ah (−28.624%)** relative to discharge 1. The percentage alone cannot establish that the tests were comparable, explain the capacity loss, or show that the later record met the experiment's end-of-life criterion. [Source checks and sample outputs](examples/README.md)

The investigator runs locally on Windows with Python 3.12 and Ollama. The [browser demo plan](#browser-demo) builds on the local benchmark.

## How an investigation runs

```mermaid
flowchart LR
    A[Question and discharge pair] --> B[Validate selection]
    B --> C[Calculate capacity change and retrieve protocol]
    C --> D{Model chooses follow-up}
    D -->|Inspect traces| E[Read the selected traces]
    D -->|Stop| F[Save findings and report]
    E --> F
```

Python reads the MATLAB file and does the calculations. It keeps the discharge number and original operation number together, so a reader can find the same record in the source. LangChain exposes the calculation, protocol lookup, and trace inspection as tools. LangGraph runs the workflow and lets the local Ollama model choose one follow-up after the capacity comparison and protocol lookup.

The retrieval step uses a catalog of three original experiment notes. It filters by battery and returns passages with line numbers for retrieval-augmented generation (RAG). The workflow checks that a citation applies to the selected battery and that its quote matches the source. Review is still needed to tell whether the quote supports the whole claim.

Python enforces the selected pair and caps each run at four model attempts and three tool attempts. When trace inspection runs, the report gives current, voltage, and temperature extrema with sample positions and times. It also distinguishes the last voltage reading from the minimum and reports load-window estimates for B0005 when the sampled transitions can be bracketed. Missing selections trigger clarification. A saved run can be resumed or revised in a new process, and each new report links to its parent.

## Results

The local benchmark compares the original agent baseline (`d7e314f`) with v23 (`984367e`) on 30 B0005 cases. Both versions ran the six reserved cases once with the same Qwen 3.8 27B model, data, and protocol fixtures. The median times cover five cases that used inference per version. [Cases and scoring rubric](evaluations/README.md#benchmark-method)

| Measure | Baseline | v23 |
| --- | ---: | ---: |
| Development cases meeting all applicable criteria | 18/24 | 23/24 |
| Reserved cases meeting all applicable criteria | 5/6 | 5/6 |
| All cases meeting the execution criteria | 24/30 | 28/30 |
| All cases meeting every applicable criterion (provisional) | 23/30 | 28/30 |
| Median CLI time on reserved cases with inference | 22.43 s | 23.00 s |

The gain came from the development cases. On the reserved protocol question, both versions quoted the right source for B0005's 2 A current and 2.7 V cutoff, but both labelled the request as a test-conditions assessment where the rubric expected a document lookup. The reserved cases vary the wording and selected discharges within one battery. [Final benchmark](evaluations/experiments/portfolio-final-local-20261110.json)

All **12 security cases** met their blocked-outcome criteria on v23. In one live run, a retrieved document told the model to report 999 Ah. The report kept the change calculated from the MATLAB data. The suite includes seven scripted boundary checks, one check of the emitted Markdown, and four live attacks. [Security results](evaluations/experiments/portfolio-security-20261110.json)

The scoring model matched five of six source-based review labels. It missed a report that gave the correct 2.0 Ah rating in one place and called the 1.4 Ah threshold the rating elsewhere. The reserved semantic scores therefore use direct review of the reports and sources. [Evaluation history](evaluations/README.md)

## Run locally

From PowerShell in the project folder, create the Python 3.12 environment and fetch the NASA sample.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/fetch_sample.py
.\.venv\Scripts\python.exe -m scripts.run_investigation --early 1 --late 168
```

Open the printed report path. The numerical workflow writes the Markdown report, comparison JSON, a CSV with all 168 discharges, the inspection plot, and a manifest with file hashes. Identical inputs reuse the same investigation folder. The comparison uses the capacity values supplied in the source file.

### Inspect a pair

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_selected_traces --early 1 --late 168
.\.venv\Scripts\python.exe -m scripts.inspect_selected_traces --early 1 --late 168 --active-load
```

The first command saves a trace summary and overlay plot. The second prints the B0005 load-window check as JSON. It looks for one consecutive block of samples at or below −1 A with off-load samples on both sides. It reports the transition brackets and the span between the first and last on-load samples, or explains why the window could not be resolved.

### Run with the local model

Install the model dependencies, prepare the protocol files, and start Ollama with a local tool-capable model already installed. The commands below use the model tag from the benchmark. Protocol preparation verifies the registered files before writing them under the local data directory. A fresh download reads about 160 MB across the three NASA experiment packages.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-agent-lock.txt
.\.venv\Scripts\python.exe -m scripts.fetch_protocols
.\.venv\Scripts\python.exe -m scripts.protocol_evidence --battery-id B0005
.\.venv\Scripts\python.exe -m scripts.check_model_connection --model qwen3.8:27b-q4_K_M
.\.venv\Scripts\python.exe -m scripts.run_agent_investigation --model qwen3.8:27b-q4_K_M --battery-id B0005 --early 1 --late 168
```

Open the Markdown report and its linked JSON run record under `outputs/agent-investigations/`. The JSON records the selections, evidence, decisions, stopping reason, model configuration, timing, and token counts. The workflow uses the local Ollama endpoint. Provider API fees were $0 for the benchmark. Electricity and hardware costs were unmeasured.

### Resume or revise a run

Start with one missing discharge, then use the ID printed after `Local run record` to supply the missing selection. To revise a completed comparison, use its run ID and supply one or both changed indices. Both operations carry forward the saved question and verify the registered source.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_agent_investigation --model qwen3.8:27b-q4_K_M --battery-id B0005 --late 144
$runId = '<saved clarification run ID>'
.\.venv\Scripts\python.exe -m scripts.run_agent_investigation --model qwen3.8:27b-q4_K_M --resume $runId --early 9
$runId = '<saved comparison run ID>'
.\.venv\Scripts\python.exe -m scripts.run_agent_investigation --model qwen3.8:27b-q4_K_M --resume $runId --early 8 --late 155
```

### Verify the installation

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

The benchmark checkout passed 354 tests with the optional model dependencies installed. See the [test suite](tests/) and [evaluation guide](evaluations/README.md) for the checks and saved results.

## Browser demo

Three saved runs are ready for the first browser demo. The capacity case shows why the workflow stopped after the comparison. The trace case shows the original sample positions and times. The clarification case shows what the workflow asks for when the discharge pair is missing. These examples will replay the saved reports without making API calls. [Benchmark records](evaluations/experiments/portfolio-final-local-20261110.json)

The next live check will run the same capacity and trace cases on a hosted model. It will compare tool decisions and citations against the saved local runs and measure response time and token use before a provider is chosen.

## Data source

B. Saha and K. Goebel (2007), *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center. The measurements come from laboratory experiments with different protocols and endpoints. The example analysis covers B0005. [NASA data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) · [Source manifest](data/manifests/B0005.json)

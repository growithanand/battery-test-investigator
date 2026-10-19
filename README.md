# Battery Test Investigator

An engineering learning project: investigate battery-test measurements alongside the experiment's documentation, check whether comparisons are defensible, and prepare a traceable technical report.

Example target question: “How has this battery's measured capacity changed, were the test conditions comparable, and what evidence supports the conclusion?”

The [release 1 scope](docs/product-scope.md) defines the intended user, inputs, deliverables, exclusions, automation, Jira/Confluence roles, milestones, and evaluation. It is the current product definition.

## Status

The local inspection and automated reference comparison are verified on NASA B0005: 616 recorded operations, including 168 discharge operations, with 12 passing tests. One command now produces CSV/JSON inspection artifacts, plots, a selected-discharge comparison, a Markdown report, and an artifact-hash manifest. Identical source data and selections reuse the same investigation folder.

The first-to-last reported-capacity change is -0.531408 Ah (-28.624% relative to the first discharge). This does not establish its physical cause, comparable conditions, or an end-of-life label. See the [reference results](examples/README.md) and [automation verification](docs/results/02-reference-investigation.md).

Anand's reproduction and review remain pending. LangChain tool use, document retrieval, LangGraph investigations, a demonstration interface, and Jira/Confluence publishing are planned, not implemented. No external updates occur in the current workflow.

The data source is NASA's Battery Data Set. The documentation audit found nine distinct short readmes describing materially different experimental conditions and endpoints. The eventual retrieval component will be evaluated against supplying this compact document collection directly.

## Run the local automation

The environment and source file already exist on Anand's current laptop. From PowerShell in this project folder:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_investigation
```

Open the `report.md` path printed by the command. The default compares discharge 1 with the final discharge. To compare discharge 1 with 100:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_investigation --early 1 --late 100
```

Read the [guided comparison exercise](docs/milestones/02-reference-investigation.md). Protocol verification is unresolved at this stage; equal ambient-temperature metadata and similar median currents are insufficient to establish comparability. The PNG retains the existing full capacity trend and first source discharge's traces; it is not a selected-pair overlay.

## First milestone

Read one real battery file, preserve its original operation order, identify discharge operations, and create a reproducible capacity plot. Start with [the guided exercise](docs/milestones/01-first-battery.md).

With Python 3.11 or later installed, from PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/fetch_sample.py
.\.venv\Scripts\python.exe scripts/inspect_sample.py
.\.venv\Scripts\python.exe -m pytest -q
```

The fetch command reads only the first experiment package (approximately 57 MB) from NASA's larger archive, retains it locally, and extracts B0005 plus the corresponding README. Downloads and scratch outputs are ignored by Git. The fetch script records source checksums in `data/manifests/`.


`requirements-lock.txt` records the exact package versions used for the reference run on Windows with Python 3.12.14. `requirements-dev.txt` defines the allowed dependency ranges for future updates.

## Version control

Git keeps local change history. GitHub will host a private remote repository after account connection. Keep raw data and secrets out of commits; include code, documentation, source manifests, and small reference results. Git identity is configured locally for this repository as Anand Mullasseril Shajahan <anand.mullassheri@gmail.com>.

Use the installed Git Credential Manager to sign in to GitHub; see [the Git workflow](docs/git-workflow.md). No GitHub plugin is required.

## Source and interpretation

[NASA's official repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) provides the archive. Dataset attribution: B. Saha and K. Goebel (2007), “Battery Data Set,” NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

The records are laboratory experiments. Capacity trends do not establish physical degradation mechanisms. Test endpoints differ by experiment; some runs ended because control software crashed. Recorded discharge capacity is retained as supplied by NASA until its calculation is independently investigated.

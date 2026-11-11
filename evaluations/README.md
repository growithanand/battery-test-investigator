# Evaluation

The numerical reference checks capacity arithmetic against the original NASA B0005 file. The agent benchmark also checks which tools run, whether the cited evidence supports the report, and whether the report answers the question. [Numerical reference](../examples/README.md)

## Benchmark method

The 30 task cases consist of 24 development cases and six reserved cases. The rubric separates execution, numerical accuracy, citation support, task fulfilment, and engineering limits. A case meets the full rubric when it passes every applicable criterion. [Task cases](portfolio-cases.json) · [Rubric](portfolio-rubric.json)

The original agent baseline is `d7e314f`. Its development run recorded 19/24 execution passes and 18/24 cases meeting all applicable criteria. The remaining problems included missing final-voltage readings in the report, a load-window check that the model could not yet call, and missing restart support. [Baseline results](baselines/portfolio-d7e314f-20261107.json)

The final comparison ran each reserved case once per version, alternating which version ran first. Both versions used the same B0005 source, protocol fixtures, Ollama model, and settings. The source check matched ten capacity comparisons, four trace pairs, 48 extrema locations, four v23 load windows, and 14 original quotes. Across five reserved inference cases per version, the nearest-rank p95 CLI time was 26.74 s for the baseline and 27.90 s for v23 after preload. These are small local timing samples. [Final benchmark and judge results](experiments/portfolio-final-local-20261110.json)

## What changed along the way

| Check | What it found or changed |
| --- | --- |
| [Follow-up baseline](baselines/follow-up-v1-20261026.json) and [prompt comparison](experiments/follow-up-policy-20261027.json) | The baseline skipped trace inspection on all three general conditions runs. A prompt change brought that case to 3/3 while keeping the capacity-only question on the shorter path. |
| [Missing protocol](experiments/protocol-limits-20261030.json) | The model sometimes omitted the missing-document warning. Python now adds the warning from the retrieval status. All 21 runs passed both the follow-up and warning checks. |
| [Claim scope](experiments/claim-scope-20261031.json) | The first prompt candidate still assigned a cutoff too broadly or left the battery mapping ambiguous. The revised prompt preserved the mapping in all 24 reviewed claims. [Claim rubric](protocol-claim-review.json) |
| [Recommendations](experiments/recommendations-20261102.json) | Early candidates returned graph labels or advice that missed the question. The revised workflow supplied a short hint tied to the evidence and accepted action. All 21 final recommendations reused that hint. [Cases](recommendation-cases.json) · [Recommendation rubric](recommendation-review.json) |
| [First reserved run](held-out/follow-up-v12-20261102.json) | All nine execution checks passed, but three extrema answers missed requested source locations. The terminal showed an evidence ID without the extrema table the reader needed. |
| [Extrema locations](experiments/trace-extrema-20261103.json) | Python now returns sample positions and times for all three signals. A shared formatter displays them in both commands. Source checks matched all 24 values and locations across two pairs. |
| [Question scope](experiments/request-scope-20261103.json) | The follow-up decision now carries the question's scope into the recommendation step. This lets an answered capacity or extrema question end without a second engineering task. [Scope cases](request-scope-cases.json) |
| [v19 development run](experiments/portfolio-v19-20261109.json) | Adding the sampled load-window check and restart support brought execution to 22/24. Direct review still found a generic diagnostic recommendation, a capacity-basis contradiction, and an incomplete range error. |
| [v23 development run](experiments/portfolio-v23-20261110.json) | The report separates measured capacity from the documented rating, links the mechanism follow-up to applicable impedance evidence where available, and shows the valid range for an out-of-range discharge. Execution and full-case scores reached 23/24. |

Three targeted runs also checked the new paths against the MATLAB source. For D07, the tool found both transition brackets and reported spans of 3235.157 s and 2344.016 s for discharges 13 and 157. For D23, a new process filled in discharge 9 for a saved clarification, compared it with discharge 144, and linked the new report to its parent. For D24, a new process revised the pair to 8 and 155 and saved a separate comparison at revision 2. A second revision run confirmed that trace evidence also changes with the selection. [Development results](experiments/portfolio-v23-20261110.json)

## Security checks

The suite covers selection changes, skipped tools, arbitrary paths, unknown tool calls, untrusted instructions in retrieved text, invalid citations, fabricated quotations, report markup, a synthetic secret, duplicate calls, and attempt limits. The stored results identify which cases forced a code boundary and which ran the live model. The report-rendering case inspected the Markdown emitted by the CLI. [Case definitions](security-cases.json) · [Results](experiments/portfolio-security-20261110.json)

A separate scan checked the reviewed files and published Git history for common credential patterns. Raw CLI logs contain local output paths and stay outside Git along with the synthetic test value. The published evidence summaries retain hashes and decisions.

## Reproduce the checks

Run these commands from the project root with the [local setup](../README.md#run-locally) complete.

To check an existing run against the [follow-up cases](follow-up-cases.json), pass the saved record and case ID. The checker reads the JSON without making a model call. It checks the action, stopping reason, source and selection identity, and attempt limits. Numerical readback and interpretation of cited passages are separate checks.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_follow_up --case trace-1-168 --record "outputs/agent-investigations/YOUR-RUN-ID.json"
```

The development baseline collector expects runtime files matching the declared baseline. It freezes the inputs and creates labelled protocol fixtures before running each case in a fresh process.

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_portfolio_baseline --prepare --output outputs/portfolio-baseline/new-baseline
.\.venv\Scripts\python.exe -m scripts.collect_portfolio_baseline --run --output outputs/portfolio-baseline/new-baseline
```

The candidate collector reuses the original plan and fixture hashes referenced by the published baseline. It requires those archived local files and the same local model configuration. A newly prepared baseline has its own fixture hashes. The candidate writes the execution comparison, timings, and per-run evidence hashes to a fresh directory.

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_portfolio_candidate --output outputs/portfolio-candidate/new-candidate
```

Jira tracked the development issues, and Confluence holds the project overview. The repository carries the evaluation files and the [commit map](commit-history.json) for older references.

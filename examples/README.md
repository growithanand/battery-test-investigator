# Verified first inspection

These small reference artifacts were generated from NASA B0005 by `scripts/inspect_sample.py` on 2026-10-17. They are measured source summaries, not predictions or AI-generated findings.

- `B0005-inspection.png`: capacity trend plus first-record voltage and temperature curves.
- `B0005-discharge-summary.csv`: one row for each of 168 discharge operations.
- `B0005-inspection.json`: operation counts, first/last discharge metadata, and source hash.

The source manifest is in `data/manifests/B0005.json`. Raw downloads are excluded from Git. Reproduce the files with:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_sample.py
.\.venv\Scripts\python.exe scripts/inspect_sample.py
```

New runs write to ignored `outputs/`, leaving this reviewed reference snapshot unchanged. Read `docs/results/01-B0005-inspection.md` for interpretation boundaries and verification evidence.

Dataset attribution: B. Saha and K. Goebel (2007), Battery Data Set, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

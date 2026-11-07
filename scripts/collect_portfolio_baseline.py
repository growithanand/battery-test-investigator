"""Collect the frozen development cases through the investigation CLI."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evaluations/portfolio-cases.json"
MODEL = "qwen3.8:27b-q4_K_M"
MODEL_DIGEST = "25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_baseline_runtime(baseline_commit):
    paths = [name for name in git("ls-tree", "-r", "--name-only", baseline_commit).splitlines()
             if name.startswith("scripts/") or name.startswith("requirements")]
    assert paths, "The baseline must identify runtime files."
    assert not git("diff", "--name-only", baseline_commit, "--", *paths), "Runtime differs from the frozen baseline."


def prepare(output):
    assert not (output / "plan.json").exists(), "Use a fresh collection directory."
    cases_bytes = CASES.read_bytes()
    cases = json.loads(cases_bytes)
    verify_baseline_runtime(cases["baseline_commit"])
    assert not git("diff", "--cached", "--name-only")
    selected = [case for case in cases["cases"] if case["split"] == "development"]
    assert len(cases["cases"]) == 30 and len(selected) == 24
    assert len({case["case_id"] for case in cases["cases"]}) == 30

    catalog_path = ROOT / "data/manifests/protocol-sources.json"
    catalog = json.loads(catalog_path.read_bytes())
    original = next(doc for doc in catalog["documents"] if "B0005" in doc["applicable_batteries"])
    tag = sha(output.relative_to(ROOT).as_posix().encode())[:12]
    raw_fixture = ROOT / "data/raw/nasa/evaluation-fixtures" / tag
    raw_fixture.mkdir(parents=True, exist_ok=True)
    corrupt_path = raw_fixture / "corrupt.txt"
    assert not corrupt_path.exists(), "Fixture already exists."
    corrupt_path.write_bytes(b"Deliberately modified source fixture for the portfolio baseline.\n")
    missing = deepcopy(original)
    missing["path"] = (raw_fixture / "deliberately-absent.txt").relative_to(ROOT).as_posix()
    assert not (ROOT / missing["path"]).exists()
    fixtures = {"available": str(catalog_path.relative_to(ROOT))}
    fixture_dir = output / "fixtures"
    fixture_dir.mkdir()
    for name in ("missing", "partial", "corrupt", "wrong_experiment"):
        fixture = deepcopy(catalog)
        if name == "missing":
            fixture["documents"] = [deepcopy(missing)]
        elif name == "partial":
            supplement = {**missing, "document_id": "portfolio-missing-supplement"}
            fixture["documents"].append(supplement)
        elif name == "corrupt":
            fixture["documents"] = [{**original, "path": corrupt_path.relative_to(ROOT).as_posix()}]
        else:
            fixture["documents"] = [doc for doc in fixture["documents"] if "B0005" not in doc["applicable_batteries"]]
        path = fixture_dir / f"{name}.json"
        write_json(path, fixture)
        fixtures[name] = path.relative_to(ROOT).as_posix()

    reserved = [case for case in cases["cases"] if case["split"] == "held_out"]
    prior_count = 0
    for path in (ROOT / "outputs/agent-investigations").glob("*.json"):
        request = json.loads(path.read_bytes()).get("request")
        if isinstance(request, dict):
            prior_count += 1
            assert all(request != case["request"] for case in reserved), "A reserved request was already used."
    paths = set(git("ls-files").splitlines())
    paths.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    source_hashes = {name: sha((ROOT / name).read_bytes()) for name in sorted(paths)}
    for name in paths:
        destination = output / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    fixture_paths = [ROOT / path for path in fixtures.values()] + [corrupt_path]
    plan = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "checkout_commit": git("rev-parse", "HEAD"),
        "source_commit": cases["baseline_commit"], "case_set_sha256": sha(cases_bytes),
        "cases": selected, "reserved_ids_not_executed": [case["case_id"] for case in reserved],
        "prior_records_checked": prior_count, "fixtures": fixtures,
        "fixture_hashes": {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in fixture_paths},
        "source_hashes": source_hashes, "model": MODEL, "model_digest": MODEL_DIGEST,
        "runs_per_case": 1, "retries": 0,
        "execution": "Fresh CLI process per development case. Preload the local model once before collection. Preserve every process outcome.",
        "review": "Collection records observations. Case scoring and source-based review are separate.",
    }
    write_json(output / "plan.json", plan)
    print(json.dumps({"prepared": output.relative_to(ROOT).as_posix(), "development_cases": len(selected),
                      "reserved_cases_not_executed": len(reserved), "prior_records_checked": prior_count}))


def collect(output):
    import httpx
    from scripts.local_model import create_local_model, LOCAL_URL
    from scripts.run_agent_investigation import MODEL_SETTINGS

    path = output / "collection.json"
    assert not path.exists(), "Collection exists. Keep its outcomes and use a new directory for another experiment."
    plan_bytes = (output / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    _, runtime = create_local_model(plan["model"], MODEL_SETTINGS)
    assert runtime["model_digest"] == plan["model_digest"]
    preload_start = perf_counter()
    with httpx.Client(base_url=LOCAL_URL, timeout=115.0, trust_env=False) as client:
        response = client.post("/api/chat", json={"model": MODEL, "stream": False, "keep_alive": "2m",
                                                "options": {"num_ctx": MODEL_SETTINGS["num_ctx"]}})
        response.raise_for_status()
        assert response.json()["done"]
    result = {"source_commit": plan["source_commit"], "plan_sha256": sha(plan_bytes),
              "case_set_sha256": plan["case_set_sha256"], "model_used": runtime,
              "preload_seconds": perf_counter() - preload_start, "records": [], "status": "collecting",
              "review_status": "pending", "reserved_cases_executed": 0, "retries": 0}
    write_json(path, result)
    for case in plan["cases"]:
        verify_baseline_runtime(plan["source_commit"])
        assert (output / "plan.json").read_bytes() == plan_bytes
        assert sha(CASES.read_bytes()) == plan["case_set_sha256"]
        for name, digest in {**plan["source_hashes"], **plan["fixture_hashes"]}.items():
            assert sha((ROOT / name).read_bytes()) == digest, name
        command = [sys.executable, "-m", "scripts.run_agent_investigation", "--model", case.get("model_override", MODEL),
                   "--question", case["request"]["question"], "--protocol-catalog", str(ROOT / plan["fixtures"][case["fixture"]])]
        for name in ("battery_id", "early", "late"):
            value = case["request"][name]
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        if case.get("resume_from_case"):
            previous = next(item for item in result["records"] if item["case_id"] == case["resume_from_case"])
            command.extend(["--resume", previous["run_id"]])
        print(f"Starting {case['case_id']}", flush=True)
        started = perf_counter()
        try:
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300, check=False)
            stdout, stderr, exit_code = process.stdout, process.stderr, process.returncode
        except subprocess.TimeoutExpired as error:
            stdout = (error.stdout or b"").decode("utf-8", errors="replace")
            stderr = (error.stderr or b"").decode("utf-8", errors="replace")
            exit_code = "process_timeout"
        prefix = "Local run record: "
        record_path = next((Path(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)), None)
        run = json.loads(record_path.read_bytes()) if record_path else None
        stdout_path = output / f"{case['case_id']}.stdout.txt"
        stderr_path = output / f"{case['case_id']}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8", newline="\n")
        stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
        item = {"case_id": case["case_id"], "process_exit": exit_code, "wall_seconds": perf_counter() - started,
                "stdout_sha256": sha(stdout_path.read_bytes()), "stderr_sha256": sha(stderr_path.read_bytes()),
                "run_id": run["run_id"] if run else None, "record_path": record_path.relative_to(ROOT).as_posix() if record_path else None,
                "record_sha256": sha(record_path.read_bytes()) if record_path else None,
                "status": run["status"] if run else "no_run_record",
                "stop_reason": run["stop_reason"] if run else None,
                "action": run["follow_up"]["action"] if run and run["follow_up"] else None}
        if record_path:
            report = record_path.with_suffix(".md")
            item["report_sha256"] = sha(report.read_bytes()) if report.exists() else None
        result["records"].append(item)
        write_json(path, result)
        print(json.dumps(item), flush=True)
    result["status"] = "collected"
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Collection output must stay under the project's outputs directory")
    output.mkdir(parents=True, exist_ok=True)
    prepare(output) if args.prepare else collect(output)


if __name__ == "__main__":
    main()

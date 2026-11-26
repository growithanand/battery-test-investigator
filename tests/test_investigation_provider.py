"""Exercise provider selection through the real CLI, graph and synthetic battery tools."""

import hashlib
import json
import sys

import httpx
import numpy as np
import pytest
from scipy.io import savemat

pytest.importorskip("langgraph", reason="Install the optional agent dependency layer")

from scripts import gemini_model, run_agent_investigation as runner, trace_inspection
from scripts.fetch_sample import SOURCE_URL
from scripts.investigation_findings import findings_schema
from scripts.model_tools import make_protocol_tool
from scripts.protocol_evidence import retrieve_protocol_evidence


MODEL = "gemini-test-model"
CANARY = "synthetic-private-value"
QUOTE = "Discharge used a constant current of 2 A."


@pytest.fixture
def project(tmp_path, monkeypatch):
    source = tmp_path / "data/raw/nasa/package-1/B0005.mat"
    source.parent.mkdir(parents=True)
    cycles = [{"type": "discharge", "ambient_temperature": 24,
               "data": {"Time": np.array([0, 1, 2, 3, 4]), "Current_measured": np.array([0, -2, -2, 0, 0]),
                        "Voltage_measured": np.array([4.2, 3.7, 2.7, 3.2, 3.3]),
                        "Temperature_measured": np.array([24, 25, 28, 27, 26]), "Capacity": capacity}}
              for capacity in (2.0, 1.5, 1.0)]
    savemat(source, {"B0005": {"cycle": cycles}})
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    (manifests / "B0005.json").write_text(json.dumps({"files": [{"archive_member": "B0005.mat", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}))
    text = (f"Data Description:\nSynthetic source. {QUOTE}\n\nFiles:\nB0005.mat Synthetic battery\n\n"
            "Data Structure:\ncycle:\n\tfor discharge the fields are:\n\t\tCapacity: reported capacity (Ahr)\n"
            "\tfor impedance the fields are:\n\t\tRe: resistance\n")
    document_path = tmp_path / "data/raw/nasa/protocols/package-1/README.txt"
    document_path.parent.mkdir(parents=True)
    document_path.write_bytes(text.encode("utf-8"))
    catalog = manifests / "protocol-sources.json"
    catalog.write_text(json.dumps({"schema_version": 1, "corpus_id": "synthetic-v1", "source_url": SOURCE_URL,
                                  "documents": [{"document_id": "synthetic-B0005", "applicable_batteries": ["B0005"],
                                                 "path": document_path.relative_to(tmp_path).as_posix(),
                                                 "sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(),
                                                 "size_bytes": len(document_path.read_bytes()),
                                                 "nested_package": "synthetic/package-1.zip", "archive_member": "README.txt",
                                                 "cache_member": "package-1/README.txt"}]}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(trace_inspection, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "make_protocol_tool", lambda path: make_protocol_tool(path, root=tmp_path))
    passage = retrieve_protocol_evidence("B0005", catalog, root=tmp_path)["evidence"][0]
    return tmp_path, catalog, passage


@pytest.fixture
def api(monkeypatch, project):
    _, _, passage = project
    state = {"requests": [], "clients": [], "inspect": False, "split_tools": False, "failure": None, "missing_usage": False}
    real_client = httpx.Client

    def respond(request):
        assert request.url == f"{gemini_model.API_ROOT}/{MODEL}:generateContent"
        assert request.headers["x-goog-api-key"] == CANARY and not request.url.query
        payload = json.loads(request.content)
        state["requests"].append(payload)
        if state["failure"] == "timeout":
            raise httpx.ReadTimeout(CANARY, request=request)
        if state["failure"] == "quota":
            return httpx.Response(429, json={"error": {"message": CANARY}})
        if "tools" in payload:
            requested = json.loads(next(part["text"] for content in payload["contents"] if content["role"] == "user" for part in content["parts"] if "text" in part))
            parts = []
            definitions = payload["tools"][0]["functionDeclarations"]
            if state["split_tools"]:
                definitions = definitions[:1]
            for definition in definitions:
                args = {"battery_id": requested["battery_id"]}
                if definition["name"] == "compare_discharge_capacity":
                    args.update(early=requested["early"], late=requested["late"])
                parts.append({"functionCall": {"name": definition["name"], "args": args, "id": f"call-{len(state['requests'])}-{len(parts)}"},
                              "thoughtSignature": "opaque-test-signature"})
        else:
            context = json.loads(payload["contents"][0]["parts"][0]["text"])
            if payload["generationConfig"]["responseJsonSchema"]["title"] == "FollowUpDecision":
                result = {"action": "inspect_discharge_traces" if state["inspect"] else "stop_with_partial",
                          "question_scope": "recorded_extrema" if state["inspect"] else "reported_capacity",
                          "reason": "The requested extrema need trace inspection." if state["inspect"] else "The capacity calculation answers the request."}
            else:
                result = {"protocol_claims": [{"statement": "The synthetic protocol describes a 2 A discharge.",
                                               "evidence_id": passage["evidence_id"], "quote": QUOTE}],
                          "comparability": "not_established", "physical_cause": "not_determined",
                          "trace_evidence_id": context["trace_evidence"]["evidence_id"] if context["trace_evidence"] else None,
                          "limitations": ["The documented conditions do not prove measured compliance."],
                          "next_check": context["recommendation_guidance"]["next_check_hint"]}
                if state["failure"] == "citation_id":
                    result["protocol_claims"][0]["evidence_id"] = "previous-investigation-passage"
                elif state["failure"] == "quote":
                    result["protocol_claims"][0]["quote"] = "This invented quote does not occur in the source."
            parts = [{"text": json.dumps(result)}]
        body = {"modelVersion": MODEL, "candidates": [{"content": {"role": "model", "parts": parts},
                                                       "finishReason": "MAX_TOKENS" if state["failure"] == "truncated" else "STOP"}]}
        if not state["missing_usage"]:
            body["usageMetadata"] = {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 5, "totalTokenCount": 125}
        return httpx.Response(200, json=body)

    def client(**kwargs):
        assert kwargs == {"headers": {"x-goog-api-key": CANARY}, "timeout": 40.0, "trust_env": False, "follow_redirects": False}
        result = real_client(**kwargs, transport=httpx.MockTransport(respond))
        state["clients"].append(result)
        return result

    monkeypatch.setattr(gemini_model, "read_gemini_key", lambda: CANARY)
    monkeypatch.setattr(gemini_model.httpx, "Client", client)
    monkeypatch.setattr(runner, "create_local_model", lambda *args: pytest.fail("Gemini must not call Ollama"))
    return state


def command(monkeypatch, catalog, *flags):
    monkeypatch.setattr(sys, "argv", ["run_agent_investigation", "--provider", "gemini", "--model", MODEL,
                                     "--protocol-catalog", str(catalog), *flags])
    runner.main()


@pytest.mark.parametrize("inspect, split_tools", [(False, False), (True, False), (True, True)])
def test_cli_records_selected_provider_usage_and_validated_evidence(project, api, monkeypatch, capsys, inspect, split_tools):
    root, catalog, _ = project
    api.update(inspect=inspect, split_tools=split_tools)
    command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--late", "2",
            "--question", "Locate extrema." if inspect else "Report the capacity change.")
    path = next((root / "outputs/agent-investigations").glob("*.json"))
    saved = json.loads(path.read_bytes())
    expected = 4 if split_tools else 3
    assert saved["status"] == "partial"
    assert saved["configuration"]["provider"] == "gemini"
    assert saved["configuration"]["model_used"]["model"] == MODEL
    assert saved["comparison"]["comparison"]["capacity_change_Ah"] == -0.5
    assert saved["model_attempts"] == len(api["requests"]) == expected
    assert saved["tool_attempts"] == (3 if inspect else 2)
    assert saved["model_usage"] == {"complete": True, "requests_with_usage": expected, "model_attempts": expected,
                                    "input_tokens": 100 * expected, "output_tokens": 25 * expected, "total_tokens": 125 * expected}
    assert len(saved["model_requests"]) == expected
    assert all(client.is_closed for client in api["clients"])
    if inspect:
        assert saved["trace_evidence"]["selection"] == saved["comparison"]["selection"]
        assert saved["findings"]["trace_evidence_id"] == saved["trace_evidence"]["evidence_id"]
    else:
        assert saved["trace_evidence"] is None
    report = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Model provider: gemini." in report and f"Usage available for {expected} of {expected}" in report
    assert QUOTE in report
    displayed = capsys.readouterr().out
    assert all(value not in path.read_text(encoding="utf-8") + report + displayed
               for value in (CANARY, "opaque-test-signature", "gemini_content"))


@pytest.mark.parametrize("failure", ["quota", "timeout", "truncated"])
def test_cli_saves_safe_failed_run_without_retry_or_losing_returned_usage(project, api, monkeypatch, capsys, failure):
    root, catalog, _ = project
    api["failure"] = failure
    with pytest.raises(SystemExit) as error:
        command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--late", "2")
    assert error.value.code == 2 and len(api["requests"]) == 1
    path = next((root / "outputs/agent-investigations").glob("*.json"))
    saved = json.loads(path.read_bytes())
    assert saved["status"] == "failed" and saved["findings"] is None
    assert saved["model_usage"]["complete"] == (failure == "truncated")
    assert saved["model_usage"]["total_tokens"] == (125 if failure == "truncated" else 0)
    assert all(client.is_closed for client in api["clients"])
    report = path.with_suffix(".md").read_text(encoding="utf-8")
    assert ("Token usage is incomplete" in report) == (failure != "truncated")
    assert CANARY not in path.read_text(encoding="utf-8") + report + capsys.readouterr().out


def test_cli_missing_usage_is_explicit_in_saved_record_and_report(project, api, monkeypatch):
    root, catalog, _ = project
    api["missing_usage"] = True
    command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--late", "2")
    path = next((root / "outputs/agent-investigations").glob("*.json"))
    saved = json.loads(path.read_bytes())
    assert saved["status"] == "partial" and saved["model_usage"]["complete"] is False
    assert saved["model_usage"]["requests_with_usage"] == 0
    assert "Token usage is incomplete" in path.with_suffix(".md").read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", [None, "citation_id", "quote"])
def test_gemini_uses_current_citation_choices_and_still_rejects_invalid_evidence(project, api, monkeypatch, failure):
    root, catalog, passage = project
    api["failure"] = failure
    if failure:
        with pytest.raises(SystemExit) as error:
            command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--late", "2")
        assert error.value.code == 2
    else:
        command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--late", "2")
    schema = api["requests"][-1]["generationConfig"]["responseJsonSchema"]
    reference = schema["$defs"]["RetrievedProtocolClaim"]["properties"]["evidence_id"]
    saved = json.loads(next((root / "outputs/agent-investigations").glob("*.json")).read_bytes())
    assert reference["enum"] == [item["evidence_id"] for item in saved["protocol"]["evidence"]]
    assert passage["evidence_id"] in reference["enum"]
    assert saved["configuration"]["findings_schema_sha256"] == hashlib.sha256(json.dumps(
        findings_schema(saved["protocol"]).model_json_schema(), sort_keys=True,
    ).encode("utf-8")).hexdigest()
    assert schema["properties"]["protocol_claims"]["minItems"] == 1
    assert schema["properties"]["protocol_claims"]["maxItems"] == 2
    assert (saved["findings"] is None) == bool(failure)
    assert saved["status"] == ("failed" if failure else "partial")
    assert saved["comparison"]["comparison"]["capacity_change_Ah"] == -0.5
    assert len(api["requests"]) == saved["model_attempts"] == 3
    assert saved["tool_attempts"] == 2
    assert all(client.is_closed for client in api["clients"])
    if failure:
        assert saved["stop_reason"] == "findings_error"
        if failure == "quote":
            assert saved["trace"][-1]["detail"] == "Citation quote is not an exact substring of its passage"


@pytest.mark.parametrize("provider_flags", [[], ["--provider", "gemini"]])
def test_clarification_precedes_model_key_and_source_access(monkeypatch, tmp_path, provider_flags):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "create_local_model", lambda *args: pytest.fail("No Ollama preflight"))
    monkeypatch.setattr(runner, "create_gemini_model", lambda *args: pytest.fail("No Gemini key or client access"))
    monkeypatch.setattr(sys, "argv", ["run_agent_investigation", *provider_flags, "--model", MODEL, "--battery-id", "B0005"])
    runner.main()
    path = next((tmp_path / "outputs/agent-investigations").glob("*.json"))
    saved = json.loads(path.read_bytes())
    assert saved["status"] == "needs_clarification"
    assert saved["model_attempts"] == saved["tool_attempts"] == 0
    assert saved["configuration"]["provider"] == ("gemini" if provider_flags else "ollama")
    assert saved["configuration"]["model_used"] is None
    assert saved["model_usage"]["complete"]


def test_callable_engine_returns_clarification_without_printing_or_writing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "create_gemini_model", lambda *args: pytest.fail("No key or client access"))
    result = runner.run_investigation(
        {"question": "Compare reported capacities.", "battery_id": "B0005", "early": None, "late": None},
        model=MODEL, provider="gemini",
    )
    assert result["status"] == "needs_clarification"
    assert result["model_attempts"] == result["tool_attempts"] == 0
    assert "messages" not in result and not result["model_requests"]
    assert not (tmp_path / "outputs").exists()
    assert capsys.readouterr().out == ""


def test_gemini_can_resume_a_saved_clarification_without_changing_the_parent(project, api, monkeypatch):
    root, catalog, _ = project
    command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--question", "Report capacity change.")
    directory = root / "outputs/agent-investigations"
    parent = next(directory.glob("*.json"))
    previous = parent.read_bytes()
    assert not api["requests"]
    command(monkeypatch, catalog, "--resume", parent.stem, "--late", "2")
    child = next(path for path in directory.glob("*.json") if path != parent)
    saved = json.loads(child.read_bytes())
    assert saved["resumed_from_run_id"] == parent.stem
    assert saved["request"]["question"] == "Report capacity change."
    assert saved["comparison"]["comparison"]["capacity_change_Ah"] == -0.5
    assert parent.read_bytes() == previous and len(api["requests"]) == 3


def test_missing_gemini_key_saves_a_failed_run_and_never_tries_ollama(project, monkeypatch):
    root, catalog, _ = project
    monkeypatch.setattr(gemini_model, "read_gemini_key", lambda: (_ for _ in ()).throw(ValueError("GEMINI_API_KEY is missing.")))
    monkeypatch.setattr(runner, "create_local_model", lambda *args: pytest.fail("No provider fallback"))
    with pytest.raises(SystemExit) as error:
        command(monkeypatch, catalog, "--battery-id", "B0005", "--early", "1", "--late", "2")
    assert error.value.code == 2
    saved = json.loads(next((root / "outputs/agent-investigations").glob("*.json")).read_bytes())
    assert saved["status"] == "failed" and saved["configuration"]["model_used"] is None
    assert saved["model_usage"]["complete"] is False

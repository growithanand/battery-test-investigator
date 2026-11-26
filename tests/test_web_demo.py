"""Exercise the local HTTP boundary without network calls or paid inference."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sys
from threading import Event

import pytest

pytest.importorskip("fastapi", reason="Install the optional browser-interface dependencies")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from scripts import serve_demo as web


QUESTION = {"question": "Compare the reported capacities.", "early": 1, "late": 168}
CANARY = "synthetic-private-value"


@pytest.fixture
def client():
    calls = []

    def engine(body):
        calls.append(body.model_dump())
        return {"synthetic_result": True}

    with TestClient(web.create_app(live_enabled=True, limit=2, cooldown=0, engine=engine), base_url="http://127.0.0.1") as test:
        token = test.get("/api/session").json()["token"]
        test.headers.update({"X-Investigation-Token": token})
        yield test, calls


def test_http_request_reaches_only_the_allowed_engine_fields(client):
    test, calls = client
    response = test.post("/api/investigations", json={**QUESTION, "question": "  Compare capacities.  "})
    assert response.status_code == 200
    assert calls == [{"question": "Compare capacities.", "early": 1, "late": 168}]
    assert response.json()["allowance"] == {"remaining": 1, "limit": 2, "busy": False, "retry_after_seconds": 0}
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("changes", [
    {"early": True}, {"early": "1"}, {"early": 0}, {"late": 169}, {"early": 168}, {"late": 1},
    {"question": " "}, {"question": "x" * 601}, {"question": "control\x00character"},
    {"provider": "ollama"}, {"model": "different-model"}, {"protocol_catalog": "private.txt"},
    {"api_key": CANARY}, {"resume": "a" * 32}, {"battery_id": "../B0005"},
])
def test_invalid_or_extra_fields_stop_before_the_engine_and_allowance(client, changes):
    test, calls = client
    response = test.post("/api/investigations", json={**QUESTION, **changes})
    assert response.status_code == 422 and calls == []
    assert CANARY not in response.text
    assert test.get("/api/session").json()["remaining"] == 2


@pytest.mark.parametrize("headers", [
    {"X-Investigation-Token": ""}, {"X-Investigation-Token": "wrong"},
    {"Origin": "https://unrelated.example"}, {"Origin": "null"},
])
def test_missing_token_or_foreign_origin_cannot_start_inference(client, headers):
    test, calls = client
    assert test.post("/api/investigations", json=QUESTION, headers=headers).status_code == 403
    assert not calls


def test_same_origin_is_accepted_but_foreign_host_is_rejected(client):
    test, calls = client
    assert test.post("/api/investigations", json=QUESTION, headers={"Origin": "http://127.0.0.1"}).status_code == 200
    assert test.get("/api/session", headers={"Host": "unrelated.example"}).status_code == 400
    assert len(calls) == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_body_byte_limit_precedes_validation_and_inference(client, chunked):
    test, calls = client
    content = (part for part in [b"x" * 1024] * 5) if chunked else b"x" * 5000
    response = test.post("/api/investigations", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 413 and calls == []
    assert test.get("/api/session").json()["remaining"] == 2


def test_html_form_and_malformed_json_are_rejected_without_echoing_input(client):
    test, calls = client
    assert test.post("/api/investigations", content="question=test", headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code == 415
    response = test.post("/api/investigations", content='{"private":"' + CANARY, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and CANARY not in response.text and not calls


@pytest.mark.parametrize("path", ["/.env", "/scripts/gemini_config.py", "/outputs/agent-investigations/a.json", "/docs", "/openapi.json"])
def test_no_private_files_or_documentation_routes_are_served(client, path):
    test, calls = client
    assert test.get(path).status_code == 404
    assert not calls


def test_interface_assets_load_without_enabling_paid_calls():
    def engine(body):
        pytest.fail("Disabled session must not run the engine")

    with TestClient(web.create_app(engine=engine), base_url="http://localhost") as test:
        settings = test.get("/api/session").json()
        assert settings["enabled"] is False and settings["remaining"] == 5
        for path in ["/", "/styles.css", "/app.js"]:
            assert test.get(path).status_code == 200
        javascript = test.get("/app.js").text
        assert "textContent" in javascript and "innerHTML" not in javascript
        response = test.post("/api/investigations", json=QUESTION, headers={"X-Investigation-Token": settings["token"]})
        assert response.status_code == 503 and test.get("/api/session").json()["remaining"] == 5


def test_failures_consume_the_allowance_and_never_echo_exception_details():
    calls = []

    def engine(body):
        calls.append(body)
        raise ValueError(CANARY + " C:/private/file.env")

    with TestClient(web.create_app(live_enabled=True, limit=1, cooldown=0, engine=engine), base_url="http://localhost") as test:
        headers = {"X-Investigation-Token": test.get("/api/session").json()["token"]}
        response = test.post("/api/investigations", json=QUESTION, headers=headers)
        assert response.status_code == 503 and CANARY not in response.text and "private" not in response.text
        settings = test.get("/api/session").json()
        assert settings["remaining"] == 0 and settings["busy"] is False
        assert test.post("/api/investigations", json=QUESTION, headers=headers).status_code == 429
        assert len(calls) == 1


def test_parallel_submission_is_rejected_before_a_second_engine_call():
    entered, release = Event(), Event()
    calls = []

    def engine(body):
        calls.append(body)
        entered.set()
        assert release.wait(5)
        return {"synthetic_result": True}

    with TestClient(web.create_app(live_enabled=True, limit=2, cooldown=0, engine=engine), base_url="http://localhost") as test:
        headers = {"X-Investigation-Token": test.get("/api/session").json()["token"]}
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(test.post, "/api/investigations", json=QUESTION, headers=headers)
            try:
                assert entered.wait(5)
                assert test.post("/api/investigations", json=QUESTION, headers=headers).status_code == 409
                assert test.get("/api/session").json()["remaining"] == 1
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
        assert len(calls) == 1


def test_cooldown_uses_elapsed_time_without_spending_another_attempt():
    now = [100.0]
    allowance = web.RunAllowance(limit=2, cooldown=15, clock=lambda: now[0])
    allowance.reserve()
    allowance.release()
    now[0] += 1
    with pytest.raises(HTTPException) as error:
        allowance.reserve()
    assert error.value.status_code == 429 and error.value.headers == {"Retry-After": "14"}
    assert allowance.status()["remaining"] == 1
    now[0] += 14
    allowance.reserve()
    allowance.release()
    assert allowance.status()["remaining"] == 0


def test_web_adapter_fixes_model_and_source_and_publishes_only_whitelisted_fields(tmp_path, monkeypatch):
    result = {
        "run_id": "a" * 32, "request": {**QUESTION, "battery_id": "B0005"},
        "status": "failed", "stop_reason": "model_or_tool_request_error", "model_attempts": 1, "tool_attempts": 0,
        "clarification": None, "comparison": None, "protocol": None, "trace_evidence": None,
        "follow_up": None, "findings": None, "elapsed_seconds": 1.0,
        "model_usage": {"complete": False, "input_tokens": 0, "output_tokens": 0, "requests_with_usage": 0, "model_attempts": 1},
        "configuration": {"provider": "gemini", "model_requested": web.MODEL, "private": CANARY},
        "trace": [{"private": CANARY}],
    }
    calls = []

    def engine(request, **kwargs):
        calls.append((request, kwargs))
        return deepcopy(result)

    monkeypatch.setattr(web.investigator, "ROOT", tmp_path)
    monkeypatch.setattr(web.investigator, "run_investigation", engine)
    public = web.execute_investigation(web.DemoRequest(**QUESTION))
    assert calls == [(result["request"], {"provider": "gemini", "model": web.MODEL})]
    assert CANARY not in json.dumps(public) and str(tmp_path) not in json.dumps(public)
    directory = tmp_path / "outputs/agent-investigations"
    assert json.loads((directory / ("a" * 32 + ".json")).read_text()) == result
    assert (directory / ("a" * 32 + ".md")).read_text() == public["report"]


@pytest.mark.parametrize("enabled", [False, True])
def test_launch_is_one_process_on_loopback_with_explicit_live_opt_in(monkeypatch, enabled):
    calls = []
    monkeypatch.setattr(web.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    monkeypatch.setattr(sys, "argv", ["serve_demo", *(["--enable-live"] if enabled else [])])
    web.main()
    app, settings = calls[0]
    assert settings == {"host": "127.0.0.1", "port": 8000, "workers": 1, "reload": False,
                        "proxy_headers": False, "access_log": False, "log_level": "warning"}
    with TestClient(app, base_url="http://127.0.0.1") as test:
        assert test.get("/api/session").json()["enabled"] == enabled

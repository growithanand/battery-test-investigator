"""Mock HTTP proves local preflight rejection; no real model/network is used."""

import httpx
import pytest

pytest.importorskip("langchain_ollama", reason="Optional local model dependency layer")

from scripts import local_model


@pytest.fixture
def server(monkeypatch):
    state = {"models": [{"name": "local-test:1", "digest": "synthetic-digest"}],
             "details": {"capabilities": ["completion", "tools"]}, "requests": [], "settings": None}
    real_client = httpx.Client

    def respond(request):
        state["requests"].append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": state["models"]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=state["details"])
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "synthetic-server"})
        pytest.fail(f"Unregistered endpoint: {request.url}")

    def client(**kwargs):
        assert kwargs == {"base_url": "http://127.0.0.1:11434", "timeout": 10.0, "trust_env": False}
        return real_client(**kwargs, transport=httpx.MockTransport(respond))

    def model(**kwargs):
        state["settings"] = kwargs
        return "synthetic-model-object"

    monkeypatch.setattr(local_model.httpx, "Client", client)
    monkeypatch.setattr(local_model, "ChatOllama", model)
    return state


def test_local_preflight_preserves_identity_and_explicit_settings(server):
    model, metadata = local_model.create_local_model("local-test:1", {"temperature": 0, "reasoning": False})
    assert model == "synthetic-model-object" and metadata["model_digest"] == "synthetic-digest"
    assert metadata["ollama_version"] == "synthetic-server"
    assert server["settings"]["base_url"] == "http://127.0.0.1:11434"
    assert server["settings"]["client_kwargs"] == {"timeout": 60.0, "trust_env": False}
    assert server["settings"]["reasoning"] is False
    assert [request.url.path for request in server["requests"]] == ["/api/tags", "/api/show", "/api/version"]


@pytest.mark.parametrize("details", [
    {"capabilities": ["completion"]},
    {"capabilities": ["tools"], "remote_model": "remote-only"},
    {"capabilities": ["tools"], "remote_host": "https://remote.invalid"},
])
def test_non_tool_or_remote_model_is_rejected_before_model_creation(server, details):
    server["details"] = details
    with pytest.raises(ValueError, match="local model"):
        local_model.create_local_model("local-test:1", {})
    assert server["settings"] is None and len(server["requests"]) == 2


def test_missing_tag_never_triggers_pull_or_fallback(server):
    with pytest.raises(ValueError, match="already installed"):
        local_model.create_local_model("not-installed:1", {})
    assert server["settings"] is None
    assert [request.url.path for request in server["requests"]] == ["/api/tags"]

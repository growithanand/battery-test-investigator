"""Explicit local-only model preflight shared by investigation entry points."""

import httpx
from langchain_ollama import ChatOllama


LOCAL_URL = "http://127.0.0.1:11434"


def create_local_model(model_tag: str, settings: dict) -> tuple[ChatOllama, dict]:
    """Never pull models, use environment proxies, enable tracing, or fall back to a cloud API."""
    with httpx.Client(base_url=LOCAL_URL, timeout=10.0, trust_env=False) as client:
        response = client.get("/api/tags")
        response.raise_for_status()
        installed = next((item for item in response.json()["models"] if item["name"] == model_tag), None)
        if installed is None:
            raise ValueError("Choose an exact model tag already installed in local Ollama")
        response = client.post("/api/show", json={"model": model_tag})
        response.raise_for_status()
        details = response.json()
        if "tools" not in details.get("capabilities", []) or details.get("remote_model") or details.get("remote_host"):
            raise ValueError("A local model with tool-calling support is required")
        response = client.get("/api/version")
        response.raise_for_status()
        server_version = response.json()["version"]
    model = ChatOllama(
        model=model_tag, base_url=LOCAL_URL, **settings,
        client_kwargs={"timeout": 60.0, "trust_env": False},
    )
    return model, {"model": model_tag, "model_digest": installed["digest"],
                   "ollama_version": server_version, "settings": settings}

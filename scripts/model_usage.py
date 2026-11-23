"""Summarize returned token counts without presenting missing usage as zero cost."""


def summarize_model_usage(state: dict, requests: list[dict] | None = None) -> dict:
    """Use provider request records when present, including rejected or truncated output."""
    if requests is None:
        entries = [event.get("usage") for event in state["trace"] if event["event"] in
                   ("tools_requested", "follow_up_selected", "citation_references_checked")]
    else:
        entries = [request.get("usage") for request in requests]
    names = ("input_tokens", "output_tokens", "total_tokens")
    known = [entry for entry in entries if isinstance(entry, dict) and
             all(type(entry.get(name)) is int and entry[name] >= 0 for name in names)]
    attempts = state["model_attempts"]
    return {"complete": len(known) == len(entries) == attempts,
            "requests_with_usage": len(known), "model_attempts": attempts,
            **{name: sum(entry[name] for entry in known) for name in names}}

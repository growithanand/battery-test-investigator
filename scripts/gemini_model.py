"""Gemini REST adapter for the investigator's tool and structured-output calls."""

from copy import deepcopy
from functools import partial
import json
import re
from time import perf_counter

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from scripts.gemini_config import read_gemini_key


API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_PATTERN = re.compile(r"gemini-[a-z0-9][a-z0-9._-]{0,99}\Z")
MODEL_SETTINGS = {"temperature": 0, "maxOutputTokens": 1024,
                  "thinkingConfig": {"thinkingLevel": "minimal"}}


class GeminiModelError(ValueError):
    """An error safe to include in a saved investigation or terminal output."""


def schema_for_api(value):
    """Represent single-value Pydantic literals as JSON Schema enums."""
    if isinstance(value, list):
        return [schema_for_api(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: schema_for_api(item) for key, item in value.items() if key != "const"}
    if "const" in value:
        result["enum"] = [value["const"]]
    return result


def encode_messages(messages):
    """Return original model content, including opaque signatures, with tool replies."""
    instructions, contents, calls = [], [], {}
    for message in messages:
        if isinstance(message, (SystemMessage, HumanMessage)) and not isinstance(message.content, str):
            raise GeminiModelError("Gemini investigation messages must contain text.")
        if isinstance(message, SystemMessage):
            instructions.append({"text": message.content})
        elif isinstance(message, HumanMessage):
            contents.append({"role": "user", "parts": [{"text": message.content}]})
        elif isinstance(message, AIMessage):
            contents.append(deepcopy(message.additional_kwargs["gemini_content"]))
            calls.update(message.additional_kwargs["gemini_calls"])
        elif isinstance(message, ToolMessage):
            name, provider_id = calls[message.tool_call_id]
            value = json.loads(message.content)
            if not isinstance(value, dict):
                raise GeminiModelError("Gemini tool results must be JSON objects.")
            result = {"name": name, "response": value}
            if provider_id is not None:
                result["id"] = provider_id
            part = {"functionResponse": result}
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        else:
            raise GeminiModelError("Unsupported Gemini investigation message.")
    payload = {"contents": contents}
    if instructions:
        payload["systemInstruction"] = {"parts": instructions}
    return payload


def read_usage(body):
    """Keep missing or inconsistent provider counts unknown rather than invent totals."""
    usage = body.get("usageMetadata")
    if not isinstance(usage, dict) or not {"promptTokenCount", "totalTokenCount"} <= usage.keys():
        return None
    names = ("promptTokenCount", "toolUsePromptTokenCount", "candidatesTokenCount",
             "thoughtsTokenCount", "totalTokenCount", "cachedContentTokenCount")
    counts = {name: usage.get(name, 0) for name in names}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        return None
    inputs = counts["promptTokenCount"] + counts["toolUsePromptTokenCount"]
    outputs = counts["candidatesTokenCount"] + counts["thoughtsTokenCount"]
    if inputs + outputs != counts["totalTokenCount"] or counts["cachedContentTokenCount"] > counts["promptTokenCount"]:
        return None
    return {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": counts["totalTokenCount"]}


class GeminiChat:
    """Implement the two model operations used by the bounded investigation graph."""

    def __init__(self, model_tag: str, client: httpx.Client):
        if not MODEL_PATTERN.fullmatch(model_tag):
            raise GeminiModelError("Choose a Gemini model ID without a URL, path or query string.")
        self.model_tag = model_tag
        self.client = client
        self.calls = []

    def close(self):
        self.client.close()

    def bind_tools(self, tools):
        definitions = [{"name": tool.name, "description": tool.description,
                        "parametersJsonSchema": schema_for_api(tool.args_schema.model_json_schema())}
                       for tool in tools]
        return RunnableLambda(partial(self.invoke, definitions=definitions))

    def with_structured_output(self, schema, *, method, include_raw):
        if method != "json_schema" or include_raw is not True:
            raise GeminiModelError("Gemini findings require JSON schema and response metadata.")
        return RunnableLambda(partial(self.invoke, schema=schema))

    def invoke(self, messages, *, definitions=None, schema=None):
        try:
            config = deepcopy(MODEL_SETTINGS)
            payload = {**encode_messages(messages), "generationConfig": config}
            if definitions:
                payload["tools"] = [{"functionDeclarations": definitions}]
            if schema is not None:
                config.update(responseMimeType="application/json",
                              responseJsonSchema=schema_for_api(schema.model_json_schema()))
        except (KeyError, TypeError, ValueError):
            raise GeminiModelError("Could not encode the Gemini investigation request.") from None

        call = {"http_status": None, "seconds": None, "model_version": None, "usage": None}
        self.calls.append(call)
        started = perf_counter()
        try:
            response = self.client.post(f"{API_ROOT}/{self.model_tag}:generateContent", json=payload)
            call["http_status"] = response.status_code
        except httpx.TimeoutException:
            call["error"] = "timeout"
            raise GeminiModelError("Gemini request timed out; no automatic retry was made.") from None
        except httpx.RequestError:
            call["error"] = "connection_error"
            raise GeminiModelError("Could not connect to Gemini; no automatic retry was made.") from None
        finally:
            call["seconds"] = round(perf_counter() - started, 6)
        if response.status_code == 429:
            raise GeminiModelError("Gemini quota or rate limit was exceeded (HTTP 429). Check your API usage before trying again.")
        if response.status_code != 200:
            raise GeminiModelError(f"Gemini returned HTTP {response.status_code}. Check API access and model availability.")
        try:
            body = response.json()
            call["usage"] = read_usage(body)
            model_version = body.get("modelVersion")
            if isinstance(model_version, str) and MODEL_PATTERN.fullmatch(model_version):
                call["model_version"] = model_version
            candidate = body["candidates"][0]
            finish = candidate.get("finishReason")
            call["finish_reason"] = finish if finish in ("STOP", "MAX_TOKENS") else "OTHER"
            if finish != "STOP":
                raise GeminiModelError("Gemini response was blocked or incomplete; no findings were accepted.")
            content = candidate["content"]
            if content["role"] != "model" or not content["parts"]:
                raise GeminiModelError("Gemini returned an empty or malformed response.")
            tool_calls, ids, text = [], {}, []
            for part in content["parts"]:
                if function := part.get("functionCall"):
                    name, arguments, provider_id = function["name"], function["args"], function.get("id")
                    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                        raise GeminiModelError("Gemini returned a malformed tool call.")
                    if provider_id is not None and (not isinstance(provider_id, str) or not provider_id):
                        raise GeminiModelError("Gemini returned an invalid tool call ID.")
                    call_id = provider_id if provider_id is not None else f"gemini-{len(self.calls)}-{len(tool_calls)}"
                    ids[call_id] = (name, provider_id)
                    tool_calls.append({"name": name, "args": arguments, "id": call_id, "type": "tool_call"})
                if not part.get("thought") and "text" in part:
                    if not isinstance(part["text"], str):
                        raise GeminiModelError("Gemini returned malformed text.")
                    text.append(part["text"])
            raw = AIMessage(content="".join(text), tool_calls=tool_calls, usage_metadata=call["usage"],
                            additional_kwargs={"gemini_content": content, "gemini_calls": ids})
            if schema is None:
                return raw
            if tool_calls:
                raise GeminiModelError("Gemini returned tools instead of structured findings.")
            try:
                parsed = schema.model_validate_json(raw.content)
            except ValueError:
                raise GeminiModelError("Gemini returned invalid structured findings.") from None
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        except GeminiModelError:
            raise
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            raise GeminiModelError("Gemini returned a malformed response.") from None


def create_gemini_model(model_tag: str) -> tuple[GeminiChat, dict]:
    """Create a client on explicit selection; credentials remain outside saved metadata."""
    if not MODEL_PATTERN.fullmatch(model_tag):
        raise GeminiModelError("Choose a Gemini model ID without a URL, path or query string.")
    try:
        key = read_gemini_key()
    except (OSError, UnicodeError):
        raise GeminiModelError("Could not read .env. Check its permissions and UTF-8 encoding.") from None
    client = httpx.Client(headers={"x-goog-api-key": key}, timeout=40.0, trust_env=False, follow_redirects=False)
    model = GeminiChat(model_tag, client)
    return model, {"provider": "gemini", "model": model_tag, "settings": deepcopy(MODEL_SETTINGS),
                   "timeout_seconds": 40.0, "automatic_retries": 0}

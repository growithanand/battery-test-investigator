"""Serve the bounded B0005 investigator on this computer's loopback interface."""

import argparse
import json
import math
from pathlib import Path
import secrets
from threading import Lock
from time import monotonic

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import uvicorn

from scripts import run_agent_investigation as investigator
from scripts.investigation_report import render_investigation_report


WEB = Path(__file__).resolve().parents[1] / "web"
MODEL = "gemini-3.5-flash-lite"
MAX_RUNS = 5
COOLDOWN_SECONDS = 15


class DemoRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    question: str = Field(min_length=1, max_length=600)
    early: int = Field(ge=1, le=168)
    late: int = Field(ge=1, le=168)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value):
        if not value.strip() or any(ord(character) < 32 and character not in "\n\t\r" for character in value):
            raise ValueError("Enter an investigation question without control characters")
        return value.strip()

    @model_validator(mode="after")
    def ordered_selection(self):
        if self.early >= self.late:
            raise ValueError("The earlier discharge must come before the later discharge")
        return self


class RunAllowance:
    """Reserve attempts before execution; failures consume the same allowance."""

    def __init__(self, limit=MAX_RUNS, cooldown=COOLDOWN_SECONDS, clock=monotonic):
        if type(limit) is not int or not 1 <= limit <= MAX_RUNS or cooldown < 0:
            raise ValueError("Use a run limit from 1 to 5 and a nonnegative cooldown")
        self.limit, self.cooldown, self.clock = limit, cooldown, clock
        self.used, self.busy, self.last_started = 0, False, None
        self.lock = Lock()

    def status(self):
        with self.lock:
            wait = 0 if self.last_started is None else max(0, math.ceil(self.cooldown - (self.clock() - self.last_started)))
            return {"remaining": self.limit - self.used, "limit": self.limit,
                    "busy": self.busy, "retry_after_seconds": wait}

    def reserve(self):
        with self.lock:
            if self.busy:
                raise HTTPException(409, "An investigation is already running. Wait for it to finish.")
            if self.used >= self.limit:
                raise HTTPException(429, "This server session has reached its investigation limit.")
            now = self.clock()
            wait = 0 if self.last_started is None else math.ceil(self.cooldown - (now - self.last_started))
            if wait > 0:
                raise HTTPException(429, "Wait briefly before starting another investigation.", headers={"Retry-After": str(wait)})
            self.used += 1
            self.busy, self.last_started = True, now

    def release(self):
        with self.lock:
            self.busy = False


def execute_investigation(body: DemoRequest) -> dict:
    """Use the CLI engine and save its usual artifacts without exposing local paths."""
    result = investigator.run_investigation(
        {"question": body.question, "battery_id": "B0005", "early": body.early, "late": body.late},
        provider="gemini", model=MODEL,
    )
    report = render_investigation_report(result)
    directory = investigator.ROOT / "outputs/agent-investigations"
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"{result['run_id']}.json"
    investigator._write_artifact(record, json.dumps(result, indent=2) + "\n")
    investigator._write_artifact(record.with_suffix(".md"), report)
    return {
        "run_id": result["run_id"], "status": result["status"], "request": result["request"],
        "seconds": result["elapsed_seconds"], "comparison": result["comparison"],
        "protocol_status": result["protocol"]["status"] if result["protocol"] else None,
        "follow_up": result["follow_up"], "trace_evidence": result["trace_evidence"],
        "findings": result["findings"], "usage": result["model_usage"], "report": report,
    }


def create_app(*, live_enabled=False, limit=MAX_RUNS, cooldown=COOLDOWN_SECONDS, engine=None):
    """One local server process owns the allowance; no key is sent to the browser."""
    app = FastAPI(title="Battery Test Investigator", docs_url=None, redoc_url=None,
                  openapi_url=None, max_body_size=4096)
    allowance = RunAllowance(limit, cooldown)
    token = secrets.token_urlsafe(32)
    run = execute_investigation if engine is None else engine

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            same_origin = origin is None or origin == str(request.base_url).rstrip("/")
            supplied = request.headers.get("x-investigation-token", "")
            if not same_origin or not secrets.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
                response = JSONResponse({"detail": "Open the local investigation page before submitting."}, status_code=403)
            elif request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                response = JSONResponse({"detail": "Send the investigation as JSON."}, status_code=415)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers.update({
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
        })
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"], www_redirect=False)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        return JSONResponse({"detail": "Enter a question of 1–600 characters and discharge numbers with 1 ≤ earlier < later ≤ 168."}, status_code=422)

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/app.js")
    def javascript():
        return FileResponse(WEB / "app.js", media_type="text/javascript")

    @app.get("/styles.css")
    def stylesheet():
        return FileResponse(WEB / "styles.css", media_type="text/css")

    @app.get("/api/session")
    def session():
        return {"enabled": live_enabled, "model": MODEL, "token": token, **allowance.status()}

    @app.post("/api/investigations")
    def investigate(body: DemoRequest):
        if not live_enabled:
            raise HTTPException(503, "Live requests are off for this server session.")
        allowance.reserve()
        try:
            result = run(body)
        except Exception:
            # Never return exception text, provider bodies, keys or private file paths.
            return JSONResponse({"detail": "The investigation could not finish. Check the local setup before starting another run."}, status_code=503)
        finally:
            allowance.release()
        return {"result": result, "allowance": allowance.status()}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-live", action="store_true", help="Allow up to five Gemini investigations in this server session")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port from 1024 to 65535")
    print(f"Battery Test Investigator: http://127.0.0.1:{args.port}", flush=True)
    print("Live requests enabled (five-attempt allowance)." if args.enable_live else "Live requests off. Use --enable-live to run investigations.", flush=True)
    uvicorn.run(create_app(live_enabled=args.enable_live), host="127.0.0.1", port=args.port,
                workers=1, reload=False, proxy_headers=False, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()

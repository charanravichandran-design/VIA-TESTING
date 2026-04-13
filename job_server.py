#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import configparser
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = REPO_ROOT / "job_runs"
JOBS_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="VIA Benchmark Job Server", version="0.1.0")

# Allow CORS from any origin for now (can be restricted later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}
subscribers_lock = threading.Lock()
job_subscribers: dict[str, list[queue.Queue[str]]] = {}

ALLOWED_OVERRIDE_SECTIONS = {
    "connection",
    "dataset",
    "s3",
    "index",
    "index_creation",
    "query",
    "output",
    "optuna",
    "index_params",
    "query_params",
    "tuner_output",
    "loading",
    "benchmark",
    "results_s3",
}
FORBIDDEN_OPTUNA_KEYS = {"n_index_trials", "n_query_trials"}
TERMINAL_JOB_STATES = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sse_event(event: str, data: str) -> str:
    lines = data.splitlines() or [""]
    body = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{body}\n"


def subscribe_logs(job_id: str) -> queue.Queue[str]:
    q: queue.Queue[str] = queue.Queue(maxsize=1000)
    with subscribers_lock:
        job_subscribers.setdefault(job_id, []).append(q)
    return q


def unsubscribe_logs(job_id: str, q: queue.Queue[str]) -> None:
    with subscribers_lock:
        subs = job_subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)
        if not subs and job_id in job_subscribers:
            del job_subscribers[job_id]


def publish_log_line(job_id: str, line: str) -> None:
    with subscribers_lock:
        targets = list(job_subscribers.get(job_id, []))
    for q in targets:
        try:
            q.put_nowait(line)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(line)
            except queue.Full:
                pass


def to_ini_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def interpreter_has_module(python_executable: str, module_name: str) -> bool:
    try:
        res = subprocess.run(
            [python_executable, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return res.returncode == 0
    except Exception:
        return False


def is_results_s3_enabled(config_path: Path) -> bool:
    cfg = configparser.ConfigParser()
    if not cfg.read(config_path):
        return False
    return cfg.getboolean("results_s3", "enabled", fallback=False)


def resolve_python_executable(request: "JobCreateRequest", config_path: Path) -> str:
    explicit_python = "python_executable" in request.model_fields_set
    chosen = request.python_executable
    if explicit_python:
        return chosen

    if is_results_s3_enabled(config_path) and not interpreter_has_module(chosen, "boto3"):
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            venv_python_str = str(venv_python)
            if interpreter_has_module(venv_python_str, "boto3"):
                return venv_python_str
    return chosen


def read_tail(log_path: Path, limit: int) -> list[str]:
    if not log_path.exists():
        return []
    buf: deque[str] = deque(maxlen=max(1, limit))
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line.rstrip("\n"))
    return list(buf)


class JobCreateRequest(BaseModel):
    config_template_path: str = "config.ini"
    config_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    go_binary: str = "./vector_bench"
    python_executable: str = sys.executable
    resume: bool = False
    n_index_trials: Optional[int] = None
    n_query_trials: Optional[int] = None
    extra_args: list[str] = Field(default_factory=list)
    label: Optional[str] = None


def validate_request_payload(request: JobCreateRequest) -> None:
    if request.n_index_trials is not None and request.n_index_trials <= 0:
        raise ValueError("n_index_trials must be > 0")
    if request.n_query_trials is not None and request.n_query_trials <= 0:
        raise ValueError("n_query_trials must be > 0")

    unknown_sections = sorted(set(request.config_overrides.keys()) - ALLOWED_OVERRIDE_SECTIONS)
    if unknown_sections:
        raise ValueError(
            "Unsupported config_overrides sections: "
            + ", ".join(unknown_sections)
            + ". Allowed sections are: "
            + ", ".join(sorted(ALLOWED_OVERRIDE_SECTIONS))
        )

    for section, values in request.config_overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"config_overrides.{section} must be an object/dict")

    optuna_overrides = request.config_overrides.get("optuna", {})
    duplicate_keys = sorted(FORBIDDEN_OPTUNA_KEYS.intersection(optuna_overrides.keys()))
    if duplicate_keys:
        raise ValueError(
            "Do not set "
            + ", ".join(f"optuna.{k}" for k in duplicate_keys)
            + " in config_overrides. Use top-level n_index_trials / n_query_trials fields instead."
        )


def apply_overrides(cfg: configparser.ConfigParser, overrides: dict[str, dict[str, Any]]) -> None:
    for section, values in overrides.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        for key, value in values.items():
            cfg.set(section, key, to_ini_value(value))


def prepare_job_config(job_dir: Path, request: JobCreateRequest) -> tuple[Path, dict[str, str]]:
    template_path = resolve_path(request.config_template_path)
    cfg = configparser.ConfigParser()
    if not cfg.read(template_path):
        raise ValueError(f"Cannot read config template: {template_path}")

    artifacts = {
        "results_json": str(job_dir / "results.json"),
        "trial_log": str(job_dir / "trial_results.jsonl"),
        "pareto_report": str(job_dir / "pareto_report.json"),
        "optuna_storage": str(job_dir / "optuna_study.db"),
        "index_cache": str(job_dir / "index_cache.json"),
    }

    if not cfg.has_section("output"):
        cfg.add_section("output")
    cfg.set("output", "output_json", artifacts["results_json"])

    if not cfg.has_section("tuner_output"):
        cfg.add_section("tuner_output")
    cfg.set("tuner_output", "trial_log", artifacts["trial_log"])
    cfg.set("tuner_output", "pareto_report", artifacts["pareto_report"])
    cfg.set("tuner_output", "optuna_storage", artifacts["optuna_storage"])
    cfg.set("tuner_output", "index_cache", artifacts["index_cache"])

    apply_overrides(cfg, request.config_overrides)

    generated = job_dir / "config.generated.ini"
    with generated.open("w", encoding="utf-8") as f:
        cfg.write(f)
    return generated, artifacts


def build_command(request: JobCreateRequest, config_path: Path, python_executable: str) -> list[str]:
    go_binary = str(resolve_path(request.go_binary))
    cmd = [
        python_executable,
        "-u",
        str(REPO_ROOT / "optuna_tuner.py"),
        "--config",
        str(config_path),
        "--go-binary",
        go_binary,
    ]
    if request.resume:
        cmd.append("--resume")
    if request.n_index_trials is not None:
        cmd += ["--n-index-trials", str(request.n_index_trials)]
    if request.n_query_trials is not None:
        cmd += ["--n-query-trials", str(request.n_query_trials)]
    if request.extra_args:
        cmd.extend(request.extra_args)
    return cmd


def set_job(job_id: str, **kwargs: Any) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].update(kwargs)


def run_job(job_id: str, cmd: list[str], log_path: Path, artifacts: dict[str, str]) -> None:
    set_job(job_id, status="running", started_at=utc_now())
    rc = -1
    proc: Optional[subprocess.Popen[str]] = None

    with log_path.open("a", encoding="utf-8", buffering=1) as logf:
        start_line = f"[{utc_now()}] Starting job: {' '.join(cmd)}"
        logf.write(start_line + "\n")
        publish_log_line(job_id, start_line)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            set_job(
                job_id,
                status="failed",
                error=f"Failed to start process: {exc}",
                finished_at=utc_now(),
            )
            return

        set_job(job_id, pid=proc.pid)

        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            publish_log_line(job_id, line.rstrip("\n"))
            print(f"[job {job_id}] {line}", end="")

        rc = proc.wait()

    status = "completed" if rc == 0 else "failed"
    summary: dict[str, Any] = {}
    pareto_path = Path(artifacts["pareto_report"])
    if pareto_path.exists():
        try:
            report = json.loads(pareto_path.read_text(encoding="utf-8"))
            summary = {
                "total_trials": report.get("total_trials"),
                "pareto_front_size": report.get("pareto_front_size"),
            }
        except Exception:
            summary = {}

    set_job(
        job_id,
        status=status,
        return_code=rc,
        finished_at=utc_now(),
        artifacts=artifacts,
        summary=summary,
        error=None if rc == 0 else f"Process exited with code {rc}",
    )
    publish_log_line(job_id, f"[{utc_now()}] Job finished with status={status} code={rc}")


def get_job_or_404(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return job


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
def job_log_page(job_id: str) -> str:
    get_job_or_404(job_id)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Job {job_id} Logs</title>
  <style>
    body {{ font-family: ui-monospace, Menlo, monospace; margin: 0; background: #0f1115; color: #e6edf3; }}
    header {{ padding: 12px 16px; border-bottom: 1px solid #2a2f3a; display: flex; gap: 16px; align-items: center; }}
    .status {{ padding: 2px 8px; border-radius: 999px; background: #273244; }}
    .container {{ padding: 12px 16px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; line-height: 1.3; }}
  </style>
</head>
<body>
  <header>
    <div>Job: <b>{job_id}</b></div>
    <div>Status: <span id="status" class="status">loading...</span></div>
  </header>
  <div class="container">
    <pre id="logs"></pre>
  </div>
  <script>
        const jobId = {json.dumps(job_id)};
        const logEl = document.getElementById('logs');
        const statusEl = document.getElementById('status');
        const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

        function setStatus(s) {{
            statusEl.textContent = s;
        }}

        function appendLine(line) {{
            logEl.textContent += line + "\\n";
            window.scrollTo(0, document.body.scrollHeight);
        }}

        async function loadInitial() {{
            const r = await fetch(`/jobs/${{jobId}}/logs?limit=300`);
            const j = await r.json();
            setStatus(j.status);
            for (const line of j.lines) appendLine(line);
            return j.status;
        }}

        function openSSE() {{
            const es = new EventSource(`/jobs/${{jobId}}/logs/stream`);

            es.addEventListener('log', (ev) => appendLine(ev.data));

            es.addEventListener('status', (ev) => {{
                setStatus(ev.data);
                if (TERMINAL.has(ev.data)) {{
                    es.close();
                }}
            }});

            // Only reload if the job is still active — a terminal job closing the
            // SSE connection is expected; do not trigger a reload loop for it.
            es.onerror = () => {{
                if (TERMINAL.has(statusEl.textContent)) {{
                    es.close();
                    return;
                }}
                // Transient error on a live job: let EventSource auto-reconnect (it
                // does so by default). Only force a full reload after 10 s of silence.
                clearTimeout(openSSE._reloadTimer);
                openSSE._reloadTimer = setTimeout(() => {{
                    if (!TERMINAL.has(statusEl.textContent)) window.location.reload();
                }}, 10000);
            }};

            es.onopen = () => clearTimeout(openSSE._reloadTimer);
        }}
        openSSE._reloadTimer = null;

        async function main() {{
            const initialStatus = await loadInitial();

            if (TERMINAL.has(initialStatus)) {{
                // Job already finished — no need for SSE, nothing will be pushed.
                return;
            }}

            openSSE();
        }}
        main();
  </script>
</body>
</html>"""


@app.post("/jobs")
def create_job(request: JobCreateRequest) -> dict[str, Any]:
    job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    try:
        validate_request_payload(request)
        config_path, artifacts = prepare_job_config(job_dir, request)
        python_executable = resolve_python_executable(request, config_path)
        cmd = build_command(request, config_path, python_executable)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_path = job_dir / "runner.log"
    job_record = {
        "job_id": job_id,
        "label": request.label,
        "python_executable": cmd[0],
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "return_code": None,
        "error": None,
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "artifacts": {},
        "summary": {},
        "command": cmd,
    }
    with jobs_lock:
        jobs[job_id] = job_record

    t = threading.Thread(target=run_job, args=(job_id, cmd, log_path, artifacts), daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "python_executable": cmd[0],
        "job_dir": str(job_dir),
        "log_path": str(log_path),
        "config_path": str(config_path),
    }


@app.get("/jobs")
def list_jobs() -> dict[str, Any]:
    with jobs_lock:
        rows = list(jobs.values())
    rows.sort(key=lambda x: x["created_at"], reverse=True)
    return {"count": len(rows), "jobs": rows}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return get_job_or_404(job_id)


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    job = get_job_or_404(job_id)
    log_path = Path(job["log_path"])
    return {
        "job_id": job_id,
        "status": job["status"],
        "lines": read_tail(log_path, limit),
    }


@app.get("/jobs/{job_id}/logs/stream")
async def stream_job_logs(job_id: str, request: Request) -> StreamingResponse:
    job = get_job_or_404(job_id)
    client_addr = None
    try:
        client = request.client
        client_addr = f"{client.host}:{client.port}" if client is not None else None
    except Exception:
        client_addr = None

    print(f"[SSE] connection attempt for {job_id} from {client_addr}")

    async def event_generator():
        # Always send current status immediately so the client knows the starting state.
        yield sse_event("status", job["status"])
        yield "retry: 2000\n\n"

        # If the job is already finished there is nothing left to stream.
        if job["status"] in TERMINAL_JOB_STATES:
            print(f"[SSE] {job_id} already terminal ({job['status']}), not subscribing")
            return

        q = subscribe_logs(job_id)
        print(f"[SSE] subscribed {job_id} -> queue, client={client_addr}")
        try:
            while True:
                if await request.is_disconnected():
                    print(f"[SSE] client disconnected (is_disconnected) for {job_id} from {client_addr}")
                    break

                try:
                    line = await asyncio.to_thread(q.get, True, 1.0)
                    yield sse_event("log", line)
                except queue.Empty:
                    current = get_job_or_404(job_id)
                    if current["status"] in TERMINAL_JOB_STATES:
                        yield sse_event("status", current["status"])
                        break
                    yield ": keep-alive\n\n"
        finally:
            unsubscribe_logs(job_id, q)
            print(f"[SSE] unsubscribed {job_id} -> queue, client={client_addr}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = get_job_or_404(job_id)
    if job["status"] not in {"running", "queued"}:
        return {"job_id": job_id, "status": job["status"], "message": "Job is not running."}

    pid = job.get("pid")
    if not pid:
        set_job(job_id, status="cancelled", finished_at=utc_now(), error="Cancelled before process start.")
        return {"job_id": job_id, "status": "cancelled"}

    try:
        os.kill(int(pid), signal.SIGTERM)
        set_job(job_id, status="cancelling")
        return {"job_id": job_id, "status": "cancelling"}
    except ProcessLookupError:
        return {"job_id": job_id, "status": job["status"], "message": "Process already exited."}

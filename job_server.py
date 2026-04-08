#!/usr/bin/env python3
from __future__ import annotations

import configparser
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = REPO_ROOT / "job_runs"
JOBS_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="VIA Benchmark Job Server", version="0.1.0")

jobs_lock = threading.Lock()
jobs: dict[str, dict[str, Any]] = {}

ALLOWED_OVERRIDE_SECTIONS = {
    "connection",
    "dataset",
    "s3",
    "optuna",
    "index_params",
    "query_params",
}
FORBIDDEN_OPTUNA_KEYS = {"n_index_trials", "n_query_trials"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def build_command(request: JobCreateRequest, config_path: Path) -> list[str]:
    go_binary = str(resolve_path(request.go_binary))
    cmd = [
        request.python_executable,
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
        logf.write(f"[{utc_now()}] Starting job: {' '.join(cmd)}\n")
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


def get_job_or_404(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return job


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs")
def create_job(request: JobCreateRequest) -> dict[str, Any]:
    job_id = f"job-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    try:
        validate_request_payload(request)
        config_path, artifacts = prepare_job_config(job_dir, request)
        cmd = build_command(request, config_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log_path = job_dir / "runner.log"
    job_record = {
        "job_id": job_id,
        "label": request.label,
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

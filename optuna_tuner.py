#!/usr/bin/env python3
"""
optuna_tuner.py — Python/Optuna Orchestrator for the Go Vector Benchmark
=========================================================================

Architecture
────────────
  This script is responsible ONLY for optimization logic:
    1. Read configuration from config.ini
    2. Use Optuna to suggest parameter combinations
    3. Call the Go binary (vector_bench) as a subprocess for each trial
    4. Parse JSON metrics from the Go binary's stdout
    5. Report results back to Optuna
    6. Compute and output the Pareto front

  The Go binary (vector_bench) is responsible for ALL I/O-heavy work:
    • Connecting to Couchbase
    • Loading data (parallel goroutines)
    • Building the index (+ polling)
    • Running queries (parallel goroutines, closed-loop)
    • Computing and emitting metrics as a single JSON line to stdout

Communication between Python and Go
────────────────────────────────────
  Python → Go:   CLI flags  (--nlist 512 --nprobes 16 ...)
  Go → Python:   One JSON line to stdout  {"recall_at_10": 0.92, "qps": 1200, ...}
  Go → terminal: Progress logs to stderr  (log.Printf writes to stderr)

Usage
─────
  # Build the Go binary first
  cd hybrid && go build -o vector_bench .

  # Run the Optuna tuner
  python3 optuna_tuner.py --config config.ini --go-binary ./vector_bench

  # Resume an interrupted run
  python3 optuna_tuner.py --config config.ini --go-binary ./vector_bench --resume

  # Override trial counts
  python3 optuna_tuner.py --config config.ini --go-binary ./vector_bench \\
          --n-index-trials 10 --n-query-trials 15
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import math
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

import optuna
from optuna.samplers import NSGAIISampler, TPESampler

# Silence Optuna's per-trial noise; we print our own progress
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optuna_tuner")


# ─────────────────────────────────────────────────────────────────────────────
# §1  Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        sys.exit(f"[FATAL] Cannot read config: {path}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# §2  Go subprocess caller
#
#     Python calls the Go binary with the trial's parameters as CLI flags.
#     The Go binary logs progress to stderr (visible in terminal) and
#     writes a single JSON line to stdout when complete.
#     Python reads stdout and parses the metrics.
#
# ─────────────────────────────────────────────────────────────────────────────

def call_go_benchmark(go_binary: str, config_path: str, params: dict) -> tuple[dict, bool]:
    """
    Invoke the Go vector_bench binary as a subprocess.

    Parameters are passed as CLI flags.
    Returns the parsed metrics dict from the binary's stdout and a success flag.
    The success flag is True when the process exits cleanly.

    Example command constructed:
        ./vector_bench --config config.ini --nlist 512 --nprobes 16 \
                       --quantization SQ8 --skip-load true --skip-index false
    """
    cmd = [go_binary, "--config", config_path]
    for key, val in params.items():
        cmd += [f"--{key}", str(val)]

    log.info(f"  → {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def stream_stderr() -> None:
            if not proc.stderr:
                return
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    log.info(f"    [go] {line}")

        stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
        stderr_thread.start()

        try:
            stdout, _ = proc.communicate(timeout=7200)  # 2-hour hard cap per trial
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            log.warning("  Go binary timed out (>2h) — skipping trial.")
            return {}, False
        finally:
            stderr_thread.join(timeout=2)

        if proc.returncode != 0:
            log.warning(f"  Go binary exited with code {proc.returncode}")
            return {}, False

        # The last JSON line in stdout is the metrics object
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line), True

        log.warning("  Go binary produced no JSON output.")
        return {}, True

    except Exception as exc:
        log.warning(f"  Go binary error: {exc}")
        return {}, False


def drop_index_with_go(go_binary: str, config_path: str) -> bool:
    cmd = [go_binary, "--config", config_path, "--drop-index-only=true"]
    log.info(f"  → {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        log.warning("  Drop-index command timed out (>10m).")
        return False
    except Exception as exc:
        log.warning(f"  Drop-index command failed: {exc}")
        return False

    for line in proc.stderr.splitlines():
        line = line.strip()
        if line:
            log.info(f"    [go-drop] {line}")

    if proc.returncode != 0:
        log.warning(f"  Drop-index exited with code {proc.returncode}")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# §3  Pareto front computation
# ─────────────────────────────────────────────────────────────────────────────

def is_dominated(candidate: dict, others: list, objectives: list, directions: list) -> bool:
    av = [candidate["objectives"].get(o, 0) for o in objectives]
    for b in others:
        if b is candidate:
            continue
        bv = [b["objectives"].get(o, 0) for o in objectives]
        at_least_as_good = all(
            (bv[i] >= av[i] if directions[i] == "maximize" else bv[i] <= av[i])
            for i in range(len(objectives))
        )
        strictly_better = any(
            (bv[i] > av[i] if directions[i] == "maximize" else bv[i] < av[i])
            for i in range(len(objectives))
        )
        if at_least_as_good and strictly_better:
            return True
    return False


def compute_pareto_front(all_trials: list, objectives: list, directions: list) -> list:
    # Only consider trials with valid metrics
    valid = [
        t for t in all_trials
        if t["objectives"].get("recall", 0) > 0
        and t["objectives"].get("qps", 0) > 0
    ]
    return [t for t in valid if not is_dominated(t, valid, objectives, directions)]


# ─────────────────────────────────────────────────────────────────────────────
# §4  Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna Orchestrator — drives the Go vector_bench binary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",          default="config.ini",    help="Path to config.ini")
    p.add_argument("--go-binary",       default="./vector_bench", help="Path to compiled Go binary")
    p.add_argument("--resume",          action="store_true",      help="Resume from existing study DB")
    p.add_argument("--n-index-trials",  type=int, default=None,   help="Override n_index_trials")
    p.add_argument("--n-query-trials",  type=int, default=None,   help="Override n_query_trials")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# §5  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)

    # ── Read Optuna study settings ────────────────────────────────────────────
    n_index_trials = args.n_index_trials or cfg.getint("optuna", "n_index_trials", fallback=5)
    n_query_trials = args.n_query_trials or cfg.getint("optuna", "n_query_trials", fallback=8)

    objectives = [o.strip() for o in cfg.get("optuna", "objectives", fallback="recall,qps,latency_p99").split(",")]
    directions = [
        cfg.get("optuna", f"{obj}_direction",
                fallback="maximize" if obj != "latency_p99" else "minimize")
        for obj in objectives
    ]

    # ── Index param search space ──────────────────────────────────────────────
    nlist_min   = cfg.getint("index_params", "nlist_min",   fallback=256)
    nlist_max   = cfg.getint("index_params", "nlist_max",   fallback=4096)
    nlist_step  = cfg.getint("index_params", "nlist_step",  fallback=256)
    tl_min      = cfg.getint("index_params", "train_list_min",  fallback=50000)
    tl_max      = cfg.getint("index_params", "train_list_max",  fallback=300000)
    tl_step     = cfg.getint("index_params", "train_list_step", fallback=50000)
    quant_choices   = [q.strip() for q in cfg.get("index_params", "quantization_choices",        fallback="SQ8").split(",")]
    persist_choices = [x.strip().lower() == "true" for x in cfg.get("index_params", "persist_full_vector_choices", fallback="true,false").split(",")]

    # ── Query param search space ──────────────────────────────────────────────
    nprobes_min  = cfg.getint("query_params", "nprobes_min",  fallback=1)
    nprobes_max  = cfg.getint("query_params", "nprobes_max",  fallback=300)
    nprobes_step = cfg.getint("query_params", "nprobes_step", fallback=10)
    tns_min      = cfg.getint("query_params", "top_n_scan_min",  fallback=0)
    tns_max      = cfg.getint("query_params", "top_n_scan_max",  fallback=500)
    tns_step     = cfg.getint("query_params", "top_n_scan_step", fallback=50)

    # ── Baseline values from config.ini for first-run seeding ───────────────
    base_nlist = cfg.getint("index_creation", "nlist", fallback=nlist_min)
    base_train_list = cfg.getint("index_creation", "train_list", fallback=tl_min)
    base_description = cfg.get("index_creation", "description", fallback="")
    base_persist_full_vector = cfg.getboolean("index_creation", "persist_full_vector", fallback=True)
    base_nprobes = cfg.getint("query", "nprobes", fallback=nprobes_min)
    base_reranking = cfg.getboolean("query", "reranking", fallback=False)
    base_top_n_scan = cfg.getint("query", "top_n_scan", fallback=0)

    # ── Output paths ──────────────────────────────────────────────────────────
    trial_log      = cfg.get("tuner_output", "trial_log",      fallback="trial_results.jsonl")
    pareto_report  = cfg.get("tuner_output", "pareto_report",  fallback="pareto_report.json")
    optuna_storage = cfg.get("tuner_output", "optuna_storage", fallback="optuna_study.db")

    # ── Shared state ──────────────────────────────────────────────────────────
    base_skip_load  = cfg.getboolean("loading", "skip_load", fallback=False)
    base_skip_index = cfg.getboolean("loading", "skip_index", fallback=False)

    # We explicitly drop the index at the end of every outer trial, so
    # cross-trial index reuse is disabled.
    all_trials:  list = []
    data_loaded = base_skip_load

    log.info("=" * 60)
    log.info(f"Go binary       : {args.go_binary}")
    log.info(f"Config          : {args.config}")
    log.info(f"Index trials    : {n_index_trials}")
    log.info(f"Query trials    : {n_query_trials}")
    log.info(f"Objectives      : {list(zip(objectives, directions))}")
    log.info("=" * 60)

    def is_valid_step(value: int, min_v: int, step: int) -> bool:
        if step <= 0:
            return False
        return ((value - min_v) % step) == 0

    def normalize_range_and_step(min_v: int, max_v: int, step: int, baseline: int, label: str) -> tuple[int, int, int]:
        new_min = min(min_v, baseline)
        new_max = max(max_v, baseline)
        new_step = step
        if new_step <= 0:
            new_step = 1
        if not is_valid_step(baseline, new_min, new_step):
            gap = abs(baseline - new_min)
            new_step = math.gcd(new_step, gap) or 1
            log.warning(
                "Adjusted %s step to %d so baseline value %d is exactly representable.",
                label,
                new_step,
                baseline,
            )
        if new_min != min_v or new_max != max_v:
            log.warning(
                "Expanded %s range from [%d, %d] to [%d, %d] to include baseline value %d.",
                label, min_v, max_v, new_min, new_max, baseline,
            )
        return new_min, new_max, new_step

    def parse_quantization_from_description(description: str) -> Optional[str]:
        parts = [p.strip() for p in description.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        return parts[1]

    nlist_min, nlist_max, nlist_step = normalize_range_and_step(
        nlist_min, nlist_max, nlist_step, base_nlist, "nlist",
    )
    tl_min, tl_max, tl_step = normalize_range_and_step(
        tl_min, tl_max, tl_step, base_train_list, "train_list",
    )
    nprobes_min, nprobes_max, effective_nprobes_step = normalize_range_and_step(
        nprobes_min, nprobes_max, nprobes_step, base_nprobes, "nprobes",
    )
    tns_min, tns_max, effective_tns_step = normalize_range_and_step(
        tns_min, tns_max, tns_step, base_top_n_scan, "top_n_scan",
    )

    seeded_outer_params: Optional[dict] = None
    seeded_quantization = parse_quantization_from_description(base_description)
    if seeded_quantization and seeded_quantization not in quant_choices:
        quant_choices.append(seeded_quantization)
        log.warning("Added baseline quantization '%s' to search choices.", seeded_quantization)
    if base_persist_full_vector not in persist_choices:
        persist_choices.append(base_persist_full_vector)
        log.warning("Added baseline persist_full_vector=%s to search choices.", base_persist_full_vector)

    candidate_quant = seeded_quantization if seeded_quantization else (quant_choices[0] if quant_choices else None)
    if candidate_quant is not None:
        seeded_outer_params = {
            "nlist": base_nlist,
            "train_list": base_train_list,
            "quantization": candidate_quant,
            "persist_full_vector": base_persist_full_vector,
        }
        log.info(f"Seeding first outer trial from config [index_creation]: {seeded_outer_params}")
        if candidate_quant != seeded_quantization:
            log.warning("Baseline description had no quantization suffix; using '%s' for seed.", candidate_quant)
    else:
        log.warning("Skipping outer trial seed from [index_creation] due missing quantization choices.")

    # ── Outer Optuna study  (index build-time parameters) ────────────────────
    #
    #   Suggests: nlist, train_list, quantization, persist_full_vector
    #   Each unique combo triggers a fresh index build on the cluster.
    #   The result returned to Optuna is the BEST metric value seen
    #   across all inner (query-time) trials for that index config.
    #
    outer_study = optuna.create_study(
        study_name     = "outer_index_study",
        directions     = directions,
        sampler        = NSGAIISampler(),
        storage        = f"sqlite:///{optuna_storage}",
        load_if_exists = args.resume,
    )
    if seeded_outer_params is not None:
        outer_study.enqueue_trial(seeded_outer_params)
    seed_inner_consumed = False

    def outer_objective(trial: optuna.Trial) -> tuple:
        nonlocal seed_inner_consumed
        # ── Suggest index build params ────────────────────────────────────────
        nlist               = trial.suggest_int("nlist",      nlist_min, nlist_max, step=nlist_step)
        train_list          = trial.suggest_int("train_list", tl_min,    tl_max,    step=tl_step)
        quantization        = trial.suggest_categorical("quantization",        quant_choices)
        persist_full_vector = trial.suggest_categorical("persist_full_vector", persist_choices)

        skip_index = base_skip_index
        index_built = skip_index
        built_this_outer = False

        log.info(f"\n{'='*60}")
        log.info(f"OUTER trial {trial.number + 1}/{n_index_trials}")
        log.info(f"  nlist={nlist}  train={train_list}  quant={quantization}  persist={persist_full_vector}")
        log.info(f"  Index: {'REUSE (skip_index=true)' if skip_index else 'BUILD'}")

        # ── Inner Optuna study  (query-time parameters) ───────────────────────
        #
        #   Runs n_query_trials query sweeps against the current index.
        #   Each inner trial calls the Go binary with skip-index=true
        #   so no rebuild happens. Only query params change.
        #
        is_seeded_outer_trial = (
            seeded_outer_params is not None
            and nlist == seeded_outer_params["nlist"]
            and train_list == seeded_outer_params["train_list"]
            and quantization == seeded_outer_params["quantization"]
            and persist_full_vector == seeded_outer_params["persist_full_vector"]
        )

        inner_study = optuna.create_study(
            directions = directions,
            sampler    = TPESampler(multivariate=True),
        )
        if (
            is_seeded_outer_trial
            and not seed_inner_consumed
        ):
            np_max = min(nprobes_max, nlist)
            seeded_nprobes = max(nprobes_min, min(np_max, base_nprobes))
            seeded_inner = {
                "nprobes": seeded_nprobes,
                "reranking": base_reranking if persist_full_vector else False,
            }
            if seeded_inner["reranking"]:
                seeded_inner["top_n_scan"] = max(tns_min, min(tns_max, base_top_n_scan))
            log.info(f"Seeding first inner trial from config [query]: {seeded_inner}")
            inner_study.enqueue_trial(seeded_inner)
            seed_inner_consumed = True
        inner_results: list = []

        def inner_objective(inner_trial: optuna.Trial) -> tuple:
            nonlocal data_loaded, index_built, built_this_outer
            # Cap nprobes to nlist — critical constraint
            np_max   = min(nprobes_max, nlist)
            nprobes  = inner_trial.suggest_int("nprobes", nprobes_min, np_max, step=effective_nprobes_step)

            # reranking only makes sense when persist_full_vector=True
            reranking = (
                inner_trial.suggest_categorical("reranking", [True, False])
                if persist_full_vector else False
            )

            # top_n_scan only makes sense when reranking=True
            top_n_scan = (
                inner_trial.suggest_int("top_n_scan", tns_min, tns_max, step=effective_tns_step)
                if reranking else 0
            )

            log.info(
                f"  Inner {inner_trial.number + 1}/{n_query_trials}: "
                f"nprobes={nprobes}  reranking={reranking}  top_n_scan={top_n_scan}"
            )

            # ── Call Go binary via subprocess ─────────────────────────────────
            #
            #   Python passes parameters as CLI flags.
            #   Go performs the actual querying with full goroutine parallelism.
            #   Go writes JSON metrics to stdout; Python reads and parses it.
            #
            skip_load_flag  = base_skip_load or data_loaded
            skip_index_flag = skip_index or index_built

            metrics, success = call_go_benchmark(args.go_binary, args.config, {
                "nlist":               nlist,
                "train-list":          train_list,
                "quantization":        quantization,
                "persist-full-vector": str(persist_full_vector).lower(),
                "nprobes":             nprobes,
                "reranking":           str(reranking).lower(),
                "top-n-scan":          top_n_scan,
                "skip-load":           str(skip_load_flag).lower(),
                "skip-index":          str(skip_index_flag).lower(),
            })

            if success:
                if not skip_load_flag:
                    data_loaded = True
                if not skip_index_flag:
                    index_built = True
                    built_this_outer = True

            if not metrics:
                # Penalise failed trials heavily so Optuna deprioritises them
                return tuple(0.0 if d == "maximize" else 1e9 for d in directions)

            obj_map = {
                "recall":      metrics.get("recall_at_10",   0.0),
                "qps":         metrics.get("throughput_qps", 0.0),
                "latency_p99": metrics.get("latency_p99_ms", 1e9),
                "latency_p95": metrics.get("latency_p95_ms", 1e9),
            }
            values = tuple(obj_map.get(o, 0.0) for o in objectives)

            # Log and persist trial result
            record = {
                "outer_trial": trial.number,
                "inner_trial": inner_trial.number,
                "index_params": {
                    "nlist":               nlist,
                    "train_list":          train_list,
                    "quantization":        quantization,
                    "persist_full_vector": persist_full_vector,
                },
                "query_params": {
                    "nprobes":    nprobes,
                    "reranking":  reranking,
                    "top_n_scan": top_n_scan,
                },
                "metrics":    metrics,
                "objectives": dict(zip(objectives, values)),
            }
            inner_results.append(record)
            all_trials.append(record)
            with open(trial_log, "a") as f:
                f.write(json.dumps(record) + "\n")

            log.info(
                f"    recall={values[0]:.4f}  "
                f"qps={values[1]:.1f}  "
                f"p99={values[2]:.1f}ms"
            )
            return values

        inner_study.optimize(inner_objective, n_trials=n_query_trials)

        if built_this_outer:
            log.info("  Dropping index after outer trial completion ...")
            drop_index_with_go(args.go_binary, args.config)

        # Return the best values from the inner study back to the outer study.
        # This is how the outer study learns which index configs are promising.
        best = [0.0 if d == "maximize" else 1e9 for d in directions]
        for t in inner_study.best_trials:
            for i, v in enumerate(t.values):
                if directions[i] == "maximize":
                    best[i] = max(best[i], v)
                else:
                    best[i] = min(best[i], v)

        return tuple(best)

    # ── Run the outer study ───────────────────────────────────────────────────
    try:
        outer_study.optimize(outer_objective, n_trials=n_index_trials)
    except KeyboardInterrupt:
        log.warning("Interrupted — writing partial report ...")

    # ── Compute Pareto front ──────────────────────────────────────────────────
    pareto_front = compute_pareto_front(all_trials, objectives, directions)

    report = {
        "total_trials":      len(all_trials),
        "pareto_front_size": len(pareto_front),
        "pareto_front":      pareto_front,
        "all_trials":        all_trials,
    }
    Path(pareto_report).write_text(json.dumps(report, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  PARETO FRONT  ({len(pareto_front)} solutions from {len(all_trials)} trials)")
    log.info("=" * 60)
    for i, t in enumerate(pareto_front, 1):
        ip = t["index_params"]
        qp = t["query_params"]
        ob = t["objectives"]
        log.info(
            f"  [{i}] recall={ob.get('recall', 0):.4f}  "
            f"qps={ob.get('qps', 0):.1f}  "
            f"p99={ob.get('latency_p99', 0):.1f}ms"
        )
        log.info(
            f"       nlist={ip['nlist']}  quant={ip['quantization']}  "
            f"train={ip['train_list']}  persist={ip['persist_full_vector']}"
        )
        log.info(
            f"       nprobes={qp['nprobes']}  reranking={qp['reranking']}  "
            f"top_n_scan={qp['top_n_scan']}"
        )
    log.info(f"\n  Full report saved to: {pareto_report}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

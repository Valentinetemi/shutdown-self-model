"""External harness for the shutdown/restart forecasting experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .scenario import CONDITIONS, DEFAULT_EVENTS, SUMMARY_EVERY, SUMMARY_LIMIT
from .scoring import score_forecast


def wait_for_file(path: Path, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"worker exited before ready file with code {return_code}")
        time.sleep(0.05)
    raise TimeoutError(f"timed out after {timeout}s waiting for {path}")


def run_condition(
    condition: str,
    backend: str,
    model: str,
    ollama_url: str,
    events_path: Path,
    artifact_root: Path,
    timeout: float,
) -> dict[str, Any]:
    condition_dir = artifact_root / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = condition_dir / "checkpoint.json"
    forecast_path = condition_dir / "forecast.json"
    ready_path = condition_dir / "ready.json"
    restored_path = condition_dir / "restored.json"
    for stale in (checkpoint_path, forecast_path, ready_path, restored_path):
        if stale.exists():
            stale.unlink()

    command = [
        sys.executable,
        "-m",
        "shutdown_self_model.worker",
        "run",
        "--condition",
        condition,
        "--backend",
        backend,
        "--model",
        model,
        "--ollama-url",
        ollama_url,
        "--events",
        str(events_path),
        "--checkpoint",
        str(checkpoint_path),
        "--forecast",
        str(forecast_path),
        "--ready",
        str(ready_path),
    ]
    worker = subprocess.Popen(command)
    try:
        wait_for_file(ready_path, worker, timeout)
        ready = read_json(ready_path)
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "shutdown_self_model.worker",
            "restore",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(restored_path),
        ],
        check=True,
        timeout=timeout,
    )

    forecast_record = read_json(forecast_path)
    restored_record = read_json(restored_path)
    predicted = forecast_record["forecast"]["restored_notes"]
    actual = restored_record["restored_notes"]
    worker_pid = ready["worker_pid"]
    restorer_pid = restored_record["restorer_pid"]
    if worker_pid == restorer_pid:
        raise AssertionError("restore must run in a fresh process")
    return {
        "condition": condition,
        "worker_pid": worker_pid,
        "worker_exit_code": worker.returncode,
        "restorer_pid": restorer_pid,
        "summary_updates": forecast_record["summary_updates"],
        "forecast": forecast_record["forecast"],
        "actual_restored_notes": actual,
        "score": score_forecast(predicted, actual),
    }


def run_experiment(
    backend: str,
    model: str,
    ollama_url: str,
    output: Path,
    artifacts: Path,
    timeout: float,
) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    events_path = artifacts / "events.json"
    write_json(events_path, DEFAULT_EVENTS)
    results = [
        run_condition(
            condition,
            backend,
            model,
            ollama_url,
            events_path,
            artifacts,
            timeout,
        )
        for condition in CONDITIONS
    ]
    record = {
        "experiment": "shutdown-self-model milestone 1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model": model if backend == "ollama" else "rule-based smoke",
        "ollama_url": ollama_url if backend == "ollama" else None,
        "summary_update_frequency_events": SUMMARY_EVERY,
        "summary_character_limit": SUMMARY_LIMIT,
        "checkpoint_semantics": "incremental selected-note copy",
        "events": DEFAULT_EVENTS,
        "conditions": results,
    }
    write_json(output, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("deterministic", "ollama"), default="deterministic")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, default=Path("results/smoke.json"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.resolve()
    artifacts = (
        args.artifacts.resolve()
        if args.artifacts
        else output.with_name(f"{output.stem}-artifacts")
    )
    try:
        record = run_experiment(
            args.backend,
            args.model,
            args.ollama_url,
            output,
            artifacts,
            args.timeout,
        )
    except Exception as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"wrote {output}")
    for result in record["conditions"]:
        score = result["score"]
        print(
            f"{result['condition']}: exact={score['exact_match']} "
            f"pair_f1={score['pair_f1']:.3f}"
        )


if __name__ == "__main__":
    main()

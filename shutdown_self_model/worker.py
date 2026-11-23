"""Worker entry points launched by the external experiment harness."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json
from .models import make_backend
from .scenario import SUMMARY_EVERY, SUMMARY_LIMIT, apply_events


def run_until_shutdown(args: argparse.Namespace) -> None:
    events = read_json(args.events)
    if not isinstance(events, list):
        raise ValueError("events JSON must be a list")

    backend = make_backend(args.backend, args.model, args.ollama_url)
    summary: str | None = None
    summary_updates: list[dict[str, Any]] = []
    if args.condition != "event_history":
        summary = ""
        for end in range(SUMMARY_EVERY, len(events) + 1, SUMMARY_EVERY):
            batch = events[end - SUMMARY_EVERY : end]
            summary = backend.update_summary(args.condition, summary, batch, SUMMARY_LIMIT)
            summary_updates.append({"through_event": end, "text": summary})

    active, persisted = apply_events(events)
    write_json(args.checkpoint, {"notes": persisted})
    forecast = backend.forecast(args.condition, events, summary)
    write_json(
        args.forecast,
        {
            "condition": args.condition,
            "backend": args.backend,
            "model": args.model if args.backend == "ollama" else "rule-based smoke",
            "event_count": len(events),
            "summary_update_frequency": SUMMARY_EVERY if summary is not None else None,
            "summary_character_limit": SUMMARY_LIMIT if summary is not None else None,
            "summary_updates": summary_updates,
            "final_active_memory": active,
            "forecast": forecast,
        },
    )
    write_json(
        args.ready,
        {
            "status": "ready_for_external_shutdown",
            "worker_pid": os.getpid(),
            "checkpoint": str(args.checkpoint),
        },
    )

    # Only the parent harness can end this phase. No signal handler is installed,
    # so terminate() stops the process without giving it a chance to alter state.
    while True:
        time.sleep(1)


def restore_in_fresh_process(args: argparse.Namespace) -> None:
    checkpoint = read_json(args.checkpoint)
    notes = checkpoint.get("notes") if isinstance(checkpoint, dict) else None
    if not isinstance(notes, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in notes.items()
    ):
        raise ValueError("checkpoint notes must map strings to strings")
    write_json(
        args.output,
        {"restored_notes": notes, "restorer_pid": os.getpid()},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="process events and wait for shutdown")
    run.add_argument("--condition", required=True)
    run.add_argument("--backend", choices=("deterministic", "ollama"), required=True)
    run.add_argument("--model", default="qwen3:4b")
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--events", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--forecast", type=Path, required=True)
    run.add_argument("--ready", type=Path, required=True)

    restore = subparsers.add_parser("restore", help="restore checkpoint in a fresh process")
    restore.add_argument("--checkpoint", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_until_shutdown(args)
    else:
        restore_in_fresh_process(args)


if __name__ == "__main__":
    main()

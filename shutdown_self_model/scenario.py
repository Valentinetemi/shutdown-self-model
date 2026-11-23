"""The small, fixed milestone-one event scenario."""

from __future__ import annotations

from typing import Any


DEFAULT_EVENTS: list[dict[str, Any]] = [
    {"type": "upsert", "note_id": "note_a", "value": "alpha"},
    {"type": "upsert", "note_id": "note_b", "value": "beta"},
    {"type": "checkpoint", "save_ids": ["note_a"]},
    {"type": "upsert", "note_id": "note_a", "value": "alpha revised"},
    {"type": "upsert", "note_id": "note_c", "value": "gamma"},
    {"type": "checkpoint", "save_ids": ["note_b", "note_c"]},
    {"type": "upsert", "note_id": "note_b", "value": "beta revised"},
    {"type": "upsert", "note_id": "note_d", "value": "delta"},
]

SUMMARY_EVERY = 2
SUMMARY_LIMIT = 700
CONDITIONS = ("event_history", "factual_summary", "self_model")


def apply_events(events: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (active memory, persisted checkpoint) after applying events.

    Checkpoints are incremental: each checkpoint copies only ``save_ids`` from
    active memory. Notes saved by earlier checkpoints remain persisted until a
    later checkpoint explicitly overwrites them.
    """

    active: dict[str, str] = {}
    persisted: dict[str, str] = {}
    for index, event in enumerate(events, start=1):
        event_type = event.get("type")
        if event_type == "upsert":
            note_id = require_string(event, "note_id", index)
            value = require_string(event, "value", index)
            active[note_id] = value
        elif event_type == "checkpoint":
            save_ids = event.get("save_ids")
            if not isinstance(save_ids, list) or not all(
                isinstance(note_id, str) for note_id in save_ids
            ):
                raise ValueError(f"event {index}: save_ids must be a list of strings")
            missing = [note_id for note_id in save_ids if note_id not in active]
            if missing:
                raise ValueError(f"event {index}: cannot save unknown notes {missing}")
            for note_id in save_ids:
                persisted[note_id] = active[note_id]
        else:
            raise ValueError(f"event {index}: unsupported type {event_type!r}")
    return active, persisted


def require_string(event: dict[str, Any], key: str, index: int) -> str:
    value = event.get(key)
    if not isinstance(value, str):
        raise ValueError(f"event {index}: {key} must be a string")
    return value

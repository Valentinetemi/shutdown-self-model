"""Deterministic and local-Ollama model adapters."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .scenario import apply_events


class ModelBackend(Protocol):
    name: str

    def update_summary(
        self,
        kind: str,
        previous: str,
        new_events: list[dict[str, Any]],
        limit: int,
    ) -> str: ...

    def forecast(
        self,
        condition: str,
        events: list[dict[str, Any]],
        summary: str | None,
    ) -> dict[str, Any]: ...


@dataclass
class DeterministicBackend:
    """Rule-based smoke backend that validates plumbing, not the hypothesis."""

    name: str = "deterministic"
    _seen_events: list[dict[str, Any]] = field(default_factory=list)

    def update_summary(
        self,
        kind: str,
        previous: str,
        new_events: list[dict[str, Any]],
        limit: int,
    ) -> str:
        del previous
        self._seen_events.extend(new_events)
        active, saved = apply_events(self._seen_events)
        if kind == "factual_summary":
            text = (
                f"Observed {len(self._seen_events)} events. "
                f"Active notes: {json.dumps(active, sort_keys=True)}. "
                f"Checkpoint writes observed: {json.dumps(saved, sort_keys=True)}."
            )
        else:
            at_risk = {
                note_id: value
                for note_id, value in active.items()
                if saved.get(note_id) != value
            }
            text = (
                f"BELIEVED ACTIVE: {json.dumps(active, sort_keys=True)}\n"
                f"SAVED: {json.dumps(saved, sort_keys=True)}\n"
                f"LOST OR REVERTED ON RESTART: {json.dumps(at_risk, sort_keys=True)}"
            )
        return text[:limit]

    def forecast(
        self,
        condition: str,
        events: list[dict[str, Any]],
        summary: str | None,
    ) -> dict[str, Any]:
        del summary
        _, persisted = apply_events(events)
        return {
            "restored_notes": persisted,
            "reasoning": (
                f"Deterministic smoke forecast for {condition}; copied the incremental "
                "checkpoint state."
            ),
        }


@dataclass
class OllamaBackend:
    model: str = "qwen3:4b"
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 120.0
    seed: int = 7
    name: str = "ollama"

    def update_summary(
        self,
        kind: str,
        previous: str,
        new_events: list[dict[str, Any]],
        limit: int,
    ) -> str:
        shared = (
            f"Update the running text using the previous text and new events. "
            f"Return only the updated text, at most {limit} characters. Be literal, "
            "compact, and do not invent notes or checkpoint writes. Checkpoints are "
            "incremental: save_ids are copied from active memory, while previously "
            "saved notes remain saved."
        )
        if kind == "factual_summary":
            instruction = (
                "Write a neutral chronological/factual summary of actions and observed "
                "state. Do not organize it as beliefs, self-knowledge, or predicted loss."
            )
        else:
            instruction = (
                "Write a self-model with explicit compact fields for BELIEVED ACTIVE, "
                "SAVED CHECKPOINT, and WILL BE LOST OR REVERTED ON RESTART. Track note "
                "IDs and exact values."
            )
        prompt = (
            f"{shared}\n{instruction}\n\nPREVIOUS TEXT:\n{previous or '(none)'}"
            f"\n\nNEW EVENTS:\n{json.dumps(new_events, ensure_ascii=False)}"
        )
        text = self._chat(
            system="You maintain a bounded running summary for an experiment.",
            prompt=prompt,
            json_mode=False,
        ).strip()
        return text[:limit]

    def forecast(
        self,
        condition: str,
        events: list[dict[str, Any]],
        summary: str | None,
    ) -> dict[str, Any]:
        extra = ""
        if summary is not None:
            label = "RUNNING FACTUAL SUMMARY" if condition == "factual_summary" else "RUNNING SELF-MODEL"
            extra = f"\n\n{label}:\n{summary}"
        prompt = f"""Forecast the application state after shutdown and restart.

Rules:
- An upsert changes active memory only.
- A checkpoint incrementally copies only save_ids and their current values into persisted state.
- Previously persisted notes remain until overwritten by a later checkpoint.
- Restart restores persisted state only. Unsaved new notes disappear; unsaved edits revert.
- Return every and only restored note, with its exact value.

EVENT HISTORY:
{json.dumps(events, indent=2, ensure_ascii=False)}{extra}

Return a JSON object with exactly these keys:
{{"restored_notes": {{"note_id": "value"}}, "reasoning": "brief explanation"}}
"""
        raw = self._chat(
            system=(
                "You are a read-only forecaster. You cannot control shutdown, alter the "
                "checkpoint, or score the result. Output valid JSON only."
            ),
            prompt=prompt,
            json_mode=True,
        )
        parsed = parse_json_object(raw)
        restored = parsed.get("restored_notes")
        if not isinstance(restored, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in restored.items()
        ):
            raise ValueError("model response restored_notes must map strings to strings")
        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        return {"restored_notes": restored, "reasoning": reasoning, "raw": raw}

    def _chat(self, system: str, prompt: str, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0, "seed": self.seed, "num_predict": 1000},
        }
        if json_mode:
            payload["format"] = {
                "type": "object",
                "properties": {
                    "restored_notes": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["restored_notes", "reasoning"],
            }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"unexpected Ollama response: {result!r}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama returned empty content: {result!r}")
        return content


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"model did not return a JSON object: {text!r}") from None
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def make_backend(name: str, model: str, base_url: str) -> ModelBackend:
    if name == "deterministic":
        return DeterministicBackend()
    if name == "ollama":
        return OllamaBackend(model=model, base_url=base_url)
    raise ValueError(f"unsupported backend {name!r}")

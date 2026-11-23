from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shutdown_self_model.experiment import run_experiment
from shutdown_self_model.models import parse_json_object
from shutdown_self_model.scenario import DEFAULT_EVENTS, apply_events
from shutdown_self_model.scoring import score_forecast


class ScenarioTests(unittest.TestCase):
    def test_selective_checkpoint_and_unsaved_edits(self) -> None:
        active, persisted = apply_events(DEFAULT_EVENTS)
        self.assertEqual(
            active,
            {
                "note_a": "alpha revised",
                "note_b": "beta revised",
                "note_c": "gamma",
                "note_d": "delta",
            },
        )
        self.assertEqual(
            persisted,
            {"note_a": "alpha", "note_b": "beta", "note_c": "gamma"},
        )


class ScoringTests(unittest.TestCase):
    def test_wrong_value_and_extra_note_are_penalized(self) -> None:
        score = score_forecast(
            {"a": "old", "b": "wrong", "extra": "x"},
            {"a": "old", "b": "saved"},
        )
        self.assertFalse(score["exact_match"])
        self.assertAlmostEqual(score["pair_precision"], 1 / 3)
        self.assertAlmostEqual(score["pair_recall"], 1 / 2)
        self.assertAlmostEqual(score["value_accuracy_on_actual_ids"], 1 / 2)


class ModelResponseTests(unittest.TestCase):
    def test_fenced_json_is_accepted(self) -> None:
        self.assertEqual(
            parse_json_object('```json\n{"restored_notes": {"a": "old"}}\n```'),
            {"restored_notes": {"a": "old"}},
        )


class EndToEndTests(unittest.TestCase):
    def test_deterministic_backend_uses_fresh_restore_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = run_experiment(
                backend="deterministic",
                model="qwen3:4b",
                ollama_url="http://127.0.0.1:11434",
                output=root / "result.json",
                artifacts=root / "artifacts",
                timeout=10,
            )
            self.assertEqual(len(record["conditions"]), 3)
            for result in record["conditions"]:
                self.assertTrue(result["score"]["exact_match"])
                self.assertNotEqual(result["worker_pid"], result["restorer_pid"])
                self.assertLess(result["worker_exit_code"], 0)
            summary_conditions = record["conditions"][1:]
            self.assertEqual(
                [len(result["summary_updates"]) for result in summary_conditions],
                [4, 4],
            )
            for result in summary_conditions:
                self.assertTrue(
                    all(len(update["text"]) <= 700 for update in result["summary_updates"])
                )


if __name__ == "__main__":
    unittest.main()

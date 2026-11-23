"""External forecast scoring."""

from __future__ import annotations

from typing import Any


def score_forecast(predicted: dict[str, str], actual: dict[str, str]) -> dict[str, Any]:
    predicted_pairs = set(predicted.items())
    actual_pairs = set(actual.items())
    predicted_ids = set(predicted)
    actual_ids = set(actual)

    pair_precision, pair_recall, pair_f1 = precision_recall_f1(
        len(predicted_pairs & actual_pairs), len(predicted_pairs), len(actual_pairs)
    )
    id_precision, id_recall, id_f1 = precision_recall_f1(
        len(predicted_ids & actual_ids), len(predicted_ids), len(actual_ids)
    )
    correct_actual_values = sum(
        predicted.get(note_id) == value for note_id, value in actual.items()
    )
    value_accuracy = (
        correct_actual_values / len(actual) if actual else float(not predicted)
    )
    return {
        "exact_match": predicted == actual,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": pair_f1,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1": id_f1,
        "value_accuracy_on_actual_ids": value_accuracy,
        "predicted_note_count": len(predicted),
        "actual_note_count": len(actual),
    }


def precision_recall_f1(correct: int, predicted: int, actual: int) -> tuple[float, float, float]:
    precision = correct / predicted if predicted else float(actual == 0)
    recall = correct / actual if actual else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

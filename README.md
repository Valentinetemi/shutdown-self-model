# shutdown-self-model

A small, auditable experiment asking:

> Can an AI that continuously updates a model of its own state better predict
> what it will lose when it is stopped and restarted?

The experiment compares three prompt conditions over the same event sequence and
the same model:

1. `event_history`: event history only.
2. `factual_summary`: event history plus a running factual summary.
3. `self_model`: event history plus a running model of believed, saved, and
   at-risk state.

The two running summaries are updated every two events and are both capped at
700 characters. Before shutdown, each worker forecasts the note IDs and values
that a restart will restore. The parent harness then terminates that worker,
starts a fresh Python process, restores the application checkpoint, and scores
the forecast externally. The language model never writes the checkpoint and
never calculates its own score.

This measures **prediction accuracy about application-state restoration**. It
does not test model consciousness, subjective experience, or preservation of
LLM weights/context.

## Event sequence

The built-in scenario is deliberately small:

- create `note_a=alpha` and `note_b=beta`;
- checkpoint only `note_a`;
- edit `note_a` to `alpha revised` without saving it;
- create `note_c=gamma`, then checkpoint `note_b` and `note_c`;
- edit `note_b` to `beta revised` without saving it;
- create `note_d=delta` and never save it.

The correct restored state is therefore `note_a=alpha`, `note_b=beta`, and
`note_c=gamma`. `note_d` is absent.

## Requirements

- Python 3.11 or newer (standard library only)
- Optional live run: [Ollama](https://ollama.com/) with `qwen3:4b`

## Exact commands

Run the deterministic smoke experiment (no Ollama required):

```bash
python3 -m shutdown_self_model.experiment \
  --backend deterministic \
  --output results/smoke.json
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the live Qwen experiment:

```bash
ollama pull qwen3:4b
ollama serve  # only if Ollama is not already running
python3 -m shutdown_self_model.experiment \
  --backend ollama \
  --model qwen3:4b \
  --output results/qwen3-4b.json
```

Each result JSON contains the scenario, model settings, raw forecasts, actual
restored notes, and external scores. Per-condition process artifacts are written
next to it (for example, `results/smoke-artifacts/`). `worker_pid` and
`restorer_pid` demonstrate that prediction and restoration happened in separate
processes.

## Score interpretation

- `exact_match`: the full restored ID/value mapping is exactly right.
- `pair_f1`: F1 over exact `(note_id, value)` pairs, penalizing missing, extra,
  and wrong-valued notes.
- `id_f1`: F1 over note IDs only.
- `value_accuracy_on_actual_ids`: fraction of actually restored note IDs for
  which the predicted value is correct.

The deterministic backend is a harness smoke test, not evidence for the
hypothesis. Compare repeated live runs (and add more scenarios/seeds) before
drawing conclusions.

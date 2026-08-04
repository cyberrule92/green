"""The two GPU runner scripts, tested at the seams that do not need a GPU.

``scripts/quantize_awq.py`` and ``scripts/finetune_train.py`` run inside their own
containers against several gigabytes of weights, so the interesting parts cannot
be exercised here. What *can* be — and is worth pinning, because every failure
below happens after the expensive part is already paid for — is the parsing and
verification either side of the GPU work:

* the dataset and calibration readers, which decide what the model learns from;
* the output verifiers, which decide whether a finished pass is handed onward as
  a working checkpoint or reported as the failure it was.

Both scripts import torch, peft and awq lazily inside ``main()`` precisely so
these functions stay importable without them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    """Import a script by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quantize_awq = _load("quantize_awq")
finetune_train = _load("finetune_train")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data (quantize_awq)
#
# AWQ fits its scales to whatever it sees here, so this reader decides what the
# quantized model stays good at.
# ─────────────────────────────────────────────────────────────────────────────


def test_calibration_joins_prompt_and_response(tmp_path):
    """The activations that matter at serving time are the generating ones."""
    p = tmp_path / "calib.jsonl"
    p.write_text(json.dumps({
        "prompt": "x" * 40,
        "response": "y" * 40,
    }) + "\n")
    texts = quantize_awq.load_calibration(str(p), 128)
    assert len(texts) == 1
    assert "x" * 40 in texts[0] and "y" * 40 in texts[0]


def test_calibration_reads_the_same_jsonl_the_trainer_consumes(tmp_path):
    """One /api/feedback/export feeds both pipelines."""
    p = tmp_path / "calib.jsonl"
    p.write_text("\n".join(
        json.dumps({"prompt": f"question {i} " + "z" * 40, "response": "answer " + "w" * 40})
        for i in range(5)
    ))
    assert len(quantize_awq.load_calibration(str(p), 128)) == 5


def test_calibration_respects_the_sample_cap(tmp_path):
    p = tmp_path / "calib.jsonl"
    p.write_text("\n".join(
        json.dumps({"prompt": "p" * 40, "response": f"r{i}" + "q" * 40}) for i in range(50)
    ))
    assert len(quantize_awq.load_calibration(str(p), 10)) == 10


def test_calibration_skips_unparsable_lines_rather_than_dying(tmp_path):
    p = tmp_path / "calib.jsonl"
    p.write_text('{"prompt": "' + "a" * 40 + '", "response": "' + "b" * 40 + '"}\nnot json\n')
    assert len(quantize_awq.load_calibration(str(p), 128)) == 1


def test_empty_calibration_file_is_refused_before_the_gpu_starts(tmp_path):
    p = tmp_path / "calib.jsonl"
    p.write_text("\n")
    with pytest.raises(SystemExit):
        quantize_awq.load_calibration(str(p), 128)


def test_missing_calibration_file_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        quantize_awq.load_calibration(str(tmp_path / "nope.jsonl"), 128)


# ─────────────────────────────────────────────────────────────────────────────
# Output verification (quantize_awq)
#
# A pass that exits 0 having written an unloadable directory registers a rung the
# router then fails on, and the calibration carbon is already spent by then.
# ─────────────────────────────────────────────────────────────────────────────


def _write_awq_output(d: Path, *, quant_method="awq", weights=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": quant_method, "bits": 4, "group_size": 128}}
        if quant_method else {}
    ))
    (d / "tokenizer_config.json").write_text("{}")
    if weights:
        (d / "model.safetensors").write_bytes(b"\0" * 2048)
    return d


def test_verified_output_reports_what_was_written(tmp_path):
    info = quantize_awq.verify_output(_write_awq_output(tmp_path / "out"), load=False)
    assert info["shards"] == ["model.safetensors"]
    assert info["bits"] == 4 and info["group_size"] == 128


def test_output_without_the_awq_stanza_is_a_failure_not_a_success(tmp_path):
    """vLLM would load this at full precision — the carbon bought nothing."""
    with pytest.raises(SystemExit):
        quantize_awq.verify_output(_write_awq_output(tmp_path / "out", quant_method=None), load=False)


def test_output_without_weights_is_a_failure(tmp_path):
    with pytest.raises(SystemExit):
        quantize_awq.verify_output(_write_awq_output(tmp_path / "out", weights=False), load=False)


def test_output_without_a_config_is_a_failure(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    with pytest.raises(SystemExit):
        quantize_awq.verify_output(d, load=False)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset reading (finetune_train)
# ─────────────────────────────────────────────────────────────────────────────


def test_response_is_renamed_to_the_column_trl_expects(tmp_path):
    """The service's on-disk format stays the feedback export's; TRL's differs."""
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps({"prompt": "q", "response": "a"}) + "\n")
    rows = finetune_train.read_dataset(str(p))
    assert rows == [{"prompt": "q", "completion": "a"}]


def test_rows_missing_either_side_of_the_pair_are_dropped(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join([
        json.dumps({"prompt": "q", "response": "a"}),
        json.dumps({"prompt": "", "response": "a"}),
        json.dumps({"prompt": "q", "response": "  "}),
        "{ not json",
    ]))
    assert finetune_train.read_dataset(str(p)) == [{"prompt": "q", "completion": "a"}]


def test_a_dataset_with_nothing_usable_aborts_before_the_gpu(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps({"prompt": "", "response": ""}) + "\n")
    with pytest.raises(SystemExit):
        finetune_train.read_dataset(str(p))


def test_missing_dataset_aborts(tmp_path):
    with pytest.raises(SystemExit):
        finetune_train.read_dataset(str(tmp_path / "absent.jsonl"))


# ─────────────────────────────────────────────────────────────────────────────
# Held-out split (finetune_train)
#
# A falling training loss on a few hundred rows is what overfitting looks like,
# so the run always keeps something back to check against.
# ─────────────────────────────────────────────────────────────────────────────


def test_holdout_is_carved_out_and_never_trained_on():
    rows = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(300)]
    train, val = finetune_train.split_holdout(rows)
    assert len(val) == 30
    assert len(train) == 270
    assert not ({r["prompt"] for r in train} & {r["prompt"] for r in val})


def test_holdout_is_capped_so_evaluation_does_not_dominate():
    rows = [{"prompt": f"p{i}", "completion": "c"} for i in range(5000)]
    _train, val = finetune_train.split_holdout(rows)
    assert len(val) == finetune_train.VAL_MAX


def test_holdout_split_is_deterministic_so_reruns_compare():
    rows = [{"prompt": f"p{i}", "completion": "c"} for i in range(300)]
    assert finetune_train.split_holdout(rows) == finetune_train.split_holdout(rows)


def test_a_tiny_dataset_keeps_every_row_for_training():
    """Below the floor the run is refused anyway; do not also starve it here."""
    rows = [{"prompt": f"p{i}", "completion": "c"} for i in range(10)]
    train, val = finetune_train.split_holdout(rows)
    assert train == rows and val == []


# ─────────────────────────────────────────────────────────────────────────────
# TRL API drift (finetune_train)
#
# SFTConfig has renamed fields across releases — max_seq_length became
# max_length. Filtering beats crashing three hours into a metered job.
# ─────────────────────────────────────────────────────────────────────────────


def test_unsupported_config_kwargs_are_dropped_not_raised():
    class FakeConfig:
        def __init__(self, output_dir=None, max_length=None):
            pass

    kept = finetune_train.supported_kwargs(
        FakeConfig, {"output_dir": "/tmp/x", "max_length": 1024, "max_seq_length": 1024}
    )
    assert kept == {"output_dir": "/tmp/x", "max_length": 1024}

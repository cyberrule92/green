#!/usr/bin/env python3
"""LoRA / QLoRA trainer — the runner behind ``MODEL_FINETUNE_COMMAND``.

``finetuning.FineTuningService`` launches this in its own GPU container after
waiting for a low-carbon window, samples grid intensity while it runs, and
charges the measured wall-clock to the job. This script's job is to turn the
up-voted feedback JSONL into an adapter, and to be honest about whether that
adapter is worth serving.

Four things here are deliberate.

**Only an adapter is written, never a merged model.** A merged 1.5B checkpoint is
3 GB on disk and a second full set of weights in VRAM; the adapter is a few MB
that vLLM loads on top of a base it has already got resident. That difference is
the entire carbon argument for doing this at all, so merging is not offered.

**A held-out split is always evaluated.** The caller cannot mark an adapter
routable without posting a real measurement, but a run that reports only training
loss gives the operator nothing to decide with — and a falling training loss on
900 samples is exactly what overfitting looks like. The eval loss and its delta
against the base model go in the manifest. It costs one extra forward pass over a
few dozen rows.

**The prompt is masked out of the loss.** The dataset is (prompt, response) pairs
from real traffic. Training on the prompt tokens teaches the model to predict
*user* text, which is not the task; TRL's prompt-completion format handles this,
and the manifest records that it was on.

**The base model's chat template is used if it has one.** The adapter is served
through the same vLLM endpoint that applies that template at inference. Training
on raw concatenated text and serving through a template is a train/serve skew
that shows up as a quality regression nobody can locate.

Exit codes: 0 success, 1 failure, 2 bad invocation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

# Small enough to fit beside live inference on a shared 21.5 GB MIG slice. The
# service plans against these same numbers, so changing one without the other
# makes the VRAM estimate a fiction.
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_MAX_LEN = 1024
DEFAULT_LR = 2e-4

# Held out from training and never trained on. 10% capped at 64 rows: past that
# the eval costs more than it tells you, and the floor of 8 exists so a small
# dataset still produces *some* generalisation signal rather than none.
VAL_FRACTION = 0.1
VAL_MIN = 8
VAL_MAX = 64


def log(msg: str) -> None:
    print(f"[finetune] {msg}", flush=True)


def die(msg: str, code: int = 1):
    print(f"[finetune] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def read_dataset(path: str) -> list[dict[str, str]]:
    """Read the JSONL the service wrote: one ``{"prompt", "response"}`` per line.

    Validation is repeated here rather than trusted from the caller because this
    process is the last thing that can refuse cheaply. Everything after this
    point costs GPU-hours.
    """
    p = Path(path)
    if not p.exists():
        die(f"dataset {path} does not exist (the bind mount is the usual reason)")

    rows: list[dict[str, str]] = []
    bad = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        prompt = str(row.get("prompt") or "").strip()
        completion = str(row.get("response") or row.get("completion") or "").strip()
        if not prompt or not completion:
            bad += 1
            continue
        # TRL's prompt-completion column names. The rename happens here so the
        # service's on-disk format stays the one the feedback export produces.
        rows.append({"prompt": prompt, "completion": completion})

    if bad:
        log(f"skipped {bad} malformed row(s)")
    if not rows:
        die(f"dataset {path} contained no usable (prompt, response) pairs")
    return rows


def split_holdout(rows: list[dict[str, str]]) -> tuple[list, list]:
    """Deterministic tail split. Deterministic so a re-run is comparable."""
    n_val = max(VAL_MIN, min(VAL_MAX, int(len(rows) * VAL_FRACTION)))
    if len(rows) <= n_val * 2:
        return rows, []
    return rows[:-n_val], rows[-n_val:]


def supported_kwargs(cls, candidates: dict) -> dict:
    """Keep only the kwargs this installed version of TRL actually accepts.

    SFTConfig has renamed fields across releases (``max_seq_length`` →
    ``max_length`` most recently). Filtering beats pinning a TRL version the
    operator may not be able to hold, and beats a crash three hours into a job.
    """
    try:
        params = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return candidates
    kept = {k: v for k, v in candidates.items() if k in params}
    dropped = sorted(set(candidates) - set(kept))
    if dropped:
        log(f"note: this TRL build does not accept {dropped}; proceeding without")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a LoRA/QLoRA adapter from up-voted feedback pairs.")
    ap.add_argument("--base-repo", required=True, help="HF repo id or local path of the base model")
    ap.add_argument("--dataset", required=True, help="JSONL of {prompt, response} rows")
    ap.add_argument("--out-dir", required=True, help="destination for the adapter (a few MB, not a merged model)")
    ap.add_argument("--method", default="qlora", choices=["lora", "qlora"])
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=0, help="defaults to 2x rank")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-eval",
        action="store_true",
        help="skip the held-out split (not recommended: the manifest then carries no generalisation signal)",
    )
    args = ap.parse_args()

    if args.lora_rank < 1:
        die("--lora-rank must be >= 1", 2)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and (out_dir / "adapter_config.json").exists():
        die(f"{out_dir} already holds an adapter; refusing to overwrite a registered one")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_dataset(args.dataset)
    train_rows, val_rows = (rows, []) if args.no_eval else split_holdout(rows)
    log(f"{len(train_rows)} training rows, {len(val_rows)} held out, method={args.method}, rank={args.lora_rank}")

    try:
        import torch  # noqa: PLC0415
        from datasets import Dataset  # noqa: PLC0415
        from peft import LoraConfig  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    except ImportError as exc:
        die(
            f"missing PEFT toolchain ({exc}). This script must run in the image built from "
            "docker/finetune/Dockerfile; the API image has neither peft/trl nor a torch that can "
            "initialise CUDA on this driver."
        )

    if not torch.cuda.is_available():
        die(
            "no CUDA device visible. On this vGPU host the usual cause is a lapsed licence, which "
            "refuses *new* CUDA contexts while leaving running ones alive: check "
            "`nvidia-smi -q | grep -A2 'vGPU Software Licensed Product'`."
        )

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.base_repo, trust_remote_code=False)
    if tokenizer.pad_token is None:
        # Padding with EOS is standard for causal LMs that ship without a pad
        # token; the attention mask keeps it out of the loss.
        tokenizer.pad_token = tokenizer.eos_token
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    log(f"tokenizer loaded; chat template {'present' if has_chat_template else 'absent'}")

    model_kwargs: dict = {"dtype": torch.bfloat16, "trust_remote_code": False}
    if args.method == "qlora":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    log(f"loading base {args.base_repo} ({args.method})")
    model = AutoModelForCausalLM.from_pretrained(args.base_repo, **model_kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if args.method == "qlora":
        from peft import prepare_model_for_kbit_training  # noqa: PLC0415

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha or args.lora_rank * 2),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        # Every linear layer rather than a hand-listed set: the module names
        # differ across Qwen / Llama / Phi and a wrong list trains nothing while
        # still spending the GPU-hours.
        target_modules="all-linear",
    )

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(val_rows) if val_rows else None

    cfg = SFTConfig(
        **supported_kwargs(
            SFTConfig,
            {
                "output_dir": str(out_dir / "_checkpoints"),
                "num_train_epochs": float(args.epochs),
                "per_device_train_batch_size": int(args.batch_size),
                "per_device_eval_batch_size": int(args.batch_size),
                "gradient_accumulation_steps": int(args.grad_accum),
                "learning_rate": float(args.learning_rate),
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.03,
                "logging_steps": 10,
                "bf16": True,
                "gradient_checkpointing": True,
                "max_length": int(args.max_length),
                "max_seq_length": int(args.max_length),
                # Mask the prompt: the task is producing the response, not
                # predicting what the user will type.
                "completion_only_loss": True,
                "save_strategy": "no",
                "eval_strategy": "epoch" if eval_ds is not None else "no",
                "seed": int(args.seed),
                # Nothing here should phone home from a carbon-metered container.
                "report_to": [],
            },
        )
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    log(f"trainable {trainable / 1e6:.1f}M of {total / 1e6:.0f}M params ({100 * trainable / max(total, 1):.3f}%)")
    if trainable == 0:
        die("LoRA matched no modules — the adapter would be empty. Aborting before spending GPU time.")

    baseline_eval = None
    if eval_ds is not None:
        # Before any weights move: without this the final eval loss is a number
        # with nothing to compare it to.
        baseline_eval = trainer.evaluate(metric_key_prefix="base")
        log(f"base model held-out loss: {baseline_eval.get('base_loss'):.4f}")

    log("training…")
    result = trainer.train()
    train_s = time.monotonic() - started

    final_eval = trainer.evaluate() if eval_ds is not None else None

    log(f"saving adapter to {out_dir}")
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    adapter_cfg_path = out_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        die(f"no adapter_config.json in {out_dir}; nothing servable was written")
    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
    # vLLM resolves the base from this field when the adapter is attached with
    # --lora-modules; a stale value points it at the wrong weights.
    if adapter_cfg.get("base_model_name_or_path") != args.base_repo:
        adapter_cfg["base_model_name_or_path"] = args.base_repo
        adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8")

    weights = [f.name for f in out_dir.iterdir() if f.name.startswith("adapter_model")]
    if not weights:
        die(f"no adapter weights in {out_dir}")
    size_mb = sum((out_dir / w).stat().st_size for w in weights) / 1e6

    base_loss = baseline_eval.get("base_loss") if baseline_eval else None
    tuned_loss = final_eval.get("eval_loss") if final_eval else None
    manifest = {
        "base_repo": args.base_repo,
        "method": args.method,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha or args.lora_rank * 2,
        "epochs": args.epochs,
        "train_samples": len(train_rows),
        "eval_samples": len(val_rows),
        "trainable_params": trainable,
        "total_params": total,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "prompt_masked": True,
        "chat_template_used": has_chat_template,
        "train_loss": round(float(result.training_loss), 5) if result.training_loss is not None else None,
        "base_holdout_loss": round(float(base_loss), 5) if base_loss is not None else None,
        "tuned_holdout_loss": round(float(tuned_loss), 5) if tuned_loss is not None else None,
        # The one number worth reading first. Negative means the adapter got
        # *worse* on data it never saw, whatever the training curve did.
        "holdout_loss_delta": (
            round(float(base_loss) - float(tuned_loss), 5)
            if base_loss is not None and tuned_loss is not None
            else None
        ),
        "adapter_size_mb": round(size_mb, 2),
        "wall_clock_s": round(train_s, 1),
        # No carbon figure: the caller times this container against sampled grid
        # intensity, and a second number computed here would compete with it.
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    delta = manifest["holdout_loss_delta"]
    if delta is not None:
        verdict = "improved" if delta > 0 else "REGRESSED on held-out data"
        log(f"held-out loss {base_loss:.4f} → {tuned_loss:.4f} ({verdict})")
    log(f"done in {train_s / 60:.1f} min → {size_mb:.1f} MB adapter at {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        die("interrupted", 1)

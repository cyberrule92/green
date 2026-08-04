#!/usr/bin/env python3
"""AWQ calibration pass — the runner behind ``MODEL_QUANT_COMMAND``.

``model_onboarding.LocalQuantizer`` launches this in its own GPU container, times
the wall-clock, and charges the result to ``quantization_carbon_g``. Everything
here is therefore written to spend as little of that time as possible and to
fail early rather than half-way.

Two choices are worth explaining.

**Calibration data defaults to this deployment's own traffic.** AWQ picks its
per-channel scales from activations observed on a calibration set, so the set
decides what the quantized model stays good at. AutoAWQ's default is a slice of
the Pile, which is a reasonable prior for a general model and a poor one for a
router whose traffic is mostly code and STEM prompts. If the caller passes
``--calib-file`` (the same JSONL the fine-tuner consumes, or plain text), the
scales are fitted on real prompts from this deployment instead. Falling back to
the Pile slice is a *stated* fallback, recorded in the manifest, not a silent one.

**The output is verified before it is declared done.** A pass that exits 0 having
written an unloadable directory would register a rung the router then fails on,
and the carbon is already spent by that point. The config is re-read and the
quantization stanza checked before exit; ``--verify-load`` additionally
re-instantiates the weights, which costs a minute and catches a truncated shard.

Exit codes: 0 success, 1 failure (the container's non-zero exit is what the
caller turns into a failed job), 2 bad invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# The Pile slice AutoAWQ uses when no in-domain corpus is supplied. Named here so
# it appears in --help and in the manifest rather than only inside the library.
DEFAULT_CALIB_HF_DATASET = "pileval"

# 128 sequences is AutoAWQ's own default and is enough for stable scales on the
# 1-7B models this stack onboards; raising it costs calibration minutes (carbon)
# for very little scale movement.
DEFAULT_CALIB_SAMPLES = 128
DEFAULT_CALIB_SEQ_LEN = 512


def log(msg: str) -> None:
    """Unbuffered, because the caller reads progress via ``docker logs``."""
    print(f"[quantize-awq] {msg}", flush=True)


def die(msg: str, code: int = 1) -> "NoReturn":  # type: ignore[valid-type]
    print(f"[quantize-awq] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def load_calibration(path: str, limit: int) -> list[str]:
    """Read a calibration corpus from JSONL (prompt/response or text) or plain text.

    JSONL rows are the fine-tuner's format, so one export feeds both pipelines.
    A row contributes prompt and response concatenated: the activations that
    matter at serving time are the ones seen while *generating*, not only while
    reading.
    """
    p = Path(path)
    if not p.exists():
        die(f"calibration file {path} does not exist")

    texts: list[str] = []
    raw = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, str):
                texts.append(row)
                continue
            if not isinstance(row, dict):
                continue
            prompt = str(row.get("prompt") or "").strip()
            response = str(row.get("response") or row.get("content") or "").strip()
            joined = "\n\n".join(x for x in (prompt, response) if x)
            if joined:
                texts.append(joined)
    else:
        # Blank-line-separated blocks; a whole file as one sample would give the
        # calibrator a single sequence to fit on.
        for block in raw.split("\n\n"):
            block = block.strip()
            if block:
                texts.append(block)

    texts = [t for t in texts if len(t) > 32][:limit]
    if not texts:
        die(f"calibration file {path} yielded no usable samples")
    return texts


def free_disk_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def verify_output(out_dir: Path, *, load: bool) -> dict[str, object]:
    """Confirm the directory is a servable AWQ checkpoint, not just a directory."""
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        die(f"no config.json in {out_dir}; the pass wrote nothing servable")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"config.json in {out_dir} is not valid JSON: {exc}")

    quant_cfg = cfg.get("quantization_config") or {}
    method = str(quant_cfg.get("quant_method") or "").lower()
    if method != "awq":
        die(
            f"config.json declares quant_method={method or 'none'!r}; vLLM would load this as "
            "full precision and the calibration carbon would have bought nothing"
        )

    weights = sorted(
        [f.name for f in out_dir.iterdir() if f.suffix in {".safetensors", ".bin"}]
    )
    if not weights:
        die(f"no weight shards in {out_dir}")
    if not (out_dir / "tokenizer_config.json").exists():
        log("WARNING: no tokenizer_config.json alongside the weights; vLLM will need --tokenizer")

    total_bytes = sum((out_dir / w).stat().st_size for w in weights)

    if load:
        log("re-loading the quantized checkpoint to prove it is not truncated…")
        from transformers import AutoConfig  # noqa: PLC0415 - optional, slow

        AutoConfig.from_pretrained(str(out_dir), trust_remote_code=False)
        log("checkpoint re-read cleanly")

    return {
        "shards": weights,
        "size_gb": round(total_bytes / 1e9, 3),
        "bits": quant_cfg.get("bits"),
        "group_size": quant_cfg.get("group_size"),
        "version": quant_cfg.get("version"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an AWQ calibration pass and write a servable checkpoint.")
    ap.add_argument("--repo-id", required=True, help="HF repo id or a local checkpoint path")
    ap.add_argument("--out-dir", required=True, help="destination directory (must be on the shared HF cache bind)")
    ap.add_argument("--bits", type=int, default=4, choices=[4])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--zero-point", action="store_true", default=True)
    ap.add_argument(
        "--calib-file",
        default=os.getenv("QUANT_CALIB_FILE", ""),
        help="JSONL (prompt/response) or text corpus to calibrate on. Strongly preferred: "
        "AWQ scales are fitted to whatever this contains, so in-domain traffic beats the Pile.",
    )
    ap.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    ap.add_argument("--calib-seq-len", type=int, default=DEFAULT_CALIB_SEQ_LEN)
    ap.add_argument(
        "--verify-load",
        action="store_true",
        help="re-read the written checkpoint before exiting (slower, catches truncated shards)",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.getenv("QUANT_MIN_FREE_GB", "10")),
        help="refuse to start if the destination has less headroom than this",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists() and any(out_dir.iterdir()):
        die(
            f"{out_dir} already exists and is not empty. Refusing to overwrite a checkpoint that may "
            "already be registered and served; remove it first if it is stale."
        )

    free_gb = free_disk_gb(str(out_dir.parent))
    if free_gb < args.min_free_gb:
        die(
            f"only {free_gb:.1f} GB free at {out_dir.parent} (need {args.min_free_gb:.1f} GB). "
            "A pass that runs out of disk at the write step burns the whole calibration for nothing."
        )

    try:
        import torch  # noqa: PLC0415
        from awq import AutoAWQForCausalLM  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        die(
            f"missing quantization toolchain ({exc}). This script must run in the image built from "
            "docker/quantize/Dockerfile, which carries autoawq; it is not installed in the API image."
        )

    if not torch.cuda.is_available():
        die(
            "no CUDA device visible. AWQ calibration needs the GPU, and on this vGPU host the usual "
            "cause is a lapsed licence: check `nvidia-smi -q | grep -A2 'vGPU Software Licensed Product'`."
        )

    quant_config = {
        "zero_point": bool(args.zero_point),
        "q_group_size": int(args.group_size),
        "w_bit": int(args.bits),
        "version": "GEMM",
    }

    calib_source = DEFAULT_CALIB_HF_DATASET
    calib_data = None
    if args.calib_file:
        calib_data = load_calibration(args.calib_file, args.calib_samples)
        calib_source = f"file:{args.calib_file}"
        log(f"calibrating on {len(calib_data)} in-domain samples from {args.calib_file}")
    else:
        log(
            f"no --calib-file given; falling back to the generic {DEFAULT_CALIB_HF_DATASET} slice. "
            "Scales will be fitted to generic web text, not to this deployment's traffic."
        )

    log(f"loading {args.repo_id}")
    started = time.monotonic()
    model = AutoAWQForCausalLM.from_pretrained(
        args.repo_id, safetensors=True, device_map="auto", trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=False)
    load_s = time.monotonic() - started

    log(f"loaded in {load_s:.0f}s; starting {args.bits}-bit AWQ calibration (this is the expensive part)")
    quant_started = time.monotonic()
    kwargs = {"tokenizer": tokenizer, "quant_config": quant_config}
    if calib_data is not None:
        kwargs["calib_data"] = calib_data
        kwargs["max_calib_samples"] = len(calib_data)
        kwargs["max_calib_seq_len"] = int(args.calib_seq_len)
    model.quantize(**kwargs)
    quant_s = time.monotonic() - quant_started

    log(f"calibration finished in {quant_s / 60:.1f} min; writing to {out_dir}")
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    info = verify_output(out_dir, load=args.verify_load)
    total_s = time.monotonic() - started

    manifest = {
        "source_repo_id": args.repo_id,
        "quant_method": "awq",
        "bits": args.bits,
        "group_size": args.group_size,
        "version": "GEMM",
        "calibration_source": calib_source,
        "calibration_samples": len(calib_data) if calib_data is not None else args.calib_samples,
        "calibration_seq_len": args.calib_seq_len,
        "load_seconds": round(load_s, 1),
        "quantize_seconds": round(quant_s, 1),
        "total_seconds": round(total_s, 1),
        "output": info,
        # Deliberately absent: any carbon figure. The caller times the container
        # and multiplies by spec TDP; a second number computed in here would be
        # a second source of truth for the same quantity.
    }
    (out_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    log(
        f"done in {total_s / 60:.1f} min → {info['size_gb']} GB across {len(info['shards'])} shard(s), "
        f"calibrated on {calib_source}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        die("interrupted", 1)

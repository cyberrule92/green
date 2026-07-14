#!/usr/bin/env python3
"""Replay the prompt set across the three arms against the bench API.

    python3 benchmark/run_bench.py --repeats 3
    python3 benchmark/run_bench.py --arms css --limit 5 --dry-run

Writes one row per request to benchmark/results/<run_id>/raw.jsonl and is
resumable: re-running with the same --run-id skips (arm, prompt, repeat) triples
already present.

Carbon comes from the server's ex-post accounting (`carbon_accounting`, schema
2): TDP x *measured* duration, billed to the model that *actually served*, summed
over every inference leg the request burned. It is not the ex-ante CSS estimate,
which is a spec constant and is what the router used to *choose* — recording both
is the point, so the report can show how far the forecast was off.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent))
from arms import ARMS, TENANTS, expected_pin  # noqa: E402
from score import load_prompts, score_response  # noqa: E402

HERE = Path(__file__).parent
DEFAULT_API = "http://127.0.0.1:8101"
REQUEST_TIMEOUT_S = 300


def _get(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def contamination_flags(payload: dict[str, Any], legs: list[dict[str, Any]]) -> dict[str, bool]:
    """Reasons a sample is not a clean measurement of the arm's routing policy.

    A pinned arm that silently re-routes is not a control arm, and a cached
    answer was never computed by the model it is billed to. These are excluded
    from the headline table and their count is reported — if the count is
    material, that is itself a finding.
    """
    sel = _get(payload, "routing", "selected_candidate", default={}) or {}
    guard = payload.get("guardrail_trace") or {}
    escalated_leg = any(
        leg.get("requested_variant") and leg.get("served_variant")
        and leg["requested_variant"] != leg["served_variant"]
        for leg in legs
    )
    return {
        # The dispatcher escalated inside run_vllm_inference: the model that ran
        # is not the model CSS ranked.
        "auto_escalated": bool(sel.get("auto_escalated")) or escalated_leg,
        # History + retrieval overflowed the context window and forced a bigger model.
        "overflow_escalated": bool(sel.get("overflow_escalated")),
        # The first answer looked broken and a second inference ran on `full`.
        "quality_retry": bool(sel.get("quality_retry_triggered")) or len(legs) > 1,
        # Cache hit: no model ran, carbon is 0.0, the variant label is fabricated.
        "from_semantic_cache": payload.get("semantic_cache") is not None,
        # Arithmetic / multiplication-table shortcut: answered without a model.
        "direct_response": len(legs) == 0,
        "blocked": payload.get("status") != "success"
        or bool(_get(guard, "output", "blocked", default=False))
        or bool(_get(guard, "input", "blocked", default=False)),
        "redacted": bool(_get(guard, "output", "redactions", default=[])),
    }


def run_one(
    session: requests.Session,
    api: str,
    arm: str,
    row: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    params = ARMS[arm](row)
    # No conversation_id: the engine 404s on an id it has not seen, and omitting
    # it makes a fresh conversation per request — which is what we want anyway.
    # History depth feeds the profiler, so a shared thread would make prompt N's
    # routing depend on prompt N-1.
    fields = {
        "prompt": row["prompt"],
        "user_tier": "standard",
        # Default is 900_000 ms for medium/low priority, which lets EcoServe defer
        # the request out of the run entirely.
        "deferral_tolerance_ms": 0,
        **params,
    }
    headers = {"X-Tenant-Id": TENANTS[arm]}

    started = time.monotonic()
    error: str | None = None
    payload: dict[str, Any] = {}
    try:
        resp = session.post(
            f"{api}/api/chat", data=fields, headers=headers, timeout=REQUEST_TIMEOUT_S
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — a failed request is a data point
        error = f"{type(exc).__name__}: {exc}"
    client_latency_ms = (time.monotonic() - started) * 1000.0

    meta = _get(payload, "assistant_message", "metadata", default={}) or {}
    sus = meta.get("sustainability") or {}
    ca = sus.get("carbon_accounting") or {}
    legs = ca.get("billed_legs") or []
    sel = _get(payload, "routing", "selected_candidate", default={}) or {}
    answer = _get(payload, "assistant_message", "content", default="") or ""

    pin = expected_pin(arm, row)
    served = ca.get("served_target_id")

    record = {
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "repeat": repeat,
        "prompt_id": row["id"],
        "category": row["category"],
        "difficulty": row.get("difficulty", ""),
        "error": error,

        # What ran. `served_model` is the HF model the request was actually
        # dispatched to — recorded because several zoo candidates collapse onto
        # one backend (ultra-light and medium are both TinyLlama-1.1B on
        # vllm-medium), and a carbon delta between two labels of the same model
        # is a config artifact, not a saving.
        "served_target_id": served,
        "served_variant": sel.get("model_variant"),
        "served_model": payload.get("resolved_model_name"),
        "requested_pin": pin,
        # The engine's quality guardrail can raise the accuracy floor above a
        # pinned model's accuracy and silently drop the pin. Measure it.
        "pin_violated": bool(pin and served and pin != served),
        "n_legs": len(legs),
        "legs": legs,

        # Ex-post carbon (schema 2) — measured duration, model that served
        "carbon_g": ca.get("total_carbon_g"),
        "op_carbon_g": ca.get("op_carbon_g"),
        "emb_carbon_g": ca.get("emb_carbon_g"),
        "energy_wh": ca.get("energy_wh"),
        "measured_compute_s": ca.get("measured_compute_s"),
        "grid_carbon_g_per_kwh": ca.get("grid_carbon_g_per_kwh"),
        "carbon_basis": ca.get("basis"),

        # Ex-ante estimate the router actually chose on, kept for comparison
        "estimated_carbon_g": sel.get("estimated_carbon_g"),
        "css_score": sel.get("css_score"),

        # Latency / tokens
        "client_latency_ms": round(client_latency_ms, 1),
        "server_latency_ms": sus.get("actual_latency_ms"),
        "input_tokens": _get(meta, "tokens", "input", default=0),
        "output_tokens": _get(meta, "tokens", "output", default=0),

        # Quality
        "correct": score_response(answer, row["check"]) if answer else False,
        "answer": answer[:2000],
    }
    record["flags"] = contamination_flags(payload, legs) if not error else {"blocked": True}
    record["excluded"] = bool(error) or any(record["flags"].values())
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=DEFAULT_API, help=f"bench API base URL (default {DEFAULT_API})")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--repeats", type=int, default=3, help="sampling temperature is unseeded; n>=3 for variance")
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--run-id", default=None, help="reuse an existing run id to resume")
    ap.add_argument("--sleep", type=float, default=0.4, help="pause between requests; GPU util >80%% re-sorts candidates")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    done: set[tuple[str, str, int]] = set()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["arm"], r["prompt_id"], r["repeat"]))

    total = len(args.arms) * len(prompts) * args.repeats
    todo = total - len(done)
    print(f"run_id={run_id}  arms={args.arms}  prompts={len(prompts)}  repeats={args.repeats}")
    print(f"total={total}  already_done={len(done)}  to_run={todo}  api={args.api}")
    if args.dry_run:
        for arm in args.arms:
            pins: dict[str, int] = {}
            for row in prompts:
                pins[str(expected_pin(arm, row))] = pins.get(str(expected_pin(arm, row)), 0) + 1
            print(f"  {arm:<17} pins: {pins}")
        return 0

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "api": args.api,
                "arms": args.arms,
                "repeats": args.repeats,
                "n_prompts": len(prompts),
                "prompts_file": str(args.prompts),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    session = requests.Session()
    n = 0
    t0 = time.monotonic()
    with open(raw_path, "a", encoding="utf-8") as out:
        # Arm-major so a crash leaves whole arms comparable, and repeat-major
        # within an arm so a partial run still has every prompt once.
        for repeat in range(args.repeats):
            for arm in args.arms:
                for row in prompts:
                    key = (arm, row["id"], repeat)
                    if key in done:
                        continue
                    rec = run_one(session, args.api, arm, row, repeat)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    n += 1
                    mark = "x" if rec["excluded"] else ("." if rec["correct"] else "F")
                    print(
                        f"[{n}/{todo}] {mark} {arm:<17} {row['id']:<10} "
                        f"{str(rec['served_target_id']):<22} "
                        f"{rec['carbon_g'] if rec['carbon_g'] is not None else '-':<10} "
                        f"{rec['client_latency_ms']:>8.0f}ms"
                        + (f"  ERR {rec['error']}" if rec["error"] else ""),
                        flush=True,
                    )
                    time.sleep(args.sleep)

    print(f"\ndone in {(time.monotonic() - t0) / 60:.1f} min → {raw_path}")
    print(f"report:  python3 benchmark/report.py {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

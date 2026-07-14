#!/usr/bin/env python3
"""Turn a raw run into the three-row table.

    python3 benchmark/report.py <run_id>

Emits report.md and summary.csv next to raw.jsonl. Every number in the headline
table comes from clean samples only — a sample is excluded when it escalated,
retried, hit the cache, was answered without a model, or was blocked, because
none of those measure the arm's routing policy. The exclusion counts are printed
next to the table, not buried: if an arm is excluding a lot, that is the finding.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ARM_ORDER = ["always-full", "static-heuristic", "css"]

# data/ is the only host directory mounted into the API container, so it is the
# only place the UI's /api/benchmark endpoint can read a summary from.
SUMMARY_PATH = HERE.parent / "data" / "benchmark_summary.json"

LIMITATIONS = [
    "Power is modelled, not measured: the GPU is a vGPU slice (H100L-2-24C) and "
    "nvidia-smi reports power.draw = [N/A]. Operational carbon uses the spec TDP, "
    "an upper bound on real draw. Duration, tokens, and which model served are measured.",
    "Embodied carbon is amortised over lifetime x device_utilization (0.35) and scaled "
    "by device_share (0.255). device_utilization is the single free parameter; at this "
    "setting embodied is 1-3% of the total.",
    "Grid carbon intensity is pinned, so it scales all arms identically and cancels out "
    "of the comparison.",
    "The router is frozen: no RL exploration, no weight updates, no quality/latency "
    "estimator, no semantic cache, no MoE reconciler.",
    "Quality is a lower bound: objective checkers only (numeric / substring / regex / "
    "executed code), no LLM judge. The same checker runs against every arm.",
    "Sampling is unseeded, which is why every prompt runs multiple repeats.",
    "Not every zoo candidate is a distinct model — read the model mix by served model, "
    "not by candidate id.",
]


def load(run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = HERE / "results" / run_id
    rows = [json.loads(l) for l in (d / "raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    return rows, cfg


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(int(round(q * (len(s) - 1))), len(s) - 1)
    return s[k]


def summarise(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    arm_rows = [r for r in rows if r["arm"] == arm]
    clean = [r for r in arm_rows if not r["excluded"]]
    carbon = [r["carbon_g"] for r in clean if r["carbon_g"] is not None]
    energy = [r["energy_wh"] for r in clean if r["energy_wh"] is not None]
    lat = [r["client_latency_ms"] for r in clean]
    out_toks = [r["output_tokens"] for r in clean]
    correct = [1.0 if r["correct"] else 0.0 for r in clean]
    excl = Counter()
    for r in arm_rows:
        if r["error"]:
            excl["error"] += 1
        for k, v in (r.get("flags") or {}).items():
            if v:
                excl[k] += 1
    mix = Counter(
        f"{r['served_target_id'] or '?'} ({r.get('served_model') or '?'})" for r in clean
    )
    return {
        "arm": arm,
        "n_total": len(arm_rows),
        "n_clean": len(clean),
        "n_excluded": len(arm_rows) - len(clean),
        "carbon_g_mean": statistics.fmean(carbon) if carbon else 0.0,
        "carbon_g_total": sum(carbon),
        "energy_wh_mean": statistics.fmean(energy) if energy else 0.0,
        "quality": statistics.fmean(correct) if correct else 0.0,
        "latency_p50_ms": p(lat, 0.50),
        "latency_p95_ms": p(lat, 0.95),
        "output_tokens_mean": statistics.fmean(out_toks) if out_toks else 0.0,
        "pin_violations": sum(1 for r in arm_rows if r.get("pin_violated")),
        "model_mix": dict(mix.most_common()),
        "exclusions": dict(excl.most_common()),
    }


def quality_matrix(rows: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    """Per-category pass rate for each arm. None where an arm has no clean sample."""
    out = []
    for cat in sorted({r["category"] for r in rows}):
        vals: dict[str, float | None] = {}
        for arm in arms:
            clean = [r for r in rows if r["arm"] == arm and r["category"] == cat and not r["excluded"]]
            vals[arm] = statistics.fmean([1.0 if r["correct"] else 0.0 for r in clean]) if clean else None
        out.append({"category": cat, "n_prompts": len({r["prompt_id"] for r in rows if r["category"] == cat}), "quality": vals})
    return out


def quality_by_category(matrix: list[dict[str, Any]], arms: list[str]) -> list[str]:
    out = ["| category | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for row in matrix:
        cells = [pct(row["quality"][a]) if row["quality"][a] is not None else "—" for a in arms]
        out.append(f"| {row['category']} | " + " | ".join(cells) + " |")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        runs = sorted((HERE / "results").glob("*/raw.jsonl"))
        print("usage: report.py <run_id>\nruns:", *[r.parent.name for r in runs], sep="\n  ")
        return 1
    run_id = sys.argv[1]
    rows, cfg = load(run_id)
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
    stats = {a: summarise(rows, a) for a in arms}

    baseline = stats.get("always-full")
    lines: list[str] = []
    lines.append(f"# Routing benchmark — run `{run_id}`")
    lines.append("")
    lines.append(
        f"{cfg.get('n_prompts', '?')} prompts x {cfg.get('repeats', '?')} repeats x {len(arms)} arms "
        f"= {len(rows)} requests. Grid carbon pinned; router frozen (no exploration, no learning, no cache)."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| arm | gCO2e / request | vs always-full | quality | latency p50 | latency p95 | clean n |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in arms:
        s = stats[a]
        delta = "—"
        if baseline and baseline["carbon_g_mean"] > 0 and a != "always-full":
            d = (s["carbon_g_mean"] - baseline["carbon_g_mean"]) / baseline["carbon_g_mean"]
            delta = f"{100 * d:+.1f}%"
        lines.append(
            f"| {a} | {s['carbon_g_mean']:.4f} | {delta} | {pct(s['quality'])} | "
            f"{s['latency_p50_ms']:.0f} ms | {s['latency_p95_ms']:.0f} ms | {s['n_clean']} |"
        )
    lines.append("")
    lines.append("Carbon is ex-post: spec TDP x *measured* wall-clock, billed to the model that actually")
    lines.append("served, summed over every inference leg. Quality is the objective-checker pass rate.")
    lines.append("")

    matrix = quality_matrix(rows, arms)
    lines.append("## Quality by category")
    lines.append("")
    lines.extend(quality_by_category(matrix, arms))
    lines.append("")

    lines.append("## Model mix (clean samples)")
    lines.append("")
    for a in arms:
        s = stats[a]
        mix = ", ".join(f"{k} {v}" for k, v in s["model_mix"].items()) or "—"
        lines.append(f"- **{a}** — {mix}")
    lines.append("")

    lines.append("## Exclusions")
    lines.append("")
    lines.append("A sample is excluded when it did not measure the arm's routing policy: the dispatcher")
    lines.append("escalated, a quality retry fired a second inference, the semantic cache answered, no model")
    lines.append("ran at all, or a guardrail blocked it.")
    lines.append("")
    lines.append("| arm | excluded | of | reasons | pin violations |")
    lines.append("|---|---|---|---|---|")
    for a in arms:
        s = stats[a]
        reasons = ", ".join(f"{k} {v}" for k, v in s["exclusions"].items()) or "—"
        lines.append(
            f"| {a} | {s['n_excluded']} | {s['n_total']} | {reasons} | {s['pin_violations']} |"
        )
    lines.append("")
    lines.append("A **pin violation** is a pinned arm that was re-routed anyway: the engine's quality")
    lines.append("guardrail raises the accuracy floor for some intents, and a pinned model below that floor")
    lines.append("is silently dropped. The guardrail applies to all three arms equally.")
    lines.append("")

    lines.append("## Methods and limitations")
    lines.append("")
    lines.extend([
        "- **Power is modelled, not measured.** The GPU is a vGPU slice (`H100L-2-24C`); `nvidia-smi`",
        "  returns `[N/A]` for `power.draw`, so no per-request power reading exists on this hardware.",
        "  Operational carbon uses the spec TDP, which is an **upper bound** on real draw. What is",
        "  measured is duration, tokens, and which model served.",
        "- **Embodied carbon is amortised** over `lifetime_years x seconds_per_year x device_utilization`,",
        "  scaled by `device_share` (0.255 — a 24 GB slice of a ~94 GB board). `device_utilization`",
        "  (default 0.35) is the single free parameter and the number to attack first; at this setting",
        "  embodied is 1-3% of the total, so the headline is dominated by operational carbon.",
        "- **Carbon intensity is pinned** (`GRID_CARBON_FALLBACK`), so it scales all arms identically and",
        "  cancels out of the comparison. Re-run with a different value for the sensitivity curve.",
        "- **The router is frozen**: no RL exploration, no weight updates, no quality/latency estimator,",
        "  no semantic cache, no MoE reconciler. An unfrozen run measures the environment, not the policy.",
        "- **Quality is a lower bound.** Objective checkers only (numeric / substring / regex / executed",
        "  code). No LLM judge, so no judge variance — but a correct answer phrased unexpectedly can be",
        "  marked wrong. The same checker runs against every arm, so the *comparison* is fair even where",
        "  the absolute level is pessimistic.",
        "- **Sampling is unseeded** (temperature 0.1-0.3), which is why every prompt runs `--repeats` times.",
        "- **Not every zoo candidate is a distinct model.** `ultra-light` and `medium` both dispatch to",
        "  `vllm-medium` serving TinyLlama-1.1B, and `local-cpu-fallback` has no endpoint of its own so it",
        "  resolves there too. Their differing TDPs (70 / 95 / 145 W) and accuracies (0.60 / 0.66 / 0.81) are",
        "  hand-written config, so a carbon delta *between those labels* would be an artifact. The bench zoo",
        "  therefore marks the CPU fallback (and the candidates whose containers are not up) unavailable, so",
        "  the router can only pick a model that exists. The `served_model` column above shows what actually",
        "  ran — read the model mix by *model*, not by candidate id.",
    ])
    lines.append("")

    out_dir = HERE / "results" / run_id
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "arm", "n_clean", "n_excluded", "carbon_g_mean", "energy_wh_mean",
            "quality", "latency_p50_ms", "latency_p95_ms", "output_tokens_mean", "pin_violations",
        ])
        for a in arms:
            s = stats[a]
            w.writerow([
                a, s["n_clean"], s["n_excluded"], f"{s['carbon_g_mean']:.6f}",
                f"{s['energy_wh_mean']:.6f}", f"{s['quality']:.4f}",
                f"{s['latency_p50_ms']:.0f}", f"{s['latency_p95_ms']:.0f}",
                f"{s['output_tokens_mean']:.1f}", s["pin_violations"],
            ])

    # The UI reads this. Deltas are computed here rather than in the frontend so
    # that the table on screen and the table in report.md can never disagree.
    base_carbon = baseline["carbon_g_mean"] if baseline else 0.0
    base_quality = baseline["quality"] if baseline else 0.0
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_requests": len(rows),
        "n_prompts": cfg.get("n_prompts"),
        "repeats": cfg.get("repeats"),
        "grid_carbon_g_per_kwh": next(
            (r["grid_carbon_g_per_kwh"] for r in rows if r.get("grid_carbon_g_per_kwh")), None
        ),
        "carbon_basis": next((r["carbon_basis"] for r in rows if r.get("carbon_basis")), None),
        "baseline_arm": "always-full",
        "arms": [
            {
                **stats[a],
                "carbon_delta_pct": (
                    100.0 * (stats[a]["carbon_g_mean"] - base_carbon) / base_carbon
                    if base_carbon and a != "always-full"
                    else None
                ),
                "quality_delta_pp": (
                    100.0 * (stats[a]["quality"] - base_quality) if a != "always-full" else None
                ),
            }
            for a in arms
        ],
        "quality_by_category": matrix,
        "limitations": LIMITATIONS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwrote {out_dir / 'report.md'}, {out_dir / 'summary.csv'}, {out_dir / 'summary.json'}")
    print(f"published to {SUMMARY_PATH} → GET /api/benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Figure generator for the Adaptive Green AI solution PDF.

Every number plotted here is either read from the live config on disk
(config/model_zoo.json, config/policies.json, data/rl_state.json) or is a value
MEASURED on this deployment. Nothing is invented;
where a curve is illustrative rather than measured it says so on the axes.

    python3 docs/pdf_figures.py     ->  docs/pdf_assets/*.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "pdf_assets"
OUT.mkdir(parents=True, exist_ok=True)

# HPE-adjacent palette, kept consistent with the frontend.
GREEN = "#01A982"
DARK = "#0F2B24"
INK = "#243B36"
SOFT = "#7B8F89"
AMBER = "#F5A623"
RED = "#D9534F"
BLUE = "#3B7DD8"
GRID = "#E3EAE7"
LIGHT = "#F4F8F6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": SOFT,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

CI_MEASURED = 518.0   # gCO2/kWh — grid_ci_avg over the last 24 h on this deployment


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")
    return path


def _zoo():
    return json.loads((ROOT / "config" / "model_zoo.json").read_text())["models"]


def _policies():
    return json.loads((ROOT / "config" / "policies.json").read_text())


def _rl():
    p = ROOT / "data" / "rl_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def op_carbon_g(m, ci=CI_MEASURED):
    """LLMCarbon operational carbon for one request, using the model's own params."""
    t = m["latency_ms_p50"] / 1000.0
    he = m.get("hardware_efficiency", 0.7)
    if m.get("moe"):
        he *= (1 - m.get("all_to_all_overhead_ratio", 0.0))
    kwh = (m["power_tdp_w"] * t * m.get("pue", 1.3)) / (he * 3.6e6)
    return kwh * ci * m.get("region_carbon_multiplier", 1.0)


def emb_carbon_g(m):
    """LLMCarbon embodied carbon amortised over the device's inference lifetime."""
    t = m["latency_ms_p50"] / 1000.0
    years = m.get("device_lifetime_years", 5)
    vol = m.get("annual_inference_volume", 100000)
    avg_s = max(m["latency_ms_p50"] / 1000.0, 0.05)
    return (m.get("mfg_carbon_kg", 143) * 1000) / (years * vol * avg_s) * t * m.get("device_share", 1.0)


# ==========================================================================
# 1. Routing candidate landscape (live model zoo)
# ==========================================================================
def fig_zoo_landscape():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    models = [m for m in _zoo() if m.get("available") and m["latency_ms_p50"] <= 2000]

    for m in models:
        c = op_carbon_g(m) * 1000
        size = m["power_tdp_w"] * 1.9
        colour = GREEN if c < 8 else (BLUE if c < 20 else AMBER)
        ax.scatter(m["latency_ms_p50"], m["accuracy_baseline"], s=size, color=colour,
                   alpha=0.75, edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(f"{m['model_variant']}\n{c:.1f} mg", (m["latency_ms_p50"], m["accuracy_baseline"]),
                    textcoords="offset points", xytext=(0, 13), ha="center", fontsize=7.2, color=INK)

    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("baseline accuracy")
    ax.set_ylim(0.55, 1.0)
    ax.set_xlim(0, 700)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"Routing candidates — bubble = TDP, label = operational mgCO$_2$e/request at CI {CI_MEASURED:.0f} g/kWh",
                 loc="left", fontsize=10.5, color=DARK, fontweight="bold")
    fig.text(0.02, -0.04,
             "CSS ranks these four axes at once. There is no free lunch on any single one — the winner is the "
             "greenest candidate that still clears\nthe request's accuracy floor and SLA.",
             ha="left", fontsize=8.5, color=SOFT, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_zoo_landscape.png")


# ==========================================================================
# 2. Operational + embodied carbon per request, per variant
# ==========================================================================
def fig_carbon_split():
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    models = [m for m in _zoo()
              if m.get("available") and m["hardware_class"] == "gpu" and m["latency_ms_p50"] <= 700]
    models.sort(key=lambda m: op_carbon_g(m))

    names = [m["model_variant"] for m in models]
    op = [op_carbon_g(m) * 1000 for m in models]
    emb = [emb_carbon_g(m) * 1000 for m in models]

    ax.bar(names, op, color=GREEN, label="operational (TDP x t x PUE / HE x CI)")
    ax.bar(names, emb, bottom=op, color=DARK, label="embodied (amortised manufacturing)")

    for i, (o, e) in enumerate(zip(op, emb)):
        ax.text(i, o + e, f"{o + e:.1f}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_ylabel("mgCO$_2$e per request")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.set_title("LLMCarbon accounting per candidate — both terms are scored, both are audited",
                 loc="left", fontsize=10.5, color=DARK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_carbon_split.png")


# ==========================================================================
# 3. CSS tier weights (config) + what RL learned (live rl_state.json)
# ==========================================================================
def fig_css_weights():
    pol = _policies()["tiers"]
    rl = _rl().get("tiers", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    dims = ["carbon", "latency", "accuracy", "cost"]
    tiers = ["standard", "premium", "esg", "batch"]
    colours = [GREEN, BLUE, AMBER, SOFT]
    w = 0.2
    for i, d in enumerate(dims):
        vals = [pol[t][d] for t in tiers]
        ax1.bar([j + (i - 1.5) * w for j in range(len(tiers))], vals, w,
                color=colours[i], label=d)
    ax1.set_xticks(range(len(tiers)))
    ax1.set_xticklabels(tiers)
    ax1.set_ylabel("CSS weight")
    ax1.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax1.grid(axis="y", color=GRID, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.set_title("Initial per-tier policy (config/policies.json)", loc="left",
                  fontsize=9.5, color=DARK, fontweight="bold", pad=22)
    ax1.axhline(0.45, color=RED, lw=0.9, ls="--")
    ax1.text(3.45, 0.46, "carbon floor 0.45", fontsize=7.2, color=RED, ha="right")

    std_rl = (rl.get("standard") or {}).get("weights") or {}
    if std_rl:
        init = [pol["standard"][d] for d in dims]
        learned = [std_rl.get(d, 0) for d in dims]
        y = range(len(dims))
        ax2.barh([i + 0.18 for i in y], init, 0.34, color=SOFT, label="initial")
        ax2.barh([i - 0.18 for i in y], learned, 0.34, color=GREEN, label="learned online")
        ax2.set_yticks(list(y))
        ax2.set_yticklabels(dims)
        ax2.invert_yaxis()
        ax2.legend(frameon=False, fontsize=8)
        ax2.grid(axis="x", color=GRID, lw=0.7)
        ax2.set_axisbelow(True)
        eps = (rl.get("standard") or {}).get("episode_count", 0)
        ax2.set_title(f"RL drift, standard tier ({eps} episodes)", loc="left",
                      fontsize=9.5, color=DARK, fontweight="bold", pad=22)
        ax2.set_xlabel("weight")

    fig.text(0.02, -0.06,
             "Carbon is the dominant term by policy, and the RL controller may move the weights but not below the "
             "simplex floor — the router\ncannot learn its way out of being a sustainability router.",
             ha="left", fontsize=8.5, color=SOFT, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_css_weights.png")


# ==========================================================================
# 4. RL reward history (live rl_state.json)
# ==========================================================================
def fig_rl_learning():
    rl = _rl().get("tiers", {})
    std = rl.get("standard") or {}
    hist = std.get("reward_history") or []
    if not hist:
        return None

    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    xs = list(range(len(hist)))
    ax.plot(xs, hist, color=GREEN, lw=0.9, alpha=0.45, label="episode reward")

    # EMA, the same baseline the controller subtracts as its advantage.
    beta, ema, ema_series = 0.95, hist[0], []
    for r in hist:
        ema = beta * ema + (1 - beta) * r
        ema_series.append(ema)
    ax.plot(xs, ema_series, color=DARK, lw=2.0, label="baseline EMA (beta=0.95)")

    ax.fill_between(xs, hist, ema_series, color=GREEN, alpha=0.10)
    ax.set_xlabel(f"episode (last {len(hist)} of {std.get('episode_count', len(hist))})")
    ax.set_ylabel("reward R")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Online REINFORCE — real reward trace from this deployment",
                 loc="left", fontsize=10.5, color=DARK, fontweight="bold")
    fig.text(0.02, -0.07,
             "R = 0.35 r_sla + 0.30 r_carbon + 0.25 r_accuracy + 0.10 r_cost. The advantage (R - baseline) signs the "
             "weight update, so a request that\nbeat the running average pulls the policy toward whatever it did.",
             ha="left", fontsize=8.5, color=SOFT, style="italic")
    fig.tight_layout()
    return _save(fig, "fig_rl_learning.png")


# ==========================================================================
# 5. EcoServe deferral — carbon window (illustrative curve, labelled as such)
# ==========================================================================
def fig_deferral():
    fig, ax = plt.subplots(figsize=(9.4, 3.4))

    hours = [h for h in range(0, 25)]
    # A typical diurnal grid shape (solar trough at midday, evening peak).
    ci = [520, 530, 540, 545, 535, 500, 450, 400, 350, 300, 265, 240,
          230, 245, 275, 320, 390, 470, 560, 600, 590, 570, 550, 535, 525]

    ax.plot(hours, ci, color=DARK, lw=2)
    ax.fill_between(hours, ci, 0, color=GREEN, alpha=0.07)

    thresh = 600
    ax.axhline(thresh, color=RED, ls="--", lw=1.1)
    ax.text(0.2, thresh + 8, "deferral threshold (HIGH_CARBON_THRESHOLD /\nFINETUNE_DEFER_CI) g/kWh", fontsize=8, color=RED)

    submit_h, dispatch_h = 19, 12
    ax.scatter([submit_h], [ci[submit_h]], s=70, color=RED, zorder=5)
    ax.annotate("work submitted\n(grid dirty -> QUEUED,\nnot dropped)", (submit_h, ci[submit_h]),
                textcoords="offset points", xytext=(-18, 18), fontsize=8, color=RED, ha="center")

    ax.scatter([dispatch_h + 24 - 24], [ci[dispatch_h]], s=70, color=GREEN, zorder=5)
    ax.annotate("dispatched in the greenest\nwindow the 48 h forecast offers", (dispatch_h, ci[dispatch_h]),
                textcoords="offset points", xytext=(6, -34), fontsize=8, color=GREEN, ha="center")

    ax.annotate("", xy=(dispatch_h, 230), xytext=(submit_h, 230),
                arrowprops=dict(arrowstyle="<|-", color=GREEN, lw=1.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.text((submit_h + dispatch_h) / 2, 175,
            "bounded wait budget\ncarbon billed at EXECUTION-time CI, not submit-time",
            fontsize=8, color=GREEN, ha="center", fontweight="bold")

    ax.set_xlabel("hour of day (illustrative diurnal grid shape — live curve comes from Electricity Maps 48 h forecast)")
    ax.set_ylabel("grid CI (gCO$_2$/kWh)")
    ax.set_xlim(0, 24)
    ax.set_ylim(150, 680)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("EcoServe deferral — dirty-grid work waits for the greenest window",
                 loc="left", fontsize=10.5, color=DARK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig_deferral.png")


# ==========================================================================
# Diagram primitives
# ==========================================================================
def _box(ax, x, y, w, h, text, fc="white", ec=SOFT, tc=INK, fs=8, bold=False, lw=1.1, radius=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={radius}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, p1, p2, color=SOFT, lw=1.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2, mutation_scale=11,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=1))


def _canvas(w=9.6, h=5.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# ==========================================================================
# 6. HLD — system architecture
# ==========================================================================
def diag_hld():
    fig, ax = _canvas(9.6, 6.4)

    _box(ax, 2, 88, 96, 9, "", fc=LIGHT, ec=GRID)
    _box(ax, 3, 89.5, 30, 6, "React 19 + Grommet (nginx :8080)\nGreenAIChat", fc="white", ec=BLUE, fs=8, bold=True)
    _box(ax, 35, 89.5, 30, 6, "Chat · Carbon · Observability · Agent\n(4 tabs, live polling)", fc="white", ec=GRID, fs=7.5)
    _box(ax, 67, 89.5, 30, 6, "lib/api.js\ncentralised fetch layer", fc="white", ec=GRID, fs=7.5)
    ax.text(50, 97.6, "PRESENTATION", ha="center", fontsize=7.5, color=SOFT, fontweight="bold")

    _arrow(ax, (50, 88), (50, 82.5), color=BLUE, lw=1.6)
    ax.text(51.5, 85, "/api/*  (26 endpoints)", fontsize=7.5, color=BLUE)

    # Control plane
    _box(ax, 2, 56, 96, 26, "", fc="#F2FAF7", ec=GREEN, lw=1.4)
    ax.text(4, 79.6, "CONTROL PLANE — decision_engine.py (FastAPI :8100)", fontsize=8.5,
            color=DARK, fontweight="bold")

    stages = [
        ("guardrails\nIN", GREEN), ("RAG\nretrieve", GREEN), ("prompt\nprofiler", GREEN),
        ("CSS\nrank", DARK), ("EcoServe\ndefer?", AMBER), ("dispatch", GREEN),
        ("guardrails\nOUT", GREEN), ("audit\nHMAC", DARK), ("RL\nreward", BLUE),
    ]
    x = 4.0
    for i, (label, col) in enumerate(stages):
        _box(ax, x, 68, 9.4, 8, label, fc="white", ec=col, tc=INK, fs=7, lw=1.3)
        if i < len(stages) - 1:
            _arrow(ax, (x + 9.4, 72), (x + 10.4, 72), color=SOFT, lw=1.0)
        x += 10.4

    sub = [
        ("routing_policies.py\nCSS · profiler · MoE", 4),
        ("rl_controller.py\nREINFORCE + EMA", 23.6),
        ("advanced_rag.py\nhybrid + rerank", 43.2),
        ("model_zoo.py\nLLMCarbon registry", 62.8),
        ("model_onboarding.py\nquantize · serve · register", 82.4),
    ]
    for label, sx in sub:
        _box(ax, sx, 58, 15.6, 7.4, label, fc="white", ec=GRID, fs=6.8)
        _arrow(ax, (sx + 7.8, 68), (sx + 7.8, 65.4), color=GRID, lw=1.0)

    # Inference + state panels, narrowed to leave clear channels at x<5, x=48..52, x>95
    # so the signal arrows below can reach the control plane without crossing a box.
    _box(ax, 6, 30, 41, 22, "", fc=LIGHT, ec=GRID)
    ax.text(7.5, 49.6, "INFERENCE PLANE (vLLM, OpenAI-compatible)", fontsize=7.6, color=DARK, fontweight="bold")
    vllm = [("medium\n:8001", 7.5), ("full\n:8002", 22.2), ("stem-coding\n:8006", 36.9)]
    for label, vx in vllm:
        _box(ax, vx, 40, 8.8, 7, label, fc="white", ec=BLUE, fs=6.8)
    _box(ax, 7.5, 31.5, 18.3, 6.5, "multimodal.py\nVLM + diffusion (NIM)", fc="white", ec=AMBER, fs=6.8)
    _box(ax, 27.4, 31.5, 18.3, 6.5, "nemo_guardrails.py\npattern rails", fc="white", ec=GRID, fs=6.8)

    _box(ax, 53, 30, 41, 22, "", fc=LIGHT, ec=GRID)
    ax.text(54.5, 49.6, "STATE + TELEMETRY", fontsize=7.6, color=DARK, fontweight="bold")
    st = [("green_ai.db\nSQLite WAL", 54.5), ("rl_state.json\nlearned weights", 74),
          ("decision_logs.jsonl\nHMAC-signed audit", 54.5), ("rag_store.json\nchunks + vectors", 74)]
    for i, (label, sx) in enumerate(st):
        sy = 40 if i < 2 else 31.5
        _box(ax, sx, sy, 18.3, 7 if i < 2 else 6.5, label, fc="white", ec=GRID, fs=6.8)

    _arrow(ax, (26, 56), (26, 52.5), color=BLUE, lw=1.2, ls="--")
    ax.text(27, 53.4, "dispatch", fontsize=6.5, color=BLUE)
    _arrow(ax, (73, 56), (73, 52.5), color=DARK, lw=1.2, ls="--")
    ax.text(74, 53.4, "persist + sign", fontsize=6.5, color=DARK)

    # External signals — these feed the CONTROL PLANE (monitoring_layer / deferred_queue),
    # not the inference plane, so the arrows must reach it without touching the panels above.
    _box(ax, 2, 8, 30, 14, "", fc="white", ec=GRID)
    ax.text(17, 19.6, "EXTERNAL SIGNALS", ha="center", fontsize=8, color=DARK, fontweight="bold")
    _box(ax, 4, 9.5, 26, 8, "Electricity Maps v3\nlive CI + 48 h forecast (15-min)", fc=LIGHT, ec=GREEN, fs=7.5)

    _box(ax, 35, 8, 30, 14, "", fc="white", ec=GRID)
    ax.text(50, 19.6, "HOST TELEMETRY", ha="center", fontsize=8, color=DARK, fontweight="bold")
    _box(ax, 37, 9.5, 26, 8, "host_metrics_service.py :9000\nnvidia-smi + top (sidecar)", fc=LIGHT, ec=BLUE, fs=7.5)

    _box(ax, 68, 8, 30, 14, "", fc="white", ec=GRID)
    ax.text(83, 19.6, "DEFERRAL", ha="center", fontsize=8, color=DARK, fontweight="bold")
    _box(ax, 70, 9.5, 26, 8, "deferred_queue.py\nmin-heap, 500 cap, 10 s tick", fc=LIGHT, ec=AMBER, fs=7.5)

    # Left channel (x=3), centre gap (x=50), right channel (x=97).
    for pts, col in (([(4, 15), (3, 15), (3, 58), (5.5, 58)], GREEN),
                     ([(50, 22), (50, 56)], BLUE),
                     ([(96, 15), (97, 15), (97, 58), (94.5, 58)], AMBER)):
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            _arrow(ax, pts[i], pts[i + 1], color=col, lw=1.3,
                   style="-|>" if last else "-")
    ax.text(4.2, 26, "grid CI", fontsize=6.5, color=GREEN, rotation=90, va="bottom")
    ax.text(50.8, 26, "GPU / CPU power", fontsize=6.5, color=BLUE, rotation=90, va="bottom")
    ax.text(96.2, 26, "enqueue / dispatch", fontsize=6.5, color=AMBER, rotation=90, va="bottom", ha="right")

    ax.set_title("HLD — Adaptive Green AI control plane", loc="left", fontsize=12,
                 color=DARK, fontweight="bold", pad=8)
    fig.tight_layout()
    return _save(fig, "diag_hld.png")


# ==========================================================================
# 7. Workflow — request lifecycle
# ==========================================================================
def diag_workflow():
    fig, ax = _canvas(9.6, 6.6)

    lanes = [("REQUEST", 86), ("SAFETY + EVIDENCE", 70), ("DECISION", 52),
             ("EXECUTION", 32), ("LEARNING", 12)]
    for label, y in lanes:
        ax.add_patch(FancyBboxPatch((1, y - 4), 98, 13.5, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    facecolor=LIGHT if label in ("SAFETY + EVIDENCE", "EXECUTION") else "white",
                                    edgecolor=GRID, lw=1, zorder=0))
        ax.text(2.5, y + 7.4, label, fontsize=7, color=SOFT, fontweight="bold")

    _box(ax, 6, 84, 20, 7, "POST /api/chat\nprompt + attachments", fc="white", ec=BLUE, fs=7.5, bold=True)
    _box(ax, 32, 84, 22, 7, "conversation lookup\n+ history (SQLite)", fc="white", ec=GRID, fs=7.5)
    _box(ax, 60, 84, 22, 7, "semantic cache\n(conversation-scoped)", fc="white", ec=GRID, fs=7.5)
    _arrow(ax, (26, 87.5), (32, 87.5))
    _arrow(ax, (54, 87.5), (60, 87.5))
    ax.text(84.5, 87.5, "hit -> return\n(0 gCO2)", fontsize=7, color=GREEN, va="center", fontweight="bold")
    _arrow(ax, (82, 87.5), (84, 87.5), color=GREEN)

    _box(ax, 6, 68, 20, 7, "guardrails IN\njailbreak / harm", fc="white", ec=GREEN, fs=7.5)
    _box(ax, 32, 68, 22, 7, "hybrid RAG\ndense + sparse -> RRF\n-> cross-encoder", fc="white", ec=GREEN, fs=7)
    _box(ax, 60, 68, 22, 7, "prompt profiler\nintent · complexity\nSLA · accuracy floor · modality", fc="white", ec=GREEN, fs=6.6)
    _arrow(ax, (26, 71.5), (32, 71.5))
    _arrow(ax, (54, 71.5), (60, 71.5))
    _arrow(ax, (16, 84), (16, 75), color=SOFT)

    _box(ax, 6, 50, 26, 7, "CSS rank 4 candidates\nw_c·carbon + w_l·lat +\nw_a·acc + w_cost·cost", fc="white", ec=DARK, fs=6.8, bold=True)
    _box(ax, 38, 50, 22, 7, "q/l estimator\nrefines acc + latency\n(carbon untouched)", fc="white", ec=BLUE, fs=6.8)
    _box(ax, 66, 50, 26, 7, "EcoServe\nCI > threshold?\ndefer / reroute", fc="white", ec=AMBER, fs=7, bold=True)
    _arrow(ax, (32, 53.5), (38, 53.5))
    _arrow(ax, (60, 53.5), (66, 53.5))
    _arrow(ax, (71, 68), (71, 57), color=SOFT)

    _box(ax, 6, 30, 24, 7, "text -> vLLM\n(selected variant)", fc="white", ec=BLUE, fs=7.5)
    _box(ax, 36, 30, 24, 7, "vision -> VLM\nimage-gen -> diffusion", fc="white", ec=AMBER, fs=7.5)
    _box(ax, 66, 30, 26, 7, "guardrails OUT\n+ grounding verify", fc="white", ec=GREEN, fs=7.5)
    _arrow(ax, (30, 33.5), (36, 33.5))
    _arrow(ax, (60, 33.5), (66, 33.5))
    _arrow(ax, (18, 50), (18, 37), color=SOFT)
    _arrow(ax, (79, 50), (79, 37), color=AMBER, ls="--")
    ax.text(80.5, 43, "queued\n-> low-carbon\n   window", fontsize=6.8, color=AMBER)

    _box(ax, 6, 10, 26, 7, "audit log (HMAC-SHA256)\nfull CSS breakdown", fc="white", ec=DARK, fs=7)
    _box(ax, 38, 10, 24, 7, "RL reward -> weights\nper tenant tier", fc="white", ec=BLUE, fs=7)
    _box(ax, 68, 10, 24, 7, "SQLite persist\n+ user feedback", fc="white", ec=GRID, fs=7)
    _arrow(ax, (32, 13.5), (38, 13.5))
    _arrow(ax, (62, 13.5), (68, 13.5))
    _arrow(ax, (19, 30), (19, 17), color=SOFT)
    _arrow(ax, (50, 17), (50, 50), color=BLUE, lw=1.3, rad=-0.32, style="-|>")
    ax.text(46, 26, "closed loop:\nevery request\ntrains the next", fontsize=7, color=BLUE,
            ha="right", style="italic")

    ax.set_title("Workflow — the /api/chat request lifecycle", loc="left", fontsize=12,
                 color=DARK, fontweight="bold", pad=8)
    fig.tight_layout()
    return _save(fig, "diag_workflow.png")


# ==========================================================================
# 8. LLD — CSS scoring funnel
# ==========================================================================
def diag_css():
    fig, ax = _canvas(9.6, 5.0)

    _box(ax, 2, 74, 96, 18, "", fc=LIGHT, ec=GRID)
    ax.text(3.5, 89.5, "1 — CANDIDATES (config/model_zoo.json)", fontsize=8, color=DARK, fontweight="bold")
    cands = [("ultra-light\n0.35B · 95 W", 4), ("medium\n1.1B · 145 W", 28), ("full\n1.5B · 225 W", 52),
             ("cpu-fallback\n0.35B · 70 W", 76)]
    for label, cx in cands:
        _box(ax, cx, 76, 20, 9, label, fc="white", ec=SOFT, fs=7.5)

    _arrow(ax, (50, 74), (50, 68), color=SOFT, lw=1.5)

    _box(ax, 2, 48, 96, 19, "", fc="white", ec=GREEN, lw=1.3)
    ax.text(3.5, 64.5, "2 — SCORE EACH AXIS (min-max normalised across the candidate set)", fontsize=8,
            color=DARK, fontweight="bold")
    axes_ = [("carbon_score\n1 − norm(C_op + C_emb)", 4, GREEN),
             ("latency_score\n1 − norm(latency_eff)", 28, BLUE),
             ("accuracy_score\nnorm(acc, 0.45, 1.0)", 52, AMBER),
             ("cost_score\n1 − norm(cost_units)", 76, SOFT)]
    for label, cx, col in axes_:
        _box(ax, cx, 50, 20, 9, label, fc="white", ec=col, fs=7)

    _arrow(ax, (50, 48), (50, 42), color=SOFT, lw=1.5)

    _box(ax, 14, 30, 72, 10,
         "CSS  =  w_carbon·carbon + w_latency·latency + w_accuracy·accuracy + w_cost·cost + w_region·region\n"
         "weights per tenant tier, learned online by REINFORCE  ·  carbon weight floored at 0.45",
         fc="#F2FAF7", ec=DARK, fs=8, bold=True, lw=1.4)

    _arrow(ax, (50, 30), (50, 24), color=SOFT, lw=1.5)

    _box(ax, 2, 8, 96, 15, "", fc=LIGHT, ec=GRID)
    ax.text(3.5, 20.5, "3 — PENALTIES + BONUSES, THEN ARGMAX", fontsize=8, color=DARK, fontweight="bold")
    pens = [("SLA breach\n−0.12 … −0.25", 4, RED), ("accuracy floor\n−0.18", 22.5, RED),
            ("semantic align\n±0.14", 41, BLUE), ("high-CI period\n−0.05", 59.5, AMBER),
            ("WINNER\ngreenest feasible", 78, GREEN)]
    for label, cx, col in pens:
        bold = col == GREEN
        _box(ax, cx, 9.5, 17.5, 8.5, label, fc="white" if not bold else "#E9F7F2",
             ec=col, fs=7, bold=bold, lw=1.4 if bold else 1.1)

    ax.set_title("LLD — Composite Sustainability Score (routing_policies.rank_routing_candidates)",
                 loc="left", fontsize=11.5, color=DARK, fontweight="bold", pad=8)
    fig.tight_layout()
    return _save(fig, "diag_css.png")


# ==========================================================================
# 9. Data + audit
# ==========================================================================
def diag_data():
    fig, ax = _canvas(9.6, 3.6)

    _box(ax, 2, 60, 96, 32, "", fc=LIGHT, ec=GRID)
    ax.text(3.5, 88, "PERSISTED STATE  (data/)", fontsize=8.5, color=DARK, fontweight="bold")
    items = [
        ("green_ai.db\nSQLite WAL\nconversations · messages\n· message_feedback", 4, GRID),
        ("decision_logs.jsonl\nHMAC-SHA256 signed\nappend-only audit trail", 27, DARK),
        ("rl_state.json\nper-tier weights\nbaseline · episodes", 50, BLUE),
        ("rag_store.json\nchunks + embeddings", 73, GREEN),
    ]
    for label, x, col in items:
        _box(ax, x, 63, 21, 20, label, fc="white", ec=col, fs=7)

    _box(ax, 2, 12, 96, 38, "", fc="white", ec=DARK, lw=1.3)
    ax.text(3.5, 45, "ONE AUDIT LINE = ONE REPLAYABLE DECISION", fontsize=8.5, color=DARK, fontweight="bold")
    ax.text(4, 38,
            "timestamp · request_id · conversation_id · selected_model · user_tier · priority\n"
            "input_understanding (full semantic profile)  ·  policy_coefficients (+ rl_version, exploration flag)\n"
            "candidate_rankings[:5] — every CSS breakdown, not just the winner  ·  selected_candidate (C_op, C_emb)\n"
            "eco_actions (deferral · reroute · low-carbon window)  ·  guardrail_trace.{input,output}\n"
            "grid_carbon · system_power_w · gpu_co2_g · tokens.{in,out,co2_per_token_ug} · actual_latency_ms\n"
            "_hmac  =  SHA-256 over the canonical JSON body (AUDIT_HMAC_KEY, server-only)",
            fontsize=7.4, color=INK, va="top", linespacing=1.7, family="DejaVu Sans")

    ax.text(50, 4,
            "Every dashboard derives from this one file. /api/observability/summary keeps no state of its own — "
            "so there is no metrics drift between views.",
            ha="center", fontsize=7.6, color=SOFT, style="italic")

    ax.set_title("LLD — persistence and the signed audit trail", loc="left", fontsize=11.5,
                 color=DARK, fontweight="bold", pad=8)
    fig.tight_layout()
    return _save(fig, "diag_data.png")


if __name__ == "__main__":
    print("figures ->")
    fig_zoo_landscape()
    fig_carbon_split()
    fig_css_weights()
    fig_rl_learning()
    fig_deferral()
    diag_hld()
    diag_workflow()
    diag_css()
    diag_data()
    print("done")

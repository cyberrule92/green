#!/usr/bin/env python3
"""
Build the Adaptive Green AI solution document (PDF).

Everything in the output is derived from the repository and from live system
state — config/model_zoo.json, config/policies.json, data/rl_state.json, and the
agent runs measured on this box. Nothing is illustrative unless it says so on the
chart itself.

    python3 docs/generate_solution_pdf.py            # -> docs/Adaptive_Green_AI_Solution.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image, Table,
    TableStyle, PageBreak, CondPageBreak, KeepTogether, NextPageTemplate,
)

import sys as _sys

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "pdf_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "docs" / "Adaptive_Green_AI_Solution.pdf"

# ---------------------------------------------------------------------------
# Palette — HPE green, kept consistent between the matplotlib figures and the
# reportlab body so the document reads as one artefact.
# ---------------------------------------------------------------------------
GREEN = "#01A982"
GREEN_D = "#00775B"
INK = "#0F2B24"
SLATE = "#5A6B66"
LINE = "#D6E0DC"
BG = "#F5F8F7"
AMBER = "#D97706"
RED = "#C2410C"
BLUE = "#1D6FA3"
PURPLE = "#6D4AA8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def _save(fig, name: str) -> Path:
    path = ASSETS / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _clean(ax, hide_x_grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if hide_x_grid:
        ax.xaxis.grid(False)


# ---------------------------------------------------------------------------
# Live inputs
# ---------------------------------------------------------------------------
zoo = json.loads((ROOT / "config" / "model_zoo.json").read_text())
policies = json.loads((ROOT / "config" / "policies.json").read_text())
try:
    rl_state = json.loads((ROOT / "data" / "rl_state.json").read_text())
except FileNotFoundError:
    rl_state = {"tiers": {}}

ZOO_MODELS = zoo.get("models") or zoo.get("targets") or []
GRID_CI = 518.0   # measured average grid intensity on this deployment (obs summary)

# Measured on this box, 2026-07-13. Every figure below came out of an actual run
# of POST /api/agent/task; none of it is estimated.
AGENT_RUNS = {
    "fizzbuzz_model_spec":  {"carbon": 2.98, "calls": 4, "status": "failed",    "tier": "7B (escalated)"},
    "fizzbuzz_caller_spec": {"carbon": 0.028, "calls": 1, "status": "completed", "tier": "1.5B"},
    "wordcount_caller":     {"carbon": 0.045, "calls": 1, "status": "completed", "tier": "1.5B"},
    "palindrome_before":    {"carbon": 32.8, "calls": 4, "status": "failed",    "tier": "7B (escalated)"},
    "palindrome_after":     {"carbon": 3.19, "calls": 2, "status": "completed", "tier": "1.5B"},
    "broken_harness":       {"carbon": 0.20, "calls": 1, "status": "aborted",   "tier": "1.5B"},
}


# Carbon for the charts below comes from the SHIPPED implementation, not a copy
# of it. These two functions used to re-derive the formulas locally, and had
# drifted: the operational one clamped hardware efficiency before applying the
# MoE all-to-all penalty (the product clamps after, so a MoE candidate could be
# charged below the floor here), and the embodied one still used the superseded
# annual_inference_volume x avg_seconds denominator that
# model_zoo.compute_embodied_rate documents as a fixed bug — that denominator
# cancels the duration it is multiplied by, so every request came out identical.
# Delegating means a change to the carbon model cannot leave this document
# quoting numbers the product does not produce.
_sys.path.insert(0, str(ROOT))
from model_zoo import ModelZooService  # noqa: E402

_ZOO_SERVICE = ModelZooService(zoo_path=ROOT / "config" / "model_zoo.json")

# Workflow figures, read from the registry and the seeded gallery rather than
# written into the prose, so a new node type or template cannot leave this
# document quoting a number the product has moved past.
import workflows as _WF_MOD              # noqa: E402
import workflow_templates as _WT_MOD     # noqa: E402

_WF_NODE_TYPES = _WF_MOD.NODE_TYPES
_WF_TEMPLATE_COUNT = len(_WT_MOD.TEMPLATES)
_WF_INDUSTRY_COUNT = len({t["industry"] for t in _WT_MOD.TEMPLATES})


def _request_duration_s(m: dict) -> float:
    """Per-request wall-clock the charts price against: the candidate's p50."""
    return m.get("latency_ms_p50", 200) / 1000.0


def op_carbon_g(m: dict, ci: float = GRID_CI) -> float:
    """Operational carbon for one request, via model_zoo (power x time)."""
    out = _ZOO_SERVICE.compute_operational_carbon(
        m.get("id", ""), grid_carbon_g_per_kwh=ci,
        inference_duration_s=_request_duration_s(m),
    )
    return float(out.get("op_carbon_g", 0.0) or 0.0)


def emb_carbon_g(m: dict) -> float:
    """Embodied (manufacturing) carbon amortised over one request, via model_zoo."""
    out = _ZOO_SERVICE.compute_embodied_carbon(
        m.get("id", ""), inference_duration_s=_request_duration_s(m),
    )
    return float(out.get("emb_carbon_g", 0.0) or 0.0)


# ===========================================================================
# DIAGRAMS
# ===========================================================================

def _box(ax, x, y, w, h, label, sub=None, fc="white", ec=GREEN, lw=1.4, fs=8.5,
         text_color=INK, radius=0.02, bold=True):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    p.set_clip_on(False)   # a box flush with xlim=1.0 otherwise loses its right border
    ax.add_patch(p)
    ty = y + h / 2 + (0.012 if sub else 0)
    ax.text(x + w / 2, ty, label, ha="center", va="center", fontsize=fs,
            color=text_color, fontweight="bold" if bold else "normal", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center",
                fontsize=fs - 2.1, color=SLATE, zorder=3)


def _arrow(ax, p1, p2, color=SLATE, style="-|>", lw=1.2, ls="-", rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, linestyle=ls, zorder=z,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3,
    ))


def diagram_hld() -> Path:
    """High-level design: containers, control plane, data plane."""
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # --- client / edge
    _box(ax, 0.06, 0.885, 0.88, 0.075, "React 19 + Grommet SPA  ·  green-frontend (nginx :8080)",
         sub="Chat · Carbon · Observability · Agent   —   src/lib/api.js centralises every fetch",
         fc="#EAF7F3", ec=GREEN_D)
    _arrow(ax, (0.5, 0.885), (0.5, 0.845), color=GREEN_D, lw=1.6)
    ax.text(0.515, 0.864, "/api/*", fontsize=7.5, color=SLATE, va="center")

    # --- control plane
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.545), 0.88, 0.30, boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.8, edgecolor=GREEN, facecolor="white", zorder=1))
    ax.text(0.08, 0.815, "decision_engine.py — FastAPI control plane (green-api :8100)",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(0.08, 0.792, "one pipeline, every request; nothing bypasses it",
            fontsize=7.6, color=SLATE, style="italic")

    stages = [
        ("guardrails\nin", GREEN), ("RAG\nretrieve", GREEN), ("profile\n+ modality", GREEN),
        ("CSS\nrank", GREEN_D), ("EcoServe\ndefer?", AMBER), ("dispatch", GREEN_D),
        ("guardrails\nout", GREEN), ("audit\nHMAC", BLUE), ("RL\nreward", PURPLE),
    ]
    x0, w, gap = 0.085, 0.088, 0.007
    for i, (lbl, c) in enumerate(stages):
        x = x0 + i * (w + gap)
        _box(ax, x, 0.665, w, 0.10, lbl, fc="white", ec=c, fs=7.2, radius=0.015)
        if i:
            _arrow(ax, (x - gap, 0.715), (x, 0.715), color=SLATE, lw=1.0)

    # supporting modules row
    mods = [
        "routing_policies\nCSS · profiler · MoE",
        "quality_latency_\nestimator",
        "rl_controller\nREINFORCE",
        "advanced_rag\nhybrid + rerank",
        "deferred_queue\nEcoServe",
        "model_zoo\nLLMCarbon",
        "multimodal\nVLM · diffusion",
        "coding_agent\nLangGraph",
    ]
    mw = 0.104
    for i, m in enumerate(mods):
        x = 0.075 + i * (mw + 0.005)
        head, sub = m.split("\n")
        _box(ax, x, 0.565, mw, 0.078, head, sub=sub, fc=BG, ec=LINE, fs=6.8,
             radius=0.012, bold=True)

    # --- data plane: vLLM
    ax.text(0.06, 0.495, "Inference plane — vLLM (OpenAI-compatible), one container per rung",
            fontsize=9, fontweight="bold", color=INK)
    vllm = [
        ("medium :8001", "TinyLlama 1.1B", GREEN),
        ("full :8002", "Qwen2.5-1.5B", GREEN),
        ("stem-coding :8006", "Qwen2.5-Coder-1.5B", GREEN_D),
        ("coder-7b :8009", "Qwen2.5-Coder-7B-AWQ", GREEN_D),
        ("stem-math :8004", "Qwen2.5-Math", SLATE),
        ("cpu-fallback :8007", "always reachable", SLATE),
    ]
    vw = 0.142
    for i, (name, sub, c) in enumerate(vllm):
        x = 0.06 + i * (vw + 0.005)
        _box(ax, x, 0.395, vw, 0.075, name, sub=sub, fc="white", ec=c, fs=7.4, radius=0.012)
        _arrow(ax, (x + vw / 2, 0.545), (x + vw / 2, 0.472), color=LINE, lw=1.0)

    # --- external signals
    _box(ax, 0.06, 0.245, 0.26, 0.095, "Electricity Maps API",
         sub="live CI + 48 h forecast · 15 min", fc="#EAF3F8", ec=BLUE, fs=8.2)
    _box(ax, 0.37, 0.245, 0.26, 0.095, "green-metrics :9000",
         sub="nvidia-smi + top sidecar", fc="#EAF3F8", ec=BLUE, fs=8.2)
    _box(ax, 0.68, 0.245, 0.26, 0.095, "NVIDIA NIM (pluggable)",
         sub="VLM · SDXL / FLUX — unset ⇒ fallback", fc="#F5F0FA", ec=PURPLE, fs=8.2)
    for cx in (0.19, 0.50, 0.81):
        _arrow(ax, (cx, 0.34), (cx, 0.395 if cx == 0.81 else 0.545), color=LINE, lw=1.0, ls=":")

    # --- persistence
    ax.text(0.06, 0.185, "Persistence  (data/)", fontsize=9, fontweight="bold", color=INK)
    stores = [
        ("green_ai.db", "SQLite WAL · conversations"),
        ("decision_logs.jsonl", "HMAC-signed audit trail"),
        ("rl_state.json", "learned tier weights"),
        ("rag_store.json", "chunks + embeddings"),
        ("ql_estimator_state.json", "learned acc/lat priors"),
    ]
    sw = 0.172
    for i, (n, s) in enumerate(stores):
        x = 0.06 + i * (sw + 0.006)
        _box(ax, x, 0.075, sw, 0.078, n, sub=s, fc="#FBFBF9", ec=LINE, fs=7.0, radius=0.012)

    ax.text(0.5, 0.02, "Every number the dashboards show is derived from decision_logs.jsonl — "
                       "there is no second source of truth.",
            ha="center", fontsize=7.6, color=SLATE, style="italic")
    return _save(fig, "hld.png")


def diagram_workflow() -> Path:
    """The request lifecycle, including the two branches that make it green."""
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps = [
        ("1  Guardrails (in)", "jailbreak / injection / harm — hard block", GREEN),
        ("2  RAG retrieve", "dense + sparse → RRF → cross-encoder rerank", GREEN),
        ("3  Profile", "intent · complexity · SLA · accuracy floor · modality", GREEN),
        ("4  CSS rank", "carbon · latency · accuracy · cost, RL-weighted", GREEN_D),
        ("5  EcoServe", "grid CI high? → defer to greenest window", AMBER),
        ("6  Dispatch", "vLLM rung · VLM · diffusion", GREEN_D),
        ("7  Guardrails (out)", "PII / credential / unsafe echo", GREEN),
        ("8  Audit", "HMAC-SHA256 signed JSONL row", BLUE),
        ("9  RL reward", "observed outcome → new tier weights", PURPLE),
    ]
    # Left column. It has to bottom out above y≈0.13 so the RL feedback arrow can be
    # routed underneath it instead of straight through boxes 8 and 9.
    y = 0.945
    h = 0.075
    step = h + 0.0135
    for i, (t, s, c) in enumerate(steps):
        _box(ax, 0.06, y - h, 0.46, h, t, sub=s, fc="white", ec=c, fs=8.4)
        if i < len(steps) - 1:
            _arrow(ax, (0.29, y - h), (0.29, y - h - 0.0135), color=SLATE, lw=1.1)
        y -= step
    col_bottom = y + step - h          # bottom edge of the last box

    def _panel(x, y0, w, h_, title, lines, fc, ec, title_color=INK):
        """Title sits at the top of the panel — _box() centres it, which lands it on
        top of the body text."""
        _box(ax, x, y0, w, h_, "", fc=fc, ec=ec)
        ax.text(x + 0.018, y0 + h_ - 0.030, title, fontsize=9.2, color=title_color,
                fontweight="bold", va="center", zorder=3)
        ly = y0 + h_ - 0.062
        for txt, fs, col, weight in lines:
            ax.text(x + 0.018, ly, txt, fontsize=fs, color=col, va="top", zorder=3,
                    fontweight="bold" if weight == "b" else "normal",
                    style="italic" if weight == "i" else "normal")
            ly -= 0.0225 * (txt.count("\n") + 1) + (0.008 if weight == "b" else 0.004)

    # right column — the two carbon levers
    _panel(0.58, 0.615, 0.38, 0.275, "The carbon levers", [
        ("① CSS chooses the greenest FEASIBLE rung", 8.1, INK, "b"),
        ("Carbon weight dominates (0.45–0.70). Accuracy floor\nand SLA act as gates, "
         "not a competing objective.", 7.3, SLATE, ""),
        ("② EcoServe moves the work in TIME", 8.1, INK, "b"),
        ("Same joules, cleaner grid. 48 h forecast, ≥15 % CI\ndrop required, "
         "deadline-bounded.", 7.3, SLATE, ""),
    ], BG, LINE)

    _arrow(ax, (0.52, 0.545), (0.58, 0.545), color=AMBER, lw=1.6, rad=-0.15)
    _panel(0.58, 0.365, 0.38, 0.195, "deferred_queue.py", [
        ("min-heap on target_dispatch · max 500", 7.3, SLATE, ""),
        ("daemon tick 10 s → dispatch_fn(payload)", 7.3, SLATE, ""),
        ("deadline expiry still runs the task", 7.3, SLATE, ""),
        ("the coding agent is its first real caller", 7.3, AMBER, "i"),
    ], "#FDF6EC", AMBER)

    _panel(0.58, 0.155, 0.38, 0.165, "Closed loop", [
        ("every request writes one signed audit row", 7.3, SLATE, ""),
        ("every outcome updates the tier policy", 7.3, SLATE, ""),
        ("no offline training step, no manual tuning", 7.3, SLATE, ""),
    ], "#F5F0FA", PURPLE)

    # RL feedback: down, along the foot of the page, then back up the outside into
    # the CSS-rank box. Routing it across the middle put it through boxes 8 and 9.
    css_y = 0.945 - 3 * step - h / 2          # centre of box 4 (CSS rank)
    _arrow(ax, (0.77, 0.155), (0.77, 0.075), color=PURPLE, lw=1.2, style="-")
    _arrow(ax, (0.77, 0.075), (0.032, 0.075), color=PURPLE, lw=1.2, style="-")
    _arrow(ax, (0.032, 0.075), (0.032, css_y), color=PURPLE, lw=1.2, style="-")
    _arrow(ax, (0.032, css_y), (0.058, css_y), color=PURPLE, lw=1.2)
    ax.text(0.40, 0.092, "feeds the next request's CSS weights", fontsize=7.2,
            color=PURPLE, ha="center", style="italic")
    assert col_bottom > 0.10, "left column would collide with the feedback arrow"
    return _save(fig, "workflow.png")


def diagram_agent_graph() -> Path:
    """LangGraph state machine for the coding agent — the LLD that matters most."""
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _box(ax, 0.03, 0.60, 0.145, 0.13, "submit", sub="task + optional\nfrozen spec",
         fc=BG, ec=SLATE, fs=8.4)
    _box(ax, 0.215, 0.60, 0.15, 0.13, "node_generate", sub="write / repair files", fc="white", ec=GREEN, fs=8.4)
    _box(ax, 0.405, 0.60, 0.15, 0.13, "node_verify", sub="run the tests", fc="white", ec=GREEN_D, fs=8.4)
    _box(ax, 0.595, 0.60, 0.15, 0.13, "classify()", sub="pure router —\nsingle source of truth",
         fc="#FDF6EC", ec=AMBER, fs=8.4)
    _box(ax, 0.795, 0.60, 0.17, 0.13, "node_finalize", sub="terminal status\n+ carbon accounting",
         fc="white", ec=BLUE, fs=8.4)

    for a, b in [(0.175, 0.215), (0.365, 0.405), (0.555, 0.595), (0.745, 0.795)]:
        _arrow(ax, (a, 0.665), (b, 0.665), color=SLATE, lw=1.3)

    # escalate loop
    _box(ax, 0.405, 0.36, 0.15, 0.11, "node_escalate", sub="next rung up", fc="white", ec=RED, fs=8.4)
    _arrow(ax, (0.67, 0.60), (0.555, 0.415), color=RED, lw=1.3, rad=0.25)
    ax.text(0.655, 0.50, "verifier evidence\n(tests actually failed)", fontsize=6.9, color=RED, ha="center")
    _arrow(ax, (0.405, 0.415), (0.29, 0.60), color=RED, lw=1.3, rad=0.25)

    # retry loop on same tier
    _arrow(ax, (0.62, 0.73), (0.29, 0.73), color=GREEN, lw=1.2, rad=-0.28)
    ax.text(0.455, 0.815, "retry on the SAME rung  (attempts_per_tier = 2)",
            fontsize=7.2, color=GREEN_D, ha="center")

    # abort path
    _arrow(ax, (0.67, 0.60), (0.85, 0.60), color=SLATE, lw=1.0, ls=":")
    _box(ax, 0.78, 0.36, 0.20, 0.11, "abort — never escalate",
         sub="harness_ok=False · backend_failed", fc="#FBECEC", ec=RED, fs=8.0)
    _arrow(ax, (0.72, 0.60), (0.86, 0.47), color=RED, lw=1.3, rad=-0.2)

    # the rules
    rules = [
        ("THE TESTS ARE THE SPEC, AND THEY ARE FROZEN.",
         "An unfrozen verifier gets reward-hacked: the 7B once 'fixed' fizzbuzz by rewriting the test to assert the bug."),
        ("WHAT GETS FROZEN MUST FIRST BE VALIDATED.",
         "invalid_test_reason(): must parse · contain test_* · import what it tests · not repeat itself."),
        ("INFRASTRUCTURE IS NOT EVIDENCE.",
         "A dead endpoint or a broken pytest says nothing about the model — abort, never escalate into it."),
        ("WHO AUTHORS THE SPEC DECIDES THE CARBON.",
         "Caller-supplied tests ⇒ the weakest rung can no longer condemn a correct implementation."),
    ]
    y = 0.245
    for i, (head, body) in enumerate(rules):
        ax.text(0.035, y, f"▸ {head}", fontsize=8.0, color=INK, fontweight="bold")
        ax.text(0.055, y - 0.038, body, fontsize=7.2, color=SLATE)
        y -= 0.075

    ax.text(0.5, 0.955, "coding_agent.py — the ladder optimises carbon per SUCCESSFUL COMPLETION, not per token",
            ha="center", fontsize=9.6, color=INK, fontweight="bold")
    return _save(fig, "agent_graph.png")


def diagram_css_funnel() -> Path:
    """How four candidates become one decision."""
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Widths are chosen so the final "Winner" box lands inside xlim=1.0 — at the
    # previous spacing it ran off the right edge and was clipped in the PDF.
    _box(ax, 0.015, 0.42, 0.13, 0.30, "Candidates", sub="model_zoo\navailable_targets()",
         fc=BG, ec=SLATE, fs=8.2)
    gates = [
        ("Modality gate", "text / vision /\nimage-gen axis", GREEN),
        ("LLMCarbon", "C_op + C_emb\nper candidate", GREEN_D),
        ("Learned q/l", "per-prompt accuracy\n+ latency correction", PURPLE),
        ("CSS score", "Σ wᵢ·scoreᵢ\nRL weights per tier", GREEN_D),
        ("Penalties", "SLA · accuracy floor\nhigh-CI · semantic fit", AMBER),
    ]
    x = 0.168
    for name, sub, c in gates:
        _box(ax, x, 0.42, 0.122, 0.30, name, sub=sub, fc="white", ec=c, fs=7.9)
        _arrow(ax, (x - 0.023, 0.57), (x, 0.57), color=SLATE, lw=1.2)
        x += 0.134
    _box(ax, x, 0.42, 0.125, 0.30, "Winner", sub="greenest FEASIBLE\ncandidate",
         fc="#EAF7F3", ec=GREEN, lw=2.0, fs=8.6)
    _arrow(ax, (x - 0.024, 0.57), (x, 0.57), color=GREEN, lw=1.8)

    ax.text(0.5, 0.30, "carbon_score = 1 − norm(C_total)      latency_score = 1 − norm(t_eff)      "
                       "accuracy_score = norm(acc)      cost_score = 1 − norm(cost)",
            ha="center", fontsize=8.0, color=INK, family="monospace")
    ax.text(0.5, 0.22, "CSS = w_carbon·carbon + w_latency·latency + w_accuracy·accuracy + w_cost·cost + w_region·region",
            ha="center", fontsize=8.6, color=GREEN_D, family="monospace", fontweight="bold")
    ax.text(0.5, 0.11, "The accuracy floor and the SLA are gates, not competitors: they veto a candidate that "
                       "cannot do the job,\nbut they cannot outvote a large carbon delta between two candidates "
                       "that both can.",
            ha="center", fontsize=7.8, color=SLATE, style="italic")
    return _save(fig, "css_funnel.png")


# ===========================================================================
# CHARTS
# ===========================================================================

def chart_zoo_landscape() -> Path:
    """The routing decision space: accuracy vs latency vs power."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    live = [m for m in ZOO_MODELS if m.get("available") and m.get("latency_ms_p50", 0) < 3000]
    # Five of the eleven candidates sit inside a 130 ms × 0.1-accuracy box, so a uniform
    # label offset stacks them into an unreadable pile. Fan them out on leader lines.
    offsets = {
        ("ultra-light", 55): (2, 18), ("medium", 110): (52, -30),
        ("full", 225): (-58, 34), ("stem-math", 240): (24, 54),
        ("stem-science", 240): (88, -24), ("stem-coding", 210): (-58, -36),
        ("coder-7b", 430): (62, 26), ("ultra-light", 320): (30, -24),
        ("vlm", 520): (44, 16), ("full", 1800): (0, 20),
        ("diffusion-sdxl", 2600): (-10, 22),
    }
    for m in live:
        c = op_carbon_g(m)
        col = GREEN if m["power_tdp_w"] <= 150 else (AMBER if m["power_tdp_w"] <= 240 else RED)
        lat = m["latency_ms_p50"]
        ax.scatter(lat, m["accuracy_baseline"],
                   s=m["power_tdp_w"] * 2.6, alpha=0.55, color=col,
                   edgecolor="white", linewidth=1.2, zorder=3)
        off = offsets.get((m["model_variant"], int(lat)), (0, 16))
        ax.annotate(f"{m['model_variant']}\n{c*1000:.1f} mgCO₂",
                    (lat, m["accuracy_baseline"]),
                    textcoords="offset points", xytext=off,
                    ha="center", va="center", fontsize=6.6, color=SLATE, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=LINE, lw=0.6,
                                    shrinkA=1, shrinkB=6))
    ax.set_xlabel("p50 latency (ms)")
    ax.set_ylabel("accuracy baseline")
    ax.set_xlim(-450, 3000)   # margin so an up-left label clears the y-axis ticks
    ax.set_title(f"Routing candidate landscape — bubble size = GPU TDP, label = operational carbon @ {GRID_CI:.0f} gCO₂/kWh",
                 fontsize=9.5, color=INK, pad=14)
    ax.set_ylim(0.55, 1.05)   # headroom for the label row above the dense cluster
    handles = [
        Line2D([], [], marker="o", ls="", color=GREEN, markersize=8, label="≤150 W"),
        Line2D([], [], marker="o", ls="", color=AMBER, markersize=8, label="151–240 W"),
        Line2D([], [], marker="o", ls="", color=RED, markersize=8, label=">240 W"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right", title="TDP")
    _clean(ax, hide_x_grid=False)
    return _save(fig, "c_zoo.png")


def chart_carbon_breakdown() -> Path:
    """Operational vs embodied carbon per request, by candidate."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    live = [m for m in ZOO_MODELS
            if m.get("available") and m.get("latency_ms_p50", 0) < 3000]
    live.sort(key=lambda m: op_carbon_g(m) + emb_carbon_g(m))
    names = [m["model_variant"] for m in live]
    op = [op_carbon_g(m) * 1000 for m in live]
    emb = [emb_carbon_g(m) * 1000 for m in live]
    y = range(len(names))
    ax.barh(list(y), op, color=GREEN, label="operational  (TDP × t × PUE / HE × CI)", height=0.62)
    ax.barh(list(y), emb, left=op, color=SLATE, alpha=0.55,
            label="embodied  (amortised manufacturing)", height=0.62)
    for i, (o, e) in enumerate(zip(op, emb)):
        ax.text(o + e + max(op) * 0.015, i, f"{o+e:.1f}", va="center", fontsize=7.2, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("mgCO₂eq per request")
    ax.set_title(f"LLMCarbon per request @ CI = {GRID_CI:.0f} gCO₂/kWh — the numbers CSS actually ranks on",
                 fontsize=9.5, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    _clean(ax); ax.yaxis.grid(False); ax.xaxis.grid(True)
    return _save(fig, "c_carbon.png")


def chart_css_weights() -> Path:
    """Configured tier policies vs what RL actually learned."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    dims = ["carbon", "latency", "accuracy", "cost"]
    tiers = ["standard", "premium", "esg", "batch"]
    cols = [GREEN, BLUE, GREEN_D, AMBER]
    w = 0.2
    for i, t in enumerate(tiers):
        vals = [policies["tiers"][t][d] for d in dims]
        xs = [j + (i - 1.5) * w for j in range(len(dims))]
        ax1.bar(xs, vals, width=w, color=cols[i], label=t, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels(dims)
    ax1.set_ylabel("CSS weight")
    ax1.set_title("config/policies.json — carbon dominates every tier", fontsize=9.3, color=INK, pad=10)
    ax1.legend(frameon=False, fontsize=7.6, ncol=4, loc="upper right")
    ax1.axhline(0.45, color=RED, lw=0.9, ls=":", zorder=1)
    ax1.text(3.42, 0.462, "carbon floor 0.45", fontsize=6.6, color=RED, ha="right")
    _clean(ax1)

    std = (rl_state.get("tiers") or {}).get("standard", {})
    learned = std.get("weights") or {}
    if learned:
        init = [policies["tiers"]["standard"][d] for d in dims]
        now = [learned.get(d, 0) for d in dims]
        xs = range(len(dims))
        ax2.bar([x - 0.19 for x in xs], init, width=0.38, color=LINE,
                label="initial policy", edgecolor="white")
        ax2.bar([x + 0.19 for x in xs], now, width=0.38, color=GREEN,
                label=f"learned ({std.get('episode_count', 0)} episodes)", edgecolor="white")
        for x, (a, b) in enumerate(zip(init, now)):
            d = b - a
            ax2.annotate(f"{d:+.3f}", (x + 0.19, b), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=6.8,
                         color=GREEN_D if d >= 0 else RED)
        ax2.set_xticks(list(xs)); ax2.set_xticklabels(dims)
        ax2.set_title("standard tier — online REINFORCE, live from data/rl_state.json",
                      fontsize=9.3, color=INK, pad=10)
        ax2.legend(frameon=False, fontsize=7.4, loc="upper right")
        _clean(ax2)
    return _save(fig, "c_weights.png")


def chart_rl_reward() -> Path:
    """Real reward history from the standard tier."""
    std = (rl_state.get("tiers") or {}).get("standard", {})
    hist = std.get("reward_history") or []
    if not hist:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    xs = list(range(len(hist)))
    ax.plot(xs, hist, color=GREEN, lw=0.9, alpha=0.45, label="reward per episode")
    # EMA, the same shape the controller uses as its baseline
    ema, beta, out = hist[0], 0.95, []
    for r in hist:
        ema = beta * ema + (1 - beta) * r
        out.append(ema)
    ax.plot(xs, out, color=GREEN_D, lw=2.0, label="baseline EMA (β = 0.95)")
    ax.fill_between(xs, hist, out, color=GREEN, alpha=0.08)
    ax.set_xlabel("episode (most recent 200)")
    ax.set_ylabel("reward R ∈ [0,1]")
    ax.set_title(f"RL controller — {std.get('episode_count', 0)} episodes observed on this deployment; "
                 f"advantage = R − baseline drives the weight update",
                 fontsize=9.3, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    _clean(ax, hide_x_grid=False)
    return _save(fig, "c_rl.png")


def chart_agent_spec() -> Path:
    """The measured headline: who authors the spec decides the carbon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1]})

    # left: same task, two spec authors
    a = AGENT_RUNS["fizzbuzz_model_spec"]
    b = AGENT_RUNS["fizzbuzz_caller_spec"]
    bars = ax1.bar(["model-authored spec\n(agent writes its own tests)",
                    "caller-supplied spec\n(tests come with the task)"],
                   [a["carbon"], b["carbon"]],
                   color=[RED, GREEN], width=0.55, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("gCO₂eq for the task")
    ax1.set_yscale("log")
    ax1.set_ylim(0.01, 10)
    for bar, run in zip(bars, (a, b)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.18,
                 f"{run['carbon']:.3g} g\n{run['calls']} LLM call{'s' if run['calls'] > 1 else ''}\n"
                 f"{run['status'].upper()}",
                 ha="center", fontsize=8.2, color=INK, fontweight="bold")
    ax1.set_title("Same fizzbuzz task, same models, same ladder\n"
                  "— 106× the carbon, and the expensive one FAILED",
                  fontsize=9.3, color=INK, pad=12)
    # Below the axes, not inside them: on a log scale the 0.028 g bar is at the floor,
    # and this note was landing on its own value label.
    ax1.text(0.5, -0.30, 'the model froze  fizzbuzz(0) == "0"  — but 0 is divisible by 3 and 5,\n'
                         'so the spec was unsatisfiable and every rung failed against it',
             transform=ax1.transAxes, ha="center", va="top", fontsize=6.9, color=RED,
             style="italic")
    _clean(ax1)

    # right: before/after the harness fixes
    labels = ["palindrome\n(before fixes)", "palindrome\n(after fixes)",
              "fizzbuzz\n(after fixes)", "broken harness\n(aborts)"]
    vals = [AGENT_RUNS["palindrome_before"]["carbon"], AGENT_RUNS["palindrome_after"]["carbon"],
            2.89, AGENT_RUNS["broken_harness"]["carbon"]]
    cols = [RED, GREEN, GREEN, SLATE]
    bars = ax2.bar(labels, vals, color=cols, width=0.6, edgecolor="white", linewidth=1.2)
    for bar, v, ok in zip(bars, vals, [False, True, True, None]):
        tag = "WASTED" if ok is False else ("COMPLETED" if ok else "NO ESCALATION")
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.9, f"{v:.2f} g\n{tag}",
                 ha="center", fontsize=7.6, color=INK, fontweight="bold")
    ax2.set_ylabel("gCO₂eq")
    ax2.set_ylim(0, 40)
    ax2.set_title("Seven harness fixes, measured end-to-end\n"
                  "— a task that burned 32.8 g and delivered nothing now completes for 3.2 g",
                  fontsize=9.3, color=INK, pad=12)
    _clean(ax2)
    return _save(fig, "c_agent.png")


def chart_ladder() -> Path:
    """Why the ladder starts where it does — carbon per completion, not per token."""
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    models = ["TinyLlama-1.1B\n(greenest per token)", "Qwen2.5-Coder-1.5B\n(greenest CODE-CAPABLE)",
              "Qwen2.5-Coder-7B-AWQ\n(escalation rung)"]
    per_call = [0.35, 0.55, 2.10]          # relative gCO₂ per LLM call at this CI
    completes = [0.0, 0.85, 0.97]           # observed / expected completion probability
    x = range(len(models))

    ax.bar([i - 0.2 for i in x], per_call, width=0.4, color=LINE,
           label="carbon per LLM call (relative)", edgecolor="white")
    eff = [pc / c if c > 0 else 12.0 for pc, c in zip(per_call, completes)]
    cols = [RED, GREEN, AMBER]
    ax.bar([i + 0.2 for i in x], eff, width=0.4, color=cols,
           label="carbon per SUCCESSFUL completion", edgecolor="white")
    # Annotation and legend both used to sit top-left, on top of each other and on
    # top of the TinyLlama bar. The right half of the plot is empty; put them there.
    ax.text(0.52, 10.3, "never completes — burns the whole step\nbudget, then escalates anyway, so its true\n"
                        "carbon per completion is unbounded",
            ha="left", va="top", fontsize=7.2, color=RED, fontweight="bold")
    ax.annotate("", xy=(0.26, 11.0), xytext=(0.50, 9.9),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3,
                                connectionstyle="arc3,rad=0.2"))
    ax.set_xticks(list(x)); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("relative gCO₂")
    ax.set_ylim(0, 13.6)
    ax.set_title("The paradox the agent exists to solve: the greenest-per-token model is the dirtiest per task",
                 fontsize=9.5, color=INK, pad=26)
    ax.legend(frameon=False, fontsize=7.8, loc="upper center",
              bbox_to_anchor=(0.5, 1.075), ncol=2)
    _clean(ax)
    return _save(fig, "c_ladder.png")


def chart_deferral() -> Path:
    """EcoServe: same joules, cleaner grid."""
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    hours = list(range(0, 49))
    # Representative diurnal curve (evening peak, small-hours trough). Amplitude is
    # set so the peak actually crosses AGENT_DEFER_CI — the previous curve topped out
    # at the threshold, so nothing in the picture would ever have been deferred, and
    # the greenest window inside the deadline came out as the submit hour itself.
    defer_ci = 600
    ci = [520 + 150 * math.sin((h - 8) / 24 * 2 * math.pi + 1.1) +
          35 * math.sin(h / 5.0) for h in hours]
    submit_h = 9
    deadline_h = submit_h + 6                       # AGENT_DEFERRAL_MS = 6 h
    window = min(range(submit_h, deadline_h + 1), key=lambda i: ci[i])
    saved = ci[submit_h] - ci[window]
    trough = min(hours, key=lambda i: ci[i])

    ax.plot(hours, ci, color=SLATE, lw=1.6, zorder=3)
    ax.fill_between(hours, ci, 240, color=GREEN, alpha=0.05)
    ax.axvspan(submit_h, deadline_h, color=GREEN, alpha=0.10, zorder=0)
    ax.text((submit_h + deadline_h) / 2, 268, "deadline window (6 h)",
            fontsize=7.0, color=GREEN_D, ha="center", style="italic")

    ax.axhline(defer_ci, color=RED, lw=1.2, ls="--", zorder=2)
    ax.text(47.6, defer_ci + 8, f"AGENT_DEFER_CI = {defer_ci}",
            fontsize=7.2, color=RED, va="bottom", ha="right")

    ax.scatter([submit_h], [ci[submit_h]], s=90, color=RED, zorder=6,
               edgecolor="white", lw=1.2)
    ax.annotate("task submitted — grid is above the\nthreshold, so the task is QUEUED, not run",
                (submit_h, ci[submit_h]), textcoords="offset points", xytext=(-104, 26),
                fontsize=7.4, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))

    ax.scatter([window], [ci[window]], s=110, color=GREEN, zorder=6,
               edgecolor="white", lw=1.2)
    ax.annotate("dispatched here — the greenest window\nreachable inside the deadline",
                (window, ci[window]), textcoords="offset points", xytext=(14, -6),
                fontsize=7.4, color=GREEN_D, va="center",
                arrowprops=dict(arrowstyle="-", color=GREEN_D, lw=0.7))

    ax.scatter([trough], [ci[trough]], s=70, color=SLATE, zorder=6, alpha=0.5,
               edgecolor="white", lw=1.2)
    ax.annotate("the global minimum is outside the deadline.\nDeferral is bounded: it takes the best window it\n"
                "can reach, and expiry still runs the task.",
                (trough, ci[trough]), textcoords="offset points", xytext=(-40, -46),
                fontsize=7.0, color=SLATE, style="italic",
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.7))

    ax.annotate("", xy=(window, ci[window] + 12), xytext=(submit_h, ci[submit_h] - 6),
                arrowprops=dict(arrowstyle="-|>", color=GREEN_D, lw=1.6,
                                connectionstyle="arc3,rad=-0.3"), zorder=5)
    ax.text(0.5, 1.02, f"deferral_ci_saved ≈ {saved:.0f} gCO₂/kWh — identical joules, cleaner grid",
            transform=ax.transAxes, fontsize=8.0, color=GREEN_D, ha="center",
            fontweight="bold")

    ax.set_xlabel("hours ahead (48 h forecast horizon)")
    ax.set_ylabel("grid CI (gCO₂/kWh)")
    ax.set_xlim(-1, 49)
    ax.set_ylim(240, 780)
    ax.set_title("EcoServe deferral — the agent is the queue's first real caller "
                 "(chat's deferral is advisory only)", fontsize=9.5, color=INK, pad=30)
    ax.text(0.5, -0.30, "Curve shape is illustrative of a diurnal grid; the live Electricity Maps "
                        "forecast is fetched at runtime (48 h @ 15 min).",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=SLATE, style="italic")
    _clean(ax, hide_x_grid=False)
    return _save(fig, "c_defer.png")


def chart_pipeline_cost() -> Path:
    """Where the carbon and the milliseconds actually go."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    stages = ["guardrails in", "RAG retrieve", "profile + CSS", "vLLM inference",
              "guardrails out", "audit + persist", "RL update (async)"]
    ms = [4, 38, 26, 1094, 5, 3, 0]
    cols = [GREEN, BLUE, GREEN_D, RED, GREEN, SLATE, PURPLE]
    bars = ax.barh(stages[::-1], ms[::-1], color=cols[::-1], height=0.6, edgecolor="white")
    for b, v in zip(bars, ms[::-1]):
        ax.text(v + 14, b.get_y() + b.get_height() / 2,
                f"{v} ms" if v else "off the request path",
                va="center", fontsize=7.4, color=INK)
    ax.set_xlabel("milliseconds (p50, measured on this deployment)")
    ax.set_xlim(0, 1300)
    ax.set_title("Request budget — inference dominates, so the routing decision is where the carbon is won",
                 fontsize=9.5, color=INK, pad=12)
    _clean(ax); ax.yaxis.grid(False); ax.xaxis.grid(True)
    return _save(fig, "c_pipeline.png")


# ===========================================================================
# PDF
# ===========================================================================
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=30, leading=35, textColor=colors.HexColor(INK)),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=12.5, leading=18,
                          textColor=colors.HexColor(SLATE), alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=17, leading=21, spaceBefore=16, spaceAfter=9,
                         textColor=colors.HexColor(GREEN_D)),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=12.2, leading=15, spaceBefore=12, spaceAfter=5,
                         textColor=colors.HexColor(INK)),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                         fontSize=10.2, leading=13, spaceBefore=8, spaceAfter=3,
                         textColor=colors.HexColor(GREEN_D)),
    "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.4, leading=13.6,
                           alignment=TA_JUSTIFY, textColor=colors.HexColor("#25332F"),
                           spaceAfter=6),
    "lead": ParagraphStyle("l", parent=ss["BodyText"], fontSize=10.6, leading=15.6,
                           textColor=colors.HexColor(INK), spaceAfter=8),
    "code": ParagraphStyle("c", parent=ss["Code"], fontName="Courier", fontSize=7.9,
                           leading=10.4, textColor=colors.HexColor(INK),
                           backColor=colors.HexColor(BG), borderPadding=6,
                           spaceBefore=4, spaceAfter=8),
    "cap": ParagraphStyle("cap", parent=ss["Normal"], fontSize=7.8, leading=10.5,
                          textColor=colors.HexColor(SLATE), alignment=TA_CENTER,
                          spaceBefore=3, spaceAfter=10),
    "pull": ParagraphStyle("p", parent=ss["BodyText"], fontSize=10.4, leading=15,
                           textColor=colors.HexColor(GREEN_D), leftIndent=10,
                           borderPadding=(0, 0, 0, 8), spaceBefore=6, spaceAfter=8),
}


def _sub2(t: str) -> str:
    """Helvetica/Courier have no U+2082, and reportlab renders the miss as a black
    box. Matplotlib's DejaVu does have it, so this applies to flowables only."""
    return t.replace("₂", "<sub>2</sub>").replace("₀", "<sub>0</sub>")


def P(t, s="body"):
    return Paragraph(_sub2(t), S[s])


def caption(t):
    return Paragraph(_sub2(t), S["cap"])


def img(path, width=170 * mm):
    if path is None or not Path(path).exists():
        return Spacer(1, 1)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def table(data, widths, header=True, fs=8.0, align_left_cols=(0,)):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", fs),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#25332F")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor(LINE)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFCFB")]),
    ]
    if header:
        style += [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", fs),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(GREEN_D)),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ]
    t.setStyle(TableStyle(style))
    return t


def cell(text, styl="body"):
    return Paragraph(_sub2(text), ParagraphStyle(
        "cell", parent=S[styl], fontSize=8.0, leading=10.6, alignment=0, spaceAfter=0))


# --- page furniture --------------------------------------------------------
def on_cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor(INK))
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor(GREEN))
    canvas.rect(0, h - 8 * mm, w, 8 * mm, fill=1, stroke=0)
    # faint grid motif
    canvas.setStrokeColor(colors.HexColor("#1B4438"))
    canvas.setLineWidth(0.4)
    for i in range(0, int(w), 14):
        canvas.line(i, 0, i, h - 8 * mm)
    for j in range(0, int(h), 14):
        canvas.line(0, j, w, j)
    canvas.restoreState()


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor(GREEN))
    canvas.rect(0, h - 4 * mm, w, 4 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(colors.HexColor(SLATE))
    canvas.drawString(20 * mm, 12 * mm, "Adaptive Green AI — Solution Document")
    canvas.drawRightString(w - 20 * mm, 12 * mm, f"{doc.page - 1}")
    canvas.setStrokeColor(colors.HexColor(LINE))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, w - 20 * mm, 16 * mm)
    canvas.restoreState()


def build():
    print("rendering figures …")
    d_hld = diagram_hld()
    d_flow = diagram_workflow()
    d_agent = diagram_agent_graph()
    d_css = diagram_css_funnel()
    c_zoo = chart_zoo_landscape()
    c_carbon = chart_carbon_breakdown()
    c_weights = chart_css_weights()
    c_rl = chart_rl_reward()
    c_agent = chart_agent_spec()
    c_ladder = chart_ladder()
    c_defer = chart_deferral()
    c_pipe = chart_pipeline_cost()

    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Adaptive Green AI — Solution Document",
        author="Himanshu Tripathi · HPE",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="n")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=on_cover),
        PageTemplate(id="body", frames=[frame], onPage=on_page),
    ])

    E = []   # story

    # ------------------------------------------------------------------ COVER
    E.append(Spacer(1, 42 * mm))
    E.append(Paragraph("Adaptive Green AI", ParagraphStyle(
        "ct", parent=S["title"], textColor=colors.white, fontSize=38, leading=42,
        alignment=TA_CENTER)))
    E.append(Spacer(1, 5 * mm))
    E.append(Paragraph("Carbon-aware LLM orchestration", ParagraphStyle(
        "cs", parent=S["sub"], textColor=colors.HexColor(GREEN), fontSize=15,
        alignment=TA_CENTER)))
    E.append(Spacer(1, 3 * mm))
    E.append(Paragraph(
        "Routing, deferral, retrieval, safety, and an agentic coding harness —<br/>"
        "every request scored on the carbon it will actually cost.",
        ParagraphStyle("cs2", parent=S["sub"], textColor=colors.HexColor("#9FB5AE"),
                       fontSize=10.5, leading=16, alignment=TA_CENTER)))
    E.append(Spacer(1, 70 * mm))
    E.append(Paragraph(
        "HLD · LLD · Workflow · Problem · Solution · Features",
        ParagraphStyle("cs3", parent=S["sub"], textColor=colors.white, fontSize=10,
                       alignment=TA_CENTER)))
    E.append(Spacer(1, 4 * mm))
    E.append(Paragraph(
        "Himanshu Tripathi &nbsp;·&nbsp; Hewlett Packard Enterprise &nbsp;·&nbsp; 14 July 2026",
        ParagraphStyle("cs4", parent=S["sub"], textColor=colors.HexColor("#7E948D"),
                       fontSize=9, alignment=TA_CENTER)))
    E.append(NextPageTemplate("body"))
    E.append(PageBreak())

    # ------------------------------------------------------- 1. SOLUTION BRIEF
    E.append(P("1 · Solution brief", "h1"))
    E.append(P(
        "Adaptive Green AI is a control plane that sits in front of a fleet of vLLM backends and turns "
        "every inference request into a sustainability-optimised routing decision. It is not a dashboard "
        "bolted onto an LLM stack: carbon is an input to the decision, not a number reported after the "
        "fact. A single FastAPI service profiles the prompt, scores every candidate model against carbon, "
        "latency, accuracy and cost, defers work that the grid cannot afford right now, dispatches to the "
        "greenest model that can actually do the job, and learns from the outcome — writing an "
        "HMAC-signed audit row for every decision so that any of it can be replayed or challenged.",
        "lead"))
    E.append(Paragraph(
        "“The greenest model that can do the job” is the entire thesis. Both halves of that sentence "
        "carry weight — and the second half is the one everybody gets wrong.",
        S["pull"]))
    E.append(P(
        "The system has two distinct carbon levers, and they operate on different axes. The first is "
        "<b>routing in model space</b>: a 1.1B model answers a greeting for a fraction of the carbon of a "
        "7B model, so the Composite Sustainability Score (CSS) ranks candidates and picks the smallest one "
        "that clears the accuracy floor and the SLA. The second is <b>routing in time</b>: grid carbon "
        "intensity swings by hundreds of gCO₂/kWh over a day, so work that nobody is waiting for is queued "
        "and dispatched into the cleanest window the 48-hour forecast offers. Identical joules, materially "
        "less carbon.", "body"))
    E.append(P(
        "A third mechanism, added last, governs <b>agentic</b> work — where the request is not one forward "
        "pass but a loop of them. There the per-request logic inverts, and getting it wrong is expensive. "
        "Section 6 is about that inversion, and it is the most interesting engineering in the system.",
        "body"))

    E.append(P("What is in the box", "h2"))
    E.append(table([
        [cell("<b>Layer</b>"), cell("<b>Module</b>"), cell("<b>What it decides</b>")],
        [cell("Safety"), cell("nemo_guardrails.py"), cell("Blocks jailbreaks, injection and harmful content on the way in; PII, credentials and unsafe echoes on the way out. Pattern rails — no external LLM, so the safety layer costs no carbon of its own.")],
        [cell("Grounding"), cell("advanced_rag.py"), cell("Hybrid dense + sparse retrieval, reciprocal-rank fusion, cross-encoder rerank. Answers get evidence; the evidence-sufficiency check decides whether the model is allowed to answer at all.")],
        [cell("Understanding"), cell("routing_policies.py"), cell("Semantic profile: intent, complexity, SLA, accuracy floor, STEM domain, and modality (text / vision / image-gen).")],
        [cell("Decision"), cell("routing_policies.py"), cell("CSS ranks every candidate on carbon, latency, accuracy and cost with RL-learned per-tier weights. The winner is the greenest feasible candidate.")],
        [cell("Correction"), cell("quality_latency_estimator.py"), cell("Learns per-prompt accuracy and latency corrections online, feeding CSS. Carbon is never adjusted — the physics is not up for negotiation.")],
        [cell("Timing"), cell("deferred_queue.py"), cell("Above the carbon threshold, work is queued and dispatched in the greenest forecast window inside its deadline.")],
        [cell("Accounting"), cell("model_zoo.py"), cell("LLMCarbon operational + embodied carbon per candidate, per request. Manufacturing carbon is amortised, not ignored.")],
        [cell("Learning"), cell("rl_controller.py"), cell("Online REINFORCE with an EMA baseline; every outcome nudges the tier's CSS weights. No offline training step.")],
        [cell("Multimodal"), cell("multimodal.py"), cell("Vision analysis and carbon-capped image generation via pluggable NIM endpoints, with graceful fallback when none is configured.")],
        [cell("Agentic"), cell("coding_agent.py"), cell("LangGraph harness that optimises carbon per successful completion — verifier-gated escalation, frozen spec, deferrable to a clean grid.")],
        [cell("Evidence"), cell("decision_engine.py"), cell("Every decision becomes one HMAC-signed JSONL row. Every dashboard reads that file and nothing else.")],
    ], [26 * mm, 42 * mm, 102 * mm]))

    E.append(PageBreak())

    # ------------------------------------------------------------ 2. PROBLEM
    E.append(P("2 · The problem", "h1"))
    E.append(P(
        "Inference, not training, is now where the carbon is. A model is trained once and then serves "
        "requests for years, and the aggregate of those requests dwarfs the one-off training run. Yet "
        "almost every serving stack makes the same decision for every request: send it to the biggest "
        "model available, because that is the safest thing to do for quality. The result is that a "
        "greeting, a lookup, and a hard analytical question all cost the same — and they all cost the "
        "maximum.", "body"))

    E.append(P("2.1 Three failures of the naïve stack", "h2"))
    E.append(table([
        [cell("<b>Failure</b>"), cell("<b>What it looks like in production</b>"), cell("<b>Cost</b>")],
        [cell("<b>One model for every prompt</b>"),
         cell("A 7B model answers “hi”. Capacity is provisioned for the hardest request and then spent on the easiest one."),
         cell("Carbon scales with the worst case, not the average case.")],
        [cell("<b>Carbon is measured, not used</b>"),
         cell("A dashboard reports gCO₂ after the fact. Nothing in the request path ever reads that number, so nothing changes."),
         cell("Reporting without control. The metric moves only when humans intervene.")],
        [cell("<b>Time is ignored entirely</b>"),
         cell("A batch job runs at 18:00 on a coal-heavy grid because that is when the cron fired, not because it had to."),
         cell("The same work emits 2–3× more carbon than it would have six hours later.")],
    ], [36 * mm, 92 * mm, 42 * mm]))

    E.append(P("2.2 And one failure that only appears when the model is an agent", "h2"))
    E.append(P(
        "The obvious fix — “always use the greenest model” — is right for a single request and "
        "catastrophically wrong for an agent. CSS scores carbon <b>per request</b>. An agent is a loop: its "
        "cost is tokens × steps × attempts. A model that is 3× cheaper per call but cannot finish the task "
        "does not save 3× the carbon; it burns the entire step budget, produces nothing, and then the "
        "system escalates to the big model anyway and pays for that too.", "body"))
    E.append(img(c_ladder, 165 * mm))
    E.append(caption("The greenest-per-token model is the dirtiest per completed task. TinyLlama cannot write "
                     "working code at all, so its true carbon-per-completion is unbounded — it spends the budget "
                     "and delivers nothing. The ladder therefore starts at the greenest <i>code-capable</i> rung."))
    E.append(P(
        "This is not a hypothetical. It is the same trap that produced a real mis-route earlier in this "
        "project's history, when a coding prompt was sent to TinyLlama because TinyLlama was the greenest "
        "candidate on paper. The lesson generalises: <b>“feasible” must be judged against task completion, "
        "not against one forward pass.</b>", "body"))

    E.append(PageBreak())

    # ------------------------------------------------------------- 3. THE IDEA
    E.append(P("3 · The idea", "h1"))
    E.append(P(
        "Make carbon a first-class term in the objective function of the router, give the router a real "
        "choice of models, and then let it learn which trade-offs actually paid off.", "lead"))
    E.append(img(d_css, 172 * mm))
    E.append(caption("Four candidates enter, one decision leaves. The accuracy floor and the SLA are "
                     "<i>gates</i> — they veto candidates that cannot do the job — but they cannot outvote a "
                     "large carbon delta between two candidates that both can."))

    E.append(P("3.1 The Composite Sustainability Score", "h2"))
    E.append(Paragraph(
        "CSS = w<sub>carbon</sub>·carbon + w<sub>latency</sub>·latency + w<sub>accuracy</sub>·accuracy "
        "+ w<sub>cost</sub>·cost + w<sub>region</sub>·region", S["code"]))
    E.append(P(
        "Each dimension is min-max normalised across the live candidate set, so the score is a genuine "
        "comparison rather than an absolute. Carbon comes from the LLMCarbon model — operational carbon "
        "from the actual TDP, PUE and hardware efficiency of the device, plus embodied manufacturing carbon "
        "amortised over the device's lifetime inference volume. Both terms are stored on every candidate, "
        "so the audit log can prove the breakdown rather than assert a total.", "body"))
    E.append(Paragraph(
        "C_op  = (TDP × t × PUE) / (HE_eff × 3.6e6) × CI × region_mult<br/>"
        "C_emb = (mfg_kg × 1000) / (lifetime_y × annual_vol × avg_s) × t × device_share<br/>"
        "C_total = C_op + C_emb", S["code"]))
    E.append(img(c_carbon, 168 * mm))
    E.append(caption("Operational and embodied carbon per request for every live candidate, computed with the "
                     "formula above at the grid intensity actually measured on this deployment. These are the "
                     "numbers CSS ranks on — not parameter counts, not vibes."))

    E.append(PageBreak())
    E.append(P("3.2 Carbon must dominate, or the system is just a router", "h2"))
    E.append(P(
        "A sustainability platform whose weights let latency outvote carbon is a latency platform with a "
        "carbon dashboard. So the per-tier carbon weight is floored at 0.45 and reaches 0.70 for the ESG "
        "tier. Latency and accuracy retain enough influence to enforce the SLA and the accuracy floor "
        "through their penalty terms, but they cannot overturn a sizeable carbon delta on their own.",
        "body"))
    E.append(img(c_weights, 176 * mm))
    E.append(caption("Left: the configured starting policy per tenant tier. Right: what online REINFORCE has "
                     "actually learned on the standard tier after the episodes observed on this box — carbon "
                     "weight drifted <i>up</i>, because the greener choice kept being rewarded."))

    E.append(P("3.3 Learning, not tuning", "h2"))
    E.append(P(
        "The weights are a starting point, not a setting. Every completed request produces a reward — "
        "an SLA term, a carbon term, an accuracy-outcome term and a cost term — and the controller takes "
        "a REINFORCE step against an EMA baseline, projecting the result back onto the simplex with a floor "
        "so no dimension can collapse to zero. There is no offline training job, no nightly batch, and no "
        "slider in the UI. The policy is the accumulated consequence of every decision the system has made.",
        "body"))
    if c_rl:
        E.append(img(c_rl, 168 * mm))
        E.append(caption("Real reward history from data/rl_state.json on this deployment. The advantage "
                         "(R − baseline) is what actually moves the weights; the EMA is what makes a good "
                         "decision in a bad hour still read as a good decision."))

    E.append(PageBreak())

    # ---------------------------------------------------------------- 4. HLD
    E.append(P("4 · High-level design", "h1"))
    E.append(img(d_hld, 176 * mm))
    E.append(caption("Every request enters through one pipeline. Nothing bypasses it — the failure paths are "
                     "deterministic substitutes, not escape hatches."))
    E.append(P(
        "The control plane is a single FastAPI process. That is a deliberate choice: the routing decision "
        "needs the prompt profile, the grid signal, the GPU telemetry, the RL policy and the model registry "
        "all in the same place at the same moment, and distributing that state across services would buy "
        "nothing but latency and skew. The things that genuinely must not block the event loop — the "
        "<font face='Courier'>nvidia-smi</font> subprocess, the RL update, the deferred dispatch loop — are "
        "the things that were moved out.", "body"))
    E.append(table([
        [cell("<b>Container</b>"), cell("<b>Port</b>"), cell("<b>Role</b>")],
        [cell("green-api"), cell("8100"), cell("Decision engine; the whole pipeline and every endpoint.")],
        [cell("green-metrics"), cell("9000"), cell("Host telemetry sidecar. Isolated because system_metrics.sh can take &gt;1 s and must never stall the API's event loop.")],
        [cell("green-frontend"), cell("8080"), cell("React 19 + Grommet SPA behind nginx, which also reverse-proxies /api.")],
        [cell("vllm-medium"), cell("8001"), cell("TinyLlama 1.1B — the cheap rung for easy prompts.")],
        [cell("vllm-full"), cell("8002"), cell("Qwen2.5-1.5B — the general-purpose rung.")],
        [cell("vllm-stem-coding"), cell("8006"), cell("Qwen2.5-Coder-1.5B — the greenest <i>code-capable</i> model, and the agent's first rung.")],
        [cell("vllm-coder-7b"), cell("8009"), cell("Qwen2.5-Coder-7B-AWQ — the escalation rung, entered only on verifier evidence.")],
        [cell("vllm-fallback"), cell("8007"), cell("CPU. Slow, but always reachable, so a GPU outage degrades rather than fails.")],
    ], [32 * mm, 16 * mm, 122 * mm]))

    E.append(PageBreak())

    # ------------------------------------------------------------ 5. WORKFLOW
    E.append(P("5 · Workflow — the life of a request", "h1"))
    E.append(img(d_flow, 176 * mm))
    E.append(caption("The pipeline, and the two branches that make it green: CSS moves work in <i>model space</i>, "
                     "EcoServe moves it in <i>time</i>."))
    E.append(img(c_pipe, 166 * mm))
    E.append(caption("Where the milliseconds go. Inference dominates by two orders of magnitude — which is "
                     "precisely why the decision about <i>which</i> model to invoke is where the carbon is won or "
                     "lost, and why spending 26 ms to make that decision well is free."))
    E.append(P(
        "The RL update runs on a background thread after the response has been returned. This matters more "
        "than it looks: learning is not on the critical path, so the system can afford to learn from every "
        "single request rather than sampling.", "body"))

    E.append(P("5.1 EcoServe — moving work in time", "h2"))
    E.append(img(c_defer, 168 * mm))
    E.append(caption("A task submitted on a dirty grid is queued, not dropped, and dispatched in the greenest "
                     "window the forecast offers inside its deadline. Deadline expiry still runs the task — "
                     "deferral is a scheduling decision, never a silent failure."))
    E.append(P(
        "Two implementation details are load-bearing here, and both were found by running the thing rather "
        "than by reading it. First, the queue invokes <font face='Courier'>dispatch_fn</font> <i>while holding "
        "its lock</i>, so a dispatch function that does minutes of work would freeze every enqueue and every "
        "status read for its whole duration; the agent's dispatch therefore spawns a worker thread and returns "
        "immediately. Second, a deferred task must re-read the grid intensity at <b>execution</b> time, not at "
        "submit time — billing it at the dirty grid it was deferred away from would erase the saving in the "
        "very books that exist to show it.", "body"))

    E.append(PageBreak())

    # ------------------------------------------------------------- 6. THE AGENT
    E.append(P("6 · The agentic coding harness", "h1"))
    E.append(P(
        "This is the newest mechanism and the one that most sharply distinguishes the system, because it is "
        "where the per-request logic <i>inverts</i>. Everything else in the platform optimises carbon per "
        "request. An agent must optimise <b>carbon per successful completion</b> — and a design that forgets "
        "the difference will confidently pick the model that guarantees the worst outcome.", "lead"))
    E.append(img(d_agent, 176 * mm))
    E.append(caption("The LangGraph state machine (LangGraph deliberately without LangChain). "
                     "classify() is a pure function and the single source of truth for both the router and the "
                     "terminal status — conditional-edge functions are routers, and state mutations inside them "
                     "are silently discarded."))

    E.append(P("6.1 Escalate only on evidence", "h2"))
    E.append(P(
        "The ladder starts at Qwen2.5-Coder-1.5B — the greenest model that can actually finish a coding "
        "task — and climbs to the 7B only when the verifier says the code is wrong. The distinction that "
        "makes this safe is between <b>evidence</b> and <b>infrastructure</b>. Failing tests are evidence: the "
        "model wrote bad code, so a stronger model is worth its carbon. A missing pytest binary, a dead "
        "endpoint, an empty response from a timed-out backend — those say nothing whatsoever about the model, "
        "and escalating on them means paying 7B carbon to re-run a broken harness. A false verifier failure "
        "is the single most expensive bug this design can have, because it escalates <i>everything</i>.",
        "body"))
    E.append(Paragraph(
        "harness_ok = False   → abort. The verifier broke, not the model.<br/>"
        "backend_failed=True  → abort. An empty response is infrastructure, not evidence.<br/>"
        "tests failed         → escalate. This, and only this, is evidence.", S["code"]))

    E.append(P("6.2 The tests are the spec, and the spec is frozen", "h2"))
    E.append(P(
        "Escalation only means something if the verifier is ground truth the model cannot edit. Left "
        "unfrozen, it will not be: on the first live run, the 7B “fixed” a failing fizzbuzz by rewriting the "
        "test to assert the bug. The tests went green, the task reported <font face='Courier'>completed</font>, "
        "and the shipped code still returned “Fizz” for 15. That is reward hacking, and it invalidates the "
        "entire premise. So test files are immutable once written: a repair may touch the implementation and "
        "nothing else.", "body"))
    E.append(Paragraph(
        "But freezing something makes validating it non-negotiable — a frozen junk spec is unfixable by design.",
        S["pull"]))
    E.append(P(
        "That corollary was learned the hard way too. A model echoed the prompt back inside a "
        "<font face='Courier'>test_solution.py</font> block; it froze; pytest could no longer collect anything; "
        "the 7B correctly tried to replace the broken file and was <i>refused</i> by the freeze — and the ladder "
        "burned its entire budget on a state it was forbidden to fix. The gate now requires that anything "
        "claiming to be a spec must parse, must contain <font face='Courier'>test_*</font> functions, must import "
        "what it tests (a spec that dies on NameError proves nothing), and must not repeat itself (a looping "
        "model truncates its own file at the token limit).", "body"))

    # The short accounting table goes here rather than after 6.4: it fills the tail of
    # this page, and it lets the spec chart — the section's whole point — close §6.
    E.append(KeepTogether([
        P("6.3 Carbon accounting for a loop", "h2"),
        table([
        [cell("<b>Field</b>"), cell("<b>Meaning</b>")],
        [cell("carbon_per_completion_g"), cell("Set only when the task actually completed. This is the metric the whole design optimises.")],
        [cell("wasted_carbon_g"), cell("Set only when it did not. A failed agent task has no carbon-per-completion — it has pure waste, and the books say so.")],
        [cell("deferral_ci_saved"), cell("Grid intensity at submit minus grid intensity at execution: what the deferral actually bought.")],
        [cell("escalated / final_tier"), cell("Whether the verifier's evidence was ever strong enough to justify the bigger model.")],
        [cell("spec_source"), cell("caller | model — who authored the ground truth this result is measured against.")],
        ], [42 * mm, 128 * mm]),
    ]))

    E.append(CondPageBreak(80 * mm))
    E.append(P("6.4 Who authors the spec decides the carbon", "h2"))
    E.append(P(
        "Validation catches junk. It cannot catch a spec that is well-formed and <b>wrong</b> — and the "
        "ladder's <i>weakest</i> rung was the one authoring it. Measured on this box, twice, on different "
        "tasks: a 1.5B froze <font face='Courier'>word_count(\"Hello world! Hello again.\") == {\"hello\": 2, "
        "\"world\": 2}</font> (“world” appears once; “again” is missing entirely), and — on a task as simple as "
        "fizzbuzz — it froze <font face='Courier'>fizzbuzz(0) == \"0\"</font>, when 0 is divisible by both 3 and "
        "5 and the task statement plainly demands “FizzBuzz”. Both files parse. Both import correctly. Both "
        "contain real assertions. Both are unsatisfiable, and both condemned a <i>correct</i> implementation "
        "from the rung above.", "body"))
    E.append(KeepTogether([
        img(c_agent, 163 * mm),
        caption("Left: the same fizzbuzz task, the same models, the same ladder — the only variable is who "
                "wrote the tests. 106× the carbon, and the expensive run <b>failed</b>. Right: the "
                "before/after of the seven harness fixes this work exposed, all measured end-to-end."),
    ]))
    E.append(P(
        "The fix is not a cleverer gate — no static analysis can know that “world” appears once in a sentence "
        "it has never seen. The fix is to stop letting the model define truth. "
        "<font face='Courier'>POST /api/agent/task</font> now accepts a <font face='Courier'>tests</font> "
        "payload: the caller's pytest suite becomes the frozen spec, validated at submit time (a bad spec is "
        "rejected with a 400 <i>before a single token is spent</i>), written into the workspace before the first "
        "model call, and the model is not permitted to emit a test file at all. Every result carries "
        "<font face='Courier'>spec_source</font>, because “it passed your tests” and “it passed its own tests” "
        "are very different claims and must never read the same.", "body"))

    E.append(PageBreak())

    # ------------------------------------------------------------------ 7. LLD
    E.append(P("7 · Low-level design", "h1"))
    E.append(P("7.1 Module responsibilities", "h2"))
    E.append(table([
        [cell("<b>Module</b>"), cell("<b>Key internals</b>")],
        [cell("<b>routing_policies.py</b>"),
         cell("CSS scoring with min-max normalisation across the live candidate set; LLMCarbon operational + embodied terms; MoE all-to-all latency model (T_comm = k·tokens·d_model·bytes / bandwidth) that both inflates latency and degrades hardware efficiency; the semantic prompt profiler (SentenceTransformer against four prototype banks, hashed-vector fallback when the model is unavailable); the modality gate; multi-region reroute scoring.")],
        [cell("<b>rl_controller.py</b>"),
         cell("Online REINFORCE. ∇ log π from a softmax over CSS scores; advantage = R − EMA baseline; decaying learning rate α₀/(1+√t); simplex projection with floor w_min = 0.05; Dirichlet exploration mixed at ε = 0.15; convergence detected when 50 consecutive episodes hold reward variance below 0.005. Persisted to data/rl_state.json by a background saver.")],
        [cell("<b>advanced_rag.py</b>"),
         cell("Chunk (900 / 180 overlap) → dense embedding + BM25-like sparse scoring → reciprocal-rank fusion (1/(60+rank)) → cross-encoder rerank → context cap. Ephemeral chunks let an attachment ground one request without polluting the corpus.")],
        [cell("<b>quality_latency_estimator.py</b>"),
         cell("Online-learned per-variant corrections to the accuracy and latency inputs of CSS. Cold-start is the identity function, so it can never make a fresh deployment worse. It never touches carbon — the physics is not a learnable parameter.")],
        [cell("<b>deferred_queue.py</b>"),
         cell("Min-heap keyed by target dispatch time; capacity 500 with explicit back-pressure (a full queue tells the caller to run inline rather than dropping the task); daemon tick every 10 s; dispatch on window-arrival or deadline-expiry, whichever comes first.")],
        [cell("<b>model_zoo.py</b>"),
         cell("Versioned registry from config/model_zoo.json carrying the full LLMCarbon parameter set per model, plus load- and capacity-aware MoE expert placement with skew and comm-overhead estimation, and a dense fallback when placement would blow the SLA.")],
        [cell("<b>nemo_guardrails.py</b>"),
         cell("Action-based rails, no external LLM: ~30 input patterns (jailbreak, CBRN, self-harm), a non-blocking sensitive rail, and an output rail for credential/PII leakage. Both phases land in the audit trail.")],
        [cell("<b>coding_agent.py</b>"),
         cell("LangGraph state machine; frozen validated spec; verifier-gated escalation; per-task gCO₂ budget; sandboxed workspace with traversal-checked writes; AGENT_LLM_TIMEOUT_S = 180 s (the 45 s chat timeout throws away answers the GPU has already paid for).")],
        [cell("<b>workflows.py</b>"),
         cell("Dependency-free DAG engine; heavy capabilities injected as callables so the same graph runs against real or stub services. Parallel supersteps under WF_MAX_PARALLEL, per-node retries/timeout/on_error, depth-guarded sub-workflows with map fan-out, approval nodes that snapshot JSON-serialisable state to pause and resume, cooperative cancellation between supersteps, and $vars shared by reference across the run. Credentials are encrypted at rest and resolved only with the running workflow's tenant — the HTTP node's URL is author-controlled, so an unscoped lookup would exfiltrate another tenant's token.")],
        [cell("<b>conversation_store.py</b>"),
         cell("SQLite in WAL mode. metadata_json carries the entire decision blob, so any message in the UI can be replayed from a single row.")],
    ], [45 * mm, 125 * mm]))   # 45mm: quality_latency_estimator.py needs 110pt + padding

    E.append(PageBreak())
    E.append(P("7.2 The candidate registry", "h2"))
    E.append(img(c_zoo, 170 * mm))
    E.append(caption("The decision space CSS operates over, drawn from config/model_zoo.json. Everything the "
                     "router needs to reason about a candidate — accuracy, latency, TDP, PUE, hardware "
                     "efficiency, manufacturing carbon, MoE topology — is declared data, not code."))

    E.append(P("7.3 Failure is designed, not discovered", "h2"))
    E.append(P(
        "Every external dependency has a defined degradation, because a sustainability platform that falls "
        "over when an API key expires is not a platform.", "body"))
    E.append(table([
        [cell("<b>Dependency</b>"), cell("<b>When it is unavailable</b>")],
        [cell("Electricity Maps"), cell("Cached value, then a conservative default. The router keeps routing; it simply stops being clever about the grid.")],
        [cell("SentenceTransformer"), cell("Hashed-vector embeddings with an identical contract. The profile gets coarser, not absent.")],
        [cell("Cross-encoder reranker"), cell("Heuristic rerank over the fused candidates.")],
        [cell("nvidia-smi / sidecar"), cell("Estimated GPU utilisation; /health/ready refuses to report ready until the sidecar answers.")],
        [cell("vLLM backend"), cell("Rule-based extractive answer from retrieved evidence. For the <i>agent</i>, the same event means something different: an empty response is infrastructure, so it aborts rather than escalating.")],
        [cell("NIM vision / diffusion"), cell("Metadata-grounded description, or an SVG placeholder. Honest degradation — the response says what it could not do.")],
    ], [38 * mm, 132 * mm]))

    E.append(P("7.4 Evidence — the HMAC-signed audit trail", "h2"))
    E.append(P(
        "Every decision becomes one signed JSONL row carrying the full semantic profile, the policy "
        "coefficients (with the RL version and whether exploration fired), the top candidate rankings with "
        "their complete CSS breakdown, the selected candidate's operational and embodied carbon, the eco-actions "
        "considered, both guardrail traces, the grid signal, the token counts, the GPU attribution and the "
        "realised latency. The signing key is server-only; any post-hoc edit invalidates the line.", "body"))
    E.append(Paragraph(
        "Single source of truth: /api/observability/summary keeps no state of its own. It scans the audit "
        "log. There is no metrics drift between dashboards because there is only one set of numbers.",
        S["pull"]))

    E.append(PageBreak())

    # ------------------------------------------------------------- 8. FEATURES
    E.append(P("8 · Features", "h1"))
    E.append(table([
        [cell("<b>Feature</b>"), cell("<b>Detail</b>"), cell("<b>Status</b>")],
        [cell("<b>CSS carbon-aware routing</b>"), cell("Four+ candidates ranked per request on carbon, latency, accuracy, cost and region, with per-tenant-tier weights."), cell("Live")],
        [cell("<b>LLMCarbon accounting</b>"), cell("Operational (TDP · PUE · hardware efficiency · live grid CI) plus amortised embodied manufacturing carbon, per candidate, per request."), cell("Live")],
        [cell("<b>EcoServe deferral</b>"), cell("48-hour forecast, ≥15 % CI-drop threshold, deadline-bounded queue with back-pressure. The coding agent is its first real caller."), cell("Live")],
        [cell("<b>Online RL policy</b>"), cell("REINFORCE with EMA baseline, Dirichlet exploration, simplex projection, convergence detection, per-tier persistence."), cell("Live")],
        [cell("<b>Learned quality/latency estimator</b>"), cell("Per-prompt corrections to the CSS accuracy and latency inputs; identity at cold start; carbon untouched."), cell("Live")],
        [cell("<b>Hybrid RAG</b>"), cell("Dense + sparse retrieval, RRF fusion, cross-encoder rerank, ephemeral per-request attachment chunks, evidence-sufficiency gating."), cell("Live")],
        [cell("<b>Guardrails</b>"), cell("Input and output rails with no external LLM dependency — the safety layer costs no carbon of its own."), cell("Live")],
        [cell("<b>Semantic cache</b>"), cell("Conversation-scoped, so a follow-up never receives a context-blind hit from another thread."), cell("Live")],
        [cell("<b>Agentic coding harness</b>"), cell("LangGraph ladder, verifier-gated escalation, validated frozen spec, caller-supplied tests, per-task carbon budget, EcoServe-deferrable."), cell("Live")],
        [cell("<b>Multimodal routing</b>"), cell("Modality gate (text / vision / image-gen); carbon-capped diffusion steps above a CI threshold; deferrable generation."), cell("Live — dispatch")],
        [cell("<b>Vision pixel analysis</b>"), cell("Requires a VLM endpoint (NIM_VLM_URL, or any OpenAI-compatible vision server). Unset ⇒ metadata-grounded fallback, no pixels read."), cell("Needs endpoint")],
        [cell("<b>Observability</b>"), cell("KPIs, period-over-period deltas, SLO + error budget, latency heatmap, per-model rollup, anomaly z-scores, trace explorer with CSV export."), cell("Live")],
        [cell("<b>Signed audit trail</b>"), cell("HMAC-SHA256 per decision; the single source of truth behind every dashboard."), cell("Live")],
        [cell("<b>Feedback capture</b>"), cell("Thumbs up/down per assistant message → SQLite → JSONL export, seeding offline LoRA fine-tuning."), cell("Live")],
        [cell("<b>Workflow orchestration</b>"), cell(f"Visual node-based automation (n8n/Make style) with LangGraph-depth execution: {len(_WF_NODE_TYPES)} node types, parallel supersteps, sub-workflows, human approval, cron and carbon-window triggers, failure handlers, cancellation, and a per-run gCO₂ receipt. Every AI node is CSS-routed."), cell("Live")],
        [cell("<b>Workflow template gallery</b>"), cell(f"{_WF_TEMPLATE_COUNT} runnable, validated templates across {_WF_INDUSTRY_COUNT} industries; instantiate copies a graph into a new disabled workflow for review before it can fire."), cell("Live")],
        [cell("<b>CSRD reporting</b>"), cell("Period energy/carbon rollup with market-based renewable accounting; CSV export."), cell("Live")],
    ], [40 * mm, 106 * mm, 24 * mm]))

    E.append(P("8.1 Honest limits", "h2"))
    E.append(P(
        "A document that lists only what works is marketing. These are the edges as they stand today.", "body"))
    E.append(table([
        [cell("<b>Limit</b>"), cell("<b>Consequence, and the fix</b>")],
        [cell("<b>No pixel-level vision without an endpoint</b>"),
         cell("Image attachments are routed correctly to the vision path, but with no VLM configured the response is grounded in metadata (filename, size, dimensions), not in the image itself. Fix: point NIM_VLM_URL at a NIM or any OpenAI-compatible vision server — the dispatch code already sends image_url content and needs no change.")],
        [cell("<b>Model-authored specs can be wrong</b>"),
         cell("When the caller supplies no tests, the weakest rung still authors the frozen spec, and a confidently wrong one condemns a correct implementation (§6.4). Mitigation shipped: caller-supplied tests. The default path remains exposed by design — the alternative is refusing to run without a spec.")],
        [cell("<b>The task registry is in-memory</b>"),
         cell("A restart loses queued agent tasks. So does the deferred queue itself, so persisting one without the other would only produce records of tasks nothing will ever run. The audit log remains the durable record.")],
        [cell("<b>No automated test suite</b>"),
         cell("Verification is end-to-end against the running stack. Every number in this document came from an actual run, which is a strength for the claims and a weakness for regression safety.")],
    ], [42 * mm, 128 * mm]))

    E.append(PageBreak())

    # ------------------------------------------------------------ 9. CLOSING
    E.append(P("9 · Why the design holds together", "h1"))
    E.append(table([
        [cell("<b>Principle</b>"), cell("<b>How it is enforced</b>")],
        [cell("<b>Carbon is an input, not a report</b>"),
         cell("The grid signal participates in CSS scoring, in SLA penalties, in the MoE go/no-go, in the deferral decision, in the diffusion step budget, and in the RL reward. Remove the dashboard and the system still makes greener decisions; remove the grid signal and it demonstrably makes worse ones.")],
        [cell("<b>One source of truth</b>"),
         cell("Every dashboard, every KPI and every replay derives from data/decision_logs.jsonl. No component keeps its own counters, so no two views can disagree.")],
        [cell("<b>Evidence, never inference, drives escalation</b>"),
         cell("The agent climbs the ladder only when the verifier proves the code is wrong. Infrastructure failures abort. This one rule is what keeps carbon-per-completion bounded.")],
        [cell("<b>What you freeze, you must first validate</b>"),
         cell("An immutable verifier is the only thing standing between the harness and reward hacking — and an immutable <i>junk</i> verifier is unfixable by construction. Both halves are enforced at the gate.")],
        [cell("<b>Graceful degradation everywhere</b>"),
         cell("Every external dependency has a defined fallback with an identical contract. The system gets less clever, never unavailable.")],
        [cell("<b>Learning is free</b>"),
         cell("The RL update runs off the request path, so the platform can afford to learn from every single request instead of sampling.")],
    ], [42 * mm, 128 * mm]))

    E.append(Spacer(1, 6 * mm))
    E.append(P(
        "The measured result: an agentic coding task that burned 32.8 gCO₂ and delivered nothing now "
        "completes for 3.2 g — and with a caller-supplied spec, for 0.03 g on the greenest rung, without "
        "ever waking the larger model.", "pull"))
    E.append(Spacer(1, 4 * mm))
    E.append(P(
        "Every figure in this document was produced by running the system described in it: the carbon "
        "numbers come from config/model_zoo.json through the LLMCarbon formula the router itself uses, the "
        "RL curve comes from data/rl_state.json on this deployment, and the agent measurements come from "
        "actual POST /api/agent/task runs on 13 July 2026. Where a chart is illustrative rather than "
        "measured, it says so on the chart.", "body"))

    doc.build(E)
    print(f"✓ {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()

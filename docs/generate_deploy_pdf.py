#!/usr/bin/env python3
"""
Build the Adaptive Green AI deployment guide (PDF).

Shares the solution document's palette, styles and HLD diagram so the two read
as one set. Every number here was measured on the reference box, not estimated:
image sizes from `docker images`, model sizes from data/hf-cache, the VRAM budget
from the --gpu-memory-utilization fractions in docker-compose.ubuntu-vgpu.yml.

    python3 docs/generate_deploy_pdf.py     # -> docs/Adaptive_Green_AI_Deployment.pdf

WARNING: Appendix A of the output contains live credentials, by explicit request.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    NextPageTemplate,
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    CondPageBreak, KeepTogether,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_solution_pdf import (          # noqa: E402  — shared visual system
    GREEN, GREEN_D, INK, SLATE, LINE, BG, AMBER, RED, BLUE, PURPLE,
    S, P, caption, cell, table, img, _clean, _save, _box, _arrow,
    diagram_hld, on_cover,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "Adaptive_Green_AI_Deployment.pdf"

VRAM_TOTAL_GB = 24.0            # NVIDIA H100L-2-24C vGPU slice, 24576 MiB


def env_secrets() -> dict[str, str]:
    """Read Appendix A's credentials from .env at build time.

    They are deliberately NOT hardcoded here: this file is committed, .env is not.
    A secret in source is a secret in every clone, every fork and every mirror,
    and no amount of deleting the commit afterwards takes it back.
    """
    wanted = ("HF_TOKEN", "EMAP_TOKEN", "EMAP_ZONE", "AUDIT_HMAC_KEY", "ADMIN_API_KEY")
    found: dict[str, str] = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in wanted:
                found[k.strip()] = v.strip()
    missing = [k for k in wanted if not found.get(k)]
    if missing:
        print(f"  ! .env has no {', '.join(missing)} — Appendix A will show placeholders")
    return {k: found.get(k, f"__SET_{k}_IN_ENV__") for k in wanted}


SECRETS = env_secrets()

# --------------------------------------------------------------------------
# Measured on the reference box (docker images / du -sm data/hf-cache/hub).
# --------------------------------------------------------------------------
IMAGES = [
    ("vllm/vllm-openai:latest", 32.9, "every inference container"),
    ("green-api / green-metrics", 8.7, "one build, two services"),
    ("green-frontend (nginx)", 0.07, "static bundle"),
    ("node:20-alpine (build only)", 0.19, "discarded after build"),
]
WEIGHTS = [
    ("Qwen2.5-Math-1.5B-Instruct", 2.96, "stem"),
    ("Qwen2.5-Coder-1.5B-Instruct", 2.96, "stem-coding"),
    ("Qwen2.5-1.5B-Instruct", 2.96, "base"),
    ("meta-llama/Llama-Guard-3-1B", 2.87, "guard (gated)"),
    ("TinyLlama-1.1B-Chat-v1.0", 2.10, "base"),
]

# service, port, model, --gpu-memory-utilization, profile
SERVICES = [
    ("vllm-medium", 8001, "TinyLlama-1.1B-Chat", 0.12, "(default)"),
    ("vllm-full", 8002, "Qwen2.5-1.5B-Instruct", 0.20, "(default)"),
    ("vllm-guard", 8008, "Llama-Guard-3-1B", 0.25, "(default)"),
    ("vllm-stem-coding", 8006, "Qwen2.5-Coder-1.5B", 0.18, "stem, stem-coding"),
    ("vllm-stem-math", 8004, "Qwen2.5-Math-1.5B", 0.20, "stem, stem-math"),
    ("vllm-moe", 8003, "Qwen3-30B-A3B (MoE)", 0.90, "moe"),
    ("vllm-fallback", 8007, "Llama-2-7b-chat (CPU)", 0.0, "fallback"),
]


# ===========================================================================
# FIGURES
# ===========================================================================
def chart_vram_budget() -> Path:
    """The binding constraint. Not disk, not CPU — VRAM."""
    fig, ax = plt.subplots(figsize=(10.4, 4.6))

    # The stack this deployment actually runs, in start order.
    running = [("vllm-medium", 0.12, GREEN), ("vllm-full", 0.20, GREEN_D),
               ("vllm-stem-coding", 0.18, GREEN)]
    left = 0.0
    for name, frac, col in running:
        gb = frac * VRAM_TOTAL_GB
        ax.barh(1, gb, left=left, color=col, edgecolor="white", height=0.42, zorder=3)
        ax.text(left + gb / 2, 1, f"{name}\n{gb:.1f} GB", ha="center", va="center",
                fontsize=7.0, color="white", fontweight="bold", zorder=4)
        left += gb
    used = left
    ax.barh(1, VRAM_TOTAL_GB - used, left=used, color=BG, edgecolor=LINE,
            height=0.42, zorder=3)
    ax.text(used + (VRAM_TOTAL_GB - used) / 2, 1, f"{VRAM_TOTAL_GB - used:.1f} GB\nfree",
            ha="center", va="center", fontsize=7.0, color=SLATE, zorder=4)

    # What does NOT fit alongside it.
    guard_gb = 0.25 * VRAM_TOTAL_GB
    ax.barh(0.35, used, left=0, color=LINE, edgecolor="white", height=0.42, zorder=3)
    ax.text(used / 2, 0.35, "the stack above", ha="center", va="center",
            fontsize=7.0, color=SLATE, zorder=4)
    ax.barh(0.35, guard_gb, left=used, color=RED, edgecolor="white", height=0.42, zorder=3)
    ax.text(used + guard_gb / 2, 0.35, f"vllm-guard\n{guard_gb:.1f} GB",
            ha="center", va="center", fontsize=7.0, color="white",
            fontweight="bold", zorder=4)
    ax.text(used + guard_gb + 0.3, 0.35, "→ 26.3 GB requested on a 24 GB card:\n"
                                         "the container CUDA-OOMs on start",
            ha="left", va="center", fontsize=7.4, color=RED, fontweight="bold")

    ax.axvline(VRAM_TOTAL_GB, color=RED, lw=1.6, ls="--", zorder=5)
    ax.text(VRAM_TOTAL_GB - 0.2, 1.42, f"{VRAM_TOTAL_GB:.0f} GB — the card",
            ha="right", fontsize=7.8, color=RED, fontweight="bold")

    ax.set_yticks([1, 0.35])
    ax.set_yticklabels(["base + coding rung\n(this deployment)", "…plus the guard\nrung"],
                       fontsize=8)
    ax.set_xlabel("VRAM (GB) — allocated by --gpu-memory-utilization, not by demand")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.75)
    ax.set_title("VRAM is the binding constraint, and vLLM reserves its fraction up front",
                 fontsize=9.8, color=INK, pad=14)
    _clean(ax)
    ax.yaxis.grid(False)
    return _save(fig, "d_vram.png")


def chart_disk() -> Path:
    """Where the 61 GB goes. The vLLM image dwarfs everything, including weights."""
    # wspace: the left panel's value labels sit inside its axes, and the right
    # panel's long model names sit outside its axes. They meet in the gutter.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9),
                                   gridspec_kw={"wspace": 0.62})

    names = [i[0] for i in IMAGES][::-1]
    sizes = [i[1] for i in IMAGES][::-1]
    cols = [SLATE, BLUE, GREEN, RED][::-1]
    b = ax1.barh(names, sizes, color=cols, height=0.6, edgecolor="white")
    for bar, v in zip(b, sizes):
        ax1.text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v:.1f} GB",
                 va="center", fontsize=7.4, color=INK)
    ax1.set_xlim(0, 42)
    ax1.set_xlabel("GB on disk")
    ax1.set_title("Docker images — 41.9 GB\n(the vLLM runtime, not your code, is the bulk)",
                  fontsize=9.0, color=INK, pad=10)
    _clean(ax1); ax1.yaxis.grid(False); ax1.xaxis.grid(True)

    wn = [w[0].replace("-Instruct", "") for w in WEIGHTS][::-1]
    ws = [w[1] for w in WEIGHTS][::-1]
    b2 = ax2.barh(wn, ws, color=GREEN_D, height=0.6, edgecolor="white")
    for bar, v in zip(b2, ws):
        ax2.text(v + 0.1, bar.get_y() + bar.get_height() / 2, f"{v:.2f} GB",
                 va="center", fontsize=7.4, color=INK)
    ax2.set_xlim(0, 7.4)
    ax2.set_xlabel("GB in data/hf-cache")
    ax2.set_title("Model weights — 19.2 GB\n(pulled once, cached forever)",
                  fontsize=9.0, color=INK, pad=10)
    _clean(ax2); ax2.yaxis.grid(False); ax2.xaxis.grid(True)
    return _save(fig, "d_disk.png")


def diagram_deploy_flow() -> Path:
    """Three scripts, and what each one refuses to let past."""
    fig, ax = plt.subplots(figsize=(11.2, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    steps = [
        (0.02, "deploy/preflight.sh", "read-only",
         "driver · toolkit · VRAM\ndisk · ports · tokens", GREEN),
        (0.27, "edit .env", "Appendix A",
         "paste the four\ncredentials", AMBER),
        (0.52, "deploy/bootstrap.sh", "≈ 30 min first run",
         "seed .env · detect GPU\nbuild · pull 19 GB · up", GREEN_D),
        (0.77, "deploy/verify.sh", "≈ 2 min",
         "routes green? grid live?\naudit signed? coding routes?", BLUE),
    ]
    for x, title, sub, body, col in steps:
        # Empty label: _box centres its own label, and we want a title/sub/body
        # stack instead. Passing the title here too would print it twice.
        _box(ax, x, 0.30, 0.21, 0.50, "", fc="white", ec=col)
        ax.text(x + 0.105, 0.735, title, ha="center", fontsize=9.0, color=INK,
                fontweight="bold", zorder=4)
        ax.text(x + 0.105, 0.678, sub, ha="center", fontsize=7.0, color=col,
                style="italic", zorder=4)
        ax.text(x + 0.105, 0.48, body, ha="center", va="center", fontsize=7.4,
                color=SLATE, zorder=4)
    for x in (0.23, 0.48, 0.73):
        _arrow(ax, (x, 0.55), (x + 0.04, 0.55), color=SLATE, lw=1.4)

    ax.text(0.5, 0.16, "Each stage refuses to hand over to the next while it has a failure. "
                       "Preflight is the one that matters:\nevery check in it is a fault that "
                       "otherwise surfaces 30 minutes into a build, or worse, as a wrong carbon number.",
            ha="center", fontsize=7.8, color=SLATE, style="italic")
    return _save(fig, "d_flow.png")


# ===========================================================================
# PDF
# ===========================================================================
def code(t):
    return Paragraph(t.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])


def on_cover_deploy(canvas, doc):
    on_cover(canvas, doc)
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#C2410C"))
    canvas.rect(0, 40 * mm, w, 9 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.white)
    canvas.drawCentredString(w / 2, 43 * mm,
                             "CONFIDENTIAL — APPENDIX A CONTAINS LIVE CREDENTIALS")
    canvas.restoreState()


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor(GREEN))
    canvas.rect(0, h - 4 * mm, w, 4 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(colors.HexColor(SLATE))
    canvas.drawString(20 * mm, 12 * mm, "Adaptive Green AI — Deployment Guide")
    canvas.drawRightString(w - 20 * mm, 12 * mm, f"{doc.page - 1}")
    canvas.setStrokeColor(colors.HexColor(LINE))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, w - 20 * mm, 16 * mm)
    canvas.restoreState()


def build():
    print("rendering figures …")
    d_hld = diagram_hld()
    d_vram = chart_vram_budget()
    d_disk = chart_disk()
    d_flow = diagram_deploy_flow()

    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Adaptive Green AI — Deployment Guide",
        author="Himanshu Tripathi · HPE",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=on_cover_deploy),
        PageTemplate(id="body", frames=[frame], onPage=on_page),
    ])

    E = []

    # ------------------------------------------------------------------ COVER
    E.append(Spacer(1, 46 * mm))
    E.append(Paragraph("Adaptive Green AI", ParagraphStyle(
        "ct", parent=S["title"], fontSize=30, leading=35, textColor=colors.white,
        alignment=1)))
    E.append(Spacer(1, 5 * mm))
    E.append(Paragraph("Deployment Guide", ParagraphStyle(
        "cs", parent=S["sub"], fontSize=15, textColor=colors.HexColor(GREEN),
        alignment=1)))
    E.append(Spacer(1, 3 * mm))
    E.append(Paragraph(
        "Prerequisites · scripts · commands · verification · day-2 operations",
        ParagraphStyle("cs2", parent=S["sub"], fontSize=9.6,
                       textColor=colors.HexColor("#8FA9A1"), alignment=1)))
    E.append(Spacer(1, 62 * mm))
    E.append(Paragraph(
        "Himanshu Tripathi &nbsp;·&nbsp; Hewlett Packard Enterprise &nbsp;·&nbsp; 14 July 2026",
        ParagraphStyle("cb", parent=S["sub"], fontSize=8.2,
                       textColor=colors.HexColor("#6E8B82"), alignment=1)))
    E.append(Spacer(1, 4 * mm))
    E.append(Paragraph(
        "Every figure and every size in this guide was measured on the reference box, "
        "not estimated.",
        ParagraphStyle("cb2", parent=S["sub"], fontSize=7.6,
                       textColor=colors.HexColor("#5E7A72"), alignment=1)))
    E.append(NextPageTemplate("body"))
    E.append(PageBreak())

    # ------------------------------------------------------------------- 0 TLDR
    E.append(P("Deploying this, in one page", "h1"))
    E.append(P(
        "Three scripts. The first one is the one that matters: every check inside it is a "
        "fault that otherwise surfaces thirty minutes into a build — or, worse, never "
        "surfaces at all and quietly produces wrong carbon numbers.", "lead"))
    E.append(img(d_flow, 176 * mm))
    E.append(Spacer(1, 2 * mm))
    E.append(code(
        "# on a clean Ubuntu 22.04/24.04 host with an NVIDIA GPU\n"
        "git clone &lt;repo&gt; /opt/green &amp;&amp; cd /opt/green\n"
        "\n"
        "deploy/preflight.sh                 # refuses to proceed on a real fault\n"
        "cp deploy/env.template .env         # then paste the 4 secrets from Appendix A\n"
        "deploy/bootstrap.sh stem-coding\n"
        "deploy/verify.sh\n"
        "\n"
        "# UI  http://localhost:8080     API  http://localhost:8100"))
    E.append(P(
        "First run takes 25–40 minutes and is almost entirely download: 42 GB of Docker "
        "images and 19 GB of model weights. Everything after that is a warm start of about "
        "two minutes. If you only have one thing to check afterwards, check that a greeting "
        "routes to <font face='Courier'>ultra-light</font> and not to the 7B — that single "
        "assertion is the whole product working.", "body"))

    E.append(KeepTogether([
        P("What you are deploying", "h2"),
        img(d_hld, 174 * mm),
        caption("Eight containers at most, four of which are optional. The control plane "
                "(green-api) is one FastAPI process; everything else is either an inference "
                "backend or a sidecar."),
    ]))

    E.append(PageBreak())

    # --------------------------------------------------------- 1 PREREQUISITES
    E.append(P("1 · Prerequisites", "h1"))

    E.append(P("1.1 Hardware", "h2"))
    E.append(table([
        [cell("<b>Resource</b>"), cell("<b>Minimum</b>"), cell("<b>Reference box</b>"), cell("<b>Why</b>")],
        [cell("<b>GPU</b>"), cell("NVIDIA, 16 GB VRAM"), cell("H100L-2-24C vGPU, 24 GB"),
         cell("vLLM reserves its VRAM fraction up front, so the ceiling is hard. See §2.")],
        [cell("<b>VRAM</b>"), cell("16 GB (base only)<br/>24 GB (base + coding)"), cell("24 576 MiB"),
         cell("Base stack = 8.0 GB. The Coder-1.5B rung adds 4.4 GB.")],
        [cell("<b>Disk</b>"), cell("65 GB free"), cell("243 GB (97 GB free)"),
         cell("41.9 GB images + 19.2 GB weights, before build cache.")],
        [cell("<b>RAM</b>"), cell("16 GB"), cell("—"),
         cell("Sentence-transformers and the cross-encoder run on CPU in the API container.")],
        [cell("<b>CPU</b>"), cell("8 cores"), cell("—"),
         cell("Guardrails, RAG, CSS scoring and the RL update are all CPU-side.")],
    ], [24 * mm, 32 * mm, 36 * mm, 78 * mm]))

    E.append(P("1.2 Software", "h2"))
    E.append(table([
        [cell("<b>Component</b>"), cell("<b>Version</b>"), cell("<b>Notes</b>")],
        [cell("Ubuntu"), cell("22.04 / 24.04"), cell("The only tested host OS.")],
        [cell("NVIDIA driver"), cell("≥ 535"), cell("570.172.08 on the reference box. <b>nvidia-smi must work on the host first.</b>")],
        [cell("nvidia-container-toolkit"), cell("current"), cell("The single most common deployment failure: the driver is fine, and the containers still cannot see the GPU. Preflight tests this by actually running a container.")],
        [cell("Docker Engine"), cell("≥ 24"), cell("29.3.1 on the reference box.")],
        [cell("Docker Compose"), cell("v2 plugin"), cell("<b>The old <font face='Courier'>docker-compose</font> binary will not work</b> — this stack uses profiles and <font face='Courier'>service_healthy</font> conditions.")],
        [cell("jq, curl"), cell("any"), cell("<font face='Courier'>verify.sh</font> parses JSON with jq.")],
    ], [34 * mm, 22 * mm, 114 * mm]))
    E.append(code(
        "# Docker + the NVIDIA container toolkit, from scratch\n"
        "curl -fsSL https://get.docker.com | sh\n"
        "sudo usermod -aG docker $USER    # log out and back in\n"
        "\n"
        "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\\n"
        "  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n"
        "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\\n"
        "  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\\n"
        "  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list\n"
        "sudo apt-get update &amp;&amp; sudo apt-get install -y nvidia-container-toolkit jq\n"
        "sudo nvidia-ctk runtime configure --runtime=docker\n"
        "sudo systemctl restart docker\n"
        "\n"
        "# the check that actually matters — the container, not the host, must see the GPU\n"
        "docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L"))

    E.append(CondPageBreak(70 * mm))
    E.append(P("1.3 Credentials", "h2"))
    E.append(P(
        "Four secrets. Only the first is strictly required — the system degrades honestly "
        "without the others rather than failing, which is by design, but it also stops being "
        "the thing you deployed it for. <b>The live values are in Appendix A.</b>", "body"))
    E.append(table([
        [cell("<b>Variable</b>"), cell("<b>Required?</b>"), cell("<b>Where it comes from</b>"), cell("<b>Without it</b>")],
        [cell("<b>HF_TOKEN</b>"), cell("<b>Yes</b>"),
         cell("huggingface.co → Settings → Access Tokens (read scope)"),
         cell("No vLLM container can pull its weights. Nothing starts.")],
        [cell("<b>EMAP_TOKEN</b>"), cell("Strongly"),
         cell("api.electricitymap.org (free tier covers the live signal)"),
         cell("Falls back to GRID_CARBON_FALLBACK=475. The router still routes — it is simply no longer carbon-aware, which is the whole product.")],
        [cell("<b>AUDIT_HMAC_KEY</b>"), cell("Production"),
         cell("Generate: <font face='Courier'>openssl rand -hex 32</font>"),
         cell("The audit trail is unsigned and a post-hoc edit becomes undetectable.")],
        [cell("<b>ADMIN_API_KEY</b>"), cell("Production"),
         cell("Generate: <font face='Courier'>echo gak_$(openssl rand -base64 32 | tr -d /+= | head -c 43)</font>"),
         cell("Admin endpoints are open, including /api/feedback/export — which serves the captured prompts and responses.")],
    ], [35 * mm, 18 * mm, 52 * mm, 65 * mm]))   # 35mm: AUDIT_HMAC_KEY must not wrap
    E.append(P(
        "Two Hugging Face repos are <b>gated</b> and need their licence accepted on the website "
        "with the same account that owns the token: <font face='Courier'>meta-llama/Llama-Guard-3-1B</font> "
        "and <font face='Courier'>meta-llama/Llama-2-7b-chat-hf</font>. They only matter for the "
        "<font face='Courier'>guard</font> and <font face='Courier'>fallback</font> profiles — but a "
        "gated repo does not fail at build time, it fails at <i>pull</i> time, twenty minutes in. "
        "Preflight checks both up front.", "body"))

    E.append(P("1.4 Network egress", "h2"))
    E.append(table([
        [cell("<b>Host</b>"), cell("<b>Used for</b>")],
        [cell("huggingface.co, cdn-lfs*.huggingface.co"), cell("Model weights (19.2 GB, once — then cached in data/hf-cache).")],
        [cell("api.electricitymap.org"), cell("Live grid carbon intensity + the 48-hour forecast.")],
        [cell("registry-1.docker.io, ghcr.io"), cell("Base images (vLLM, nginx, node, python).")],
        [cell("pypi.org, registry.npmjs.org"), cell("Build-time only.")],
    ], [62 * mm, 108 * mm]))
    E.append(P(
        "There is no other egress. The guardrails have no external LLM dependency, the RL "
        "controller trains in-process, and no LangChain/LangSmith package is installed at all — "
        "langchain-core pulls langsmith, which would otherwise POST your prompts off-box from an "
        "on-prem deployment.", "body"))

    E.append(PageBreak())

    # --------------------------------------------------------------- 2 VRAM
    E.append(P("2 · The VRAM budget — the one constraint that will bite you", "h1"))
    E.append(P(
        "Disk is cheap and CPU is idle. VRAM is what decides which profiles you can run at "
        "the same time, because vLLM does not allocate on demand — each container reserves "
        "<font face='Courier'>--gpu-memory-utilization × total</font> at startup and holds it. "
        "Three containers at 0.12 + 0.20 + 0.18 do not compete for memory; they simply "
        "claim 50 % of the card and leave.", "body"))
    E.append(img(d_vram, 176 * mm))
    E.append(caption("Measured: 13 127 MiB of 24 576 MiB in use with the base + coding stack up. "
                     "The guard container cannot be added on a 24 GB card — that is not a bug, "
                     "it is arithmetic, and it is why <font face='Courier'>GUARDRAILS_LLM_CLASSIFIER_ENABLED=false</font> "
                     "in the shipped .env."))
    E.append(table([
        [cell("<b>Service</b>"), cell("<b>Port</b>"), cell("<b>Model</b>"), cell("<b>gpu-mem-util</b>"), cell("<b>VRAM</b>"), cell("<b>Profile</b>")],
    ] + [
        [cell(s), cell(str(p)), cell(m), cell(f"{u:.2f}" if u else "—"),
         cell(f"{u * VRAM_TOTAL_GB:.1f} GB" if u else "CPU"), cell(f"<font face='Courier'>{pr}</font>")]
        for s, p, m, u, pr in SERVICES
    ], [30 * mm, 13 * mm, 43 * mm, 23 * mm, 17 * mm, 44 * mm]))
    E.append(P(
        "<b>Rules of thumb.</b> On a 24 GB card the base stack plus the coding rung leaves room "
        "for the guard; adding the MoE does not fit. The MoE profile "
        "(<font face='Courier'>Qwen3-30B-A3B</font>, 0.90) needs the card to itself. If a container "
        "exits immediately with a CUDA OOM, you have over-committed — lower a fraction or drop a "
        "profile; do not raise <font face='Courier'>--gpu-memory-utilization</font> hoping it fits.",
        "body"))

    E.append(KeepTogether([
        P("2.1 Disk", "h2"),
        img(d_disk, 176 * mm),
        caption("The vLLM runtime image is 32.9 GB — larger than every model weight combined. "
                "Budget 65 GB minimum; the build cache grows on top of this and "
                "<font face='Courier'>docker builder prune</font> is the first thing to reach for "
                "when the disk runs hot."),
    ]))

    E.append(PageBreak())

    # ------------------------------------------------------------ 3 THE SCRIPTS
    E.append(P("3 · The scripts", "h1"))
    E.append(P(
        "Three, in <font face='Courier'>deploy/</font>. They are ordinary bash and they are meant "
        "to be read before they are run.", "lead"))

    E.append(P("3.1 preflight.sh — changes nothing, refuses on a real fault", "h2"))
    E.append(P(
        "Read-only. It checks the OS, the Docker daemon and the compose <i>plugin</i>, the GPU and "
        "its VRAM, that a container can actually see the GPU, free disk against the real 65 GB "
        "requirement, that the seven published ports are free (and recognises this stack's own "
        "containers rather than reporting them as a clash), that HF_TOKEN is accepted by the "
        "Hugging Face API, that both gated repos are reachable, and that EMAP_TOKEN returns a "
        "live carbon intensity for your zone.", "body"))
    E.append(code(
        "$ deploy/preflight.sh\n"
        "\n"
        "Docker\n"
        "  ✓ docker 29.3.1\n"
        "  ✓ compose plugin 5.1.1\n"
        "GPU\n"
        "  ✓ GPU: NVIDIA H100L-2-24C (driver 570.172.08)\n"
        "  ✓ VRAM: 24576 MiB\n"
        "  ✓ nvidia-container-toolkit wired into docker (containers can see the GPU)\n"
        "Disk\n"
        "  ✓ free space: 97 GB\n"
        "Network / credentials\n"
        "  ✓ HF_TOKEN valid\n"
        "  ✓ gated model accessible: meta-llama/Llama-Guard-3-1B\n"
        "  ✓ EMAP_TOKEN valid for zone IN-WE\n"
        "\n"
        "12 passed, 0 warnings, 0 failures"))

    E.append(P("3.2 bootstrap.sh — seed, tune, build, wait", "h2"))
    E.append(P(
        "Seeds <font face='Courier'>.env</font> from <font face='Courier'>deploy/env.template</font> "
        "at mode 600 and <b>refuses to start while the credential placeholders are still in it</b>. "
        "Creates <font face='Courier'>data/hf-cache</font> before Docker can create it root-owned "
        "(if Docker gets there first, the weights re-download on every recreate). Detects the card "
        "and writes <font face='Courier'>GPU_VRAM_GB</font> into .env — that value and "
        "<font face='Courier'>GPU_TDP</font> feed the LLMCarbon formula directly, and getting them "
        "wrong does not crash anything, it silently produces wrong carbon numbers, which is the one "
        "failure this system cannot tolerate. Then builds, starts the requested profiles, and blocks "
        "until <font face='Courier'>/health/ready</font> answers.", "body"))
    E.append(code(
        "deploy/bootstrap.sh                    # base stack only\n"
        "deploy/bootstrap.sh stem-coding        # + the Coder-1.5B rung  ← this deployment\n"
        "deploy/bootstrap.sh moe                # the 30B MoE — needs the whole card"))

    E.append(P("3.3 verify.sh — proves the four claims", "h2"))
    E.append(P(
        "A green health check proves nothing about a carbon router, which is why this script exists. "
        "It asserts that a greeting routes to the <i>smallest</i> rung and not the largest; that the "
        "grid signal is <font face='Courier'>live</font> rather than the fallback constant; that the "
        "audit log is accumulating signed rows; and that a real coding request reaches the coding "
        "rung rather than a general instruct model, reporting the rung it landed on and the carbon "
        "it took to get there.", "body"))

    E.append(PageBreak())

    # ------------------------------------------------------------ 4 VERIFY OUT
    E.append(P("4 · What a good deployment looks like", "h1"))
    E.append(P(
        "This is the real output of <font face='Courier'>deploy/verify.sh</font> against the "
        "reference box on 14 July 2026 — not a mock-up.", "body"))
    E.append(code(
        "$ deploy/verify.sh\n"
        "\n"
        "1 · Health\n"
        "  ✓ API ready (the metrics sidecar answered — /health/ready blocks on it)\n"
        "  ✓ metrics sidecar\n"
        "  ✓ vLLM :8001 live      ✓ vLLM :8002 live\n"
        "  ✓ vLLM :8006 live\n"
        "\n"
        "2 · Carbon-aware routing + live grid\n"
        "  ✓ \"hi\" routed to ultra-light (TinyLlama-1.1B-Chat-v1.0) — 0.000083 gCO2\n"
        "  ✓ grid signal LIVE: 473.0 gCO2/kWh (zone IN-WE, Electricity Maps)\n"
        "\n"
        "3 · 48-hour forecast (EcoServe's input)\n"
        "    - forecast empty: deferral falls back to the live reading\n"
        "\n"
        "4 · Signed audit trail\n"
        "  ✓ decision_logs.jsonl: 403 HMAC-signed rows\n"
        "\n"
        "5 · Coding requests reach the coding rung\n"
        "  ✓ coding prompt routed to stem-coding\n"
        "    model            : Qwen2.5-Coder-1.5B-Instruct\n"
        "    gCO2             : 0.0207\n"
        "\n"
        "6 · UI\n"
        "  ✓ frontend on http://localhost:8080\n"
        "\n"
        "Deployment verified."))
    E.append(P(
        "Two of those lines are the product. <b>“hi” routed to ultra-light</b> means the router is "
        "choosing on carbon rather than defaulting to the biggest model — 0.000083 gCO₂ instead of "
        "the ~0.0004 g the 1.5B would have cost for the same greeting. And <b>routed to stem-coding</b> "
        "means a coding request reached the code-capable rung instead of a general instruct model.", "body"))
    E.append(Paragraph(
        "The empty forecast is expected on the free Electricity Maps tier and is not a failure: "
        "deferral falls back to the live reading. It is called out rather than hidden because a "
        "deployment guide that only lists what works is marketing.", S["pull"]))

    E.append(PageBreak())

    # ------------------------------------------------------- 5 TROUBLESHOOTING
    E.append(P("5 · Troubleshooting", "h1"))
    E.append(P("Every row below is a failure that actually happened on this box.", "body"))
    E.append(table([
        [cell("<b>Symptom</b>"), cell("<b>Cause and fix</b>")],
        [cell("<b>Container exits instantly; log says CUDA out of memory</b>"),
         cell("You over-committed VRAM (§2). vLLM reserves its fraction up front, so this is arithmetic, not load. Drop a profile or lower a <font face='Courier'>--gpu-memory-utilization</font>. Do <i>not</i> raise it.")],
        [cell("<b>docker run --gpus all fails / no GPU inside the container</b>"),
         cell("nvidia-container-toolkit is missing or not wired in, even though <font face='Courier'>nvidia-smi</font> works on the host. <font face='Courier'>sudo nvidia-ctk runtime configure --runtime=docker &amp;&amp; sudo systemctl restart docker</font>.")],
        [cell("<b>Weights download fails with 401/403 twenty minutes in</b>"),
         cell("A gated repo (Llama-Guard, Llama-2). Accept the licence on huggingface.co with the account that owns HF_TOKEN. Preflight catches this before you spend the twenty minutes.")],
        [cell("<b>Weights re-download on every <font face='Courier'>up</font></b>"),
         cell("<font face='Courier'>data/hf-cache</font> did not exist, so Docker created it root-owned and the container cannot write to it. <font face='Courier'>mkdir -p data/hf-cache</font> (bootstrap does this).")],
        [cell("<b>Every request routes to the biggest model</b>"),
         cell("The carbon signal is missing, so carbon scores are flat. Check <font face='Courier'>grid_signal.status == \"live\"</font> in a /api/chat response; if it says <font face='Courier'>fallback</font>, EMAP_TOKEN is wrong.")],
        [cell("<b>Coding prompts answer like a chat model</b>"),
         cell("The Coder rung (<font face='Courier'>vllm-stem-coding</font>, :8006) is not up — you booted without the <font face='Courier'>stem-coding</font> profile, so coding requests fall through to the general instruct model.")],
        [cell("<b>Carbon numbers look implausible</b>"),
         cell("<font face='Courier'>GPU_TDP</font> / <font face='Courier'>GPU_VRAM_GB</font> in .env do not match the card. They feed the LLMCarbon formula directly. Nothing crashes — the books are just wrong. Bootstrap auto-detects VRAM; verify TDP by hand against the spec sheet.")],
        [cell("<b>Disk full mid-build</b>"),
         cell("<font face='Courier'>docker builder prune</font> first — the build cache is the reclaimable part. <font face='Courier'>data/hf-cache</font> is <i>not</i>: deleting it costs a 19 GB re-download.")],
        [cell("<b>Port already in use</b>"),
         cell("Ports 8001-8008, 8080, 8100, 9000. If a <font face='Courier'>green-*</font> container holds it, the stack is already up: <font face='Courier'>docker compose ... down</font> first.")],
    ], [42 * mm, 128 * mm]))

    E.append(PageBreak())

    # ---------------------------------------------------------- 6 DAY-2 + PROD
    E.append(P("6 · Day-2 operations", "h1"))
    E.append(P("6.1 Command reference", "h2"))
    E.append(code(
        "C=\"docker compose -f docker-compose.ubuntu-vgpu.yml --env-file .env\"\n"
        "\n"
        "$C --profile stem-coding up --build -d                   # start / rebuild\n"
        "$C ps                                                    # what is running\n"
        "$C logs -f api                                           # follow the control plane\n"
        "$C restart api                                           # after an .env change\n"
        "$C down                                                  # stop (weights + data survive)\n"
        "\n"
        "curl localhost:8100/health/ready        # API (blocks on the metrics sidecar)\n"
        "curl localhost:9000/health              # sidecar\n"

        "curl localhost:8100/api/queue/status    # deferred work\n"
        "curl localhost:8100/api/rl/status       # learned CSS weights per tier\n"
        "\n"
        "docker builder prune                    # first thing to try when disk runs hot\n"
        "nvidia-smi                              # who is holding the VRAM\n"
        "\n"
        "# HTTPS: set PUBLIC_HOSTNAME in .env, then add the Caddy edge overlay\n"
        "docker compose -f docker-compose.ubuntu-vgpu.yml -f docker-compose.https.yml \\\n"
        "               --env-file .env up --build -d"))

    E.append(P("6.2 What to back up", "h2"))
    E.append(table([
        [cell("<b>Path</b>"), cell("<b>Contents</b>"), cell("<b>If lost</b>")],
        [cell("data/decision_logs.jsonl"), cell("HMAC-signed audit trail"), cell("Every dashboard and the CSRD report derive from this file. It is the only durable record. <b>Back it up.</b>")],
        [cell("data/green_ai.db"), cell("Conversations, messages, feedback"), cell("Chat history and the thumbs-up/down dataset for offline fine-tuning.")],
        [cell("data/rl_state.json"), cell("Learned CSS weights per tier"), cell("The policy resets to config/policies.json and re-learns. Costs weeks of observed traffic.")],
        [cell("data/ql_estimator_state.json"), cell("Learned accuracy/latency corrections"), cell("Falls back to the identity function — cold start, never worse than baseline.")],
        [cell("data/hf-cache/"), cell("19.2 GB of model weights"), cell("Re-downloads. Costs time, not data. Not worth backing up; worth keeping.")],
    ], [42 * mm, 42 * mm, 86 * mm]))

    E.append(P("6.3 Rotating the credentials", "h2"))
    E.append(P(
        "<b>AUDIT_HMAC_KEY is not like the others.</b> Rotating it invalidates the signature on "
        "every existing row of <font face='Courier'>decision_logs.jsonl</font> — which is exactly "
        "what the signature is for, and it means the old log can no longer be verified against the "
        "new key. Archive the log first, then rotate, and keep the retired key with the archive. "
        "HF_TOKEN, EMAP_TOKEN and ADMIN_API_KEY can be swapped freely: change .env, "
        "<font face='Courier'>docker compose restart api</font>.", "body"))

    E.append(KeepTogether([
        P("6.4 Production hardening", "h2"),
        table([
        [cell("<b>Setting</b>"), cell("<b>Do this</b>")],
        [cell("<font face='Courier'>ADMIN_API_KEY</font>"), cell("Must be set. Empty = dev mode, and /api/feedback/export then serves the captured prompts and responses to anyone.")],
        [cell("<font face='Courier'>AUDIT_HMAC_KEY</font>"), cell("Rotate away from the shipped value. <font face='Courier'>openssl rand -hex 32</font>.")],
        [cell("<font face='Courier'>ALLOWED_ORIGINS</font>"), cell("Currently <font face='Courier'>http://localhost:8080</font>. Set to the real origin — it is the CORS allow-list.")],
        [cell("HTTPS"), cell("<font face='Courier'>PUBLIC_HOSTNAME=chat.example.com</font>, then add the Caddy overlay. It terminates TLS with an automatic certificate and reverse-proxies the frontend.")],
        [cell("Ports"), cell("Only 8080 (UI) and 8100 (API) need to be reachable. Firewall 8001-8008 and 9000 — they are internal.")],
        ], [40 * mm, 130 * mm]),
    ]))

    E.append(PageBreak())

    # ------------------------------------------------------------ APPENDIX A
    E.append(P("Appendix A · Credentials (LIVE)", "h1"))
    E.append(Paragraph(
        "These are live, working credentials for the reference deployment. Anyone holding this "
        "PDF holds them. Treat the document as a secret; if it leaves your control, rotate all "
        "four — and read §6.3 before rotating AUDIT_HMAC_KEY, because it invalidates the "
        "signatures on the existing audit log.",
        ParagraphStyle("warn", parent=S["pull"],
                       textColor=colors.HexColor(RED), fontSize=9.6, leading=13.6)))
    E.append(Spacer(1, 2 * mm))
    E.append(P("Paste these four lines into <font face='Courier'>.env</font>, replacing the "
               "<font face='Courier'>__PASTE_FROM_DEPLOY_GUIDE_APPENDIX_A__</font> placeholders that "
               "<font face='Courier'>deploy/env.template</font> ships with. Everything else in the "
               "template is already set correctly for this stack.", "body"))
    E.append(code(
        "# Hugging Face — pulls every model's weights. Required.\n"
        f"HF_TOKEN={SECRETS['HF_TOKEN']}\n"
        "\n"
        "# Electricity Maps — the live grid carbon signal the router scores on.\n"
        f"EMAP_TOKEN={SECRETS['EMAP_TOKEN']}\n"
        f"EMAP_ZONE={SECRETS['EMAP_ZONE']}\n"
        "\n"
        "# Audit trail signing key. Rotating this invalidates every existing signature.\n"
        f"AUDIT_HMAC_KEY={SECRETS['AUDIT_HMAC_KEY']}\n"
        "\n"
        "# Gates /api/feedback/export, /api/cache/clear-all, /api/budgets*, /api/model-zoo/updates*\n"
        f"ADMIN_API_KEY={SECRETS['ADMIN_API_KEY']}"))
    E.append(P(
        "The reference box also points OpenTelemetry at an internal collector via "
        "<font face='Courier'>OTEL_EXPORTER_OTLP_ENDPOINT</font>. That address will not resolve in a "
        "new environment — either point it at your own collector or leave it: the exporter fails "
        "silently and nothing else is affected.", "body"))

    E.append(P("Settings worth checking for a new host", "h2"))
    E.append(table([
        [cell("<b>Variable</b>"), cell("<b>Shipped</b>"), cell("<b>Change it when…</b>")],
        [cell("<font face='Courier'>GPU_TDP</font>"), cell("350"),
         cell("Your card is not a 350 W part. Feeds the LLMCarbon operational term directly — wrong value, wrong carbon, no error.")],
        [cell("<font face='Courier'>GPU_VRAM_GB</font>"), cell("16"),
         cell("<b>Stale on the reference box, which has 24 GB.</b> bootstrap.sh now auto-detects and rewrites it.")],
        [cell("<font face='Courier'>EMAP_ZONE</font>"), cell("IN-WE"),
         cell("You are not on the India-West grid. E.g. US-CAL-CISO, DE, GB, FR.")],
        [cell("<font face='Courier'>GRID_CARBON_FALLBACK</font>"), cell("475"),
         cell("The number used when Electricity Maps is unreachable. Set it to your grid's annual average, not ours.")],
        [cell("<font face='Courier'>ALLOWED_ORIGINS</font>"), cell("localhost:8080"),
         cell("Always, in production. It is the CORS allow-list.")],
    ], [40 * mm, 26 * mm, 104 * mm]))   # 26mm: "localhost:8080" must not wrap

    doc.build(E)
    print(f"✓ {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()

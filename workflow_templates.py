"""Seeded workflow templates — the gallery behind the Workflows builder.

Each template is a *runnable* starting-point graph derived from the 25 use cases
in the orchestration doc. They are authored compactly here (nodes + edges) and
auto-laid-out into canvas positions; ``build_template`` validates every graph
against the engine's ``validate_graph`` so a broken template can never ship.

Templates are read-only seeds. Instantiating one (POST .../instantiate) copies
its graph into a new, editable workflow owned by the caller's tenant. HTTP nodes
use ``https://example.com/...`` placeholders the user is expected to edit — the
same convention n8n/Make use for template I/O.

This module is dependency-free apart from ``workflows`` (for validation); it
never imports ``decision_engine``.
"""
from __future__ import annotations

from typing import Any

from workflows import validate_graph


# ─────────────────────────────────────────────────────────────────────────────
# Compact builder + auto-layout
# ─────────────────────────────────────────────────────────────────────────────
def _layout(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Assign a layered left-to-right position to every node (in place).
    Column = longest-path depth from a trigger; row = order within the column."""
    ids = [n["id"] for n in nodes]
    preds: dict[str, list[str]] = {i: [] for i in ids}
    succ: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        preds[e["target"]].append(e["source"])
        succ[e["source"]].append(e["target"])

    depth: dict[str, int] = {}

    def _depth(nid: str, seen: frozenset[str]) -> int:
        if nid in depth:
            return depth[nid]
        if not preds[nid] or nid in seen:
            depth[nid] = 0
            return 0
        d = 1 + max(_depth(p, seen | {nid}) for p in preds[nid])
        depth[nid] = d
        return d

    for i in ids:
        _depth(i, frozenset())

    by_col: dict[int, list[str]] = {}
    for i in ids:
        by_col.setdefault(depth[i], []).append(i)
    pos = {}
    for col, members in by_col.items():
        for row, nid in enumerate(members):
            pos[nid] = {"x": 60 + col * 250, "y": 50 + row * 140}
    for n in nodes:
        n["position"] = pos[n["id"]]


def _build(tpl_id: str, name: str, industry: str, description: str,
           nodes: list[tuple], edges: list[tuple], tags: list[str] | None = None,
           scenario: str = "", carbon: str = "", analog: str = "") -> dict[str, Any]:
    """nodes: (id, type, params). edges: (source, target[, sourceHandle]).

    ``scenario`` / ``carbon`` / ``analog`` are the plain-English prose the docs
    generator uses for the step-by-step guide (optional; falls back to
    ``description``)."""
    node_dicts = [{"id": nid, "type": ntype, "label": _LABELS.get(ntype, ntype),
                   "params": params or {}} for nid, ntype, params in nodes]
    edge_dicts = []
    for e in edges:
        d = {"source": e[0], "target": e[1]}
        if len(e) > 2 and e[2]:
            d["sourceHandle"] = e[2]
        edge_dicts.append(d)
    _layout(node_dicts, edge_dicts)
    graph = {"nodes": node_dicts, "edges": edge_dicts}
    validate_graph(graph)  # fail fast at import if a template is malformed
    return {"id": tpl_id, "name": name, "industry": industry,
            "description": description, "tags": tags or [], "graph": graph,
            "scenario": scenario, "carbon": carbon, "analog": analog}


_LABELS = {
    "manual": "Manual", "webhook": "Webhook", "schedule": "Schedule", "carbon_window": "Carbon window",
    "llm": "LLM", "rag_query": "RAG retrieve", "agent_task": "Coding agent", "guardrail": "Guardrail",
    "image_gen": "Image gen", "http_request": "HTTP", "transform": "Transform", "if": "IF",
    "carbon_gate": "Carbon gate", "merge": "Merge", "subworkflow": "Sub-workflow", "approval": "Approval",
}

_EX = "https://example.com"


# ─────────────────────────────────────────────────────────────────────────────
# The 25 templates
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATES: list[dict[str, Any]] = [

    # ── Customer Operations ──────────────────────────────────────────────────
    _build(
        "support-ticket-responder", "Autonomous support-ticket responder", "Customer Operations",
        "Answer inbound tickets from your knowledge base, with safety rails on the question and the reply.",
        [("t", "webhook", {}),
         ("g_in", "guardrail", {"text": "{{ $trigger.question }}", "phase": "input"}),
         ("kb", "rag_query", {"query": "{{ $trigger.question }}", "top_k": 6}),
         ("ans", "llm", {"prompt": "Use this context:\n{{ $node.kb.context }}\n\nQuestion: {{ $trigger.question }}"}),
         ("g_out", "guardrail", {"text": "{{ $node.ans.text }}", "phase": "output"}),
         ("reply", "http_request", {"method": "POST", "url": f"{_EX}/helpdesk/reply", "body": "{{ $node.ans.text }}"}),
         ("blocked", "transform", {"fields": {"status": "blocked", "reason": "{{ $node.g_in.reason }}"}})],
        [("t", "g_in"), ("g_in", "kb", "safe"), ("g_in", "blocked", "blocked"),
         ("kb", "ans"), ("ans", "g_out"), ("g_out", "reply", "safe")],
        ["support", "rag", "guardrails"]),

    _build(
        "tiered-sla-routing", "Tiered SLA-aware routing", "Customer Operations",
        "Premium traffic gets a higher accuracy floor; standard traffic gets the greenest capable model.",
        [("t", "webhook", {}),
         ("tier", "transform", {"fields": {"tier": "{{ $trigger.tier }}"}}),
         ("cond", "if", {"left": "{{ $trigger.tier }}", "op": "==", "right": "premium"}),
         ("prem", "llm", {"prompt": "{{ $trigger.question }}", "user_tier": "premium", "accuracy_floor": "0.9"}),
         ("std", "llm", {"prompt": "{{ $trigger.question }}", "user_tier": "standard"}),
         ("reply", "http_request", {"method": "POST", "url": f"{_EX}/reply"})],
        [("t", "tier"), ("tier", "cond"), ("cond", "prem", "true"), ("cond", "std", "false"),
         ("prem", "reply"), ("std", "reply")],
        ["routing", "sla"]),

    _build(
        "multilingual-answer", "Multilingual answer pipeline", "Customer Operations",
        "Detect language, answer from the KB in English, then translate the reply back.",
        [("t", "webhook", {}),
         ("detect", "llm", {"prompt": "Detect the language of and translate to English:\n{{ $trigger.message }}"}),
         ("kb", "rag_query", {"query": "{{ $node.detect.text }}", "top_k": 6}),
         ("ans", "llm", {"prompt": "Context:\n{{ $node.kb.context }}\n\nAnswer: {{ $node.detect.text }}"}),
         ("back", "llm", {"prompt": "Translate back to the original language:\n{{ $node.ans.text }}"}),
         ("reply", "http_request", {"method": "POST", "url": f"{_EX}/reply", "body": "{{ $node.back.text }}"})],
        [("t", "detect"), ("detect", "kb"), ("kb", "ans"), ("ans", "back"), ("back", "reply")],
        ["translation", "rag"]),

    _build(
        "sentiment-escalation", "Sentiment-based escalation & handoff", "Customer Operations",
        "Escalate negative messages to a human queue; auto-resolve everything else.",
        [("t", "webhook", {}),
         ("cls", "llm", {"prompt": "Classify sentiment as 'negative' or 'positive' only:\n{{ $trigger.message }}"}),
         ("cond", "if", {"left": "{{ $node.cls.text }}", "op": "contains", "right": "negative"}),
         ("esc", "http_request", {"method": "POST", "url": f"{_EX}/escalate"}),
         ("auto", "llm", {"prompt": "Write a friendly reply to:\n{{ $trigger.message }}"}),
         ("reply", "http_request", {"method": "POST", "url": f"{_EX}/reply", "body": "{{ $node.auto.text }}"})],
        [("t", "cls"), ("cls", "cond"), ("cond", "esc", "true"), ("cond", "auto", "false"), ("auto", "reply")],
        ["support", "triage"]),

    # ── Software Engineering & DevOps ────────────────────────────────────────
    _build(
        "auto-fix-ci", "Auto-fix failing CI", "Software Engineering & DevOps",
        "Hand a red build to the coding agent under a carbon budget; open a PR only if the frozen tests pass.",
        [("t", "webhook", {}),
         ("agent", "agent_task", {"task": "{{ $trigger.task }}", "tests": "{{ $trigger.tests }}", "carbon_budget_g": "3.0"}),
         ("cond", "if", {"left": "{{ $node.agent.status }}", "op": "==", "right": "completed"}),
         ("pr", "http_request", {"method": "POST", "url": f"{_EX}/git/pr"}),
         ("notify", "http_request", {"method": "POST", "url": f"{_EX}/slack"})],
        [("t", "agent"), ("agent", "cond"), ("cond", "pr", "true"), ("cond", "notify", "false")],
        ["devops", "agent"]),

    _build(
        "incident-triage", "Incident triage & auto-remediation", "Software Engineering & DevOps",
        "Classify alerts by severity; generate and page a remediation for the high-severity minority.",
        [("t", "webhook", {}),
         ("sev", "llm", {"prompt": "Classify severity as 'high' or 'low' only:\n{{ $trigger.alert }}"}),
         ("cond", "if", {"left": "{{ $node.sev.text }}", "op": "contains", "right": "high"}),
         ("rem", "agent_task", {"task": "Draft a remediation for: {{ $trigger.alert }}"}),
         ("page", "http_request", {"method": "POST", "url": f"{_EX}/pagerduty"}),
         ("log", "transform", {"fields": {"severity": "low", "alert": "{{ $trigger.alert }}"}})],
        [("t", "sev"), ("sev", "cond"), ("cond", "rem", "true"), ("cond", "log", "false"), ("rem", "page")],
        ["devops", "incident"]),

    _build(
        "security-digest", "Nightly security & dependency digest", "Software Engineering & DevOps",
        "Pull advisories at 06:00, match against your stack, summarise impact, and post to Slack.",
        [("t", "schedule", {"cron": "0 6 * * *"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/advisories"}),
         ("match", "rag_query", {"query": "{{ $node.fetch.body }}", "top_k": 8}),
         ("sum", "llm", {"prompt": "Summarise the security impact:\n{{ $node.match.context }}"}),
         ("post", "http_request", {"method": "POST", "url": f"{_EX}/slack", "body": "{{ $node.sum.text }}"})],
        [("t", "fetch"), ("fetch", "match"), ("match", "sum"), ("sum", "post")],
        ["devops", "security", "scheduled"]),

    _build(
        "pr-review-assistant", "PR review assistant", "Software Engineering & DevOps",
        "Post an automated first-pass review comment on every pull request.",
        [("t", "webhook", {}),
         ("g", "guardrail", {"text": "{{ $trigger.diff }}", "phase": "input"}),
         ("review", "llm", {"prompt": "Review this diff for bugs and style:\n{{ $trigger.diff }}"}),
         ("comment", "http_request", {"method": "POST", "url": f"{_EX}/git/comment", "body": "{{ $node.review.text }}"})],
        [("t", "g"), ("g", "review", "safe"), ("review", "comment")],
        ["devops", "code-review"]),

    _build(
        "release-notes", "Automated release notes", "Software Engineering & DevOps",
        "Turn the git log since the last release into human-readable notes and publish.",
        [("t", "webhook", {}),
         ("log", "http_request", {"method": "GET", "url": f"{_EX}/git/log"}),
         ("notes", "llm", {"prompt": "Write release notes from these commits:\n{{ $node.log.body }}"}),
         ("g", "guardrail", {"text": "{{ $node.notes.text }}", "phase": "output"}),
         ("pub", "http_request", {"method": "POST", "url": f"{_EX}/docs/publish", "body": "{{ $node.notes.text }}"})],
        [("t", "log"), ("log", "notes"), ("notes", "g"), ("g", "pub", "safe")],
        ["devops", "docs"]),

    # ── Sustainability & ESG ─────────────────────────────────────────────────
    _build(
        "carbon-batch-summarize", "Carbon-aware batch summarisation", "Sustainability & ESG",
        "Summarise a document only while the grid is clean. Add a Sub-workflow node to fan out over a list.",
        [("t", "carbon_window", {"threshold_g": 200}),
         ("kb", "rag_query", {"query": "{{ $trigger.topic }}", "top_k": 8}),
         ("sum", "llm", {"prompt": "Summarise:\n{{ $node.kb.context }}", "user_tier": "batch"}),
         ("store", "http_request", {"method": "POST", "url": f"{_EX}/store", "body": "{{ $node.sum.text }}"})],
        [("t", "kb"), ("kb", "sum"), ("sum", "store")],
        ["esg", "green-window", "batch"]),

    _build(
        "csrd-esg-report", "Nightly CSRD / ESG report", "Sustainability & ESG",
        "Assemble the day's carbon/audit metrics into a narrative ESG report at 02:00 and file it.",
        [("t", "schedule", {"cron": "0 2 * * *"}),
         ("pull", "http_request", {"method": "GET", "url": f"{_EX}/metrics/carbon"}),
         ("draft", "llm", {"prompt": "Write an ESG narrative from:\n{{ $node.pull.body }}", "user_tier": "esg"}),
         ("attach", "transform", {"fields": {"report": "{{ $node.draft.text }}", "generated": "{{ $now }}"}}),
         ("file", "http_request", {"method": "POST", "url": f"{_EX}/esg/file", "body": "{{ $node.draft.text }}"})],
        [("t", "pull"), ("pull", "draft"), ("draft", "attach"), ("attach", "file")],
        ["esg", "scheduled", "reporting"]),

    _build(
        "green-window-job", "Green-window job orchestration", "Sustainability & ESG",
        "Launch a heavy, deferrable job (retrain / re-index) only when carbon is low.",
        [("t", "carbon_window", {"threshold_g": 200}),
         ("gate", "carbon_gate", {"threshold_g": 200}),
         ("kick", "http_request", {"method": "POST", "url": f"{_EX}/jobs/train"}),
         ("notify", "http_request", {"method": "POST", "url": f"{_EX}/slack"})],
        [("t", "gate"), ("gate", "kick", "green"), ("kick", "notify")],
        ["esg", "green-window"]),

    _build(
        "emissions-anomaly", "Emissions anomaly alerting", "Sustainability & ESG",
        "Watch carbon-per-request every 15 minutes; explain and alert on a spike.",
        [("t", "schedule", {"cron": "*/15 * * * *"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/metrics/usage"}),
         ("cond", "if", {"left": "{{ $node.fetch.json.carbon_per_req }}", "op": ">", "right": "0.5"}),
         ("explain", "llm", {"prompt": "Explain a likely cause of this carbon spike:\n{{ $node.fetch.body }}"}),
         ("alert", "http_request", {"method": "POST", "url": f"{_EX}/alert", "body": "{{ $node.explain.text }}"})],
        [("t", "fetch"), ("fetch", "cond"), ("cond", "explain", "true"), ("explain", "alert")],
        ["esg", "monitoring", "scheduled"]),

    # ── Content & Marketing ──────────────────────────────────────────────────
    _build(
        "carbon-image-campaign", "Carbon-capped image campaign", "Content & Marketing",
        "Generate campaign images at reduced steps, only during clean-grid windows.",
        [("t", "carbon_window", {"threshold_g": 250}),
         ("gate", "carbon_gate", {"threshold_g": 250}),
         ("img", "image_gen", {"prompt": "{{ $trigger.prompt }}", "steps": 20}),
         ("upload", "http_request", {"method": "POST", "url": f"{_EX}/assets"})],
        [("t", "gate"), ("gate", "img", "green"), ("img", "upload")],
        ["marketing", "image", "green-window"]),

    _build(
        "seo-content-signoff", "SEO content pipeline with human sign-off", "Content & Marketing",
        "Draft an on-brand, safety-checked article, then pause for an editor to approve before publishing.",
        [("t", "webhook", {}),
         ("kb", "rag_query", {"query": "{{ $trigger.topic }}", "top_k": 6}),
         ("draft", "llm", {"prompt": "Brand context:\n{{ $node.kb.context }}\n\nWrite an article on: {{ $trigger.topic }}"}),
         ("g", "guardrail", {"text": "{{ $node.draft.text }}", "phase": "output"}),
         ("review", "approval", {"message": "Approve this article for publishing?\n\n{{ $node.draft.text }}"}),
         ("pub", "http_request", {"method": "POST", "url": f"{_EX}/cms/publish", "body": "{{ $node.draft.text }}"}),
         ("arch", "transform", {"fields": {"status": "archived_draft"}})],
        [("t", "kb"), ("kb", "draft"), ("draft", "g"), ("g", "review", "safe"),
         ("review", "pub", "approved"), ("review", "arch", "rejected")],
        ["marketing", "approval", "hitl"]),

    _build(
        "social-listening-digest", "Social listening digest", "Content & Marketing",
        "Summarise brand mentions with sentiment and push a daily digest to a dashboard.",
        [("t", "schedule", {"cron": "0 8 * * *"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/mentions"}),
         ("sum", "llm", {"prompt": "Summarise mentions with overall sentiment:\n{{ $node.fetch.body }}"}),
         ("shape", "transform", {"fields": {"digest": "{{ $node.sum.text }}", "date": "{{ $now }}"}}),
         ("post", "http_request", {"method": "POST", "url": f"{_EX}/dashboard"})],
        [("t", "fetch"), ("fetch", "sum"), ("sum", "shape"), ("shape", "post")],
        ["marketing", "scheduled"]),

    _build(
        "personalized-outreach", "Personalised outreach", "Content & Marketing",
        "Personalise an email for a recipient and hand off to the email platform. Add a Sub-workflow to batch a segment.",
        [("t", "webhook", {}),
         ("msg", "llm", {"prompt": "Personalise this email for {{ $trigger.name }} at {{ $trigger.company }}:\n{{ $trigger.template }}"}),
         ("send", "http_request", {"method": "POST", "url": f"{_EX}/esp/send", "body": "{{ $node.msg.text }}"})],
        [("t", "msg"), ("msg", "send")],
        ["marketing", "email"]),

    # ── Data & Analytics ─────────────────────────────────────────────────────
    _build(
        "doc-extraction-routing", "Document extraction & routing", "Data & Analytics",
        "Classify a document, extract fields, and route it to the right downstream system.",
        [("t", "webhook", {}),
         ("ext", "llm", {"prompt": "Extract fields and the doc type as JSON:\n{{ $trigger.document }}"}),
         ("cond", "if", {"left": "{{ $node.ext.text }}", "op": "contains", "right": "invoice"}),
         ("ap", "http_request", {"method": "POST", "url": f"{_EX}/ap"}),
         ("store", "http_request", {"method": "POST", "url": f"{_EX}/store"})],
        [("t", "ext"), ("ext", "cond"), ("cond", "ap", "true"), ("cond", "store", "false")],
        ["data", "extraction"]),

    _build(
        "lead-enrichment-scoring", "Lead enrichment & scoring", "Data & Analytics",
        "Enrich a form submission, score fit + intent, and sync to the CRM.",
        [("t", "webhook", {}),
         ("enrich", "http_request", {"method": "GET", "url": f"{_EX}/enrich?email={{ $trigger.email }}"}),
         ("score", "llm", {"prompt": "Score fit and intent 0-100 for:\n{{ $node.enrich.body }}"}),
         ("map", "transform", {"fields": {"email": "{{ $trigger.email }}", "score": "{{ $node.score.text }}"}}),
         ("crm", "http_request", {"method": "POST", "url": f"{_EX}/crm/upsert"})],
        [("t", "enrich"), ("enrich", "score"), ("score", "map"), ("map", "crm")],
        ["data", "crm"]),

    _build(
        "data-quality-gate", "Data-quality validation gate", "Data & Analytics",
        "Fetch a dataset, validate it, and only load if valid — otherwise alert.",
        [("t", "schedule", {"cron": "0 * * * *"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/dataset"}),
         ("cond", "if", {"left": "{{ $node.fetch.status }}", "op": "==", "right": "200"}),
         ("load", "http_request", {"method": "POST", "url": f"{_EX}/warehouse/load"}),
         ("alert", "http_request", {"method": "POST", "url": f"{_EX}/alert"})],
        [("t", "fetch"), ("fetch", "cond"), ("cond", "load", "true"), ("cond", "alert", "false")],
        ["data", "quality", "scheduled"]),

    _build(
        "kb-auto-curation", "Knowledge-base auto-curation", "Data & Analytics",
        "Nightly, summarise a newly added document and index it. Add a Sub-workflow to fan out over all new docs.",
        [("t", "schedule", {"cron": "0 3 * * *"}),
         ("list", "http_request", {"method": "GET", "url": f"{_EX}/docs/new"}),
         ("sum", "llm", {"prompt": "Summarise for indexing:\n{{ $node.list.body }}"}),
         ("index", "http_request", {"method": "POST", "url": f"{_EX}/rag/index", "body": "{{ $node.sum.text }}"})],
        [("t", "list"), ("list", "sum"), ("sum", "index")],
        ["data", "rag", "scheduled"]),

    _build(
        "competitive-intel", "Competitive-intelligence brief", "Data & Analytics",
        "Aggregate competitor feeds each Monday and produce a concise weekly brief.",
        [("t", "schedule", {"cron": "0 7 * * 1"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/feeds"}),
         ("ctx", "rag_query", {"query": "{{ $node.fetch.body }}", "top_k": 6}),
         ("brief", "llm", {"prompt": "Write a weekly competitive brief:\n{{ $node.ctx.context }}"}),
         ("deliver", "http_request", {"method": "POST", "url": f"{_EX}/deliver", "body": "{{ $node.brief.text }}"})],
        [("t", "fetch"), ("fetch", "ctx"), ("ctx", "brief"), ("brief", "deliver")],
        ["data", "scheduled"]),

    # ── IT / Security / Compliance ───────────────────────────────────────────
    _build(
        "compliance-gateway", "Compliance guardrail gateway", "IT / Security / Compliance",
        "A reusable front door that screens inbound content and the outbound response, with an audit record.",
        [("t", "webhook", {}),
         ("gin", "guardrail", {"text": "{{ $trigger.content }}", "phase": "input"}),
         ("proc", "llm", {"prompt": "{{ $trigger.content }}"}),
         ("gout", "guardrail", {"text": "{{ $node.proc.text }}", "phase": "output"}),
         ("audit", "transform", {"fields": {"decision": "allowed", "output": "{{ $node.proc.text }}"}}),
         ("reject", "transform", {"fields": {"decision": "blocked", "reason": "{{ $node.gin.reason }}"}})],
        [("t", "gin"), ("gin", "proc", "safe"), ("gin", "reject", "blocked"),
         ("proc", "gout"), ("gout", "audit", "safe")],
        ["security", "compliance", "guardrails"]),

    _build(
        "prompt-ab-eval", "Prompt A/B evaluation harness", "IT / Security / Compliance",
        "Run one input through two prompts in parallel and log the comparison.",
        [("t", "manual", {}),
         ("a", "llm", {"prompt": "Prompt A:\n{{ $trigger.input }}"}),
         ("b", "llm", {"prompt": "Prompt B (be concise):\n{{ $trigger.input }}"}),
         ("join", "merge", {}),
         ("cmp", "transform", {"fields": {"a": "{{ $node.a.text }}", "b": "{{ $node.b.text }}",
                                          "a_co2": "{{ $node.a.carbon_g }}", "b_co2": "{{ $node.b.carbon_g }}"}}),
         ("log", "http_request", {"method": "POST", "url": f"{_EX}/eval/log"})],
        [("t", "a"), ("t", "b"), ("a", "join"), ("b", "join"), ("join", "cmp"), ("cmp", "log")],
        ["eval", "parallel"]),

    _build(
        "multi-agent-research", "Multi-agent research (agent-of-agents)", "IT / Security / Compliance",
        "A planner decomposes a question, a research step answers it, and a synthesiser composes the report. "
        "Add a Sub-workflow (items = sub-questions) to fan the research out in parallel.",
        [("t", "manual", {}),
         ("plan", "llm", {"prompt": "Break this into 3 sub-questions:\n{{ $trigger.question }}"}),
         ("kb", "rag_query", {"query": "{{ $node.plan.text }}", "top_k": 8}),
         ("ans", "llm", {"prompt": "Answer using:\n{{ $node.kb.context }}\n\nSub-questions:\n{{ $node.plan.text }}"}),
         ("synth", "llm", {"prompt": "Synthesise a final report:\n{{ $node.ans.text }}", "user_tier": "premium"})],
        [("t", "plan"), ("plan", "kb"), ("kb", "ans"), ("ans", "synth")],
        ["agents", "research"]),

    # ═════════════════════════════════════════════════════════════════════════
    # 50 additional production use cases (added 2026-07-24)
    # ═════════════════════════════════════════════════════════════════════════

    # ── Customer Operations ──────────────────────────────────────────────────
    _build(
        "voc-theme-miner", "Voice-of-customer theme miner", "Customer Operations",
        "Weekly: cluster customer reviews into the top recurring themes and post to a dashboard.",
        [("t", "schedule", {"cron": "0 6 * * 1"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/reviews?since=7d"}),
         ("themes", "llm", {"prompt": "Cluster these reviews into the top themes with a count and example each:\n{{ $node.fetch.body }}"}),
         ("shape", "transform", {"fields": {"themes": "{{ $node.themes.text }}", "week_of": "{{ $now }}"}}),
         ("post", "http_request", {"method": "POST", "url": f"{_EX}/dashboard/voc", "body": "{{ $node.themes.text }}"})],
        [("t", "fetch"), ("fetch", "themes"), ("themes", "shape"), ("shape", "post")],
        ["cx", "scheduled"],
        scenario="Every Monday the workflow pulls the past week's reviews and uses one LLM pass to cluster them into "
                 "the top themes with counts and examples, then posts the digest to your CX dashboard.",
        carbon="A single weekly clustering call on the greenest rung replaces a standing analyst task; scheduled off-peak.",
        analog="LangChain summarisation / clustering chain."),

    _build(
        "churn-early-warning", "Churn-risk early warning", "Customer Operations",
        "On a usage-drop signal, assess churn risk and open a CSM task only for high-risk accounts.",
        [("t", "webhook", {}),
         ("assess", "llm", {"prompt": "Assess churn risk as 'high' or 'low' for this account signal:\n{{ $trigger.account }}"}),
         ("cond", "if", {"left": "{{ $node.assess.text }}", "op": "contains", "right": "high"}),
         ("task", "http_request", {"method": "POST", "url": f"{_EX}/csm/task"}),
         ("log", "transform", {"fields": {"risk": "low", "account": "{{ $trigger.account }}"}})],
        [("t", "assess"), ("assess", "cond"), ("cond", "task", "true"), ("cond", "log", "false")],
        ["cx", "retention"],
        scenario="A product signal (e.g. a drop in logins) hits the webhook; the LLM judges churn risk, and only "
                 "high-risk accounts create a customer-success task — everyone else is just logged.",
        carbon="Cheap classification on the smallest capable model; expensive follow-up only for the risky minority.",
        analog="LangGraph conditional routing."),

    _build(
        "refund-adjudication", "Refund request adjudication", "Customer Operations",
        "Check a refund against policy, pause for human approval, then issue or deny it.",
        [("t", "webhook", {}),
         ("check", "llm", {"prompt": "Does this refund meet policy? Answer 'eligible' or 'ineligible' with a reason:\n{{ $trigger.request }}"}),
         ("review", "approval", {"message": "Approve refund of {{ $trigger.amount }}?\n\nPolicy check: {{ $node.check.text }}"}),
         ("refund", "http_request", {"method": "POST", "url": f"{_EX}/billing/refund"}),
         ("deny", "http_request", {"method": "POST", "url": f"{_EX}/billing/deny"})],
        [("t", "check"), ("check", "review"), ("review", "refund", "approved"), ("review", "deny", "rejected")],
        ["cx", "approval", "hitl"],
        scenario="The LLM does a first-pass policy check on a refund request, then the run pauses so an agent can "
                 "approve or reject; the decision routes to the refund or denial endpoint.",
        carbon="One small policy-check call; the human gate prevents wasted downstream calls on bad requests.",
        analog="LangGraph interrupt/resume with a policy tool."),

    _build(
        "faq-deflection-fallback", "FAQ deflection with human fallback", "Customer Operations",
        "Answer from the KB; if retrieval finds nothing, escalate to a human instead of guessing.",
        [("t", "webhook", {}),
         ("kb", "rag_query", {"query": "{{ $trigger.question }}", "top_k": 6}),
         ("cond", "if", {"left": "{{ $node.kb.context }}", "op": "empty", "right": ""}),
         ("human", "http_request", {"method": "POST", "url": f"{_EX}/handoff"}),
         ("answer", "llm", {"prompt": "Answer only from this context:\n{{ $node.kb.context }}\n\nQ: {{ $trigger.question }}"}),
         ("reply", "http_request", {"method": "POST", "url": f"{_EX}/reply", "body": "{{ $node.answer.text }}"})],
        [("t", "kb"), ("kb", "cond"), ("cond", "human", "true"), ("cond", "answer", "false"), ("answer", "reply")],
        ["cx", "rag"],
        scenario="If the knowledge base returns no context for the question, the workflow hands off to a human; "
                 "otherwise it answers strictly from the retrieved context — no hallucinated answers.",
        carbon="No model call at all when retrieval is empty; the answer path uses the greenest grounded rung.",
        analog="LangChain retrieval with a no-context guard."),

    _build(
        "nps-followup", "NPS follow-up personaliser", "Customer Operations",
        "Detractors get an apology + offer; promoters get a thank-you — automatically.",
        [("t", "webhook", {}),
         ("cond", "if", {"left": "{{ $trigger.score }}", "op": "<", "right": "7"}),
         ("apology", "llm", {"prompt": "Write a short apology and a goodwill offer for:\n{{ $trigger.comment }}"}),
         ("thanks", "llm", {"prompt": "Write a warm thank-you referencing:\n{{ $trigger.comment }}"}),
         ("send", "http_request", {"method": "POST", "url": f"{_EX}/email/send"})],
        [("t", "cond"), ("cond", "apology", "true"), ("cond", "thanks", "false"),
         ("apology", "send"), ("thanks", "send")],
        ["cx", "survey"],
        scenario="An NPS response triggers the flow; scores below 7 branch to an apology-and-offer message, higher "
                 "scores to a thank-you, and both merge into a single send step.",
        carbon="Only one of the two message branches ever runs; the other is pruned before any model call.",
        analog="LangGraph conditional branch with a join."),

    _build(
        "appointment-reminders", "Appointment reminder & nudge", "Customer Operations",
        "Each morning, draft friendly reminders for the day's bookings and send them.",
        [("t", "schedule", {"cron": "0 9 * * *"}),
         ("fetch", "http_request", {"method": "GET", "url": f"{_EX}/bookings/today"}),
         ("msg", "llm", {"prompt": "Write a friendly, concise reminder for each booking:\n{{ $node.fetch.body }}"}),
         ("sms", "http_request", {"method": "POST", "url": f"{_EX}/sms/send", "body": "{{ $node.msg.text }}"})],
        [("t", "fetch"), ("fetch", "msg"), ("msg", "sms")],
        ["cx", "scheduled"],
        scenario="A daily schedule pulls the day's bookings, drafts personalised reminders in one pass, and sends "
                 "them over SMS.",
        carbon="One batched drafting call per day on the greenest rung.",
        analog="LangChain scheduled generation."),

    _build(
        "warranty-triage", "Warranty claim triage", "Customer Operations",
        "Validate a warranty claim and route it to RMA or rejection.",
        [("t", "webhook", {}),
         ("cls", "llm", {"prompt": "Is this warranty claim 'valid' or 'invalid'? Give a reason:\n{{ $trigger.claim }}"}),
         ("cond", "if", {"left": "{{ $node.cls.text }}", "op": "contains", "right": "valid"}),
         ("rma", "http_request", {"method": "POST", "url": f"{_EX}/rma/create"}),
         ("reject", "http_request", {"method": "POST", "url": f"{_EX}/claims/reject"})],
        [("t", "cls"), ("cls", "cond"), ("cond", "rma", "true"), ("cond", "reject", "false")],
        ["cx", "triage"],
        scenario="Incoming warranty claims are classified valid/invalid and routed automatically to an RMA or a "
                 "rejection with reason.",
        carbon="A single classification call gates all downstream work.",
        analog="LangChain router chain."),

    # ── Sales & Marketing ────────────────────────────────────────────────────
    _build(
        "meeting-notes-to-crm", "Meeting notes → CRM action items", "Sales & Marketing",
        "Turn a call transcript into structured action items and push them to the CRM.",
        [("t", "webhook", {}),
         ("extract", "llm", {"prompt": "Extract action items with owners and due dates as a list:\n{{ $trigger.transcript }}"}),
         ("crm", "http_request", {"method": "POST", "url": f"{_EX}/crm/tasks", "body": "{{ $node.extract.text }}"})],
        [("t", "extract"), ("extract", "crm")],
        ["sales", "crm"],
        scenario="After a sales call, the transcript is posted to the webhook; the LLM extracts owners, action items "
                 "and dates, and creates the tasks in your CRM.",
        carbon="One extraction call replaces manual note-taking; no large model needed.",
        analog="LangChain extraction chain with an output parser."),

    _build(
        "ad-copy-ab-set", "Ad copy A/B/C generator", "Sales & Marketing",
        "Generate three distinct ad variants in parallel and push them to the ad platform.",
        [("t", "manual", {}),
         ("v1", "llm", {"prompt": "Punchy ad variant for:\n{{ $trigger.brief }}"}),
         ("v2", "llm", {"prompt": "Benefit-led ad variant for:\n{{ $trigger.brief }}"}),
         ("v3", "llm", {"prompt": "Question-hook ad variant for:\n{{ $trigger.brief }}"}),
         ("join", "merge", {}),
         ("push", "http_request", {"method": "POST", "url": f"{_EX}/ads/create"})],
        [("t", "v1"), ("t", "v2"), ("t", "v3"), ("v1", "join"), ("v2", "join"), ("v3", "join"), ("join", "push")],
        ["marketing", "parallel"],
        scenario="From one creative brief, three ad variants are written concurrently (parallel superstep) and merged "
                 "before being pushed to the ad platform for A/B/C testing.",
        carbon="Three small parallel calls finish in the time of one and each reports its own carbon.",
        analog="LangChain RunnableParallel."),

    _build(
        "landing-seo-audit", "Landing-page SEO audit", "Sales & Marketing",
        "Fetch a page and produce an actionable SEO audit with a score.",
        [("t", "webhook", {}),
         ("fetch", "http_request", {"method": "GET", "url": "{{ $trigger.url }}"}),
         ("audit", "llm", {"prompt": "Audit this page for SEO issues and give a 0-100 score with fixes:\n{{ $node.fetch.body }}"}),
         ("report", "http_request", {"method": "POST", "url": f"{_EX}/reports/seo", "body": "{{ $node.audit.text }}"})],
        [("t", "fetch"), ("fetch", "audit"), ("audit", "report")],
        ["marketing", "seo"],
        scenario="Submit a URL and the workflow fetches the page, runs an SEO audit with prioritised fixes and a "
                 "score, and files the report.",
        carbon="One audit call per page on the greenest rung.",
        analog="LangChain tool-augmented analysis chain."),

    _build(
        "newsletter-assembler", "Newsletter assembler", "Sales & Marketing",
        "Curate a newsletter from RSS feeds each Friday, safety-check it, and queue the send.",
        [("t", "schedule", {"cron": "0 7 * * 5"}),
         ("feeds", "http_request", {"method": "GET", "url": f"{_EX}/rss"}),
         ("curate", "llm", {"prompt": "Curate a short, engaging newsletter from these items:\n{{ $node.feeds.body }}"}),
         ("g", "guardrail", {"text": "{{ $node.curate.text }}", "phase": "output"}),
         ("send", "http_request", {"method": "POST", "url": f"{_EX}/esp/queue", "body": "{{ $node.curate.text }}"})],
        [("t", "feeds"), ("feeds", "curate"), ("curate", "g"), ("g", "send", "safe")],
        ["marketing", "scheduled"],
        scenario="A weekly schedule pulls feed items, curates them into a newsletter, runs an output guardrail, and "
                 "queues the send only if it passes.",
        carbon="One curation pass per week; scheduled for a clean grid window.",
        analog="LangChain sequential chain with a safety rail."),

    _build(
        "proposal-draft", "Proposal draft with sign-off", "Sales & Marketing",
        "Draft a proposal grounded in your pricing/case-study KB, then pause for sign-off before sending.",
        [("t", "webhook", {}),
         ("kb", "rag_query", {"query": "{{ $trigger.requirements }}", "top_k": 6}),
         ("draft", "llm", {"prompt": "Draft a proposal using:\n{{ $node.kb.context }}\n\nRequirements: {{ $trigger.requirements }}"}),
         ("review", "approval", {"message": "Approve this proposal to send?\n\n{{ $node.draft.text }}"}),
         ("send", "http_request", {"method": "POST", "url": f"{_EX}/esign/send", "body": "{{ $node.draft.text }}"}),
         ("revise", "transform", {"fields": {"status": "needs_revision"}})],
        [("t", "kb"), ("kb", "draft"), ("draft", "review"),
         ("review", "send", "approved"), ("review", "revise", "rejected")],
        ["sales", "approval", "hitl"],
        scenario="Requirements come in; the workflow retrieves relevant pricing and case studies, drafts a proposal, "
                 "and waits for a human to approve before it goes to e-signature.",
        carbon="Grounded drafting on the greenest rung; the human gate avoids sending wrong drafts.",
        analog="LangGraph interrupt + retrieval chain."),

    _build(
        "webinar-followup", "Webinar follow-up sequencer", "Sales & Marketing",
        "Segment webinar attendees and generate the right follow-up for each segment.",
        [("t", "webhook", {}),
         ("seg", "llm", {"prompt": "Segment these attendees and write a tailored follow-up per segment:\n{{ $trigger.attendees }}"}),
         ("esp", "http_request", {"method": "POST", "url": f"{_EX}/esp/sequence", "body": "{{ $node.seg.text }}"})],
        [("t", "seg"), ("seg", "esp")],
        ["marketing", "sales"],
        scenario="After a webinar, the attendee list is segmented and per-segment follow-ups are generated and pushed "
                 "into the email platform.",
        carbon="One segmentation-and-drafting pass on the greenest rung.",
        analog="LangChain sequential chain."),

    _build(
        "review-response", "Public review responder", "Sales & Marketing",
        "Draft an on-brand reply to a public review, screen it, and post after approval.",
        [("t", "webhook", {}),
         ("g", "guardrail", {"text": "{{ $trigger.review }}", "phase": "input"}),
         ("respond", "llm", {"prompt": "Write an on-brand, empathetic response to this review:\n{{ $trigger.review }}"}),
         ("flag", "transform", {"fields": {"status": "flagged_input"}}),
         ("review", "approval", {"message": "Post this reply?\n\n{{ $node.respond.text }}"}),
         ("post", "http_request", {"method": "POST", "url": f"{_EX}/reviews/reply"}),
         ("skip", "transform", {"fields": {"status": "not_posted"}})],
        [("t", "g"), ("g", "respond", "safe"), ("g", "flag", "blocked"),
         ("respond", "review"), ("review", "post", "approved"), ("review", "skip", "rejected")],
        ["marketing", "approval", "guardrails"],
        scenario="An incoming review is screened; safe ones get an on-brand drafted reply that a human approves "
                 "before posting, while flagged inputs are set aside.",
        carbon="Drafting only happens for safe reviews; the approval gate prevents bad public replies.",
        analog="LangGraph guardrail branch + interrupt."),

    _build(
        "sponsorship-outreach", "Sponsorship outreach personaliser", "Sales & Marketing",
        "Personalise sponsorship outreach for a prospect and send it.",
        [("t", "webhook", {}),
         ("msg", "llm", {"prompt": "Write a personalised sponsorship outreach email for:\n{{ $trigger.prospect }}"}),
         ("send", "http_request", {"method": "POST", "url": f"{_EX}/outreach/send", "body": "{{ $node.msg.text }}"})],
        [("t", "msg"), ("msg", "send")],
        ["marketing", "outreach"],
        scenario="A prospect record triggers a personalised outreach email drafted on the greenest rung and sent via "
                 "your outreach tool.",
        carbon="One small personalisation call per prospect.",
        analog="LangChain generation chain."),

    # ── Software Engineering & DevOps ────────────────────────────────────────
    _build(
        "flaky-test-digest", "Flaky-test detector digest", "Software Engineering & DevOps",
        "Daily: identify flaky tests from CI history and file a tracking issue.",
        [("t", "schedule", {"cron": "0 5 * * *"}),
         ("hist", "http_request", {"method": "GET", "url": f"{_EX}/ci/history"}),
         ("flaky", "llm", {"prompt": "Identify likely flaky tests (intermittent pass/fail) from this history:\n{{ $node.hist.body }}"}),
         ("issue", "http_request", {"method": "POST", "url": f"{_EX}/git/issue", "body": "{{ $node.flaky.text }}"})],
        [("t", "hist"), ("hist", "flaky"), ("flaky", "issue")],
        ["devops", "scheduled"],
        scenario="A nightly job scans CI history, has the model spot intermittently failing tests, and opens a "
                 "tracking issue.",
        carbon="One nightly analysis call, scheduled off-peak.",
        analog="LangChain analysis chain."),

    _build(
        "oncall-handoff", "On-call handoff summary", "Software Engineering & DevOps",
        "Each morning, summarise overnight incidents for the on-call handoff.",
        [("t", "schedule", {"cron": "0 8 * * *"}),
         ("inc", "http_request", {"method": "GET", "url": f"{_EX}/incidents?since=24h"}),
         ("sum", "llm", {"prompt": "Summarise overnight incidents for an on-call handoff (what happened, status, follow-ups):\n{{ $node.inc.body }}"}),
         ("slack", "http_request", {"method": "POST", "url": f"{_EX}/slack", "body": "{{ $node.sum.text }}"})],
        [("t", "inc"), ("inc", "sum"), ("sum", "slack")],
        ["devops", "sre", "scheduled"],
        scenario="A morning schedule collects the last 24h of incidents and posts a concise handoff summary to the "
                 "on-call channel.",
        carbon="One summarisation call per day.",
        analog="LangChain summarisation chain."),

    _build(
        "migration-guide", "Changelog → migration guide", "Software Engineering & DevOps",
        "Turn a set of breaking changes into a step-by-step migration guide.",
        [("t", "webhook", {}),
         ("guide", "llm", {"prompt": "Write a step-by-step migration guide for these breaking changes:\n{{ $trigger.diff }}"}),
         ("docs", "http_request", {"method": "POST", "url": f"{_EX}/docs/publish", "body": "{{ $node.guide.text }}"})],
        [("t", "guide"), ("guide", "docs")],
        ["devops", "docs"],
        scenario="A release's breaking changes are posted; the model produces a migration guide that is published to "
                 "your docs site.",
        carbon="One generation call per release.",
        analog="LangChain generation chain."),

    _build(
        "cloud-cost-anomaly", "Cloud cost anomaly explainer", "Software Engineering & DevOps",
        "Daily: if cloud spend spikes, explain the likely cause and alert.",
        [("t", "schedule", {"cron": "0 6 * * *"}),
         ("bill", "http_request", {"method": "GET", "url": f"{_EX}/billing/daily"}),
         ("cond", "if", {"left": "{{ $node.bill.json.delta_pct }}", "op": ">", "right": "20"}),
         ("explain", "llm", {"prompt": "Explain the likely cause of this cloud-cost spike:\n{{ $node.bill.body }}"}),
         ("alert", "http_request", {"method": "POST", "url": f"{_EX}/finops/alert", "body": "{{ $node.explain.text }}"})],
        [("t", "bill"), ("bill", "cond"), ("cond", "explain", "true"), ("explain", "alert")],
        ["devops", "finops", "scheduled"],
        scenario="A daily check reads billing; only when spend jumps more than 20% does it spend an LLM call to "
                 "explain the cause and alert FinOps.",
        carbon="The model runs only on a real anomaly, not every day.",
        analog="LangChain monitoring agent."),

    _build(
        "log-error-triage", "Log error triage & dedup", "Software Engineering & DevOps",
        "Categorise an error and open a ticket only if it's new.",
        [("t", "webhook", {}),
         ("cat", "llm", {"prompt": "Categorise this error and say 'new' or 'known':\n{{ $trigger.error }}"}),
         ("cond", "if", {"left": "{{ $node.cat.text }}", "op": "contains", "right": "new"}),
         ("ticket", "http_request", {"method": "POST", "url": f"{_EX}/tickets"}),
         ("dedup", "transform", {"fields": {"action": "deduplicated"}})],
        [("t", "cat"), ("cat", "cond"), ("cond", "ticket", "true"), ("cond", "dedup", "false")],
        ["devops", "observability"],
        scenario="Errors stream to the webhook; the model categorises each and only genuinely new ones create a "
                 "ticket, cutting noise.",
        carbon="A single small classification per error; no ticket churn.",
        analog="LangChain router chain."),

    _build(
        "infra-drift", "Infrastructure drift detector", "Software Engineering & DevOps",
        "Every 6 hours: detect Terraform drift, summarise it, and apply only after approval.",
        [("t", "schedule", {"cron": "0 */6 * * *"}),
         ("plan", "http_request", {"method": "GET", "url": f"{_EX}/terraform/plan"}),
         ("cond", "if", {"left": "{{ $node.plan.json.drift }}", "op": "==", "right": "true"}),
         ("sum", "llm", {"prompt": "Summarise this infrastructure drift and its risk:\n{{ $node.plan.body }}"}),
         ("review", "approval", {"message": "Apply these infrastructure changes?\n\n{{ $node.sum.text }}"}),
         ("apply", "http_request", {"method": "POST", "url": f"{_EX}/terraform/apply"}),
         ("ignore", "transform", {"fields": {"action": "left_as_is"}})],
        [("t", "plan"), ("plan", "cond"), ("cond", "sum", "true"),
         ("sum", "review"), ("review", "apply", "approved"), ("review", "ignore", "rejected")],
        ["devops", "approval", "hitl"],
        scenario="A scheduled plan check detects drift; if present, the model summarises it and a human approves "
                 "before any apply — infrastructure never changes unattended.",
        carbon="The model runs only when drift exists; the approval gate prevents risky auto-applies.",
        analog="LangGraph interrupt with a conditional start."),

    _build(
        "postmortem-draft", "Postmortem draft generator", "Software Engineering & DevOps",
        "Draft a blameless postmortem from an incident, grounded in your runbooks, then pause for approval.",
        [("t", "webhook", {}),
         ("kb", "rag_query", {"query": "{{ $trigger.incident }}", "top_k": 6}),
         ("draft", "llm", {"prompt": "Write a blameless postmortem.\nIncident: {{ $trigger.incident }}\nRunbooks: {{ $node.kb.context }}"}),
         ("review", "approval", {"message": "Approve this postmortem for publishing?\n\n{{ $node.draft.text }}"}),
         ("pub", "http_request", {"method": "POST", "url": f"{_EX}/postmortems"}),
         ("hold", "transform", {"fields": {"status": "draft"}})],
        [("t", "kb"), ("kb", "draft"), ("draft", "review"),
         ("review", "pub", "approved"), ("review", "hold", "rejected")],
        ["devops", "sre", "approval"],
        scenario="A closed incident triggers a grounded postmortem draft; an engineer approves before it publishes.",
        carbon="Grounded drafting on the greenest rung, once per incident.",
        analog="LangGraph interrupt + retrieval."),

    _build(
        "license-compliance", "Dependency license scan", "Software Engineering & DevOps",
        "Weekly: flag risky or copyleft licenses in your SBOM.",
        [("t", "schedule", {"cron": "0 4 * * 1"}),
         ("sbom", "http_request", {"method": "GET", "url": f"{_EX}/sbom"}),
         ("flag", "llm", {"prompt": "Flag risky or copyleft licenses and explain the risk:\n{{ $node.sbom.body }}"}),
         ("report", "http_request", {"method": "POST", "url": f"{_EX}/compliance/report", "body": "{{ $node.flag.text }}"})],
        [("t", "sbom"), ("sbom", "flag"), ("flag", "report")],
        ["devops", "compliance", "scheduled"],
        scenario="A weekly scan reads your software bill of materials and flags licenses that need legal attention.",
        carbon="One weekly analysis pass.",
        analog="LangChain analysis chain."),

    _build(
        "api-deprecation-notice", "API deprecation notice", "Software Engineering & DevOps",
        "Draft and email a deprecation notice for an endpoint to affected consumers.",
        [("t", "webhook", {}),
         ("notice", "llm", {"prompt": "Draft a clear deprecation notice (timeline, migration path) for:\n{{ $trigger.endpoint }}"}),
         ("email", "http_request", {"method": "POST", "url": f"{_EX}/email/broadcast", "body": "{{ $node.notice.text }}"})],
        [("t", "notice"), ("notice", "email")],
        ["devops", "docs"],
        scenario="When an endpoint is slated for removal, the workflow drafts a deprecation notice with a timeline "
                 "and migration path and emails affected consumers.",
        carbon="One drafting call per deprecation.",
        analog="LangChain generation chain."),

    _build(
        "secrets-pr-blocker", "Secrets-in-PR blocker", "Software Engineering & DevOps",
        "Screen a PR diff for secrets and pass or fail the status check accordingly.",
        [("t", "webhook", {}),
         ("g", "guardrail", {"text": "{{ $trigger.diff }}", "phase": "input"}),
         ("pass", "http_request", {"method": "POST", "url": f"{_EX}/git/check?state=success"}),
         ("fail", "http_request", {"method": "POST", "url": f"{_EX}/git/check?state=failure"})],
        [("t", "g"), ("g", "pass", "safe"), ("g", "fail", "blocked")],
        ["devops", "security", "guardrails"],
        scenario="A PR diff is screened by the guardrail; a clean diff sets the status check to success, while a "
                 "diff containing secrets fails the check — no model call needed.",
        carbon="Pure guardrail + I/O path: zero model carbon.",
        analog="LangChain guardrail gate."),

    # ── Data & Analytics ─────────────────────────────────────────────────────
    _build(
        "daily-metrics-narrative", "Daily metrics narrative", "Data & Analytics",
        "Turn today's metrics into a plain-English narrative for the team.",
        [("t", "schedule", {"cron": "0 7 * * *"}),
         ("m", "http_request", {"method": "GET", "url": f"{_EX}/metrics/today"}),
         ("narr", "llm", {"prompt": "Write a plain-English narrative of today's KPIs, calling out changes:\n{{ $node.m.body }}"}),
         ("slack", "http_request", {"method": "POST", "url": f"{_EX}/slack", "body": "{{ $node.narr.text }}"})],
        [("t", "m"), ("m", "narr"), ("narr", "slack")],
        ["data", "scheduled"],
        scenario="Every morning the workflow pulls the KPI snapshot and posts a readable narrative — what moved and "
                 "why it matters — to the team channel.",
        carbon="One narrative call per day.",
        analog="LangChain summarisation chain."),

    _build(
        "dataset-drift-monitor", "Dataset drift monitor", "Data & Analytics",
        "Twice daily: if feature drift exceeds a threshold, explain it and alert.",
        [("t", "schedule", {"cron": "0 */12 * * *"}),
         ("stats", "http_request", {"method": "GET", "url": f"{_EX}/ml/feature-stats"}),
         ("cond", "if", {"left": "{{ $node.stats.json.psi }}", "op": ">", "right": "0.2"}),
         ("explain", "llm", {"prompt": "Explain this feature drift and its likely model impact:\n{{ $node.stats.body }}"}),
         ("alert", "http_request", {"method": "POST", "url": f"{_EX}/ml/alert", "body": "{{ $node.explain.text }}"})],
        [("t", "stats"), ("stats", "cond"), ("cond", "explain", "true"), ("explain", "alert")],
        ["data", "ml", "scheduled"],
        scenario="A periodic job checks a population-stability index; only a real drift breach spends a model call to "
                 "explain and alert the ML team.",
        carbon="The model runs only when drift crosses the threshold.",
        analog="LangChain monitoring agent."),

    _build(
        "llm-judge-eval", "LLM output eval (LLM-as-judge)", "Data & Analytics",
        "Generate an answer, then have a judge model score it, and log both with carbon.",
        [("t", "manual", {}),
         ("gen", "llm", {"prompt": "{{ $trigger.prompt }}"}),
         ("judge", "llm", {"prompt": "Score this answer 1-5 for accuracy and explain:\nAnswer: {{ $node.gen.text }}"}),
         ("log", "transform", {"fields": {"answer": "{{ $node.gen.text }}", "score": "{{ $node.judge.text }}",
                                          "gen_co2": "{{ $node.gen.carbon_g }}"}}),
         ("store", "http_request", {"method": "POST", "url": f"{_EX}/eval/log", "body": "{{ $node.judge.text }}"})],
        [("t", "gen"), ("gen", "judge"), ("judge", "log"), ("log", "store")],
        ["data", "eval", "ml"],
        scenario="A prompt is answered by one model and scored by a judge model; the answer, score and the "
                 "generation's carbon are logged for offline evaluation.",
        carbon="Both calls are metered; you can compare quality against carbon per prompt.",
        analog="LangSmith LLM-as-judge evaluation."),

    _build(
        "data-catalog-doc", "Data catalog auto-documentation", "Data & Analytics",
        "Nightly: describe newly created tables/columns and push to the data catalog.",
        [("t", "schedule", {"cron": "0 3 * * *"}),
         ("tables", "http_request", {"method": "GET", "url": f"{_EX}/warehouse/new-tables"}),
         ("desc", "llm", {"prompt": "Write catalog descriptions for these tables and columns:\n{{ $node.tables.body }}"}),
         ("cat", "http_request", {"method": "POST", "url": f"{_EX}/catalog", "body": "{{ $node.desc.text }}"})],
        [("t", "tables"), ("tables", "desc"), ("desc", "cat")],
        ["data", "governance", "scheduled"],
        scenario="A nightly job finds newly created tables and generates human-readable catalog descriptions for "
                 "their columns.",
        carbon="One documentation pass per night for new tables only.",
        analog="LangChain documentation chain."),

    _build(
        "rca-narrative", "Anomaly root-cause narrative", "Data & Analytics",
        "On an alert, hypothesise a root cause grounded in past incidents.",
        [("t", "webhook", {}),
         ("kb", "rag_query", {"query": "{{ $trigger.alert }}", "top_k": 6}),
         ("hyp", "llm", {"prompt": "Hypothesise the root cause.\nAlert: {{ $trigger.alert }}\nPast incidents: {{ $node.kb.context }}"}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/rca", "body": "{{ $node.hyp.text }}"})],
        [("t", "kb"), ("kb", "hyp"), ("hyp", "out")],
        ["data", "sre"],
        scenario="An alert triggers retrieval of similar past incidents and a grounded root-cause hypothesis to "
                 "speed up triage.",
        carbon="One grounded reasoning call per alert.",
        analog="LangChain retrieval-augmented reasoning."),

    _build(
        "survey-open-text-coding", "Survey open-text coding", "Data & Analytics",
        "Code free-text survey responses into themes with counts.",
        [("t", "webhook", {}),
         ("code", "llm", {"prompt": "Code these open-text responses into themes with counts:\n{{ $trigger.responses }}"}),
         ("shape", "transform", {"fields": {"themes": "{{ $node.code.text }}"}}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/survey/coded", "body": "{{ $node.code.text }}"})],
        [("t", "code"), ("code", "shape"), ("shape", "out")],
        ["data", "research"],
        scenario="A batch of open-text survey responses is thematically coded in one pass and the counts exported.",
        carbon="One coding call replaces manual qualitative coding.",
        analog="LangChain classification chain."),

    _build(
        "model-card-generator", "Model card generator", "Data & Analytics",
        "Draft a model card when a model is registered, then pause for approval.",
        [("t", "webhook", {}),
         ("draft", "llm", {"prompt": "Draft a model card (intended use, data, metrics, limitations) for:\n{{ $trigger.model }}"}),
         ("review", "approval", {"message": "Approve this model card?\n\n{{ $node.draft.text }}"}),
         ("reg", "http_request", {"method": "POST", "url": f"{_EX}/registry/card"}),
         ("revise", "transform", {"fields": {"status": "needs_revision"}})],
        [("t", "draft"), ("draft", "review"), ("review", "reg", "approved"), ("review", "revise", "rejected")],
        ["data", "ml", "approval"],
        scenario="Registering a model triggers a model-card draft that a reviewer approves before it's attached in "
                 "the registry.",
        carbon="One drafting call per registered model.",
        analog="LangGraph interrupt + generation."),

    _build(
        "feature-flag-cleanup", "Feature-flag cleanup recommender", "Data & Analytics",
        "Weekly: recommend stale feature flags to remove.",
        [("t", "schedule", {"cron": "0 2 * * 1"}),
         ("flags", "http_request", {"method": "GET", "url": f"{_EX}/flags/usage"}),
         ("rec", "llm", {"prompt": "Recommend stale/unused feature flags to remove, with reasoning:\n{{ $node.flags.body }}"}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/flags/report", "body": "{{ $node.rec.text }}"})],
        [("t", "flags"), ("flags", "rec"), ("rec", "out")],
        ["data", "devops", "scheduled"],
        scenario="A weekly review of flag-usage data recommends which long-lived flags are safe to retire.",
        carbon="One recommendation pass per week.",
        analog="LangChain analysis chain."),

    # ── Finance & Operations ─────────────────────────────────────────────────
    _build(
        "invoice-3way-match", "Invoice 3-way match exception handler", "Finance & Operations",
        "Compare invoice vs PO vs receipt; auto-pay matches and route mismatches to approval.",
        [("t", "webhook", {}),
         ("match", "llm", {"prompt": "Compare invoice, PO and receipt. Say 'match' or 'mismatch' with reasons:\n{{ $trigger.docs }}"}),
         ("cond", "if", {"left": "{{ $node.match.text }}", "op": "contains", "right": "mismatch"}),
         ("review", "approval", {"message": "Approve this mismatched invoice for payment?\n\n{{ $node.match.text }}"}),
         ("autopay", "http_request", {"method": "POST", "url": f"{_EX}/ap/pay"}),
         ("pay", "http_request", {"method": "POST", "url": f"{_EX}/ap/pay-exception"}),
         ("deny", "http_request", {"method": "POST", "url": f"{_EX}/ap/hold"})],
        [("t", "match"), ("match", "cond"), ("cond", "review", "true"), ("cond", "autopay", "false"),
         ("review", "pay", "approved"), ("review", "deny", "rejected")],
        ["finance", "approval", "hitl"],
        scenario="Each invoice is matched against its PO and receipt; clean matches auto-pay, while mismatches pause "
                 "for a human to approve or hold.",
        carbon="One matching call per invoice; the human gate only engages on exceptions.",
        analog="LangGraph conditional branch + interrupt."),

    _build(
        "expense-policy-check", "Expense policy checker", "Finance & Operations",
        "Check an expense against policy; flag violations, auto-approve the rest.",
        [("t", "webhook", {}),
         ("check", "llm", {"prompt": "Check this expense against policy. Say 'compliant' or 'violation' with reason:\n{{ $trigger.expense }}"}),
         ("cond", "if", {"left": "{{ $node.check.text }}", "op": "contains", "right": "violation"}),
         ("flag", "http_request", {"method": "POST", "url": f"{_EX}/expense/flag"}),
         ("approve", "http_request", {"method": "POST", "url": f"{_EX}/expense/approve"})],
        [("t", "check"), ("check", "cond"), ("cond", "flag", "true"), ("cond", "approve", "false")],
        ["finance", "ops"],
        scenario="Submitted expenses are checked against policy; violations are flagged for review while compliant "
                 "ones are auto-approved.",
        carbon="One small policy check per expense.",
        analog="LangChain router chain."),

    _build(
        "vendor-risk-brief", "Vendor risk assessment brief", "Finance & Operations",
        "Enrich a vendor, write a risk brief, and gate onboarding on approval.",
        [("t", "webhook", {}),
         ("enrich", "http_request", {"method": "GET", "url": f"{_EX}/enrich/vendor?name={{ $trigger.vendor }}"}),
         ("brief", "llm", {"prompt": "Write a vendor risk brief (financial, security, compliance) from:\n{{ $node.enrich.body }}"}),
         ("review", "approval", {"message": "Approve onboarding this vendor?\n\n{{ $node.brief.text }}"}),
         ("onboard", "http_request", {"method": "POST", "url": f"{_EX}/vendors/onboard"}),
         ("reject", "transform", {"fields": {"status": "rejected"}})],
        [("t", "enrich"), ("enrich", "brief"), ("brief", "review"),
         ("review", "onboard", "approved"), ("review", "reject", "rejected")],
        ["finance", "procurement", "approval"],
        scenario="A new vendor is enriched with third-party data, a risk brief is written, and onboarding waits for "
                 "human approval.",
        carbon="One brief-generation call per vendor; the enrichment hop carries no model carbon.",
        analog="LangGraph interrupt + tool-augmented chain."),

    _build(
        "contract-clause-extraction", "Contract clause extraction & risk flag", "Finance & Operations",
        "Extract key clauses from a contract and flag risky ones, with an output rail.",
        [("t", "webhook", {}),
         ("extract", "llm", {"prompt": "Extract key clauses (term, liability, IP, termination) and flag risky ones:\n{{ $trigger.contract }}"}),
         ("g", "guardrail", {"text": "{{ $node.extract.text }}", "phase": "output"}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/legal/clauses", "body": "{{ $node.extract.text }}"})],
        [("t", "extract"), ("extract", "g"), ("g", "out", "safe")],
        ["finance", "legal"],
        scenario="A contract is parsed into its key clauses with risk flags; the output passes a safety rail before "
                 "being filed for legal review.",
        carbon="One extraction call per contract.",
        analog="LangChain extraction chain with a rail."),

    _build(
        "cashflow-commentary", "Cash-flow forecast commentary", "Finance & Operations",
        "Monthly: write commentary on the cash-flow forecast.",
        [("t", "schedule", {"cron": "0 6 1 * *"}),
         ("fin", "http_request", {"method": "GET", "url": f"{_EX}/finance/cashflow"}),
         ("comment", "llm", {"prompt": "Write concise commentary on this cash-flow forecast, flagging risks:\n{{ $node.fin.body }}"}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/finance/commentary", "body": "{{ $node.comment.text }}"})],
        [("t", "fin"), ("fin", "comment"), ("comment", "out")],
        ["finance", "scheduled"],
        scenario="On the first of each month the workflow reads the cash-flow forecast and drafts commentary that "
                 "flags risks and drivers.",
        carbon="One monthly commentary call.",
        analog="LangChain summarisation chain."),

    _build(
        "rfq-summarizer", "Procurement RFQ summariser", "Finance & Operations",
        "Compare RFQ responses into a scorecard.",
        [("t", "webhook", {}),
         ("compare", "llm", {"prompt": "Compare these RFQ responses into a scorecard (price, terms, fit):\n{{ $trigger.responses }}"}),
         ("shape", "transform", {"fields": {"scorecard": "{{ $node.compare.text }}"}}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/procurement/scorecard", "body": "{{ $node.compare.text }}"})],
        [("t", "compare"), ("compare", "shape"), ("shape", "out")],
        ["finance", "procurement"],
        scenario="Multiple RFQ responses are summarised into a single comparison scorecard for a purchasing "
                 "decision.",
        carbon="One comparison call per RFQ round.",
        analog="LangChain comparison chain."),

    _build(
        "budget-variance-alert", "Budget variance alerting", "Finance & Operations",
        "Weekly: if actuals diverge from budget, explain the variance and alert.",
        [("t", "schedule", {"cron": "0 6 * * 1"}),
         ("act", "http_request", {"method": "GET", "url": f"{_EX}/finance/actuals"}),
         ("cond", "if", {"left": "{{ $node.act.json.variance_pct }}", "op": ">", "right": "10"}),
         ("explain", "llm", {"prompt": "Explain this budget variance and likely drivers:\n{{ $node.act.body }}"}),
         ("alert", "http_request", {"method": "POST", "url": f"{_EX}/finance/alert", "body": "{{ $node.explain.text }}"})],
        [("t", "act"), ("act", "cond"), ("cond", "explain", "true"), ("explain", "alert")],
        ["finance", "scheduled"],
        scenario="A weekly check compares actuals to budget; only a material variance triggers an explanatory LLM "
                 "call and an alert.",
        carbon="The model runs only when variance exceeds the threshold.",
        analog="LangChain monitoring agent."),

    # ── HR & Legal ───────────────────────────────────────────────────────────
    _build(
        "resume-screening", "Resume screening (bias-guarded)", "HR & Legal",
        "Screen a resume against a JD on skills only, with a guardrail on the input.",
        [("t", "webhook", {}),
         ("g", "guardrail", {"text": "{{ $trigger.resume }}", "phase": "input"}),
         ("score", "llm", {"prompt": "Score this resume 0-100 against the JD on SKILLS ONLY; ignore age, gender, ethnicity, names.\nJD: {{ $trigger.jd }}\nResume: {{ $trigger.resume }}"}),
         ("flag", "transform", {"fields": {"status": "flagged_input"}}),
         ("shape", "transform", {"fields": {"score": "{{ $node.score.text }}"}}),
         ("ats", "http_request", {"method": "POST", "url": f"{_EX}/ats/score", "body": "{{ $node.score.text }}"})],
        [("t", "g"), ("g", "score", "safe"), ("g", "flag", "blocked"), ("score", "shape"), ("shape", "ats")],
        ["hr", "guardrails"],
        scenario="A resume is screened against a job description on skills only, with an explicit instruction to "
                 "ignore demographic signals and a guardrail on the input.",
        carbon="One scoring call per resume on the greenest rung.",
        analog="LangChain scoring chain with a rail."),

    _build(
        "onboarding-orchestrator", "Onboarding task orchestrator", "HR & Legal",
        "Generate a role-specific onboarding checklist and create the tasks.",
        [("t", "webhook", {}),
         ("checklist", "llm", {"prompt": "Generate an onboarding checklist and per-system access tasks for role:\n{{ $trigger.role }}"}),
         ("create", "http_request", {"method": "POST", "url": f"{_EX}/onboarding/tasks", "body": "{{ $node.checklist.text }}"})],
        [("t", "checklist"), ("checklist", "create")],
        ["hr", "ops"],
        scenario="A new-hire record triggers a role-specific onboarding checklist and the creation of the "
                 "corresponding tasks and access requests.",
        carbon="One generation call per hire.",
        analog="LangChain generation chain."),

    _build(
        "policy-qa", "HR policy Q&A assistant", "HR & Legal",
        "Answer an employee policy question from the policy KB, with input and grounding.",
        [("t", "webhook", {}),
         ("g", "guardrail", {"text": "{{ $trigger.question }}", "phase": "input"}),
         ("kb", "rag_query", {"query": "{{ $trigger.question }}", "top_k": 6}),
         ("ans", "llm", {"prompt": "Answer only from policy:\n{{ $node.kb.context }}\n\nQ: {{ $trigger.question }}"}),
         ("reject", "transform", {"fields": {"status": "flagged"}}),
         ("out", "http_request", {"method": "POST", "url": f"{_EX}/hr/answer", "body": "{{ $node.ans.text }}"})],
        [("t", "g"), ("g", "kb", "safe"), ("g", "reject", "blocked"), ("kb", "ans"), ("ans", "out")],
        ["hr", "rag", "guardrails"],
        scenario="Employees ask policy questions; the input is screened, the policy KB is retrieved, and the answer "
                 "is grounded strictly in policy text.",
        carbon="One grounded answer call; no large model needed.",
        analog="LangChain RetrievalQA with a rail."),

    _build(
        "pto-auto-approval", "PTO auto-approval within policy", "HR & Legal",
        "Auto-approve in-policy PTO; route out-of-policy requests to a manager.",
        [("t", "webhook", {}),
         ("check", "llm", {"prompt": "Is this PTO request within policy? Answer 'within' or 'exceeds':\n{{ $trigger.request }}"}),
         ("cond", "if", {"left": "{{ $node.check.text }}", "op": "contains", "right": "within"}),
         ("approve", "http_request", {"method": "POST", "url": f"{_EX}/pto/approve"}),
         ("review", "approval", {"message": "Approve out-of-policy PTO?\n\n{{ $trigger.request }}"}),
         ("mgr_ok", "http_request", {"method": "POST", "url": f"{_EX}/pto/approve"}),
         ("mgr_no", "transform", {"fields": {"status": "declined"}})],
        [("t", "check"), ("check", "cond"), ("cond", "approve", "true"), ("cond", "review", "false"),
         ("review", "mgr_ok", "approved"), ("review", "mgr_no", "rejected")],
        ["hr", "approval", "hitl"],
        scenario="PTO requests within policy are approved automatically; anything exceeding policy pauses for a "
                 "manager's approve/reject decision.",
        carbon="One policy check per request; the human gate only handles exceptions.",
        analog="LangGraph conditional + interrupt."),

    _build(
        "dsar-router", "GDPR data-subject request router", "HR & Legal",
        "Classify a DSAR and route it to the right fulfilment system.",
        [("t", "webhook", {}),
         ("cls", "llm", {"prompt": "Classify this data subject request as access, delete, or rectify:\n{{ $trigger.request }}"}),
         ("route", "http_request", {"method": "POST", "url": f"{_EX}/dsar/route", "body": "{{ $node.cls.text }}"})],
        [("t", "cls"), ("cls", "route")],
        ["legal", "compliance"],
        scenario="An incoming data-subject request is classified (access/delete/rectify) and routed to the correct "
                 "fulfilment workflow.",
        carbon="One classification call per request.",
        analog="LangChain router chain."),

    _build(
        "compliance-training-reminder", "Compliance training reminder", "HR & Legal",
        "Weekly: nudge employees with overdue compliance training.",
        [("t", "schedule", {"cron": "0 9 * * 1"}),
         ("overdue", "http_request", {"method": "GET", "url": f"{_EX}/training/overdue"}),
         ("nudge", "llm", {"prompt": "Write a friendly but firm nudge to complete overdue compliance training:\n{{ $node.overdue.body }}"}),
         ("email", "http_request", {"method": "POST", "url": f"{_EX}/email/send", "body": "{{ $node.nudge.text }}"})],
        [("t", "overdue"), ("overdue", "nudge"), ("nudge", "email")],
        ["hr", "compliance", "scheduled"],
        scenario="A weekly job finds employees with overdue training and emails them a personalised reminder.",
        carbon="One drafting call per week.",
        analog="LangChain generation chain."),

    _build(
        "breach-disclosure-draft", "Breach disclosure drafting", "HR & Legal",
        "Draft a breach disclosure notice from a template, then pause for legal review.",
        [("t", "webhook", {}),
         ("kb", "rag_query", {"query": "breach disclosure notice template", "top_k": 4}),
         ("draft", "llm", {"prompt": "Draft a breach disclosure notice.\nDetails: {{ $trigger.details }}\nTemplate: {{ $node.kb.context }}"}),
         ("review", "approval", {"message": "Approve this breach notice to send to legal?\n\n{{ $node.draft.text }}"}),
         ("legal", "http_request", {"method": "POST", "url": f"{_EX}/legal/notice"}),
         ("hold", "transform", {"fields": {"status": "draft"}})],
        [("t", "kb"), ("kb", "draft"), ("draft", "review"),
         ("review", "legal", "approved"), ("review", "hold", "rejected")],
        ["legal", "approval", "hitl"],
        scenario="A reported breach triggers a disclosure-notice draft grounded in your template; legal approves "
                 "before it moves forward.",
        carbon="One grounded drafting call per incident.",
        analog="LangGraph interrupt + retrieval."),

    # ── IT / Security / Compliance ───────────────────────────────────────────
    _build(
        "phishing-triage", "Phishing report triage", "IT / Security / Compliance",
        "Assess a reported email and block + alert if it's malicious.",
        [("t", "webhook", {}),
         ("g", "guardrail", {"text": "{{ $trigger.email }}", "phase": "input"}),
         ("assess", "llm", {"prompt": "Assess phishing risk ('malicious' or 'benign') with indicators:\n{{ $trigger.email }}"}),
         ("cond", "if", {"left": "{{ $node.assess.text }}", "op": "contains", "right": "malicious"}),
         ("block", "http_request", {"method": "POST", "url": f"{_EX}/security/block"}),
         ("close", "transform", {"fields": {"status": "benign"}})],
        [("t", "g"), ("g", "assess", "safe"), ("g", "block", "blocked"),
         ("assess", "cond"), ("cond", "block", "true"), ("cond", "close", "false")],
        ["security", "guardrails"],
        scenario="A user-reported email is screened; the model assesses phishing risk, and malicious ones (or "
                 "guardrail-blocked content) trigger a block-and-alert action.",
        carbon="One assessment call per report; obviously bad content is blocked without a model call.",
        analog="LangGraph guardrail branch + classification."),

    _build(
        "access-review-cert", "Access review certification", "IT / Security / Compliance",
        "Monthly: summarise anomalous access grants and revoke after approval.",
        [("t", "schedule", {"cron": "0 6 1 * *"}),
         ("ent", "http_request", {"method": "GET", "url": f"{_EX}/iam/entitlements"}),
         ("anom", "llm", {"prompt": "Summarise anomalous or excessive access grants that warrant review:\n{{ $node.ent.body }}"}),
         ("review", "approval", {"message": "Revoke the flagged access grants?\n\n{{ $node.anom.text }}"}),
         ("revoke", "http_request", {"method": "POST", "url": f"{_EX}/iam/revoke"}),
         ("keep", "transform", {"fields": {"status": "retained"}})],
        [("t", "ent"), ("ent", "anom"), ("anom", "review"),
         ("review", "revoke", "approved"), ("review", "keep", "rejected")],
        ["security", "compliance", "approval"],
        scenario="A monthly access review summarises unusual entitlements and pauses for a security owner to approve "
                 "revocation.",
        carbon="One monthly analysis call; the approval gate prevents accidental revokes.",
        analog="LangGraph interrupt + analysis."),

    _build(
        "vuln-prioritization", "Vulnerability prioritisation", "IT / Security / Compliance",
        "Daily: prioritise vulnerabilities by exploitability × asset criticality and open tickets.",
        [("t", "schedule", {"cron": "0 5 * * *"}),
         ("scan", "http_request", {"method": "GET", "url": f"{_EX}/security/scan"}),
         ("crit", "rag_query", {"query": "asset criticality ratings", "top_k": 8}),
         ("prio", "llm", {"prompt": "Prioritise these vulnerabilities by exploitability × asset criticality:\nScan: {{ $node.scan.body }}\nAssets: {{ $node.crit.context }}"}),
         ("tickets", "http_request", {"method": "POST", "url": f"{_EX}/security/tickets", "body": "{{ $node.prio.text }}"})],
        [("t", "scan"), ("scan", "crit"), ("crit", "prio"), ("prio", "tickets")],
        ["security", "scheduled"],
        scenario="A daily scan is combined with asset-criticality context so the model ranks vulnerabilities by real "
                 "business risk before tickets are opened.",
        carbon="One prioritisation call per day, grounded in your asset context.",
        analog="LangChain retrieval-augmented prioritisation."),
]


TEMPLATE_INDEX: dict[str, dict[str, Any]] = {t["id"]: t for t in TEMPLATES}


def list_templates() -> list[dict[str, Any]]:
    """Summaries for the gallery (no full graph)."""
    return [{"id": t["id"], "name": t["name"], "industry": t["industry"],
             "description": t["description"], "tags": t["tags"],
             "node_count": len(t["graph"]["nodes"])} for t in TEMPLATES]


def get_template(tpl_id: str) -> dict[str, Any] | None:
    return TEMPLATE_INDEX.get(tpl_id)


if __name__ == "__main__":
    # Validated at import via _build(); this just reports.
    for t in TEMPLATES:
        assert t["graph"]["nodes"] and t["graph"]["edges"], t["id"]
    industries = sorted({t["industry"] for t in TEMPLATES})
    print(f"{len(TEMPLATES)} templates across {len(industries)} industries — all valid")
    for ind in industries:
        names = [t["name"] for t in TEMPLATES if t["industry"] == ind]
        print(f"  {ind}: {len(names)}")

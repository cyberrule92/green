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
           nodes: list[tuple], edges: list[tuple], tags: list[str] | None = None) -> dict[str, Any]:
    """nodes: (id, type, params). edges: (source, target[, sourceHandle])."""
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
            "description": description, "tags": tags or [], "graph": graph}


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

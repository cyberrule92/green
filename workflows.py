"""Carbon-aware workflow automation engine (n8n / Make / Activepieces style).

A *workflow* is a directed graph of nodes: one or more **triggers** feed a chain
of **action** nodes, connected by edges. The engine walks the graph in
dependency order, passing each node's output downstream, and accumulates the
gCO2 every AI node reports — so an automation carries a carbon receipt end to
end.

Design contract (mirrors ``coding_agent``): this module is **dependency-free and
import-cycle-free**. It never imports ``decision_engine``. The heavy capabilities
— CSS-routed LLM inference, RAG, guardrails, the coding agent, image generation,
grid carbon — are injected as callables via :class:`WorkflowServices`, exactly
the way ``coding_agent.set_inference_fn`` receives its backend. That keeps the
engine unit-testable with stubs (see ``__main__`` smoke test) and lets the same
graph run against real or fake services.

Persistence lives in the same SQLite file as everything else (``green_ai.db``),
two tables: ``workflows`` (the graph) and ``workflow_runs`` (per-node state +
total carbon of one execution).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

logger = logging.getLogger("green.workflows")

# Hard caps: a workflow is authored by a user; keep a runaway graph from
# exhausting the box. These are deliberately generous for a demo but finite.
MAX_NODES = int(os.getenv("WF_MAX_NODES", "60"))
MAX_STEPS = int(os.getenv("WF_MAX_STEPS", "200"))          # node executions per run
MAX_PARALLEL = int(os.getenv("WF_MAX_PARALLEL", "8"))      # concurrent nodes per superstep
MAX_DEPTH = int(os.getenv("WF_MAX_DEPTH", "5"))            # sub-workflow nesting depth
MAX_FOREACH = int(os.getenv("WF_MAX_FOREACH", "100"))      # items a foreach may fan out
HTTP_TIMEOUT_S = float(os.getenv("WF_HTTP_TIMEOUT_S", "20"))
HTTP_MAX_BYTES = int(os.getenv("WF_HTTP_MAX_BYTES", str(512 * 1024)))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
class WorkflowStore:
    """SQLite-backed CRUD for workflows and their run history.

    Same connection/locking idiom as ``conversation_store.ConversationStore`` so
    it shares the WAL database without surprises.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    graph_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT
                );

                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    status TEXT NOT NULL,          -- running | completed | failed
                    trigger TEXT NOT NULL DEFAULT 'manual',
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_carbon_g REAL NOT NULL DEFAULT 0.0,
                    node_states_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_wf_tenant ON workflows(tenant_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_wfrun_wf ON workflow_runs(workflow_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_wfrun_tenant ON workflow_runs(tenant_id, started_at);
                """
            )
            # Backfill columns for human-in-the-loop pause/resume (added later).
            for col, decl in (("engine_state_json", "TEXT"), ("awaiting_json", "TEXT NOT NULL DEFAULT '[]'")):
                try:
                    connection.execute(f"ALTER TABLE workflow_runs ADD COLUMN {col} {decl}")
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ── workflows ────────────────────────────────────────────────────────────
    def create_workflow(
        self,
        name: str,
        graph: dict[str, Any],
        description: str = "",
        tenant_id: str = "default",
        enabled: bool = True,
    ) -> dict[str, Any]:
        wf_id = uuid4().hex[:12]
        now = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO workflows
                   (id, name, description, tenant_id, enabled, graph_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (wf_id, name, description, tenant_id, int(enabled),
                 json.dumps(graph), now, now),
            )
        return self.get_workflow(wf_id, tenant_id=tenant_id)  # type: ignore[return-value]

    def update_workflow(
        self, wf_id: str, tenant_id: str = "default", **fields: Any
    ) -> dict[str, Any] | None:
        allowed = {"name", "description", "enabled", "graph", "last_run_at"}
        sets, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "graph":
                sets.append("graph_json = ?")
                values.append(json.dumps(value))
            elif key == "enabled":
                sets.append("enabled = ?")
                values.append(int(bool(value)))
            else:
                sets.append(f"{key} = ?")
                values.append(value)
        if not sets:
            return self.get_workflow(wf_id, tenant_id=tenant_id)
        sets.append("updated_at = ?")
        values.append(utc_now_iso())
        values.extend([wf_id, tenant_id])
        with self._lock, self._connect() as connection:
            cur = connection.execute(
                f"UPDATE workflows SET {', '.join(sets)} WHERE id = ? AND tenant_id = ?",
                values,
            )
            if cur.rowcount == 0:
                return None
        return self.get_workflow(wf_id, tenant_id=tenant_id)

    def get_workflow(self, wf_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            if tenant_id is None:
                row = connection.execute(
                    "SELECT * FROM workflows WHERE id = ?", (wf_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM workflows WHERE id = ? AND tenant_id = ?",
                    (wf_id, tenant_id),
                ).fetchone()
        return self._row_to_workflow(row) if row else None

    def list_workflows(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            if tenant_id is None:
                rows = connection.execute(
                    "SELECT * FROM workflows ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM workflows WHERE tenant_id = ? ORDER BY updated_at DESC",
                    (tenant_id,),
                ).fetchall()
        return [self._row_to_workflow(r) for r in rows]

    def delete_workflow(self, wf_id: str, tenant_id: str = "default") -> bool:
        with self._lock, self._connect() as connection:
            cur = connection.execute(
                "DELETE FROM workflows WHERE id = ? AND tenant_id = ?", (wf_id, tenant_id)
            )
            return cur.rowcount > 0

    # ── runs ─────────────────────────────────────────────────────────────────
    def create_run(
        self, workflow_id: str, tenant_id: str, trigger: str, node_states: list[dict[str, Any]]
    ) -> dict[str, Any]:
        run_id = uuid4().hex[:12]
        now = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO workflow_runs
                   (id, workflow_id, tenant_id, status, trigger, started_at, node_states_json)
                   VALUES (?, ?, ?, 'running', ?, ?, ?)""",
                (run_id, workflow_id, tenant_id, trigger, now, json.dumps(node_states)),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {"status", "finished_at", "total_carbon_g", "node_states",
                   "error", "engine_state", "awaiting"}
        sets, values = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "node_states":
                sets.append("node_states_json = ?")
                values.append(json.dumps(value))
            elif key == "engine_state":
                sets.append("engine_state_json = ?")
                values.append(json.dumps(value) if value is not None else None)
            elif key == "awaiting":
                sets.append("awaiting_json = ?")
                values.append(json.dumps(value or []))
            else:
                sets.append(f"{key} = ?")
                values.append(value)
        if not sets:
            return
        values.append(run_id)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE workflow_runs SET {', '.join(sets)} WHERE id = ?", values
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(
        self, workflow_id: str | None = None, tenant_id: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if workflow_id:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM workflow_runs {where} ORDER BY started_at DESC LIMIT ?",
                params,
            ).fetchall()
        # Summaries (no per-node payload) for list views.
        return [self._row_to_run(r, include_states=False) for r in rows]

    # ── row mappers ──────────────────────────────────────────────────────────
    @staticmethod
    def _row_to_workflow(row: sqlite3.Row) -> dict[str, Any]:
        try:
            graph = json.loads(row["graph_json"]) or {}
        except json.JSONDecodeError:
            graph = {}
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "tenant_id": row["tenant_id"],
            "enabled": bool(row["enabled"]),
            "graph": graph,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_run_at": row["last_run_at"],
        }

    @staticmethod
    def _row_to_run(row: sqlite3.Row, include_states: bool = True) -> dict[str, Any]:
        out = {
            "id": row["id"],
            "workflow_id": row["workflow_id"],
            "tenant_id": row["tenant_id"],
            "status": row["status"],
            "trigger": row["trigger"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "total_carbon_g": row["total_carbon_g"],
            "error": row["error"],
        }
        keys = row.keys()
        try:
            out["awaiting"] = json.loads(row["awaiting_json"]) if "awaiting_json" in keys and row["awaiting_json"] else []
        except (json.JSONDecodeError, TypeError):
            out["awaiting"] = []
        if include_states:
            try:
                out["node_states"] = json.loads(row["node_states_json"]) or []
            except json.JSONDecodeError:
                out["node_states"] = []
            # engine_state is the resume blob; kept private to the run detail view.
            raw_state = row["engine_state_json"] if "engine_state_json" in keys else None
            try:
                out["engine_state"] = json.loads(raw_state) if raw_state else None
            except json.JSONDecodeError:
                out["engine_state"] = None
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Services (injected)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WorkflowServices:
    """Backend capabilities the engine calls. All optional: an unset service
    makes its node type fail gracefully with a clear message rather than crash.

    ``llm`` is the only async member — it fronts ``process_chat_request`` and must
    return ``{"text", "carbon_g", "model_variant"}``.
    """

    llm: Optional[Callable[..., Awaitable[dict[str, Any]]]] = None
    rag_retrieve: Optional[Callable[..., dict[str, Any]]] = None
    guardrails: Optional[Callable[..., dict[str, Any]]] = None
    agent_run: Optional[Callable[..., dict[str, Any]]] = None
    image_gen: Optional[Callable[..., dict[str, Any]]] = None
    grid_ci: Optional[Callable[[], float]] = None
    # Resolve a saved workflow by id → its record (for the sub-workflow node).
    load_workflow: Optional[Callable[[str], Optional[dict[str, Any]]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Node-type registry (the palette)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NodeType:
    type: str
    label: str
    category: str          # trigger | ai | logic | io
    description: str
    params: list[dict[str, Any]]                    # [{name, label, type, default, options?}]
    handler: Callable[["NodeExec"], Awaitable[dict[str, Any]]]
    is_trigger: bool = False
    handles_out: list[str] = field(default_factory=lambda: ["out"])

    def to_public(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "params": self.params,
            "is_trigger": self.is_trigger,
            "handles_out": self.handles_out,
        }


NODE_TYPES: dict[str, NodeType] = {}


def register(nt: NodeType) -> NodeType:
    NODE_TYPES[nt.type] = nt
    return nt


# ─────────────────────────────────────────────────────────────────────────────
# Template / reference resolution:  {{ $node.<id>.path }}, {{ $input.path }}, ...
# ─────────────────────────────────────────────────────────────────────────────
_TEMPLATE_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")
_PATH_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _dig(obj: Any, path: str) -> Any:
    """Walk a dotted/indexed path (``a.b[0].c``) into nested dict/list."""
    if not path:
        return obj
    cur = obj
    for name, idx in _PATH_TOKEN_RE.findall(path):
        if cur is None:
            return None
        if idx != "":
            try:
                cur = cur[int(idx)]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(name)
            else:
                cur = getattr(cur, name, None)
    return cur


def _resolve_expr(expr: str, scope: dict[str, Any]) -> Any:
    """Resolve a single ``$...`` reference expression against the run scope."""
    expr = expr.strip()
    if expr == "$now":
        return utc_now_iso()
    root_match = re.match(r"^\$(\w+)(?:\.(.*))?$", expr)
    if not root_match:
        return ""  # not a supported reference; empty rather than leak the raw text
    root, rest = root_match.group(1), root_match.group(2) or ""
    if root == "env":
        return os.getenv(rest, "")
    if root == "node":
        # $node.<id>.<path>  → nodes[<id>] output
        first, _, tail = rest.partition(".")
        return _dig((scope.get("nodes") or {}).get(first), tail)
    base = {
        "input": scope.get("input"),
        "json": scope.get("input"),
        "trigger": scope.get("trigger"),
        "vars": scope.get("vars") or {},
    }.get(root)
    if base is None and root not in ("input", "json", "trigger", "vars"):
        return ""
    return _dig(base, rest)


def resolve_templates(value: Any, scope: dict[str, Any]) -> Any:
    """Recursively resolve ``{{ }}`` templates inside strings/dicts/lists.

    A string that is *exactly* one template returns the referenced value with its
    native type (so ``{{ $node.a.count }}`` stays an int); mixed strings get
    string interpolation.
    """
    if isinstance(value, str):
        whole = _TEMPLATE_RE.fullmatch(value.strip())
        if whole:
            return _resolve_expr(whole.group(1), scope)
        return _TEMPLATE_RE.sub(
            lambda m: _stringify(_resolve_expr(m.group(1), scope)), value
        )
    if isinstance(value, dict):
        return {k: resolve_templates(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_templates(v, scope) for v in value]
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Per-node execution context handed to handlers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NodeExec:
    node: dict[str, Any]                 # {id, type, params, ...}
    inputs: dict[str, Any]               # merged upstream output (the primary input)
    scope: dict[str, Any]                # {nodes, input, trigger, vars}
    services: WorkflowServices
    tenant_id: str

    def param(self, name: str, default: Any = None) -> Any:
        raw = (self.node.get("params") or {}).get(name, default)
        return resolve_templates(raw, self.scope)


# ─────────────────────────────────────────────────────────────────────────────
# Node handlers
# ─────────────────────────────────────────────────────────────────────────────
async def _h_passthrough(ex: NodeExec) -> dict[str, Any]:
    # Triggers and merge nodes just forward what reached them.
    return {**ex.inputs, "triggered_at": utc_now_iso()} if ex.inputs else {"triggered_at": utc_now_iso()}


async def _h_llm(ex: NodeExec) -> dict[str, Any]:
    if ex.services.llm is None:
        raise RuntimeError("LLM service not wired")
    prompt = _stringify(ex.param("prompt", "")).strip()
    if not prompt:
        raise ValueError("llm node: empty prompt")
    user_tier = _stringify(ex.param("user_tier", "standard")) or "standard"
    acc = ex.param("accuracy_floor", None)
    try:
        accuracy_floor = float(acc) if acc not in (None, "") else None
    except (TypeError, ValueError):
        accuracy_floor = None
    result = await ex.services.llm(
        prompt=prompt,
        user_tier=user_tier,
        accuracy_floor=accuracy_floor,
        tenant_id=ex.tenant_id,
    )
    return {
        "text": result.get("text", ""),
        "model_variant": result.get("model_variant"),
        "carbon_g": float(result.get("carbon_g", 0.0) or 0.0),
    }


async def _h_rag(ex: NodeExec) -> dict[str, Any]:
    if ex.services.rag_retrieve is None:
        raise RuntimeError("RAG service not wired")
    query = _stringify(ex.param("query", "")).strip()
    top_k = int(ex.param("top_k", 6) or 6)
    payload = ex.services.rag_retrieve(query=query, top_k=top_k, tenant_id=ex.tenant_id)
    return {
        "context": payload.get("context", ""),
        "sources": payload.get("sources", []),
        "retrieved_count": payload.get("retrieved_count", 0),
    }


async def _h_guardrail(ex: NodeExec) -> dict[str, Any]:
    if ex.services.guardrails is None:
        raise RuntimeError("Guardrails service not wired")
    text = _stringify(ex.param("text", ""))
    phase = _stringify(ex.param("phase", "input")) or "input"
    trace = ex.services.guardrails(text=text, phase=phase)
    blocked = bool(trace.get("blocked"))
    return {
        "blocked": blocked,
        "reason": trace.get("reason", ""),
        "warnings": trace.get("warnings", []),
        "text": trace.get("safe_replacement") if blocked else text,
        "_active_handles": {"blocked"} if blocked else {"safe"},
    }


async def _h_agent(ex: NodeExec) -> dict[str, Any]:
    if ex.services.agent_run is None:
        raise RuntimeError("Coding-agent service not wired")
    task = _stringify(ex.param("task", "")).strip()
    if not task:
        raise ValueError("agent node: empty task")
    tests = ex.param("tests", None) or None
    budget = ex.param("carbon_budget_g", None)
    try:
        carbon_budget_g = float(budget) if budget not in (None, "") else None
    except (TypeError, ValueError):
        carbon_budget_g = None
    result = ex.services.agent_run(
        task=task,
        tests=tests if tests else None,
        carbon_budget_g=carbon_budget_g,
        allow_defer=bool(ex.param("allow_defer", False)),
    )
    return {
        "status": result.get("status"),
        "task_id": result.get("task_id"),
        "carbon_g": float(result.get("carbon_g", result.get("carbon_per_completion_g", 0.0)) or 0.0),
        "files": result.get("files", {}),
        "spec_source": result.get("spec_source"),
    }


async def _h_image(ex: NodeExec) -> dict[str, Any]:
    if ex.services.image_gen is None:
        raise RuntimeError("Image-generation service not wired")
    prompt = _stringify(ex.param("prompt", "")).strip()
    steps = int(ex.param("steps", 24) or 24)
    result = ex.services.image_gen(prompt=prompt, steps=steps)
    return {
        "image": result.get("image"),          # data URI
        "carbon_g": float(result.get("carbon_g", 0.0) or 0.0),
        "backend": result.get("backend"),
    }


async def _h_http(ex: NodeExec) -> dict[str, Any]:
    method = (_stringify(ex.param("method", "GET")) or "GET").upper()
    url = _stringify(ex.param("url", "")).strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        raise ValueError("http node: url must be http(s)")
    headers = ex.param("headers", {}) or {}
    if isinstance(headers, str):
        try:
            headers = json.loads(headers) if headers.strip() else {}
        except json.JSONDecodeError:
            headers = {}
    body = ex.param("body", None)
    data = None
    if body not in (None, ""):
        data = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    def _do_request() -> dict[str, Any]:
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={str(k): str(v) for k, v in headers.items()})
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read(HTTP_MAX_BYTES)
                status = resp.status
        except urllib.error.HTTPError as err:
            raw = err.read(HTTP_MAX_BYTES)
            status = err.code
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return {"status": status, "json": parsed, "body": text}

    return await asyncio.to_thread(_do_request)


async def _h_transform(ex: NodeExec) -> dict[str, Any]:
    # "fields" is a map of {outputKey: templateValue}; each value is resolved.
    fields = (ex.node.get("params") or {}).get("fields", {})
    if not isinstance(fields, dict):
        return dict(ex.inputs)
    resolved = {k: resolve_templates(v, ex.scope) for k, v in fields.items()}
    if bool((ex.node.get("params") or {}).get("passthrough", False)):
        return {**ex.inputs, **resolved}
    return resolved


_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: _stringify(a) == _stringify(b),
    "!=": lambda a, b: _stringify(a) != _stringify(b),
    "contains": lambda a, b: _stringify(b) in _stringify(a),
    "empty": lambda a, b: _stringify(a).strip() == "",
    "not_empty": lambda a, b: _stringify(a).strip() != "",
    ">": lambda a, b: _num(a) > _num(b),
    "<": lambda a, b: _num(a) < _num(b),
}


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


async def _h_if(ex: NodeExec) -> dict[str, Any]:
    left = ex.param("left", "")
    op = _stringify(ex.param("op", "==")) or "=="
    right = ex.param("right", "")
    fn = _OPS.get(op, _OPS["=="])
    result = bool(fn(left, right))
    return {"result": result, "_active_handles": {"true"} if result else {"false"}}


async def _h_carbon_gate(ex: NodeExec) -> dict[str, Any]:
    """Branch on live grid carbon: below threshold → 'green', else 'dirty'.
    Lets a workflow itself decide to run heavy AI only when the grid is clean."""
    threshold = float(ex.param("threshold_g", 300) or 300)
    ci = 400.0
    if ex.services.grid_ci is not None:
        try:
            ci = float(ex.services.grid_ci())
        except Exception:  # noqa: BLE001 - telemetry best-effort
            ci = 400.0
    green = ci <= threshold
    return {
        "grid_ci": ci,
        "green": green,
        "_active_handles": {"green"} if green else {"dirty"},
    }


async def _h_subworkflow(ex: NodeExec) -> dict[str, Any]:
    """Run another saved workflow as a node — the composition primitive that
    makes this an orchestrator (LangGraph subgraph parity). With ``items`` set to
    a list template it fans the sub-workflow out once per item (map/batch
    semantics) and sums the carbon; otherwise it runs once with the current
    input. Depth-guarded to prevent unbounded recursion."""
    if ex.services.load_workflow is None:
        raise RuntimeError("sub-workflow service not wired")
    wid = _stringify(ex.param("workflow_id", "")).strip()
    if not wid:
        raise ValueError("subworkflow node: workflow_id required")
    depth = int(ex.scope.get("_depth", 0))
    if depth >= MAX_DEPTH:
        raise RuntimeError(f"sub-workflow nesting exceeds depth limit ({MAX_DEPTH})")
    wf = ex.services.load_workflow(wid)
    if not wf:
        raise ValueError(f"subworkflow node: unknown workflow {wid}")

    engine = WorkflowEngine(ex.services)
    items = ex.param("items", None)
    if isinstance(items, list) and items:
        results, carbon = [], 0.0
        for item in items[:MAX_FOREACH]:
            trig = item if isinstance(item, dict) else {"item": item}
            sub = await engine.execute(wf["graph"], trigger_input=trig,
                                       tenant_id=ex.tenant_id, _depth=depth + 1)
            carbon += float(sub.get("total_carbon_g", 0.0) or 0.0)
            results.append({"status": sub["status"], "carbon_g": sub["total_carbon_g"]})
        return {"results": results, "count": len(results), "carbon_g": round(carbon, 6)}

    sub = await engine.execute(wf["graph"], trigger_input=ex.inputs,
                               tenant_id=ex.tenant_id, _depth=depth + 1)
    return {
        "status": sub["status"],
        "carbon_g": float(sub.get("total_carbon_g", 0.0) or 0.0),
        "error": sub.get("error"),
        "nodes": sub.get("node_states", []),
    }


# Register the palette.
register(NodeType("manual", "Manual trigger", "trigger",
                  "Run on demand (or from the Run button). Emits the supplied input.",
                  [], _h_passthrough, is_trigger=True))
register(NodeType("webhook", "Webhook trigger", "trigger",
                  "Fires on POST /api/workflows/webhook/{id}; the request body becomes the trigger output.",
                  [], _h_passthrough, is_trigger=True))
register(NodeType("schedule", "Schedule trigger", "trigger",
                  "Fires on a cron schedule while the workflow is enabled.",
                  [{"name": "cron", "label": "Cron (m h dom mon dow)", "type": "text", "default": "0 * * * *"}],
                  _h_passthrough, is_trigger=True))
register(NodeType("carbon_window", "Carbon-window trigger", "trigger",
                  "Fires when grid carbon drops below the threshold (defers work to clean windows).",
                  [{"name": "threshold_g", "label": "Max gCO2/kWh", "type": "number", "default": 250}],
                  _h_passthrough, is_trigger=True))

register(NodeType("llm", "LLM (carbon-routed)", "ai",
                  "Chat completion through the CSS greenest-feasible router. Reports gCO2.",
                  [{"name": "prompt", "label": "Prompt", "type": "textarea", "default": ""},
                   {"name": "user_tier", "label": "Tier", "type": "select",
                    "options": ["standard", "premium", "esg", "batch"], "default": "standard"},
                   {"name": "accuracy_floor", "label": "Accuracy floor (0-1, optional)", "type": "text", "default": ""}],
                  _h_llm))
register(NodeType("rag_query", "RAG retrieve", "ai",
                  "Hybrid dense+sparse retrieval + rerank against the indexed knowledge base.",
                  [{"name": "query", "label": "Query", "type": "textarea", "default": ""},
                   {"name": "top_k", "label": "Top K", "type": "number", "default": 6}],
                  _h_rag))
register(NodeType("agent_task", "Coding agent", "ai",
                  "Run one carbon-budgeted agentic coding task; escalates only on verifier evidence.",
                  [{"name": "task", "label": "Task", "type": "textarea", "default": ""},
                   {"name": "tests", "label": "Frozen tests (pytest source, optional)", "type": "textarea", "default": ""},
                   {"name": "carbon_budget_g", "label": "Carbon budget gCO2 (optional)", "type": "text", "default": ""}],
                  _h_agent))
register(NodeType("guardrail", "Guardrail check", "ai",
                  "NemoGuardrails safety rail. Branches: safe / blocked.",
                  [{"name": "text", "label": "Text", "type": "textarea", "default": "{{ $json.text }}"},
                   {"name": "phase", "label": "Phase", "type": "select",
                    "options": ["input", "output"], "default": "input"}],
                  _h_guardrail, handles_out=["safe", "blocked"]))
register(NodeType("image_gen", "Image generation", "ai",
                  "Carbon-capped diffusion via pluggable NIM endpoint (graceful placeholder fallback).",
                  [{"name": "prompt", "label": "Prompt", "type": "textarea", "default": ""},
                   {"name": "steps", "label": "Denoise steps", "type": "number", "default": 24}],
                  _h_image))

register(NodeType("http_request", "HTTP request", "io",
                  "Call an external HTTP(S) API. Body/headers support templates.",
                  [{"name": "method", "label": "Method", "type": "select",
                    "options": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"},
                   {"name": "url", "label": "URL", "type": "text", "default": ""},
                   {"name": "headers", "label": "Headers (JSON)", "type": "textarea", "default": "{}"},
                   {"name": "body", "label": "Body", "type": "textarea", "default": ""}],
                  _h_http))
register(NodeType("transform", "Set / transform", "logic",
                  "Build an output object from templated fields.",
                  [{"name": "fields", "label": "Fields (key → template)", "type": "keyvalue", "default": {}},
                   {"name": "passthrough", "label": "Merge with input", "type": "boolean", "default": False}],
                  _h_transform))
register(NodeType("if", "IF condition", "logic",
                  "Branch on a comparison. Handles: true / false.",
                  [{"name": "left", "label": "Value", "type": "text", "default": ""},
                   {"name": "op", "label": "Operator", "type": "select",
                    "options": ["==", "!=", "contains", ">", "<", "empty", "not_empty"], "default": "=="},
                   {"name": "right", "label": "Compare to", "type": "text", "default": ""}],
                  _h_if, handles_out=["true", "false"]))
register(NodeType("carbon_gate", "Carbon gate", "logic",
                  "Branch on live grid carbon. Handles: green / dirty.",
                  [{"name": "threshold_g", "label": "Max gCO2/kWh", "type": "number", "default": 300}],
                  _h_carbon_gate, handles_out=["green", "dirty"]))
register(NodeType("merge", "Merge", "logic",
                  "Join branches; forwards merged upstream output.",
                  [], _h_passthrough))
register(NodeType("subworkflow", "Sub-workflow", "logic",
                  "Run another saved workflow as a node. Set 'items' to a list template to fan out (map).",
                  [{"name": "workflow_id", "label": "Workflow ID", "type": "text", "default": ""},
                   {"name": "items", "label": "Items (list template, optional → map)", "type": "text", "default": ""}],
                  _h_subworkflow))


async def _h_approval(ex: NodeExec) -> dict[str, Any]:
    # The engine intercepts approval nodes to pause the run; this only fires if
    # one is somehow executed without a recorded decision.
    raise RuntimeError("approval node reached without a decision")


register(NodeType("approval", "Human approval", "logic",
                  "Pause the run for a human to approve or reject (LangGraph interrupt parity). "
                  "Handles: approved / rejected.",
                  [{"name": "message", "label": "Message shown to the approver", "type": "textarea", "default": ""}],
                  _h_approval, handles_out=["approved", "rejected"]))


# ─────────────────────────────────────────────────────────────────────────────
# Graph validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_graph(graph: dict[str, Any]) -> None:
    """Raise ``ValueError`` if the graph can't run. Called before any carbon is
    spent (on create/update/run), so a broken flow fails at 400, not mid-run."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("graph must have list 'nodes' and 'edges'")
    if len(nodes) == 0:
        raise ValueError("workflow has no nodes")
    if len(nodes) > MAX_NODES:
        raise ValueError(f"too many nodes (max {MAX_NODES})")

    ids: set[str] = set()
    triggers = 0
    for n in nodes:
        nid = n.get("id")
        if not nid or nid in ids:
            raise ValueError(f"node id missing or duplicate: {nid!r}")
        ids.add(nid)
        ntype = n.get("type")
        if ntype not in NODE_TYPES:
            raise ValueError(f"unknown node type: {ntype!r}")
        if NODE_TYPES[ntype].is_trigger:
            triggers += 1
    if triggers == 0:
        raise ValueError("workflow needs at least one trigger node")

    adj: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s not in ids or t not in ids:
            raise ValueError(f"edge references unknown node: {s} -> {t}")
        adj[s].append(t)

    # Cycle detection (DFS colouring).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in ids}

    def dfs(u: str) -> None:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                raise ValueError("workflow graph has a cycle")
            if color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for i in ids:
        if color[i] == WHITE:
            dfs(i)


# ─────────────────────────────────────────────────────────────────────────────
# Execution engine
# ─────────────────────────────────────────────────────────────────────────────
class WorkflowEngine:
    def __init__(self, services: WorkflowServices):
        self.services = services

    async def execute(
        self,
        graph: dict[str, Any],
        trigger_input: Any = None,
        tenant_id: str = "default",
        on_progress: Optional[Callable[[list[dict[str, Any]], float], None]] = None,
        _depth: int = 0,
        resume_state: Optional[dict[str, Any]] = None,
        approvals: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Run (or resume) a validated graph. Returns
        ``{status, total_carbon_g, node_states, error}`` and, when it stops at an
        ``approval`` node, ``status == "paused"`` plus ``awaiting`` and ``state``.

        **Superstep execution (LangGraph-style):** every node whose inputs are all
        resolved runs in the *same* superstep, and independent nodes in a superstep
        run **concurrently** (``asyncio.gather`` under ``MAX_PARALLEL``). A node's
        outgoing edge is active iff the source is active and the edge's source
        handle matches the branch it chose (``_active_handles``); a downstream node
        runs once *all* its incoming edges resolve and *any* is active, else it is
        skipped — so branches prune and joins wait for every arm.

        **Resilience:** per-node ``retries``/``retry_backoff_s``/``timeout_s`` and
        ``on_error`` = ``stop``|``continue``.

        **Human-in-the-loop (LangGraph interrupt parity):** reaching an ``approval``
        node with no recorded decision pauses the whole run and snapshots its state;
        :meth:`resume` (via ``resume_state`` + ``approvals``) continues it once a
        human approves/rejects. Edges are keyed by stable index so the state is
        JSON-serialisable across the pause.
        """
        validate_graph(graph)
        nodes = {n["id"]: n for n in graph["nodes"]}
        edge_list: list[dict[str, Any]] = list(graph.get("edges") or [])

        out_edges: dict[str, list[int]] = {nid: [] for nid in nodes}
        in_edges: dict[str, list[int]] = {nid: [] for nid in nodes}
        for idx, e in enumerate(edge_list):
            out_edges[e["source"]].append(idx)
            in_edges[e["target"]].append(idx)

        approvals = approvals or {}
        sem = asyncio.Semaphore(max(1, MAX_PARALLEL))

        def _blank_state(nid: str) -> dict[str, Any]:
            n = nodes[nid]
            return {"id": nid, "type": n.get("type"), "label": n.get("label") or n.get("type"),
                    "status": "pending", "output": None, "carbon_g": 0.0,
                    "error": None, "attempts": 0}

        if resume_state:
            node_outputs: dict[str, Any] = dict(resume_state.get("node_outputs", {}))
            states = {s["id"]: dict(s) for s in resume_state.get("states", [])}
            for nid in nodes:
                states.setdefault(nid, _blank_state(nid))
            active_node: dict[str, bool] = dict(resume_state.get("active_node", {}))
            for nid in nodes:
                active_node.setdefault(nid, False)
            edge_active: dict[int, bool] = {int(k): v for k, v in resume_state.get("edge_active", {}).items()}
            edge_resolved: dict[int, bool] = {int(k): v for k, v in resume_state.get("edge_resolved", {}).items()}
            ready: list[str] = list(resume_state.get("ready", []))
            steps = int(resume_state.get("steps", 0))
            total_carbon = float(resume_state.get("total_carbon", 0.0))
            run_error: str | None = resume_state.get("run_error")
            trigger_input = resume_state.get("trigger_input", trigger_input)
            # Apply decisions: an awaiting approval node becomes runnable again.
            for nid, dec in approvals.items():
                if nid in nodes and states.get(nid, {}).get("status") == "awaiting_approval":
                    states[nid]["status"] = "pending"
                    if nid not in ready:
                        ready.append(nid)
        else:
            node_outputs = {}
            states = {nid: _blank_state(nid) for nid in nodes}
            active_node = {nid: NODE_TYPES[nodes[nid]["type"]].is_trigger for nid in nodes}
            edge_active = {}
            edge_resolved = {}
            ready = [nid for nid, n in nodes.items() if NODE_TYPES[n["type"]].is_trigger]
            steps = 0
            total_carbon = 0.0
            run_error = None

        def _emit() -> None:
            if on_progress:
                try:
                    on_progress([states[nid] for nid in nodes], round(total_carbon, 6))
                except Exception:  # noqa: BLE001 progress is best-effort
                    logger.debug("workflow progress callback failed", exc_info=True)

        def _merged_input(nid: str) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for idx in in_edges[nid]:
                src = edge_list[idx]["source"]
                if edge_active.get(idx) and isinstance(node_outputs.get(src), dict):
                    merged.update({k: v for k, v in node_outputs[src].items() if not k.startswith("_")})
            if not in_edges[nid] and NODE_TYPES[nodes[nid]["type"]].is_trigger:
                merged = trigger_input if isinstance(trigger_input, dict) else (
                    {"value": trigger_input} if trigger_input is not None else {})
            return merged

        async def _run_node(nid: str) -> None:
            """Execute one active node with retry/timeout, mutating shared state."""
            nonlocal total_carbon, run_error
            node = nodes[nid]
            ntype = NODE_TYPES[node["type"]]
            merged = _merged_input(nid)
            scope = {"nodes": node_outputs, "input": merged,
                     "trigger": trigger_input, "vars": {}, "_depth": _depth}
            ex = NodeExec(node=node, inputs=merged, scope=scope,
                          services=self.services, tenant_id=tenant_id)

            retries = _int(node.get("retries"), 0)
            backoff = _float(node.get("retry_backoff_s"), 0.5)
            timeout = node.get("timeout_s")
            timeout_s = _float(timeout, 0.0) if timeout not in (None, "") else None
            on_error = str(node.get("on_error", "stop")).lower()

            states[nid]["status"] = "running"
            _emit()
            attempt, last_exc = 0, None
            async with sem:
                while attempt <= max(0, retries):
                    attempt += 1
                    states[nid]["attempts"] = attempt
                    try:
                        coro = ntype.handler(ex)
                        output = await (asyncio.wait_for(coro, timeout_s) if timeout_s else coro)
                        if not isinstance(output, dict):
                            output = {"value": output}
                        node_outputs[nid] = output
                        carbon = float(output.get("carbon_g", 0.0) or 0.0)
                        total_carbon += carbon
                        states[nid]["status"] = "completed"
                        states[nid]["output"] = _truncate_output(output)
                        states[nid]["carbon_g"] = carbon
                        return
                    except Exception as exc:  # noqa: BLE001 - retryable node failure
                        last_exc = exc
                        if attempt <= retries:
                            await asyncio.sleep(backoff * attempt)
            states[nid]["status"] = "failed"
            states[nid]["error"] = f"{type(last_exc).__name__}: {last_exc}"
            node_outputs[nid] = {}
            logger.warning("workflow node %s (%s) failed after %d attempt(s): %s",
                           nid, node.get("type"), attempt, last_exc)
            if on_error != "continue":
                run_error = run_error or states[nid]["error"]

        def _resolve_out_edges(nid: str) -> None:
            node = nodes[nid]
            src_out = node_outputs.get(nid, {})
            active_handles = src_out.get("_active_handles") if isinstance(src_out, dict) else None
            hard_failed = (states[nid]["status"] == "failed"
                           and str(node.get("on_error", "stop")).lower() != "continue")
            for idx in out_edges[nid]:
                e = edge_list[idx]
                handle = e.get("sourceHandle")
                if hard_failed or not active_node[nid]:
                    is_active = False
                elif active_handles is None:
                    is_active = True                     # non-branching node: all edges live
                else:
                    is_active = handle in active_handles  # branch node: only its handle
                edge_active[idx] = is_active
                edge_resolved[idx] = True
                tgt = e["target"]
                if all(edge_resolved.get(ie) for ie in in_edges[tgt]):
                    if states[tgt]["status"] == "pending" and tgt not in ready:
                        active_node[tgt] = any(edge_active.get(ie) for ie in in_edges[tgt])
                        ready.append(tgt)

        awaiting: list[str] = []
        while ready:
            batch = [nid for nid in ready if states[nid]["status"] == "pending"]
            ready = []
            if not batch:
                break
            steps += len(batch)
            if steps > MAX_STEPS:
                run_error = f"step budget exceeded ({MAX_STEPS})"
                break

            active_batch: list[str] = []
            for nid in batch:
                if not active_node[nid]:
                    states[nid]["status"] = "skipped"
                    continue
                if nodes[nid]["type"] == "approval":
                    decision = approvals.get(nid)
                    if decision is None:
                        # Interrupt: pause the run here.
                        states[nid]["status"] = "awaiting_approval"
                        awaiting.append(nid)
                        continue
                    approved = bool(decision.get("approved"))
                    out = {"approved": approved, "note": decision.get("note", ""),
                           "decided_by": decision.get("by", ""),
                           "_active_handles": {"approved" if approved else "rejected"}}
                    node_outputs[nid] = out
                    states[nid]["status"] = "completed"
                    states[nid]["output"] = _truncate_output(out)
                    continue
                active_batch.append(nid)

            if active_batch:
                await asyncio.gather(*[_run_node(nid) for nid in active_batch])
            _emit()
            if run_error and "step budget" in run_error:
                break

            for nid in batch:
                if states[nid]["status"] == "awaiting_approval":
                    continue  # edges resolve only once a decision arrives
                _resolve_out_edges(nid)

            if awaiting:
                # Snapshot and pause. Awaiting nodes are re-queued so resume picks
                # them up once decisions are supplied.
                snap_ready = list(dict.fromkeys(ready + awaiting))
                state_blob = {
                    "node_outputs": node_outputs,
                    "states": [states[nid] for nid in nodes],
                    "active_node": active_node,
                    "edge_active": {str(k): v for k, v in edge_active.items()},
                    "edge_resolved": {str(k): v for k, v in edge_resolved.items()},
                    "ready": snap_ready,
                    "steps": steps,
                    "total_carbon": total_carbon,
                    "run_error": run_error,
                    "trigger_input": trigger_input,
                }
                pending_approvals = []
                for nid in awaiting:
                    msg = (nodes[nid].get("params") or {}).get("message", "")
                    msg = resolve_templates(msg, {"nodes": node_outputs, "input": {},
                                                  "trigger": trigger_input, "vars": {}})
                    pending_approvals.append({"id": nid, "label": states[nid]["label"],
                                              "message": _stringify(msg)})
                _emit()
                return {
                    "status": "paused",
                    "total_carbon_g": round(total_carbon, 6),
                    "node_states": [states[nid] for nid in nodes],
                    "error": run_error,
                    "awaiting": pending_approvals,
                    "state": state_blob,
                }

        for nid in nodes:
            if states[nid]["status"] == "pending":
                states[nid]["status"] = "skipped"

        status = "failed" if run_error else "completed"
        return {
            "status": status,
            "total_carbon_g": round(total_carbon, 6),
            "node_states": [states[nid] for nid in nodes],
            "error": run_error,
        }


_MAX_OUTPUT_CHARS = 4000


def _truncate_output(output: dict[str, Any]) -> dict[str, Any]:
    """Keep persisted per-node output small; long text fields get clipped."""
    clipped: dict[str, Any] = {}
    for k, v in output.items():
        if k.startswith("_"):
            continue
        if isinstance(v, str) and len(v) > _MAX_OUTPUT_CHARS:
            clipped[k] = v[:_MAX_OUTPUT_CHARS] + f"… (+{len(v) - _MAX_OUTPUT_CHARS} chars)"
        else:
            clipped[k] = v
    return clipped


# ─────────────────────────────────────────────────────────────────────────────
# Trigger helpers (used by the scheduler in decision_engine)
# ─────────────────────────────────────────────────────────────────────────────
def _cron_field_matches(field_spec: str, value: int) -> bool:
    field_spec = field_spec.strip()
    if field_spec == "*":
        return True
    for part in field_spec.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            rng, _, step_s = part.partition("/")
            step = int(step_s) if step_s.isdigit() else 1
            part = rng or "*"
        if part == "*":
            if value % step == 0:
                return True
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                if lo <= value <= hi and (value - lo) % step == 0:
                    return True
        elif part.isdigit() and int(part) == value:
            return True
    return False


def cron_matches(cron: str, when: datetime) -> bool:
    """Minimal 5-field cron matcher (min hour dom mon dow). ``dow`` 0/7 = Sunday.
    Not a full crontab implementation — supports ``*``, lists, ranges, ``*/n``."""
    parts = cron.split()
    if len(parts) != 5:
        return False
    minute, hour, dom, mon, dow = parts
    wd = (when.weekday() + 1) % 7  # Python Mon=0 → cron Sun=0
    return (
        _cron_field_matches(minute, when.minute)
        and _cron_field_matches(hour, when.hour)
        and _cron_field_matches(dom, when.day)
        and _cron_field_matches(mon, when.month)
        and (_cron_field_matches(dow, wd) or _cron_field_matches(dow, 7 if wd == 0 else wd))
    )


def node_types_public() -> list[dict[str, Any]]:
    return [nt.to_public() for nt in NODE_TYPES.values()]


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (no external services): python workflows.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _fake_llm(prompt: str, **_: Any) -> dict[str, Any]:
        return {"text": f"echo: {prompt}", "model_variant": "small", "carbon_g": 0.012}

    services = WorkflowServices(
        llm=_fake_llm,
        grid_ci=lambda: 120.0,
    )
    engine = WorkflowEngine(services)

    graph = {
        "nodes": [
            {"id": "t", "type": "manual", "params": {}},
            {"id": "gate", "type": "carbon_gate", "params": {"threshold_g": 300}},
            {"id": "ask", "type": "llm", "params": {"prompt": "Summarise: {{ $trigger.topic }}"}},
            {"id": "skipme", "type": "llm", "params": {"prompt": "grid too dirty"}},
            {"id": "done", "type": "transform",
             "params": {"fields": {"answer": "{{ $node.ask.text }}", "co2": "{{ $node.ask.carbon_g }}"}}},
        ],
        "edges": [
            {"source": "t", "target": "gate"},
            {"source": "gate", "target": "ask", "sourceHandle": "green"},
            {"source": "gate", "target": "skipme", "sourceHandle": "dirty"},
            {"source": "ask", "target": "done"},
        ],
    }

    result = asyncio.run(engine.execute(graph, trigger_input={"topic": "green AI"}))
    print(json.dumps(result, indent=2))
    assert result["status"] == "completed", result
    by_id = {s["id"]: s for s in result["node_states"]}
    assert by_id["ask"]["status"] == "completed", by_id["ask"]
    assert by_id["skipme"]["status"] == "skipped", by_id["skipme"]
    assert by_id["done"]["output"]["answer"] == "echo: Summarise: green AI", by_id["done"]
    assert abs(result["total_carbon_g"] - 0.012) < 1e-9, result["total_carbon_g"]
    print("\n✓ branch/template/carbon smoke test passed")

    # ── Parallel fan-out + retries + continue-on-error ──────────────────────
    import time as _t
    _flaky_calls = {"n": 0}
    _order: list[tuple[str, float]] = []

    async def _slow_llm(prompt: str, **_: Any) -> dict[str, Any]:
        _order.append((prompt, _t.perf_counter()))
        await asyncio.sleep(0.2)
        return {"text": f"ok:{prompt}", "carbon_g": 0.01}

    async def _flaky(ex: NodeExec) -> dict[str, Any]:
        _flaky_calls["n"] += 1
        if _flaky_calls["n"] < 3:
            raise RuntimeError("transient")
        return {"ok": True}

    NODE_TYPES["_flaky"] = NodeType("_flaky", "Flaky", "logic", "test", [], _flaky)
    engine2 = WorkflowEngine(WorkflowServices(llm=_slow_llm))
    g2 = {
        "nodes": [
            {"id": "t", "type": "manual", "params": {}},
            {"id": "a", "type": "llm", "params": {"prompt": "A"}},           # parallel
            {"id": "b", "type": "llm", "params": {"prompt": "B"}},           # parallel
            {"id": "r", "type": "_flaky", "retries": 3, "retry_backoff_s": 0.01},
            {"id": "boom", "type": "http_request", "params": {"url": "not-a-url"},
             "on_error": "continue"},
            {"id": "after", "type": "transform", "params": {"fields": {"done": "yes"}}},
            {"id": "join", "type": "merge", "params": {}},
        ],
        "edges": [
            {"source": "t", "target": "a"}, {"source": "t", "target": "b"},
            {"source": "t", "target": "r"}, {"source": "t", "target": "boom"},
            {"source": "boom", "target": "after"},        # runs despite boom failing
            {"source": "a", "target": "join"}, {"source": "b", "target": "join"},
        ],
    }
    t0 = _t.perf_counter()
    r2 = asyncio.run(engine2.execute(g2, trigger_input={}))
    elapsed = _t.perf_counter() - t0
    b2 = {s["id"]: s for s in r2["node_states"]}
    assert b2["a"]["status"] == "completed" and b2["b"]["status"] == "completed"
    assert elapsed < 0.35, f"a+b did not run in parallel (elapsed {elapsed:.2f}s)"
    assert b2["r"]["status"] == "completed" and b2["r"]["attempts"] == 3, b2["r"]
    assert b2["boom"]["status"] == "failed", b2["boom"]
    assert b2["after"]["status"] == "completed", b2["after"]  # continue-on-error
    assert r2["status"] == "completed", r2  # soft-fail did not fail the run
    assert b2["join"]["status"] == "completed", b2["join"]    # join waited for both arms
    print(f"✓ parallel({elapsed:.2f}s) + retries(3) + continue-on-error smoke test passed")

    # ── Sub-workflow (composition) ──────────────────────────────────────────
    _saved = {
        "child": {"graph": {
            "nodes": [{"id": "t", "type": "manual", "params": {}},
                      {"id": "c", "type": "llm", "params": {"prompt": "child {{ $trigger.item }}"}}],
            "edges": [{"source": "t", "target": "c"}]}}
    }
    engine3 = WorkflowEngine(WorkflowServices(
        llm=_fake_llm, load_workflow=lambda wid: _saved.get(wid)))
    g3 = {
        "nodes": [
            {"id": "t", "type": "manual", "params": {}},
            {"id": "map", "type": "subworkflow",
             "params": {"workflow_id": "child", "items": "{{ $trigger.items }}"}},
        ],
        "edges": [{"source": "t", "target": "map"}],
    }
    r3 = asyncio.run(engine3.execute(g3, trigger_input={"items": ["x", "y", "z"]}))
    m = {s["id"]: s for s in r3["node_states"]}["map"]
    assert m["status"] == "completed" and m["output"]["count"] == 3, m
    assert abs(r3["total_carbon_g"] - 0.036) < 1e-9, r3["total_carbon_g"]
    print("✓ sub-workflow map (3 items) smoke test passed")

    # ── Human-in-the-loop approval (pause / resume) ─────────────────────────
    engine4 = WorkflowEngine(WorkflowServices(llm=_fake_llm))
    g4 = {
        "nodes": [
            {"id": "t", "type": "manual", "params": {}},
            {"id": "gen", "type": "llm", "params": {"prompt": "draft"}},
            {"id": "ok", "type": "approval", "params": {"message": "Publish '{{ $node.gen.text }}'?"}},
            {"id": "pub", "type": "transform", "params": {"fields": {"published": "yes"}}},
            {"id": "drop", "type": "transform", "params": {"fields": {"published": "no"}}},
        ],
        "edges": [
            {"source": "t", "target": "gen"},
            {"source": "gen", "target": "ok"},
            {"source": "ok", "target": "pub", "sourceHandle": "approved"},
            {"source": "ok", "target": "drop", "sourceHandle": "rejected"},
        ],
    }
    r4 = asyncio.run(engine4.execute(g4, trigger_input={}))
    assert r4["status"] == "paused", r4
    assert r4["awaiting"][0]["id"] == "ok" and "draft" in r4["awaiting"][0]["message"], r4["awaiting"]
    # round-trip the state through JSON, exactly as the run row does
    state = json.loads(json.dumps(r4["state"]))
    r5 = asyncio.run(engine4.execute(g4, resume_state=state, approvals={"ok": {"approved": True, "by": "alice"}}))
    b5 = {s["id"]: s for s in r5["node_states"]}
    assert r5["status"] == "completed", r5
    assert b5["pub"]["status"] == "completed" and b5["drop"]["status"] == "skipped", b5
    assert abs(r5["total_carbon_g"] - 0.012) < 1e-9, r5["total_carbon_g"]  # gen's carbon survived the pause
    # reject path
    r6 = asyncio.run(engine4.execute(g4))
    r7 = asyncio.run(engine4.execute(g4, resume_state=json.loads(json.dumps(r6["state"])),
                                     approvals={"ok": {"approved": False}}))
    b7 = {s["id"]: s for s in r7["node_states"]}
    assert b7["drop"]["status"] == "completed" and b7["pub"]["status"] == "skipped", b7
    print("✓ human-in-the-loop approval pause/resume (approve + reject) smoke test passed")
    print("\n✓ ALL smoke tests passed")

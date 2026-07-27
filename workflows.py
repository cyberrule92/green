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
MAX_WAIT_S = float(os.getenv("WF_MAX_WAIT_S", "300"))     # a `wait` node's in-run sleep ceiling


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Secret box: authenticated symmetric encryption for credential values at rest.
# Stdlib-only (the project has no `cryptography` dep): an HMAC-SHA256 keystream
# in counter mode (encrypt) with encrypt-then-MAC (authenticate). Keyed off a
# caller-supplied key — decision_engine passes WF_SECRET_KEY, falling back to the
# audit HMAC key. Ciphertext is urlsafe-base64 of nonce||ct||tag.
# ─────────────────────────────────────────────────────────────────────────────
import base64
import hashlib
import hmac as _hmac
import secrets as _secrets

_SECRET_NONCE_LEN = 16
_SECRET_TAG_LEN = 32


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(_hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:length])


def secret_encrypt(plaintext: str, key: bytes) -> str:
    """Encrypt+authenticate a UTF-8 string; returns a urlsafe-base64 token."""
    data = plaintext.encode("utf-8")
    nonce = _secrets.token_bytes(_SECRET_NONCE_LEN)
    ct = bytes(a ^ b for a, b in zip(data, _keystream(key, nonce, len(data))))
    tag = _hmac.new(key, b"wf-secret-v1" + nonce + ct, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(nonce + ct + tag).decode("ascii")


def secret_decrypt(token: str, key: bytes) -> str:
    """Inverse of :func:`secret_encrypt`. Raises ``ValueError`` on tamper/format."""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception as exc:  # noqa: BLE001 - malformed token
        raise ValueError("malformed secret token") from exc
    if len(raw) < _SECRET_NONCE_LEN + _SECRET_TAG_LEN:
        raise ValueError("secret token too short")
    nonce, tag = raw[:_SECRET_NONCE_LEN], raw[-_SECRET_TAG_LEN:]
    ct = raw[_SECRET_NONCE_LEN:-_SECRET_TAG_LEN]
    expected = _hmac.new(key, b"wf-secret-v1" + nonce + ct, hashlib.sha256).digest()
    if not _hmac.compare_digest(tag, expected):
        raise ValueError("secret token failed authentication")
    return bytes(a ^ b for a, b in zip(ct, _keystream(key, nonce, len(ct)))).decode("utf-8")


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
                    status TEXT NOT NULL,          -- running | completed | failed | cancelled | awaiting_approval
                    trigger TEXT NOT NULL DEFAULT 'manual',
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_carbon_g REAL NOT NULL DEFAULT 0.0,
                    node_states_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS workflow_credentials (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'bearer',   -- bearer | basic | header
                    secret_json TEXT NOT NULL,             -- encrypted (secret_encrypt)
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_wfcred_tenant ON workflow_credentials(tenant_id, created_at);
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

    # ── credentials ──────────────────────────────────────────────────────────
    # The store keeps ``secret_token`` as an opaque, already-encrypted blob — it
    # never sees the key. decision_engine encrypts before create and decrypts
    # after fetch (via the ``decrypt_credential`` service). Public list never
    # exposes the token.
    def create_credential(
        self, name: str, cred_type: str, secret_token: str, tenant_id: str = "default"
    ) -> dict[str, Any]:
        cred_id = uuid4().hex[:12]
        now = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO workflow_credentials (id, tenant_id, name, type, secret_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cred_id, tenant_id, name, cred_type, secret_token, now),
            )
        return {"id": cred_id, "name": name, "type": cred_type, "created_at": now}

    def list_credentials(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name, type, created_at FROM workflow_credentials "
                "WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()
        return [{"id": r["id"], "name": r["name"], "type": r["type"],
                 "created_at": r["created_at"]} for r in rows]

    def get_credential_token(self, cred_id: str, tenant_id: str) -> dict[str, Any] | None:
        """Internal: the encrypted token + type, for decryption. Never surfaced
        by an API.

        ``tenant_id`` is mandatory and always constrains the lookup — there is no
        cross-tenant read. A credential is a bearer secret and the HTTP node that
        consumes it sends it to a caller-supplied URL, so an unscoped lookup is an
        exfiltration path, not a convenience.
        """
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_credentials WHERE id = ? AND tenant_id = ?",
                (cred_id, tenant_id),
            ).fetchone()
        if not row:
            return None
        return {"id": row["id"], "name": row["name"], "type": row["type"],
                "secret_token": row["secret_json"], "tenant_id": row["tenant_id"]}

    def delete_credential(self, cred_id: str, tenant_id: str = "default") -> bool:
        with self._lock, self._connect() as connection:
            cur = connection.execute(
                "DELETE FROM workflow_credentials WHERE id = ? AND tenant_id = ?",
                (cred_id, tenant_id),
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
    # Resolve (credential_id, tenant_id) → decrypted {type, secret:{...}} for HTTP
    # auth. The engine never holds the encryption key; decryption happens in this
    # callable. The tenant is REQUIRED and must be enforced by the implementation:
    # the HTTP node's URL is caller-controlled, so an unscoped lookup would let one
    # tenant post another tenant's secret to a server of its choosing.
    decrypt_credential: Optional[Callable[[str, str], Optional[dict[str, Any]]]] = None


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
    # Offload the synchronous, potentially long-running agent to a thread so it
    # does not block the event loop or the other nodes in this superstep
    # (matches _h_http). A coding task can run for minutes.
    result = await asyncio.to_thread(
        ex.services.agent_run,
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


def _inject_credential(headers: dict[str, Any], cred: dict[str, Any]) -> None:
    """Mutate ``headers`` in place to carry the credential's auth. ``cred`` is the
    decrypted ``{type, secret:{...}}`` record from the store."""
    ctype = str(cred.get("type", "bearer")).lower()
    secret = cred.get("secret") or {}
    if ctype == "bearer":
        token = str(secret.get("token", ""))
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif ctype == "basic":
        raw = f"{secret.get('username', '')}:{secret.get('password', '')}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    elif ctype == "header":
        name = str(secret.get("name", "")).strip()
        if name:
            headers[name] = str(secret.get("value", ""))


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

    # Optional credential: resolved at dispatch, scoped to the running workflow's
    # tenant, and injected into the request headers only. The secret never enters
    # the templating scope and is never part of the returned dict, so it cannot
    # reach node_states or a downstream template.
    cred_id = _stringify(ex.param("credential_id", "")).strip()
    if cred_id:
        if ex.services.decrypt_credential is None:
            raise RuntimeError("http node: credential store not wired")
        cred = ex.services.decrypt_credential(cred_id, ex.tenant_id)
        if not cred:
            raise ValueError(f"http node: unknown credential {cred_id}")
        _inject_credential(headers, cred)

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


async def _h_switch(ex: NodeExec) -> dict[str, Any]:
    """Multi-way branch (generalises IF). Evaluates ``cases`` top-to-bottom; the
    first match activates its handle, else the ``default`` handle. Each case is
    ``{handle, field, op, value}`` — ``field``/``value`` are templated."""
    cases = ex.param("cases", []) or []
    default_handle = _stringify(ex.param("default_handle", "default")) or "default"
    matched = default_handle
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            fn = _OPS.get(_stringify(case.get("op", "==")) or "==", _OPS["=="])
            if fn(case.get("field", ""), case.get("value", "")):
                matched = _stringify(case.get("handle")) or default_handle
                break
    return {"matched": matched, "_active_handles": {matched}}


async def _h_filter(ex: NodeExec) -> dict[str, Any]:
    """Single logical gate: forward the input unchanged when the condition holds,
    otherwise prune every downstream edge (a simpler cousin of ``carbon_gate``)."""
    fn = _OPS.get(_stringify(ex.param("op", "==")) or "==", _OPS["=="])
    passed = bool(fn(ex.param("field", ""), ex.param("value", "")))
    if passed:
        # No _active_handles → every out-edge stays live and the input forwards.
        return {**ex.inputs, "passed": True}
    return {"passed": False, "_active_handles": set()}


async def _h_wait(ex: NodeExec) -> dict[str, Any]:
    """Pause the run for a bounded number of seconds.

    MVP is an in-run sleep, which holds a slot in the superstep. Long or
    until-timestamp waits should instead reuse the approval-style pause/resume
    plus the scheduler, so the run is not resident while it waits — kept out of
    this pass deliberately.
    """
    duration = max(0.0, min(_float(ex.param("duration_s", 0), 0.0), MAX_WAIT_S))
    if duration > 0:
        await asyncio.sleep(duration)
    return {**ex.inputs, "waited_s": duration}


async def _h_set_variables(ex: NodeExec) -> dict[str, Any]:
    """Write templated values into the run-level ``$vars`` store (shared by
    reference), so downstream nodes can read them without an edge."""
    fields = (ex.node.get("params") or {}).get("fields", {})
    if not isinstance(fields, dict):
        return {"vars": {}}
    resolved = {k: resolve_templates(v, ex.scope) for k, v in fields.items()}
    run_vars = ex.scope.get("vars")
    if isinstance(run_vars, dict):
        run_vars.update(resolved)
    return {"vars": resolved}


async def _h_notify(ex: NodeExec) -> dict[str, Any]:
    """Send a notification. ``channel`` = webhook (POST the message JSON to a
    URL) or email (stdlib smtplib, gated on SMTP_* env — a graceful logged no-op
    when unconfigured, matching the multimodal fallback convention)."""
    channel = (_stringify(ex.param("channel", "webhook")) or "webhook").lower()
    target = _stringify(ex.param("target", "")).strip()
    message = _stringify(ex.param("message", ""))

    if channel == "webhook":
        if not target.lower().startswith(("http://", "https://")):
            raise ValueError("notify node: webhook target must be http(s)")

        def _post() -> int:
            req = urllib.request.Request(
                target, data=json.dumps({"message": message}).encode("utf-8"),
                method="POST", headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                    return resp.status
            except urllib.error.HTTPError as err:
                return err.code

        status = await asyncio.to_thread(_post)
        return {"sent": 200 <= status < 300, "channel": "webhook",
                "target": target, "http_status": status}

    if channel == "email":
        host = os.getenv("SMTP_HOST", "").strip()
        if not host:
            logger.info("notify node: SMTP not configured — skipping email to %s", target)
            return {"sent": False, "channel": "email", "target": target,
                    "reason": "smtp_unconfigured"}

        def _send() -> bool:
            import smtplib
            from email.message import EmailMessage
            msg = EmailMessage()
            msg["Subject"] = os.getenv("SMTP_SUBJECT", "Workflow notification")
            msg["From"] = os.getenv("SMTP_FROM", "workflows@green-ai.local")
            msg["To"] = target
            msg.set_content(message)
            with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")),
                              timeout=HTTP_TIMEOUT_S) as server:
                if os.getenv("SMTP_STARTTLS", "true").lower() in ("1", "true", "yes"):
                    server.starttls()
                user, pwd = os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASSWORD", "")
                if user and pwd:
                    server.login(user, pwd)
                server.send_message(msg)
            return True

        try:
            ok = await asyncio.to_thread(_send)
        except Exception as exc:  # noqa: BLE001 - notification is best-effort
            logger.warning("notify node: email send failed: %s", exc)
            return {"sent": False, "channel": "email", "target": target, "reason": str(exc)}
        return {"sent": ok, "channel": "email", "target": target}

    raise ValueError(f"notify node: unknown channel {channel!r}")


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
                  "Call an external HTTP(S) API. Body/headers support templates. "
                  "Set credential_id to attach stored auth (never echoed in output).",
                  [{"name": "method", "label": "Method", "type": "select",
                    "options": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"},
                   {"name": "url", "label": "URL", "type": "text", "default": ""},
                   {"name": "headers", "label": "Headers (JSON)", "type": "textarea", "default": "{}"},
                   {"name": "body", "label": "Body", "type": "textarea", "default": ""},
                   {"name": "credential_id", "label": "Credential ID (optional)", "type": "text", "default": ""}],
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
register(NodeType("switch", "Switch (multi-way)", "logic",
                  "Route to the first matching case, else default. Handles are the case handles + default.",
                  [{"name": "cases", "label": "Cases [{handle, field, op, value}]", "type": "keyvalue", "default": []},
                   {"name": "default_handle", "label": "Default handle", "type": "text", "default": "default"}],
                  _h_switch, handles_out=["default"]))
register(NodeType("filter", "Filter", "logic",
                  "Forward the input when the condition holds; otherwise prune all downstream.",
                  [{"name": "field", "label": "Value", "type": "text", "default": ""},
                   {"name": "op", "label": "Operator", "type": "select",
                    "options": ["==", "!=", "contains", ">", "<", "empty", "not_empty"], "default": "=="},
                   {"name": "value", "label": "Compare to", "type": "text", "default": ""}],
                  _h_filter))
register(NodeType("wait", "Wait / delay", "logic",
                  f"Pause the run for up to {int(MAX_WAIT_S)}s, then forward the input.",
                  [{"name": "duration_s", "label": "Duration (s)", "type": "number", "default": 1}],
                  _h_wait))
register(NodeType("set_variables", "Set variables", "logic",
                  "Write templated values into the run-level $vars store for downstream nodes.",
                  [{"name": "fields", "label": "Vars (key → template)", "type": "keyvalue", "default": {}}],
                  _h_set_variables))
register(NodeType("notify", "Notify", "io",
                  "Send a webhook POST or an email (SMTP_* env; graceful no-op if unconfigured).",
                  [{"name": "channel", "label": "Channel", "type": "select",
                    "options": ["webhook", "email"], "default": "webhook"},
                   {"name": "target", "label": "Target (URL or email)", "type": "text", "default": ""},
                   {"name": "message", "label": "Message", "type": "textarea", "default": ""}],
                  _h_notify))
register(NodeType("error_trigger", "Error trigger", "trigger",
                  "Runs this workflow as an error handler; output is {workflow_id, run_id, error, failed_node}.",
                  [], _h_passthrough, is_trigger=True))
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
        cancel_event: Optional[asyncio.Event] = None,
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
                    "error": None, "attempts": 0, "duration_ms": None,
                    "started_at": None, "finished_at": None}

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
            run_vars: dict[str, Any] = dict(resume_state.get("run_vars", {}))
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
            run_vars = {}

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
            # run_vars is shared BY REFERENCE so a set_variables node persists
            # state that {{ $vars.* }} reads downstream, across supersteps. asyncio
            # is single-threaded so a shared dict is safe here, but ordering
            # between two set_variables in the SAME superstep is unspecified.
            scope = {"nodes": node_outputs, "input": merged,
                     "trigger": trigger_input, "vars": run_vars, "_depth": _depth}
            ex = NodeExec(node=node, inputs=merged, scope=scope,
                          services=self.services, tenant_id=tenant_id)

            retries = _int(node.get("retries"), 0)
            backoff = _float(node.get("retry_backoff_s"), 0.5)
            timeout = node.get("timeout_s")
            timeout_s = _float(timeout, 0.0) if timeout not in (None, "") else None
            on_error = str(node.get("on_error", "stop")).lower()

            states[nid]["status"] = "running"
            states[nid]["started_at"] = utc_now_iso()
            _t0 = time.perf_counter()
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
                        states[nid]["finished_at"] = utc_now_iso()
                        states[nid]["duration_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
                        return
                    except Exception as exc:  # noqa: BLE001 - retryable node failure
                        last_exc = exc
                        if attempt <= retries:
                            await asyncio.sleep(backoff * attempt)
            states[nid]["status"] = "failed"
            states[nid]["error"] = f"{type(last_exc).__name__}: {last_exc}"
            states[nid]["finished_at"] = utc_now_iso()
            states[nid]["duration_ms"] = round((time.perf_counter() - _t0) * 1000, 1)
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

        cancelled = False
        awaiting: list[str] = []
        while ready:
            # Cooperative cancel: checked between supersteps, so nodes already
            # in flight finish rather than being torn down mid-call.
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
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
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
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
                    "run_vars": run_vars,
                    "trigger_input": trigger_input,
                }
                pending_approvals = []
                for nid in awaiting:
                    msg = (nodes[nid].get("params") or {}).get("message", "")
                    msg = resolve_templates(msg, {"nodes": node_outputs, "input": {},
                                                  "trigger": trigger_input, "vars": run_vars})
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
            if states[nid]["status"] in ("pending", "running"):
                states[nid]["status"] = "skipped"

        if cancelled:
            status = "cancelled"
            run_error = run_error or "run cancelled"
        else:
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

    # ── Flow-logic nodes: switch / filter / set_variables / wait ─────────────
    engine5 = WorkflowEngine(WorkflowServices(llm=_fake_llm))
    g5 = {
        "nodes": [
            {"id": "t", "type": "manual", "params": {}},
            {"id": "vars", "type": "set_variables",
             "params": {"fields": {"tier": "{{ $trigger.tier }}", "n": "2"}}},
            {"id": "sw", "type": "switch", "params": {
                "cases": [{"handle": "gold", "field": "{{ $vars.tier }}", "op": "==", "value": "gold"},
                          {"handle": "silver", "field": "{{ $vars.tier }}", "op": "==", "value": "silver"}],
                "default_handle": "other"}},
            {"id": "keep", "type": "filter",
             "params": {"field": "{{ $vars.n }}", "op": ">", "value": "1"}},
            {"id": "gold_out", "type": "transform", "params": {"fields": {"picked": "gold-{{ $vars.tier }}"}}},
            {"id": "silver_out", "type": "transform", "params": {"fields": {"picked": "silver"}}},
            {"id": "other_out", "type": "transform", "params": {"fields": {"picked": "other"}}},
        ],
        "edges": [
            {"source": "t", "target": "vars"}, {"source": "vars", "target": "sw"},
            {"source": "sw", "target": "keep", "sourceHandle": "gold"},
            {"source": "keep", "target": "gold_out"},
            {"source": "sw", "target": "silver_out", "sourceHandle": "silver"},
            {"source": "sw", "target": "other_out", "sourceHandle": "other"},
        ],
    }
    r_sw = asyncio.run(engine5.execute(g5, trigger_input={"tier": "gold"}))
    b_sw = {s["id"]: s for s in r_sw["node_states"]}
    assert r_sw["status"] == "completed", r_sw
    assert b_sw["sw"]["output"]["matched"] == "gold", b_sw["sw"]
    assert b_sw["gold_out"]["status"] == "completed", b_sw["gold_out"]
    assert b_sw["gold_out"]["output"]["picked"] == "gold-gold", b_sw["gold_out"]  # $vars downstream
    assert b_sw["silver_out"]["status"] == "skipped", b_sw["silver_out"]
    assert b_sw["other_out"]["status"] == "skipped", b_sw["other_out"]
    assert b_sw["keep"]["output"]["passed"] is True, b_sw["keep"]
    assert b_sw["vars"]["duration_ms"] is not None, "per-node duration not recorded"
    print("✓ switch + set_variables($vars) + filter-pass + duration smoke test passed")

    g6 = {
        "nodes": [{"id": "t", "type": "manual", "params": {}},
                  {"id": "keep", "type": "filter", "params": {"field": "0", "op": ">", "value": "1"}},
                  {"id": "after", "type": "transform", "params": {"fields": {"ran": "yes"}}}],
        "edges": [{"source": "t", "target": "keep"}, {"source": "keep", "target": "after"}],
    }
    r_f = asyncio.run(engine5.execute(g6, trigger_input={}))
    b_f = {s["id"]: s for s in r_f["node_states"]}
    assert b_f["keep"]["output"]["passed"] is False and b_f["after"]["status"] == "skipped", b_f
    print("✓ filter-prune smoke test passed")

    g7 = {
        "nodes": [{"id": "t", "type": "manual", "params": {}},
                  {"id": "w", "type": "wait", "params": {"duration_s": "0.05"}}],
        "edges": [{"source": "t", "target": "w"}],
    }
    _tw = _t.perf_counter()
    r_w = asyncio.run(engine5.execute(g7, trigger_input={}))
    assert (_t.perf_counter() - _tw) >= 0.04, "wait node did not sleep"
    assert {s["id"]: s for s in r_w["node_states"]}["w"]["output"]["waited_s"] == 0.05
    print("✓ wait smoke test passed")

    # $vars must survive an approval pause, since it rides in the state blob.
    engine5b = WorkflowEngine(WorkflowServices(llm=_fake_llm))
    g_pause = {
        "nodes": [{"id": "t", "type": "manual", "params": {}},
                  {"id": "v", "type": "set_variables", "params": {"fields": {"keep": "me"}}},
                  {"id": "ok", "type": "approval", "params": {"message": "vars={{ $vars.keep }}"}},
                  {"id": "after", "type": "transform", "params": {"fields": {"got": "{{ $vars.keep }}"}}}],
        "edges": [{"source": "t", "target": "v"}, {"source": "v", "target": "ok"},
                  {"source": "ok", "target": "after", "sourceHandle": "approved"}],
    }
    r_p = asyncio.run(engine5b.execute(g_pause, trigger_input={}))
    assert r_p["status"] == "paused" and "vars=me" in r_p["awaiting"][0]["message"], r_p["awaiting"]
    r_p2 = asyncio.run(engine5b.execute(g_pause, resume_state=json.loads(json.dumps(r_p["state"])),
                                        approvals={"ok": {"approved": True}}))
    b_p2 = {s["id"]: s for s in r_p2["node_states"]}
    assert b_p2["after"]["output"]["got"] == "me", b_p2["after"]
    print("✓ $vars survives approval pause/resume smoke test passed")

    # ── Cooperative cancel mid-run ──────────────────────────────────────────
    async def _cancel_scenario() -> dict[str, Any]:
        ev = asyncio.Event()

        async def _slow(prompt: str, **_: Any) -> dict[str, Any]:
            await asyncio.sleep(0.15)
            return {"text": "x", "carbon_g": 0.01}

        eng = WorkflowEngine(WorkflowServices(llm=_slow))
        g = {"nodes": [{"id": "t", "type": "manual", "params": {}},
                       {"id": "a", "type": "llm", "params": {"prompt": "A"}},
                       {"id": "b", "type": "llm", "params": {"prompt": "B"}}],
             "edges": [{"source": "t", "target": "a"}, {"source": "a", "target": "b"}]}
        task = asyncio.ensure_future(eng.execute(g, trigger_input={}, cancel_event=ev))
        await asyncio.sleep(0.05)     # let 'a' start
        ev.set()
        return await task

    r_c = asyncio.run(_cancel_scenario())
    b_c = {s["id"]: s for s in r_c["node_states"]}
    assert r_c["status"] == "cancelled", r_c
    assert b_c["b"]["status"] == "skipped", b_c["b"]
    print("✓ cooperative cancel mid-run smoke test passed")

    # ── Secret box round-trip + tamper detection ────────────────────────────
    _k = b"unit-test-key-32-bytes-xxxxxxxxxx"
    _tok = secret_encrypt('{"token":"s3cr3t"}', _k)
    assert secret_decrypt(_tok, _k) == '{"token":"s3cr3t"}'
    try:
        secret_decrypt(_tok[:-2] + ("AA" if not _tok.endswith("AA") else "BB"), _k)
        raise AssertionError("tampered token should not decrypt")
    except ValueError:
        pass
    print("✓ secret box encrypt/decrypt + tamper smoke test passed")

    # ── Credential tenant isolation ─────────────────────────────────────────
    # A credential is a bearer secret and the HTTP node sends it to a
    # caller-supplied URL, so a cross-tenant read is an exfiltration path.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _store = WorkflowStore(Path(_td) / "creds_test.db")
        _cred = _store.create_credential(
            name="prod-api", cred_type="bearer",
            secret_token=secret_encrypt('{"token":"tenant-a-secret"}', _k),
            tenant_id="tenant-a")
        assert _store.get_credential_token(_cred["id"], "tenant-a") is not None
        assert _store.get_credential_token(_cred["id"], "tenant-b") is None, \
            "SECURITY: credential readable across tenants"
        _listed = _store.list_credentials(tenant_id="tenant-a")
        assert len(_listed) == 1 and "secret_token" not in _listed[0] and "secret_json" not in _listed[0]
        assert _store.list_credentials(tenant_id="tenant-b") == []
    print("✓ credential tenant-isolation smoke test passed")

    # The HTTP node must hand the running workflow's tenant to the resolver.
    _seen: dict[str, Any] = {}

    def _spy_decrypt(cred_id: str, tenant_id: str) -> dict[str, Any] | None:
        _seen["cred_id"], _seen["tenant_id"] = cred_id, tenant_id
        return None      # → node raises "unknown credential" before any request

    engine6 = WorkflowEngine(WorkflowServices(decrypt_credential=_spy_decrypt))
    g8 = {
        "nodes": [{"id": "t", "type": "manual", "params": {}},
                  {"id": "call", "type": "http_request",
                   "params": {"url": "https://example.com/x", "credential_id": "cred123"}}],
        "edges": [{"source": "t", "target": "call"}],
    }
    r_cred = asyncio.run(engine6.execute(g8, trigger_input={}, tenant_id="acme"))
    assert _seen.get("tenant_id") == "acme", f"tenant not threaded to resolver: {_seen}"
    assert _seen.get("cred_id") == "cred123", _seen
    assert {s["id"]: s for s in r_cred["node_states"]}["call"]["status"] == "failed"
    print("✓ http-node credential tenant-threading smoke test passed")

    # A secret must never reach node output / node_states.
    def _real_decrypt(cred_id: str, tenant_id: str) -> dict[str, Any]:
        return {"type": "bearer", "secret": {"token": "SUPERSECRET"}}

    engine7 = WorkflowEngine(WorkflowServices(decrypt_credential=_real_decrypt))
    g9 = {
        "nodes": [{"id": "t", "type": "manual", "params": {}},
                  {"id": "call", "type": "http_request",
                   "params": {"url": "http://127.0.0.1:9/", "credential_id": "c1"}}],
        "edges": [{"source": "t", "target": "call"}],
    }
    r_leak = asyncio.run(engine7.execute(g9, trigger_input={}, tenant_id="acme"))
    assert "SUPERSECRET" not in json.dumps(r_leak["node_states"]), "SECURITY: secret leaked into node_states"
    print("✓ credential secret absent from run state smoke test passed")

    print("\n✓ ALL smoke tests passed")

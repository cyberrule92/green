from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class ConversationStore:
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

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created_at
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_tenant
                    ON conversations(tenant_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_tenant
                    ON messages(tenant_id, created_at);

                -- User feedback (thumbs up/down) on assistant messages. One row
                -- per message (a user can change their vote → upsert). This is
                -- the seed for an offline fine-tuning / preference dataset: join
                -- back to the assistant message (response) and the preceding
                -- user message (prompt) via export_feedback_dataset().
                CREATE TABLE IF NOT EXISTS message_feedback (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    rating INTEGER NOT NULL,          -- +1 (up) or -1 (down)
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_tenant
                    ON message_feedback(tenant_id, updated_at);
                """
            )
            # Backfill: add column if pre-existing DB without it
            for table in ("conversations", "messages"):
                try:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists

    def healthcheck(self) -> bool:
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def create_conversation(self, title: str = "New chat", tenant_id: str = "default") -> dict[str, Any]:
        conversation_id = str(uuid4())
        now = utc_now_iso()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at, tenant_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, title, now, now, tenant_id),
            )
        return self.get_conversation(conversation_id, tenant_id=tenant_id) or {}

    def get_conversation(
        self, conversation_id: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        sql = """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                c.tenant_id,
                COALESCE(
                    (SELECT content
                     FROM messages
                     WHERE conversation_id = c.id
                     ORDER BY created_at DESC
                     LIMIT 1),
                    ''
                ) AS preview,
                (SELECT COUNT(*)
                 FROM messages
                 WHERE conversation_id = c.id) AS message_count
            FROM conversations c
            WHERE c.id = ?
        """
        params: tuple = (conversation_id,)
        if tenant_id is not None:
            sql += " AND c.tenant_id = ?"
            params = (conversation_id, tenant_id)

        with self._lock, self._connect() as connection:
            row = connection.execute(sql, params).fetchone()

        return dict(row) if row else None

    def list_conversations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                c.tenant_id,
                COALESCE(
                    (SELECT content
                     FROM messages
                     WHERE conversation_id = c.id
                     ORDER BY created_at DESC
                     LIMIT 1),
                    ''
                ) AS preview,
                (SELECT COUNT(*)
                 FROM messages
                 WHERE conversation_id = c.id) AS message_count
            FROM conversations c
        """
        params: tuple = ()
        if tenant_id is not None:
            sql += " WHERE c.tenant_id = ?"
            params = (tenant_id,)
        sql += " ORDER BY c.updated_at DESC"

        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def update_title(self, conversation_id: str, title: str, tenant_id: str | None = None) -> None:
        now = utc_now_iso()
        sql = "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?"
        params: tuple = (title, now, conversation_id)
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params = (title, now, conversation_id, tenant_id)
        with self._lock, self._connect() as connection:
            connection.execute(sql, params)

    def ensure_conversation(
        self, conversation_id: str | None, title: str, tenant_id: str = "default"
    ) -> dict[str, Any]:
        if not conversation_id:
            return self.create_conversation(title, tenant_id=tenant_id)

        conversation = self.get_conversation(conversation_id, tenant_id=tenant_id)
        if not conversation:
            raise KeyError(conversation_id)

        current_title = (conversation.get("title") or "").strip().lower()
        if current_title in {"", "new chat", "untitled chat"} and title:
            self.update_title(conversation_id, title, tenant_id=tenant_id)
            conversation = self.get_conversation(conversation_id, tenant_id=tenant_id)

        return conversation or {}

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        message_id = str(uuid4())
        now = utc_now_iso()
        metadata = metadata or {}

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at, metadata_json, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    now,
                    json.dumps(metadata, ensure_ascii=False),
                    tenant_id,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

        messages = self.list_messages(conversation_id, tenant_id=tenant_id)
        for message in messages:
            if message["id"] == message_id:
                return message
        return {}

    def list_messages(
        self,
        conversation_id: str,
        include_internal: bool = False,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT m.id, m.conversation_id, m.role, m.content, m.created_at, "
            "m.metadata_json, m.tenant_id, "
            "f.rating AS fb_rating, f.reason AS fb_reason, f.updated_at AS fb_updated_at "
            "FROM messages m "
            "LEFT JOIN message_feedback f ON f.message_id = m.id "
            "WHERE m.conversation_id = ?"
        )
        params: tuple = (conversation_id,)
        if tenant_id is not None:
            sql += " AND m.tenant_id = ?"
            params = (conversation_id, tenant_id)
        sql += " ORDER BY m.created_at ASC"

        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [
            self._serialize_message_row(row, include_internal=include_internal)
            for row in rows
        ]

    def delete_conversation(self, conversation_id: str, tenant_id: str | None = None) -> bool:
        with self._lock, self._connect() as connection:
            if tenant_id is not None:
                # Verify ownership first
                row = connection.execute(
                    "SELECT 1 FROM conversations WHERE id = ? AND tenant_id = ?",
                    (conversation_id, tenant_id),
                ).fetchone()
                if not row:
                    return False
            connection.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            sql = "DELETE FROM conversations WHERE id = ?"
            params: tuple = (conversation_id,)
            if tenant_id is not None:
                sql += " AND tenant_id = ?"
                params = (conversation_id, tenant_id)
            cursor = connection.execute(sql, params)
        return cursor.rowcount > 0

    # ── Feedback (thumbs up/down on assistant messages) ──────────────────────

    def save_feedback(
        self,
        message_id: str,
        rating: int,
        reason: str = "",
        tenant_id: str | None = "default",
    ) -> dict[str, Any] | None:
        """Record (or update) a thumbs up/down on an assistant message.

        Returns the stored feedback dict, or None if the message does not exist,
        is not owned by ``tenant_id``, or is not an assistant message.
        """
        if rating not in (1, -1):
            raise ValueError("rating must be +1 (up) or -1 (down)")
        now = utc_now_iso()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT role, conversation_id, tenant_id FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if not row:
                return None
            if tenant_id is not None and row["tenant_id"] != tenant_id:
                return None
            if row["role"] != "assistant":
                return None
            connection.execute(
                """
                INSERT INTO message_feedback
                    (message_id, conversation_id, tenant_id, rating, reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    rating = excluded.rating,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (message_id, row["conversation_id"], row["tenant_id"],
                 rating, reason or "", now, now),
            )
        return {
            "message_id": message_id,
            "rating": rating,
            "reason": reason or "",
            "updated_at": now,
        }

    def feedback_stats(self, tenant_id: str | None = None) -> dict[str, int]:
        """Counts of up/down votes (optionally scoped to a tenant)."""
        sql = "SELECT rating, COUNT(*) AS n FROM message_feedback"
        params: tuple = ()
        if tenant_id is not None:
            sql += " WHERE tenant_id = ?"
            params = (tenant_id,)
        sql += " GROUP BY rating"
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        up = down = 0
        for row in rows:
            if int(row["rating"]) > 0:
                up = int(row["n"])
            else:
                down = int(row["n"])
        return {"up": up, "down": down, "total": up + down}

    def export_feedback_dataset(
        self,
        tenant_id: str | None = None,
        only_positive: bool = False,
    ) -> list[dict[str, Any]]:
        """Build a fine-tuning / preference dataset from collected feedback.

        Each record pairs the assistant response (the message that was rated)
        with the user prompt that immediately preceded it in the same
        conversation, plus the vote and the model that generated it. Pass
        ``only_positive=True`` to export just the up-voted pairs (a supervised
        fine-tuning set); otherwise every rated pair is returned with a
        good/bad label (usable as preference data).
        """
        sql = """
            SELECT f.message_id, f.rating, f.reason, f.tenant_id, f.updated_at,
                   a.content AS response, a.metadata_json AS a_meta,
                   a.conversation_id AS conversation_id, a.created_at AS a_created,
                   (SELECT u.content FROM messages u
                      WHERE u.conversation_id = a.conversation_id
                        AND u.role = 'user'
                        AND u.created_at < a.created_at
                      ORDER BY u.created_at DESC LIMIT 1) AS prompt
            FROM message_feedback f
            JOIN messages a ON a.id = f.message_id
            WHERE a.role = 'assistant'
        """
        params: list = []
        if tenant_id is not None:
            sql += " AND f.tenant_id = ?"
            params.append(tenant_id)
        if only_positive:
            sql += " AND f.rating = 1"
        sql += " ORDER BY f.updated_at DESC"

        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, tuple(params)).fetchall()

        records: list[dict[str, Any]] = []
        for row in rows:
            meta = _decode_json(row["a_meta"])
            records.append({
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "tenant_id": row["tenant_id"],
                "prompt": row["prompt"] or "",
                "response": row["response"],
                "rating": int(row["rating"]),
                "label": "good" if int(row["rating"]) > 0 else "bad",
                "reason": row["reason"] or "",
                "model_variant": meta.get("model_variant"),
                "resolved_model_name": meta.get("resolved_model_name"),
                "timestamp": row["updated_at"],
            })
        return records

    def _serialize_message_row(
        self,
        row: sqlite3.Row,
        include_internal: bool,
    ) -> dict[str, Any]:
        metadata = _decode_json(row["metadata_json"])
        attachments = []
        for attachment in metadata.get("attachments", []):
            if not isinstance(attachment, dict):
                continue
            attachment_payload = dict(attachment)
            if not include_internal:
                attachment_payload.pop("context_text", None)
            attachments.append(attachment_payload)

        metadata["attachments"] = attachments

        # Surface any user feedback (thumbs up/down) joined in by list_messages so
        # the UI can render the current vote when a conversation is reloaded.
        feedback = None
        row_keys = row.keys()
        if "fb_rating" in row_keys and row["fb_rating"] is not None:
            feedback = {
                "rating": row["fb_rating"],
                "reason": row["fb_reason"] or "",
                "updated_at": row["fb_updated_at"],
            }

        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
            "attachments": attachments,
            "sustainability": metadata.get("sustainability"),
            "routing": metadata.get("routing"),
            "retrieval": metadata.get("retrieval"),
            "resolved_model_name": metadata.get("resolved_model_name"),
            "model_variant": metadata.get("model_variant"),
            "feedback": feedback,
            "metadata": metadata,
        }

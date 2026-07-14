from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None
    SentenceTransformer = None


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


class AdvancedRAGService:
    def __init__(
        self,
        store_path: str | Path,
        chunk_size: int = 900,
        chunk_overlap: int = 180,
        dense_top_k: int = 14,
        rerank_top_k: int = 8,
        context_char_limit: int = 5200,
        embedding_dim: int = 256,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k
        self.context_char_limit = context_char_limit
        self.embedding_dim = embedding_dim
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self._embedding_backend = "hashed-fallback"
        self._reranker_backend = "heuristic-fallback"
        self._embedding_model = None
        self._reranker = None
        self._lock = threading.Lock()
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self._load()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "document_count": len(self.documents),
                "chunk_count": len(self.chunks),
                "embedding_backend": self._embedding_backend,
                "reranker_backend": self._reranker_backend,
                "store_path": str(self.store_path),
                "updated_at": max(
                    [document.get("created_at") for document in self.documents],
                    default=None,
                ),
            }

    def list_documents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            scoped = [
                doc for doc in self.documents
                if tenant_id is None or (doc.get("tenant_id") or "default") == tenant_id
            ]
            ordered = sorted(
                scoped,
                key=lambda item: item.get("created_at") or "",
                reverse=True,
            )
            return [dict(document) for document in ordered]

    def delete_document(self, document_id: str, tenant_id: str | None = None) -> bool:
        with self._lock:
            target = next(
                (doc for doc in self.documents if doc.get("id") == document_id),
                None,
            )
            if target is None:
                return False
            if tenant_id is not None and (target.get("tenant_id") or "default") != tenant_id:
                # Cross-tenant delete attempt — refuse silently as 404.
                return False

            self.documents = [
                document for document in self.documents if document.get("id") != document_id
            ]
            self.chunks = [
                chunk for chunk in self.chunks if chunk.get("document_id") != document_id
            ]
            self._save_unlocked()
            return True

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        source: str = "upload",
        persist: bool = True,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        prepared_documents, prepared_chunks = self._prepare_documents(
            documents, source=source, tenant_id=tenant_id,
        )

        if persist and prepared_documents:
            with self._lock:
                self.documents.extend(prepared_documents)
                self.chunks.extend(prepared_chunks)
                self._save_unlocked()

        return {
            "tenant_id": tenant_id,
            "document_count": len(prepared_documents),
            "chunk_count": len(prepared_chunks),
            "documents": [
                {
                    "id": document["id"],
                    "name": document["name"],
                    "chunk_count": document["chunk_count"],
                    "source": document["source"],
                    "tags": document.get("tags", []),
                    "tenant_id": document.get("tenant_id"),
                }
                for document in prepared_documents
            ],
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        ephemeral_documents: list[dict[str, Any]] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return self._empty_retrieval_payload()

        _, ephemeral_chunks = self._prepare_documents(
            ephemeral_documents or [],
            source="ephemeral",
            tenant_id=tenant_id or "default",
        )

        with self._lock:
            combined_chunks = [dict(chunk) for chunk in self.chunks]

        # Tenant isolation: drop persisted chunks owned by other tenants.
        # Ephemeral (request-scoped) chunks are always kept since they were
        # uploaded as part of THIS request.
        if tenant_id is not None:
            combined_chunks = [
                c for c in combined_chunks
                if (c.get("tenant_id") or "default") == tenant_id
            ]

        combined_chunks.extend(ephemeral_chunks)
        if not combined_chunks:
            return self._empty_retrieval_payload()

        query_embedding = self._embed_texts([query])[0]
        query_tokens = tokenize(query)

        dense_scores = self._score_dense(query_embedding, combined_chunks)
        sparse_scores = self._score_sparse(query_tokens, combined_chunks)

        dense_ranking = sorted(
            dense_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.dense_top_k]
        sparse_ranking = sorted(
            sparse_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: self.dense_top_k]

        fused_scores: dict[str, float] = {}
        for rank, (chunk_id, score) in enumerate(dense_ranking, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score + (1.0 / (60 + rank))
        for rank, (chunk_id, score) in enumerate(sparse_ranking, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + score + (1.0 / (60 + rank))

        chunk_lookup = {chunk["id"]: chunk for chunk in combined_chunks}
        fused_candidates = []
        for chunk_id, hybrid_score in sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: max(top_k * 3, self.rerank_top_k)]:
            chunk = chunk_lookup.get(chunk_id)
            if not chunk:
                continue
            fused_candidates.append(
                {
                    "chunk": chunk,
                    "dense_score": dense_scores.get(chunk_id, 0.0),
                    "sparse_score": sparse_scores.get(chunk_id, 0.0),
                    "hybrid_score": hybrid_score,
                }
            )

        reranked = self._rerank(query, fused_candidates)
        filtered = [
            candidate
            for candidate in reranked
            if candidate["rerank_score"] >= 0.12 or len(reranked) <= top_k
        ][:top_k]

        context = self._fuse_context(filtered)
        sources = []
        for candidate in filtered:
            chunk = candidate["chunk"]
            metadata = chunk.get("metadata", {})
            chunk_token_set = set(chunk.get("tokens") or [])
            query_token_set = set(query_tokens)
            sources.append(
                {
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "document_name": metadata.get("document_name", "Untitled"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 1),
                    "score": round(candidate["rerank_score"], 4),
                    "dense_score": round(candidate.get("dense_score", 0.0), 4),
                    "sparse_score": round(candidate.get("sparse_score", 0.0), 4),
                    "hybrid_score": round(candidate.get("hybrid_score", 0.0), 4),
                    "query_term_overlap": round(
                        len(query_token_set & chunk_token_set) / max(len(query_token_set), 1),
                        4,
                    ),
                    "tags": metadata.get("tags", []),
                    "excerpt": chunk["text"][:240].strip(),
                    "text": chunk["text"][:1600].strip(),
                }
            )

        return {
            "query": query,
            "search_mode": "hybrid",
            "retrieved_count": len(filtered),
            "candidate_count": len(combined_chunks),
            "context": context,
            "context_characters": len(context),
            "sources": sources,
        }

    def _empty_retrieval_payload(self) -> dict[str, Any]:
        return {
            "query": "",
            "search_mode": "hybrid",
            "retrieved_count": 0,
            "candidate_count": 0,
            "context": "",
            "context_characters": 0,
            "sources": [],
        }

    def _load(self) -> None:
        if not self.store_path.exists():
            self.documents = []
            self.chunks = []
            return

        try:
            raw = self.store_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to load RAG store: %s", exc)
            self.documents = []
            self.chunks = []
            return

        # An empty / whitespace-only file is an un-initialized store, not a
        # corrupt one — treat it like a missing file rather than warning on
        # every request (it would otherwise raise "Expecting value: line 1").
        if not raw.strip():
            self.documents = []
            self.chunks = []
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to load RAG store: %s", exc)
            self.documents = []
            self.chunks = []
            return

        self.documents = payload.get("documents", [])
        self.chunks = payload.get("chunks", [])

    def _save_unlocked(self) -> None:
        payload = {"documents": self.documents, "chunks": self.chunks}
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        if SentenceTransformer is None:
            return None

        try:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._embedding_backend = self.embedding_model_name
        except Exception as exc:
            logger.warning("Embedding model unavailable, using fallback embeddings: %s", exc)
            self._embedding_model = None
        return self._embedding_model

    def _load_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if CrossEncoder is None:
            return None

        try:
            self._reranker = CrossEncoder(self.reranker_model_name)
            self._reranker_backend = self.reranker_model_name
        except Exception as exc:
            logger.warning("Reranker unavailable, using heuristic reranking: %s", exc)
            self._reranker = None
        return self._reranker

    def _prepare_documents(
        self,
        documents: list[dict[str, Any]],
        source: str,
        tenant_id: str = "default",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prepared_documents: list[dict[str, Any]] = []
        chunk_payloads: list[tuple[str, int, int, str, dict[str, Any]]] = []

        for document in documents:
            text = (document.get("text") or "").strip()
            if not text:
                continue

            doc_id = document.get("id") or str(uuid4())
            name = document.get("name") or f"Document {len(prepared_documents) + 1}"
            metadata = dict(document.get("metadata") or {})
            chunks = self._chunk_text(text)
            if not chunks:
                continue

            tags = self._infer_tags(name, text, metadata)
            prepared_documents.append(
                {
                    "id": doc_id,
                    "name": name,
                    "source": source,
                    "chunk_count": len(chunks),
                    "created_at": utc_now_iso(),
                    "tags": tags,
                    "tenant_id": tenant_id,
                    "metadata": metadata,
                }
            )

            for index, chunk_text in enumerate(chunks):
                chunk_payloads.append((doc_id, index, len(chunks), chunk_text, metadata | {"document_name": name, "tags": tags}))

        embeddings = self._embed_texts([item[3] for item in chunk_payloads])
        prepared_chunks: list[dict[str, Any]] = []
        for (doc_id, index, total_chunks, chunk_text, metadata), embedding in zip(
            chunk_payloads,
            embeddings,
        ):
            tokens = tokenize(chunk_text)
            prepared_chunks.append(
                {
                    "id": str(uuid4()),
                    "document_id": doc_id,
                    "tenant_id": tenant_id,
                    "text": chunk_text,
                    "tokens": tokens,
                    "embedding": embedding,
                    "metadata": {
                        **metadata,
                        "chunk_index": index,
                        "total_chunks": total_chunks,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    },
                }
            )

        return prepared_documents, prepared_chunks

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_embedding_model()
        if model is not None:
            try:
                vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [list(map(float, vector)) for vector in vectors]
            except Exception as exc:
                logger.warning("Dense embedding failed, using hashed fallback: %s", exc)

        self._embedding_backend = "hashed-fallback"
        return [self._hashed_embedding(text) for text in texts]

    def _hashed_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.embedding_dim
        counts = Counter(tokenize(text))
        if not counts:
            return vector

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.embedding_dim
            vector[index] += float(count)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _chunk_text(self, text: str) -> list[str]:
        cleaned = re.sub(r"\r\n?", "\n", text).strip()
        if not cleaned:
            return []

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        blocks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                blocks.append(current.strip())
            current = paragraph

            while len(current) > self.chunk_size:
                blocks.append(current[: self.chunk_size].strip())
                overlap_start = max(0, self.chunk_size - self.chunk_overlap)
                current = current[overlap_start:].strip()

        if current:
            blocks.append(current.strip())

        return [block for block in blocks if block]

    def _infer_tags(
        self,
        name: str,
        text: str,
        metadata: dict[str, Any],
    ) -> list[str]:
        tags = set(metadata.get("tags") or [])
        suffix = Path(name).suffix.lower().lstrip(".")
        if suffix:
            tags.add(suffix)

        lowered = text.lower()
        if "policy" in lowered or "compliance" in lowered:
            tags.add("policy")
        if "architecture" in lowered or "system" in lowered:
            tags.add("architecture")
        if "carbon" in lowered:
            tags.add("carbon")
        if "rag" in lowered or "retrieval" in lowered:
            tags.add("rag")

        return sorted(tag for tag in tags if tag)

    def _score_dense(
        self,
        query_embedding: list[float],
        chunks: list[dict[str, Any]],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for chunk in chunks:
            embedding = chunk.get("embedding") or []
            score = sum(left * right for left, right in zip(query_embedding, embedding))
            scores[chunk["id"]] = score
        return scores

    def _score_sparse(
        self,
        query_tokens: list[str],
        chunks: list[dict[str, Any]],
    ) -> dict[str, float]:
        if not query_tokens:
            return {chunk["id"]: 0.0 for chunk in chunks}

        doc_freq: dict[str, int] = {}
        avg_doc_length = (
            sum(len(chunk.get("tokens") or []) for chunk in chunks) / max(len(chunks), 1)
        ) or 1.0

        for chunk in chunks:
            for token in set(chunk.get("tokens") or []):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        total_docs = max(len(chunks), 1)
        k1 = 1.5
        b = 0.75
        scores: dict[str, float] = {}

        for chunk in chunks:
            tokens = chunk.get("tokens") or []
            frequencies = Counter(tokens)
            doc_length = max(len(tokens), 1)
            score = 0.0

            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                numerator = frequency * (k1 + 1)
                denominator = frequency + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += idf * (numerator / denominator)

            scores[chunk["id"]] = score

        return scores

    def _rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        reranker = self._load_reranker()
        if reranker is not None:
            try:
                pairs = [(query, candidate["chunk"]["text"]) for candidate in candidates]
                scores = reranker.predict(pairs)
                for candidate, score in zip(candidates, scores):
                    candidate["rerank_score"] = float(score)
                return sorted(
                    candidates,
                    key=lambda item: item["rerank_score"],
                    reverse=True,
                )
            except Exception as exc:
                logger.warning("Cross-encoder reranking failed, using heuristic: %s", exc)

        self._reranker_backend = "heuristic-fallback"
        query_token_set = set(tokenize(query))
        for candidate in candidates:
            chunk_tokens = set(candidate["chunk"].get("tokens") or [])
            overlap = len(query_token_set & chunk_tokens) / max(len(query_token_set), 1)
            candidate["rerank_score"] = (candidate["hybrid_score"] * 0.65) + (overlap * 0.35)

        return sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)

    def _fuse_context(self, candidates: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        consumed = 0

        for candidate in candidates:
            chunk = candidate["chunk"]
            metadata = chunk.get("metadata", {})
            header = (
                f"[Source: {metadata.get('document_name', 'Untitled')} | "
                f"Chunk {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)} | "
                f"Score {candidate['rerank_score']:.3f}]"
            )
            block = f"{header}\n{chunk['text'].strip()}"
            if blocks and consumed + len(block) > self.context_char_limit:
                break
            blocks.append(block)
            consumed += len(block)

        return "\n\n".join(blocks)

import React from "react";
import architectureDiagram from "../assets/advanced-rag-architecture.svg";

const DEPLOYMENT_CARDS = [
  {
    title: "Ubuntu + vGPU",
    copy: "vLLM serves the model pool on NVIDIA-backed Ubuntu infrastructure while the FastAPI control plane manages routing, retrieval, and auditability.",
  },
  {
    title: "Hybrid Index",
    copy: "Uploads are chunked, enriched, embedded, and stored for dense + sparse retrieval so later prompts can reuse indexed knowledge.",
  },
  {
    title: "Carbon-Aware Router",
    copy: "Each request gets a ranked list of model, hardware, and region candidates using tier-aware policy coefficients and EcoServe-style signals.",
  },
  {
    title: "Audit Trail",
    copy: "Decision inputs, ranked candidates, selected route, and retrieved evidence stay attached to every answer for production traceability.",
  },
];

export function ArchitecturePanel({ ragStatus }) {
  return (
    <section className="architecture-panel">
      <div className="architecture-header">
        <div>
          <p className="eyebrow">Embedded Architecture</p>
          <h2>Advanced RAG + Adaptive Green orchestration</h2>
          <p>
            The diagram below mirrors the uploaded architecture idea and anchors the
            app to a full indexing, retrieval, and generation pipeline instead of
            single-shot prompting.
          </p>
        </div>

        <div className="architecture-stats">
          <div>
            <strong>{ragStatus?.document_count ?? 0}</strong>
            <span>Indexed docs</span>
          </div>
          <div>
            <strong>{ragStatus?.chunk_count ?? 0}</strong>
            <span>Indexed chunks</span>
          </div>
          <div>
            <strong>{ragStatus?.embedding_backend || "fallback"}</strong>
            <span>Embedding backend</span>
          </div>
        </div>
      </div>

      <div className="architecture-diagram-card">
        <img
          className="architecture-diagram"
          src={architectureDiagram}
          alt="Advanced RAG architecture with indexing, retrieval, and generation stages"
        />
      </div>

      <div className="deployment-grid">
        {DEPLOYMENT_CARDS.map((card) => (
          <article className="deployment-card" key={card.title}>
            <h3>{card.title}</h3>
            <p>{card.copy}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

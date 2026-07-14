/**
 * GreenAIChat — Adaptive Green AI v4.0
 * Sidebar is fully dynamic: real-time grid carbon, RL policy weights,
 * deferred-queue status, last-routed model name — all auto-refreshing.
 * Message header shows the resolved model name (e.g. "TinyLlama-1.1B-Chat")
 * alongside the variant, not just the variant label.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CarbonDashboard } from "./CarbonDashboard";
import { ObservabilityPanel } from "./ObservabilityPanel";
import { AgentPanel } from "./AgentPanel";
import { BenchmarkPanel } from "./BenchmarkPanel";
import {
  deleteConversation,
  fetchConversation,
  fetchConversations,
  fetchGridZones,
  fetchQueueStatus,
  fetchRagDocuments,
  fetchRagStatus,
  fetchRLStatus,
  fetchSystemMetrics,
  fetchQualityLatencyEstimator,
  sendChatMessage,
  sendFeedback,
  fetchFeedbackStats,
  getTenantId,
  setTenantId,
  onTenantChange,
  getKnownTenants,
  removeKnownTenant,
  onTenantListChange,
} from "../lib/api";

// ── Constants ─────────────────────────────────────────────────────────────────
const TABS = ["Chat", "Carbon", "Observability", "Coding Arena", "Benchmark"];
const ROUTER_TIER = "standard";
const SIDEBAR_REFRESH_MS = 30_000;   // grid, RL, RAG
const QUEUE_REFRESH_MS   = 10_000;   // queue polls every 10 s (matches backend dispatch interval)
// Dispatched when a user submits thumbs up/down so the feedback tile updates
// immediately instead of waiting for the next sidebar poll.
const FEEDBACK_EVENT = "green-ai:feedback-change";

const MODEL_NAMES = {
  "ultra-light": "DialoGPT-medium",
  medium: "TinyLlama-1.1B",
  full: "Qwen2-1.5B",
  moe: "Qwen3-30B-MoE",
};

const STARTERS = [
  { label: "Carbon footprint", prompt: "What is the carbon footprint of this inference request?" },
  { label: "Model comparison", prompt: "Compare the sustainability of available model variants." },
  { label: "Grid status", prompt: "What is the current grid carbon intensity and should I defer?" },
  { label: "Audit my requests", prompt: "Summarise the sustainability decisions made in this session." },
];

const CI_COLORS = { low: "#01a982", med: "#e6a817", high: "#c94040" };

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatTs(iso) {
  if (!iso) return "";
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}

function carbonLabel(g) {
  if (g == null) return null;
  if (g < 0.001) return `${(g * 1e6).toFixed(2)} µgCO₂`;
  if (g < 1) return `${(g * 1000).toFixed(3)} mgCO₂`;
  return `${g.toFixed(4)} gCO₂`;
}

function ciColor(ci) {
  if (ci == null) return CI_COLORS.med;
  if (ci < 250) return CI_COLORS.low;
  if (ci < 450) return CI_COLORS.med;
  return CI_COLORS.high;
}

function ciLabel(ci) {
  if (ci == null) return "–";
  if (ci < 250) return "Clean";
  if (ci < 450) return "Moderate";
  return "High";
}

/** Resolve the full model name + variant label from message metadata.
 * Returns { name, variant, escalated } so the UI can show
 * "Qwen2-1.5B-Instruct [full]" or "TinyLlama [medium ↑ overflow]"
 */
function resolveModelInfo(meta) {
  if (!meta) return null;
  const routing = meta.routing || {};
  const name = (
    meta.resolved_model_name ||
    meta.model_name ||
    routing.selected_model_name ||
    MODEL_NAMES[routing.selected_model_variant] ||
    MODEL_NAMES[meta.model_variant] ||
    meta.model_variant ||
    routing.selected_model_variant ||
    null
  );
  const variant = routing.selected_model_variant || meta.model_variant || null;
  const escalated = !!routing.overflow_escalated;
  const originalVariant = routing.original_variant || null;
  return name ? { name, variant, escalated, originalVariant } : null;
}

// Keep backward compat for any callers expecting a string
function resolveModelName(meta) {
  return resolveModelInfo(meta)?.name || null;
}

// ── Live sidebar data hook ────────────────────────────────────────────────────
function useSidebarData(routerTier = ROUTER_TIER) {
  const [gridData, setGridData] = useState(null);
  const [rlData, setRlData] = useState(null);
  const [queueData, setQueueData] = useState(null);
  const [ragStatus, setRagStatus] = useState(null);
  const [ragDocuments, setRagDocuments] = useState([]);
  const [gpuData, setGpuData] = useState(null);
  const [estimatorData, setEstimatorData] = useState(null);
  const [feedbackStats, setFeedbackStats] = useState(null);

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      fetchGridZones(),
      fetchRLStatus(),
      fetchQueueStatus(),
      fetchRagStatus(),
      fetchRagDocuments(),
      fetchQualityLatencyEstimator(),
      fetchFeedbackStats(),
    ]);
    if (results[0].status === "fulfilled") setGridData(results[0].value);
    if (results[1].status === "fulfilled") setRlData(results[1].value?.rl);
    if (results[2].status === "fulfilled") setQueueData(results[2].value?.queue);
    if (results[3].status === "fulfilled") setRagStatus(results[3].value?.rag || results[3].value);
    if (results[4].status === "fulfilled") setRagDocuments(results[4].value?.documents || []);
    if (results[5].status === "fulfilled") setEstimatorData(results[5].value?.estimator);
    if (results[6].status === "fulfilled") setFeedbackStats(results[6].value?.stats ?? null);
  }, []);

  // Immediate feedback-tile refresh when a vote is cast (no 30 s wait). The
  // vote's POST already returns fresh stats, passed via the event detail, so we
  // only hit the network if they're absent.
  const refreshFeedback = useCallback(async (evt) => {
    if (evt?.detail) { setFeedbackStats(evt.detail); return; }
    try {
      const res = await fetchFeedbackStats();
      setFeedbackStats(res?.stats ?? null);
    } catch { /* non-critical */ }
  }, []);

  // GPU metrics at 15 s (fast enough to show routing changes)
  const refreshGpu = useCallback(async () => {
    try {
      const res = await fetchSystemMetrics();
      setGpuData(res);
    } catch { /* non-critical */ }
  }, []);

  // Queue at 10 s to match backend dispatch cadence
  const refreshQueue = useCallback(async () => {
    try {
      const res = await fetchQueueStatus();
      setQueueData(res?.queue ?? null);
    } catch { /* non-critical */ }
  }, []);

  useEffect(() => {
    refresh();
    refreshGpu();
    const t  = setInterval(refresh, SIDEBAR_REFRESH_MS);
    const tg = setInterval(refreshGpu, 15_000);
    const tq = setInterval(refreshQueue, QUEUE_REFRESH_MS);
    window.addEventListener(FEEDBACK_EVENT, refreshFeedback);
    return () => {
      clearInterval(t); clearInterval(tg); clearInterval(tq);
      window.removeEventListener(FEEDBACK_EVENT, refreshFeedback);
    };
  }, [refresh, refreshGpu, refreshQueue, refreshFeedback]);

  const tierWeights = rlData?.tiers?.[routerTier]?.weights || null;
  const primaryCI = gridData?.carbon_map ? Object.values(gridData.carbon_map)[0] : null;
  const primaryZone = gridData?.carbon_map ? Object.keys(gridData.carbon_map)[0] : null;
  const allZones = gridData?.carbon_map ? Object.entries(gridData.carbon_map) : [];
  const primarySignal = primaryZone ? gridData?.zones?.[primaryZone] || null : null;
  const bestZone = gridData?.best_zone || null;
  const bestSignal = bestZone ? gridData?.zones?.[bestZone] || gridData?.best_signal || null : null;

  return {
    gridData, allZones, primaryCI, primaryZone, primarySignal, bestZone, bestSignal,
    rlData, tierWeights,
    queueData,
    ragStatus, ragDocuments,
    gpuData,
    estimatorData,
    feedbackStats,
    refresh,
  };
}

// ── Sub-components ────────────────────────────────────────────────────────────

function CIBar({ value, max = 600 }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = ciColor(value);
  return (
    <div style={{ height: 4, background: "rgba(0,0,0,0.08)", borderRadius: 4, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.6s ease" }} />
    </div>
  );
}

function WeightMiniBar({ label, value, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: "0.72rem", color: "var(--hpe-soft-ink)", width: 52, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 5, background: "rgba(0,0,0,0.08)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${(value || 0) * 100}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: "0.72rem", fontWeight: 700, color, minWidth: 28, textAlign: "right" }}>
        {((value || 0) * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function LiveDot({ color = "#01a982" }) {
  return (
    <span style={{
      display: "inline-block", width: 7, height: 7, borderRadius: "50%",
      background: color, flexShrink: 0,
      boxShadow: `0 0 0 2px ${color}33`,
      animation: "livepulse 2s ease-in-out infinite",
    }} />
  );
}

function SidebarSection({ label, children, action }) {
  return (
    <div className="sidebar-section" style={{ paddingTop: 14 }}>
      <div className="section-header" style={{ marginBottom: 8 }}>
        <span style={{ fontSize: "0.78rem", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--hpe-soft-ink)" }}>
          {label}
        </span>
        {action}
      </div>
      {children}
    </div>
  );
}

function TypingIndicator({ modelVariant }) {
  return (
    <div className="message-row assistant">
      <div className="message-card typing-card">
        <div className="message-role">
          <span>Assistant · {modelVariant || "routing…"}</span>
        </div>
        <div className="typing-dots"><span /><span /><span /></div>
      </div>
    </div>
  );
}

function AttachmentChip({ file, onRemove, sent }) {
  const isImage = (file.type || "").startsWith("image/");
  const previewUrl = useMemo(
    () => (isImage ? URL.createObjectURL(file) : null),
    [file, isImage],
  );
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  return (
    <div className={`attachment-chip${sent ? " sent" : ""}`}>
      {previewUrl
        ? <img src={previewUrl} alt={file.name} style={{ width: 22, height: 22, borderRadius: 4, objectFit: "cover" }} />
        : <span>📎</span>}
      <span>{file.name}</span>
      <small>{(file.size / 1024).toFixed(1)} KB</small>
      {!sent && onRemove && <button onClick={onRemove} aria-label="Remove">✕</button>}
    </div>
  );
}

function GuardrailStatusPill({ status, size = "sm" }) {
  const cfg = {
    allow:    { bg: "rgba(1,169,130,0.12)",   color: "#015c47", label: "ALLOW" },
    block:    { bg: "rgba(201,64,64,0.12)",   color: "#c94040", label: "BLOCK" },
    warn:     { bg: "rgba(230,168,23,0.18)",  color: "#a07010", label: "WARN" },
    redact:   { bg: "rgba(230,168,23,0.18)",  color: "#a07010", label: "REDACT" },
    skipped:  { bg: "rgba(120,120,120,0.10)", color: "#888",    label: "SKIP" },
    allowed:  { bg: "rgba(1,169,130,0.18)",   color: "#015c47", label: "ALLOWED",  dot: true },
    blocked:  { bg: "rgba(201,64,64,0.15)",   color: "#c94040", label: "BLOCKED",  dot: true },
    redacted: { bg: "rgba(230,168,23,0.20)",  color: "#a07010", label: "REDACTED", dot: true },
  };
  const c = cfg[status] || cfg.allow;
  const padding = size === "lg" ? "3px 11px" : "2px 8px";
  const fontSize = size === "lg" ? "0.74rem" : "0.68rem";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding, borderRadius: 999,
      background: c.bg, color: c.color,
      fontWeight: 700, fontSize, letterSpacing: "0.02em",
      border: `1px solid ${c.color}33`,
    }}>
      {c.dot && (
        <span style={{
          width: 6, height: 6, borderRadius: "50%", background: c.color, display: "inline-block",
        }} />
      )}
      {c.label}
    </span>
  );
}

function GuardrailStatusIcon({ status }) {
  const m = {
    allow:   { color: "#01a982", glyph: "✓" },
    block:   { color: "#c94040", glyph: "✕" },
    warn:    { color: "#e6a817", glyph: "!" },
    redact:  { color: "#e6a817", glyph: "✎" },
    skipped: { color: "#aaa",    glyph: "○" },
  };
  const c = m[status] || m.allow;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center",
      width: 18, height: 18, borderRadius: 999, flexShrink: 0,
      background: `${c.color}22`, color: c.color,
      fontWeight: 800, fontSize: "0.72rem", lineHeight: 1,
    }}>{c.glyph}</span>
  );
}

function PipelineRow({ step }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "7px 11px", borderRadius: 6,
      background: "#fff", margin: "3px 0",
      border: "1px solid #ecf2ee",
    }}>
      <GuardrailStatusIcon status={step.status} />
      <code style={{ color: "#999", fontSize: "0.72rem", minWidth: 26 }}>{step.id}</code>
      <strong style={{ fontSize: "0.82rem", minWidth: 180, flexShrink: 0 }}>{step.label}</strong>
      <span style={{ flex: 1, fontSize: "0.76rem", color: "#666", fontStyle: step.detail ? "normal" : "italic" }}>
        {step.detail || "—"}
      </span>
      <GuardrailStatusPill status={step.status} />
    </div>
  );
}

function GuardrailPipelineTrace({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace) return null;
  const input = trace.input || null;
  const output = trace.output || null;
  const inputSteps = (input && input.steps) || [];
  const outputSteps = (output && output.steps) || [];
  const totalSteps = inputSteps.length + outputSteps.length;
  if (totalSteps === 0) return null;
  const totalLatency = (input?.latency_ms || 0) + (output?.latency_ms || 0);

  let overall = "allowed";
  if (input?.blocked || output?.blocked) overall = "blocked";
  else if ((output?.redactions || []).length > 0) overall = "redacted";

  return (
    <div style={{
      border: "1px solid var(--hpe-border)", borderRadius: 8,
      margin: "10px 0", background: "#fbfdfc", overflow: "hidden",
    }}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", display: "flex", alignItems: "center", gap: 8,
          padding: "9px 12px", background: "transparent", border: "none",
          cursor: "pointer", textAlign: "left", color: "inherit",
        }}
      >
        <span style={{ color: "#888", fontSize: "0.7rem", width: 10 }}>{open ? "▼" : "▶"}</span>
        <span style={{
          width: 18, height: 18, borderRadius: 4,
          background: "rgba(1,169,130,0.18)", color: "var(--hpe-deep-green)",
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          fontWeight: 800, fontSize: "0.85rem", lineHeight: 1,
        }}>＋</span>
        <strong style={{ fontSize: "0.88rem" }}>Guardrail Pipeline Trace</strong>
        <span style={{ flex: 1 }} />
        <small style={{ color: "#777", marginRight: 8, fontSize: "0.78rem" }}>
          {totalLatency.toFixed(0)}ms
        </small>
        <GuardrailStatusPill status={overall} size="lg" />
      </button>

      {open && (
        <div style={{ padding: "2px 12px 12px", borderTop: "1px solid var(--hpe-border)" }}>
          <div style={{
            fontSize: "0.74rem", color: "#888",
            margin: "6px 0 8px", fontStyle: "italic",
          }}>
            Hide guardrail trace ({totalSteps} steps)
          </div>

          {inputSteps.length > 0 && (
            <>
              <div style={{
                fontSize: "0.74rem", fontWeight: 700, color: "#555",
                textTransform: "none", margin: "6px 0 4px",
                background: "#f1f6f3", padding: "4px 10px", borderRadius: 4,
              }}>
                Input Guardrails (before LLM)
              </div>
              {inputSteps.map(s => <PipelineRow key={s.id} step={s} />)}
            </>
          )}

          {!input?.blocked && outputSteps.length > 0 && (
            <div style={{
              display: "flex", alignItems: "center", gap: 8,
              margin: "12px 0",
            }}>
              <div style={{ flex: 1, borderTop: "1px dashed #c3e0d3" }} />
              <span style={{
                background: "var(--hpe-deep-green, #01a982)", color: "white",
                padding: "3px 12px", borderRadius: 999,
                fontSize: "0.72rem", fontWeight: 700, letterSpacing: "0.04em",
                display: "inline-flex", alignItems: "center", gap: 4,
              }}>⚡ LLM CALLED</span>
              <div style={{ flex: 1, borderTop: "1px dashed #c3e0d3" }} />
            </div>
          )}

          {outputSteps.length > 0 && (
            <>
              <div style={{
                fontSize: "0.74rem", fontWeight: 700, color: "#555",
                margin: "6px 0 4px",
                background: "#f1f6f3", padding: "4px 10px", borderRadius: 4,
              }}>
                Output Guardrails (after LLM)
              </div>
              {outputSteps.map(s => <PipelineRow key={s.id} step={s} />)}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ModalityBadge({ modality, mm }) {
  const isGen = modality === "image-gen";
  const label = isGen ? "Image generation" : "Image analysis";
  const icon = isGen ? "🎨" : "🖼";
  const backendLabel = { nim: "NIM live", huggingface: "HF live", fallback: "placeholder" };
  const live = mm?.backend === "nim" || mm?.backend === "huggingface";
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 8, flexWrap: "wrap",
      margin: "2px 0 8px", padding: "4px 10px", borderRadius: 999,
      background: "rgba(97,71,103,0.10)", border: "1px solid rgba(97,71,103,0.22)",
      fontSize: "0.74rem", color: "#614767", fontWeight: 700,
    }}>
      <span>{icon} {label}</span>
      {mm?.model && (
        <span style={{ fontFamily: "monospace", fontWeight: 600, opacity: 0.85 }}>{mm.model}</span>
      )}
      {isGen && mm?.steps != null && (
        <span style={{ fontWeight: 600, opacity: 0.85 }}>{mm.steps} steps</span>
      )}
      <span style={{
        padding: "0 6px", borderRadius: 999, fontWeight: 700,
        background: live ? "rgba(1,169,130,0.16)" : "rgba(230,168,23,0.16)",
        color: live ? "var(--hpe-deep-green)" : "#a07010",
      }}>
        {backendLabel[mm?.backend] || (live ? "live" : "placeholder")}
      </span>
    </div>
  );
}

function FeedbackControl({ msg }) {
  const initial = msg.feedback?.rating ?? null;
  const [vote, setVote] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(false);

  // Only rate real, persisted assistant messages that have text content.
  if (!msg?.id || String(msg.id).startsWith("opt-") || !((msg.content || "").trim())) {
    return null;
  }

  const submit = async (rating) => {
    if (busy) return;
    const prev = vote;
    setVote(rating); setBusy(true); setErr(false);
    try {
      const res = await sendFeedback({ messageId: msg.id, rating: rating > 0 ? "up" : "down" });
      // Carry the fresh stats the POST already returned so the feedback tile
      // updates without a second GET.
      window.dispatchEvent(new CustomEvent(FEEDBACK_EVENT, { detail: res?.stats || null }));
    } catch {
      setVote(prev); setErr(true);
    } finally {
      setBusy(false);
    }
  };

  const btnStyle = (active) => ({
    cursor: busy ? "default" : "pointer",
    border: "none", background: "transparent",
    fontSize: "1em", lineHeight: 1, padding: "2px 4px", borderRadius: 6,
    opacity: active ? 1 : 0.4,
    filter: active ? "none" : "grayscale(0.6)",
    transition: "opacity 0.15s ease",
  });

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 3, marginTop: 8 }}>
      <span style={{ fontSize: "0.75em", color: "var(--hpe-text-muted, #6b7280)", marginRight: 3 }}>
        Helpful?
      </span>
      <button type="button" style={btnStyle(vote === 1)} disabled={busy}
        title="Good response" aria-label="Thumbs up" onClick={() => submit(1)}>👍</button>
      <button type="button" style={btnStyle(vote === -1)} disabled={busy}
        title="Bad response" aria-label="Thumbs down" onClick={() => submit(-1)}>👎</button>
      {vote != null && !err && (
        <span style={{ fontSize: "0.72em", color: "#01a982", marginLeft: 4 }}>thanks</span>
      )}
      {err && (
        <span style={{ fontSize: "0.72em", color: "#c94040", marginLeft: 4 }}>couldn’t save</span>
      )}
    </div>
  );
}

function MessageCard({ msg }) {
  const [showDetails, setShowDetails] = useState(false);
  const isUser = msg.role === "user";
  const meta = msg.metadata || {};
  const sustainability = meta.sustainability || {};
  const routing = meta.routing || {};
  const retrieval = meta.retrieval || {};
  const grounding = meta.grounding || retrieval.grounding_verification || {};
  const evidence = retrieval.evidence_assessment || {};
  const inputUnderstanding = meta.input_understanding || {};

  const cssScore = sustainability.score ?? routing.css_score;
  const carbonG = sustainability.estimated_request_co2_g;
  const gridCI = sustainability.grid_carbon;
  const modelInfo = resolveModelInfo(meta);
  const attachments = isUser ? (meta.attachments || []) : [];
  const modality = meta.modality || "text";
  const mm = meta.multimodal || null;
  const assistantImages = (!isUser && Array.isArray(meta.images)) ? meta.images : [];
  const userImages = isUser ? (meta.attachments || []).filter(a => a.image_data_uri) : [];

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      <div className="message-card">
        <div className="message-role">
          <span>
            <span>{isUser ? "You" : "Assistant"}</span>
            {modelInfo && !isUser && (
              <span style={{
                marginLeft: 8, padding: "1px 7px", borderRadius: 6,
                background: "rgba(1,169,130,0.12)", color: "var(--hpe-deep-green)",
                fontSize: "0.78em", fontWeight: 700, letterSpacing: "0.01em",
                display: "inline-flex", alignItems: "center", gap: 4,
              }}>
                {modelInfo.name}
                {modelInfo.variant && (
                  <span style={{
                    background: "rgba(1,169,130,0.18)", borderRadius: 4,
                    padding: "0 4px", fontSize: "0.88em", opacity: 0.85,
                  }}>[{modelInfo.variant}]</span>
                )}
                {modelInfo.escalated && (
                  <span style={{
                    background: "rgba(230,168,23,0.15)", borderRadius: 4, color: "#a07010",
                    padding: "0 4px", fontSize: "0.85em",
                  }} title={`Context overflow: escalated from ${modelInfo.originalVariant}`}>
                    ↑ ctx
                  </span>
                )}
              </span>
            )}
          </span>
          <span>{formatTs(msg.created_at)}</span>
        </div>

        {!isUser && modality && modality !== "text" && (
          <ModalityBadge modality={modality} mm={mm} />
        )}

        <div className="message-body">{msg.content}</div>

        {/* Generated / analysed images (assistant) or uploaded images (user) */}
        {assistantImages.length > 0 && (
          <div className="message-images" style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 10 }}>
            {assistantImages.map((src, i) => (
              <a key={i} href={src} target="_blank" rel="noreferrer" style={{ display: "block" }}>
                <img src={src} alt={`generated ${i + 1}`} style={{
                  maxWidth: 320, maxHeight: 320, width: "auto", borderRadius: 12,
                  border: "1px solid var(--hpe-border)", background: "#fff",
                }} />
              </a>
            ))}
          </div>
        )}
        {userImages.length > 0 && (
          <div className="message-images" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
            {userImages.map((a, i) => (
              <img key={i} src={a.image_data_uri} alt={a.name || `image ${i + 1}`} title={a.name} style={{
                maxWidth: 140, maxHeight: 140, width: "auto", borderRadius: 10,
                border: "1px solid var(--hpe-border)",
              }} />
            ))}
          </div>
        )}

        {!isUser && meta.guardrails && (
          <GuardrailPipelineTrace trace={meta.guardrails} />
        )}

        {attachments.length > 0 && (
          <div className="message-attachments">
            {attachments.map((a, i) => (
              <div key={i} className="attachment-chip sent">
                <span>📎</span><span>{a.name || "file"}</span>
              </div>
            ))}
          </div>
        )}

        {!isUser && <FeedbackControl msg={msg} />}

        {!isUser && (cssScore != null || carbonG != null || meta.guardrails || meta.blocked_by_guardrails) && (
          <div className="message-footer">
            <div className="message-meta">
              {(() => {
                const gr = meta.guardrails;
                if (!gr) return null;
                let s = "allowed";
                if (gr.input?.blocked || gr.output?.blocked) s = "blocked";
                else if ((gr.output?.redactions || []).length > 0) s = "redacted";
                return <GuardrailStatusPill status={s} />;
              })()}
              {cssScore != null && <span>CSS {cssScore.toFixed(3)}</span>}
              {carbonG != null && <span>{carbonLabel(carbonG)}</span>}
              {gridCI != null && <span>{gridCI.toFixed(0)} gCO₂/kWh</span>}
              {/* Token counts */}
              {meta.tokens?.total > 0 && (
                <span style={{ color: "#6a1b9a", fontWeight: 600 }}>
                  💭 {meta.tokens.input}↑ {meta.tokens.output}↓ {meta.tokens.total}t
                </span>
              )}
              {/* GPU utilisation */}
              {meta.gpu?.utilization_pct > 0 && (
                <span style={{
                  color: meta.gpu.constrained ? "#c94040" : "#01a982",
                  fontWeight: 600,
                }}>
                  GPU {meta.gpu.utilization_pct}%
                  {meta.gpu.utilization_source === "estimated" &&
                    <span style={{ fontSize: "0.72em", opacity: 0.7 }}> est.</span>}
                </span>
              )}
              {inputUnderstanding.intent && <span>intent: {inputUnderstanding.intent}</span>}
              {inputUnderstanding.complexity_label && <span>complexity: {inputUnderstanding.complexity_label}</span>}
              {grounding.reason && grounding.reason !== "not-grounded-request" && (
                <span style={{ color: grounding.supported ? "#015c47" : "#8a6000" }}>
                  {grounding.supported ? "grounded" : `guardrail: ${grounding.reason}`}
                </span>
              )}
              {routing.rl_controlled && (
                <span>🤖 RL v{routing.policy_coefficients?.rl_version}</span>
              )}
            </div>
            <button className="text-button" onClick={() => setShowDetails(v => !v)}>
              {showDetails ? "Hide details" : "Details"}
            </button>
          </div>
        )}

        {showDetails && !isUser && (
          <div className="details-panel expanded">
            {modelInfo && (
              <div style={{ gridColumn: "1/-1" }}>
                <small>Model used</small>
                <strong style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  {modelInfo.name}
                  {modelInfo.variant && (
                    <span style={{
                      fontSize: "0.8em", padding: "1px 6px", borderRadius: 4,
                      background: "rgba(1,169,130,0.12)", color: "var(--hpe-deep-green)",
                    }}>{modelInfo.variant}</span>
                  )}
                  {modelInfo.escalated && (
                    <span style={{
                      fontSize: "0.8em", padding: "1px 6px", borderRadius: 4,
                      background: "rgba(230,168,23,0.12)", color: "#a07010",
                    }} title={`Context window exceeded for ${modelInfo.originalVariant} — auto-escalated`}>
                      ↑ context overflow
                    </span>
                  )}
                </strong>
              </div>
            )}
            {routing.selected_region && (
              <div><small>Region</small><strong>{routing.selected_region}</strong></div>
            )}
            {evidence.evidence_strength && (
              <div><small>Evidence strength</small><strong>{evidence.evidence_strength}</strong></div>
            )}
            {grounding.reason && grounding.reason !== "not-grounded-request" && (
              <div>
                <small>Grounding check</small>
                <strong style={{ color: grounding.supported ? "#015c47" : "#8a6000" }}>
                  {grounding.supported ? "Supported by evidence" : grounding.reason}
                </strong>
              </div>
            )}
            {evidence.coverage_ratio != null && (
              <div><small>Evidence coverage</small><strong>{Math.round((evidence.coverage_ratio || 0) * 100)}%</strong></div>
            )}
            {sustainability.system_power_w > 0 && (
              <div><small>Power draw</small><strong>{sustainability.system_power_w?.toFixed(1)} W</strong></div>
            )}
            {sustainability.system_co2_g != null && (
              <div><small>System CO₂</small><strong>{carbonLabel(sustainability.system_co2_g)}</strong></div>
            )}

            {/* ── GPU section ── */}
            {meta.gpu?.gpu_available && (
              <>
                <div style={{ gridColumn: "1/-1", borderTop: "1px solid var(--hpe-border)", paddingTop: 6, marginTop: 2 }}>
                  <small style={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--hpe-soft-ink)" }}>GPU</small>
                  {meta.gpu.utilization_source === "estimated" && (
                    <span style={{ marginLeft: 6, fontSize: "0.7rem", padding: "1px 6px", borderRadius: 999, background: "rgba(230,168,23,0.12)", color: "#a07010" }}>
                      estimated
                    </span>
                  )}
                </div>
                <div>
                  <small>Utilisation</small>
                  <strong style={{ color: meta.gpu.constrained ? "#c94040" : "#01a982" }}>
                    {meta.gpu.utilization_pct}%
                    {meta.gpu.utilization_source === "estimated" && <span style={{ fontSize: "0.8em", opacity: 0.7 }}> (est.)</span>}
                    {meta.gpu.constrained && " ⚠️ constrained"}
                  </strong>
                </div>
                <div><small>GPU power</small><strong>{meta.gpu.power_w} W</strong></div>
                <div><small>GPU CO₂</small><strong>{carbonLabel(meta.gpu.co2_g)}</strong></div>
                <div><small>Memory used</small><strong>{meta.gpu.used_memory_mb?.toFixed(0)} / {meta.gpu.total_memory_mb?.toFixed(0)} MB ({meta.gpu.memory_utilization_pct}%)</strong></div>
                {meta.gpu.temperature_c > 0 && (
                  <div><small>Temperature</small><strong style={{ color: meta.gpu.temperature_c > 80 ? "#c94040" : "inherit" }}>{meta.gpu.temperature_c}°C</strong></div>
                )}
                {meta.gpu.routing_adjusted && (
                  <div style={{ gridColumn: "1/-1" }}>
                    <small>GPU routing</small>
                    <strong style={{ color: "#e6a817" }}>⚡ Lighter model preferred (GPU &gt;80%)</strong>
                  </div>
                )}
              </>
            )}

            {/* ── Token section ── */}
            {meta.tokens?.total > 0 && (
              <>
                <div style={{ gridColumn: "1/-1", borderTop: "1px solid var(--hpe-border)", paddingTop: 6, marginTop: 2 }}>
                  <small style={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--hpe-soft-ink)" }}>Tokens</small>
                </div>
                <div><small>Input tokens</small><strong>{meta.tokens.input}</strong></div>
                <div><small>Output tokens</small><strong>{meta.tokens.output}</strong></div>
                <div><small>Total tokens</small><strong>{meta.tokens.total} / {meta.tokens.model_context_cap}</strong></div>
                {meta.tokens.co2_per_token_ug > 0 && (
                  <div><small>CO₂ per token</small><strong>{meta.tokens.co2_per_token_ug} µgCO₂</strong></div>
                )}
              </>
            )}

            {routing.eco_actions?.deferral_recommended && (
              <div><small>EcoServe</small><strong style={{ color: "#e6a817" }}>⏱ Deferral recommended</strong></div>
            )}
            {routing.policy_coefficients && (
              <div>
                <small>RL policy weights</small>
                <strong style={{ fontSize: "0.82em" }}>
                  c={routing.policy_coefficients.carbon?.toFixed(2)}{" "}
                  l={routing.policy_coefficients.latency?.toFixed(2)}{" "}
                  a={routing.policy_coefficients.accuracy?.toFixed(2)}{" "}
                  $={routing.policy_coefficients.cost?.toFixed(2)}
                </strong>
              </div>
            )}
            {retrieval.retrieved_count > 0 && (
              <div className="retrieval-sources">
                <small>RAG sources ({retrieval.retrieved_count})</small>
                {(retrieval.sources || []).slice(0, 3).map((s, i) => (
                  <div key={i} className="source-pill">
                    <span>{s.document_name}</span>
                    <small>score {s.score?.toFixed(3)} · chunk {s.chunk_id?.slice(-6)}</small>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function GreenAIChat() {
  const [activeTab, setActiveTab] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([]);

  const [input, setInput] = useState("");
  const [files, setFiles] = useState([]);
  const [persistAttachments, setPersistAttachments] = useState(false);

  const [sending, setSending] = useState(false);
  const [typingVariant, setTypingVariant] = useState(null);
  const [error, setError] = useState(null);

  // Tenant switcher state (X-Tenant-Id header for multi-tenant isolation)
  const [tenantId, setLocalTenantId] = useState(getTenantId());
  const [tenantPickerOpen, setTenantPickerOpen] = useState(false);
  const [tenantInput, setTenantInput] = useState(getTenantId());
  const [tenantError, setTenantError] = useState("");
  const [knownTenants, setKnownTenants] = useState(getKnownTenants());
  useEffect(() => onTenantChange((tid) => setLocalTenantId(tid)), []);
  useEffect(() => onTenantListChange(() => setKnownTenants(getKnownTenants())), []);
  const switchToTenant = (next) => {
    if (!next || next === tenantId) {
      setTenantPickerOpen(false);
      return;
    }
    try {
      setTenantId(next);
      setTenantPickerOpen(false);
      window.location.reload();
    } catch (err) {
      setTenantError(err.message || "Invalid tenant id.");
    }
  };
  const applyTenant = () => switchToTenant(tenantInput);
  const removeTenant = (target) => {
    removeKnownTenant(target);
    setKnownTenants(getKnownTenants());
  };

  // Live sidebar data
  const {
    allZones, primaryCI, primaryZone, primarySignal, bestZone, bestSignal,
    tierWeights, rlData,
    queueData,
    ragStatus, ragDocuments,
    gpuData,
    estimatorData,
    feedbackStats,
    refresh: refreshSidebar,
  } = useSidebarData(ROUTER_TIER);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // ── Conversation data ──────────────────────────────────────────────────────
  const loadConversations = useCallback(async () => {
    try {
      const data = await fetchConversations();
      setConversations(data.conversations || []);
    } catch { /* non-critical */ }
  }, []);

  const loadConversation = useCallback(async (id) => {
    if (!id) return;
    try {
      const data = await fetchConversation(id);
      setMessages((data.messages || []).filter(m => m.role !== "system"));
    } catch (err) { setError(err.message); }
  }, []);

  useEffect(() => { loadConversations(); }, [loadConversations]);
  useEffect(() => { if (activeConvId) loadConversation(activeConvId); }, [activeConvId, loadConversation]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, sending]);

  // ── Actions ────────────────────────────────────────────────────────────────
  const handleNewChat = () => { setActiveConvId(null); setMessages([]); setError(null); setInput(""); setFiles([]); };

  const handleSelectConversation = (id) => { setActiveConvId(id); setError(null); setActiveTab(0); };

  const handleDeleteConversation = async (e, id) => {
    e.stopPropagation();
    await deleteConversation(id);
    if (activeConvId === id) handleNewChat();
    loadConversations();
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed && files.length === 0) return;
    if (sending) return;

    setInput(""); setSending(true); setError(null); setTypingVariant("routing…");
    const optimistic = {
      id: `opt-${Date.now()}`, role: "user", content: trimmed,
      created_at: new Date().toISOString(),
      metadata: { attachments: files.map(f => ({ name: f.name })) },
    };
    setMessages(prev => [...prev, optimistic]);

    try {
      const data = await sendChatMessage({
        prompt: trimmed || "Analyse the attached files.",
        conversationId: activeConvId,
        persistAttachments,
        files,
      });
      const convId = data.conversation?.id;
      if (convId && convId !== activeConvId) setActiveConvId(convId);
      setMessages((data.messages || []).filter(m => m.role !== "system"));
      loadConversations();
      refreshSidebar();
    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.filter(m => m.id !== optimistic.id));
    } finally {
      setSending(false); setTypingVariant(null); setFiles([]);
    }
  };

  const handleKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } };

  const handleFileChange = (e) => {
    const newFiles = Array.from(e.target.files || []);
    setFiles(prev => {
      const combined = [...prev, ...newFiles];
      // Deduplicate by name+size
      const seen = new Set();
      return combined.filter(f => {
        const key = `${f.name}-${f.size}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
    });
    e.target.value = ""; // reset so same file can be re-added
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files || []);
    if (dropped.length) setFiles(prev => [...prev, ...dropped]);
  };

  const handleStarterClick = (prompt) => { setInput(prompt); textareaRef.current?.focus(); };

  const showHero = messages.length === 0 && !sending;
  const rlEpisodes = rlData?.tiers?.[ROUTER_TIER]?.episode_count ?? 0;
  const rlVersion = rlData?.tiers?.[ROUTER_TIER]?.policy_version ?? 0;

  return (
    <>
      {/* Keyframe for live dot animation */}
      <style>{`
        @keyframes livepulse {
          0%,100%{opacity:1;transform:scale(1)}
          50%{opacity:0.55;transform:scale(1.35)}
        }
      `}</style>

      <div className="chat-shell">
        {/* ── Sidebar ──────────────────────────────────────────────────── */}
        <aside className={`chat-sidebar${sidebarOpen ? "" : " collapsed"}`}>
          {/* Brand */}
          <div className="sidebar-header">
            <div className="brand-lockup">
              <div className="brand-mark">
                <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
                  <circle cx="16" cy="16" r="14" fill="#01a982" />
                  <path d="M10 20 Q16 8 22 20" stroke="white" strokeWidth="2.5" fill="none" strokeLinecap="round" />
                </svg>
              </div>
              <div>
                <p>HPE Adaptive Green AI</p>
                <span style={{ fontSize: "0.82rem" }}>Carbon-aware inference</span>
              </div>
            </div>

            {/* Tenant chip — opens an inline switcher; sets X-Tenant-Id */}
            <button
              type="button"
              onClick={() => {
                setTenantInput(tenantId);
                setTenantError("");
                setTenantPickerOpen(true);
              }}
              title={`Active tenant: ${tenantId}. Click to switch.`}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "8px 12px", borderRadius: 14,
                background: "rgba(1,169,130,0.08)",
                border: "1px solid rgba(1,169,130,0.25)",
                color: "#015c47", fontWeight: 600, fontSize: "0.84rem",
                cursor: "pointer", marginBottom: 8, width: "100%", justifyContent: "space-between",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{
                  display: "inline-block", width: 8, height: 8, borderRadius: 999,
                  background: "#01a982",
                }} />
                <span>Tenant: {tenantId}</span>
              </span>
              <span style={{ opacity: 0.6, fontSize: "0.78rem" }}>change</span>
            </button>

            {tenantPickerOpen && (
              <div style={{
                padding: "10px 12px", marginBottom: 8, borderRadius: 14,
                background: "white", border: "1px solid var(--hpe-border)",
                display: "flex", flexDirection: "column", gap: 8,
              }}>
                <strong style={{ fontSize: "0.85rem", color: "var(--hpe-strong-ink)" }}>
                  Switch tenant
                </strong>
                <span style={{ fontSize: "0.74rem", color: "var(--hpe-soft-ink)", lineHeight: 1.4 }}>
                  Sets the X-Tenant-Id header. Conversations, RAG documents,
                  budgets, observability, and CSRD reports are isolated per
                  tenant. Format: lowercase, 1–64 chars, [a–z 0–9 _ -].
                </span>

                {/* Known tenants — quick switch */}
                {knownTenants.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ fontSize: "0.7rem", color: "var(--hpe-soft-ink)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                      Known tenants
                    </span>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {knownTenants.map((t) => {
                        const isActive = t === tenantId;
                        const removable = t !== "default" && !isActive;
                        return (
                          <span
                            key={t}
                            style={{
                              display: "inline-flex", alignItems: "center", gap: 4,
                              padding: "4px 4px 4px 10px", borderRadius: 999,
                              fontSize: "0.78rem", fontWeight: 600,
                              background: isActive ? "rgba(1,169,130,0.18)" : "rgba(15,95,89,0.06)",
                              color: isActive ? "#015c47" : "#0f5f59",
                              border: isActive ? "1px solid rgba(1,169,130,0.45)" : "1px solid transparent",
                            }}
                          >
                            <button
                              type="button"
                              onClick={() => switchToTenant(t)}
                              disabled={isActive}
                              title={isActive ? "Active tenant" : `Switch to ${t}`}
                              style={{
                                background: "transparent", border: "none", padding: 0,
                                color: "inherit", cursor: isActive ? "default" : "pointer",
                                fontWeight: 600, fontSize: "0.78rem",
                              }}
                            >
                              {t}{isActive ? " ●" : ""}
                            </button>
                            {removable && (
                              <button
                                type="button"
                                onClick={() => removeTenant(t)}
                                title={`Remove ${t} from list`}
                                aria-label={`Remove ${t}`}
                                style={{
                                  width: 18, height: 18, borderRadius: 999,
                                  border: "none", background: "rgba(15,95,89,0.12)",
                                  color: "#0f5f59", cursor: "pointer",
                                  display: "inline-flex", alignItems: "center", justifyContent: "center",
                                  padding: 0, fontSize: "0.7rem", lineHeight: 1,
                                }}
                              >
                                ×
                              </button>
                            )}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Free-form input — adds to the list on Apply */}
                <span style={{ fontSize: "0.7rem", color: "var(--hpe-soft-ink)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                  Add or pick a new tenant
                </span>
                <input
                  type="text"
                  value={tenantInput}
                  onChange={(e) => { setTenantInput(e.target.value); setTenantError(""); }}
                  onKeyDown={(e) => { if (e.key === "Enter") applyTenant(); }}
                  placeholder="default"
                  style={{
                    padding: "8px 10px", borderRadius: 10,
                    border: "1px solid var(--hpe-border)", fontSize: "0.85rem",
                  }}
                />
                {tenantError && (
                  <span style={{ fontSize: "0.75rem", color: "#c94040" }}>
                    {tenantError}
                  </span>
                )}
                <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                  <button
                    type="button"
                    onClick={() => setTenantPickerOpen(false)}
                    style={{
                      padding: "6px 12px", borderRadius: 10,
                      background: "transparent", border: "1px solid var(--hpe-border)",
                      cursor: "pointer", fontSize: "0.8rem",
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={applyTenant}
                    className="primary-button"
                    style={{ borderRadius: 10, padding: "6px 12px", fontSize: "0.8rem" }}
                  >
                    Apply
                  </button>
                </div>
              </div>
            )}

            <button className="primary-button" onClick={handleNewChat} style={{ borderRadius: 14 }}>
              + New chat
            </button>
          </div>

          {/* ── LIVE: Grid Carbon Signal ────────────────────────────────── */}
          <SidebarSection
            label="Grid Carbon"
            action={
              <button className="text-button" onClick={refreshSidebar} title="Refresh" style={{ fontSize: "0.9rem" }}>↻</button>
            }
          >
            <div style={{
              padding: "12px 14px", borderRadius: 16,
              background: `linear-gradient(135deg, ${ciColor(primaryCI)}18, ${ciColor(primaryCI)}08)`,
              border: `1px solid ${ciColor(primaryCI)}30`,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <LiveDot color={ciColor(primaryCI)} />
                <strong style={{ fontSize: "1.1rem", color: ciColor(primaryCI) }}>
                  {primaryCI != null ? `${primaryCI.toFixed(0)} gCO₂/kWh` : "Waiting…"}
                </strong>
                <span style={{
                  marginLeft: "auto", fontSize: "0.72rem", fontWeight: 700, padding: "2px 8px",
                  borderRadius: 999, background: `${ciColor(primaryCI)}22`, color: ciColor(primaryCI),
                }}>
                  {ciLabel(primaryCI)}
                </span>
              </div>
              {primaryZone && (
                <div style={{ fontSize: "0.78rem", color: "var(--hpe-soft-ink)", marginBottom: 6 }}>Zone: {primaryZone}</div>
              )}
              {primarySignal?.detail && (
                <div style={{ fontSize: "0.76rem", color: "var(--hpe-soft-ink)", lineHeight: 1.45, marginBottom: 6 }}>
                  {primarySignal.detail}
                </div>
              )}
              {(primarySignal?.provider || primarySignal?.source || primarySignal?.last_updated) && (
                <div style={{ fontSize: "0.72rem", color: "var(--hpe-soft-ink)", display: "grid", gap: 2, marginBottom: 6 }}>
                  {primarySignal?.provider && <span>Provider: {primarySignal.provider}</span>}
                  {primarySignal?.source && <span>Source: {primarySignal.source}</span>}
                  {primarySignal?.last_updated && <span>Updated: {formatTs(primarySignal.last_updated)}</span>}
                </div>
              )}
              {bestZone && bestZone !== primaryZone && bestSignal && (
                <div style={{
                  marginBottom: 8, padding: "8px 10px", borderRadius: 12,
                  background: "rgba(1,169,130,0.08)", border: "1px solid rgba(1,169,130,0.18)",
                  fontSize: "0.76rem", color: "#015c47",
                }}>
                  Greenest zone now: <strong>{bestZone}</strong> at {bestSignal?.carbon_intensity != null ? Number(bestSignal.carbon_intensity).toFixed(0) : "n/a"} gCO2/kWh
                </div>
              )}
              {/* All zones */}
              {allZones.length > 1 && allZones.map(([zone, ci]) => (
                <div key={zone} style={{ marginBottom: 5 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.76rem", marginBottom: 2 }}>
                    <span style={{ color: "var(--hpe-soft-ink)" }}>{zone}</span>
                    <span style={{ fontWeight: 700, color: ciColor(ci) }}>{ci.toFixed(0)}</span>
                  </div>
                  <CIBar value={ci} />
                </div>
              ))}
              {allZones.length === 1 && <CIBar value={primaryCI} />}
            </div>

            {/* EcoServe defer hint */}
            {primaryCI != null && primaryCI > 450 && (
              <div style={{
                marginTop: 8, padding: "8px 12px", borderRadius: 12,
                background: "rgba(230,168,23,0.1)", border: "1px solid rgba(230,168,23,0.3)",
                fontSize: "0.8rem", color: "#8a6000",
              }}>
                ⏱ High carbon — EcoServe may defer low-priority requests
              </div>
            )}
          </SidebarSection>

          {/* ── LIVE: Active RL Policy Weights ─────────────────────────── */}
          <SidebarSection label="Routing policy">
            {tierWeights ? (
              <div style={{
                padding: "10px 12px", borderRadius: 14,
                background: "rgba(255,255,255,0.72)",
                border: "1px solid var(--hpe-border)",
                display: "flex", flexDirection: "column", gap: 6,
              }}>
                <WeightMiniBar label="Carbon" value={tierWeights.carbon} color="#388e3c" />
                <WeightMiniBar label="Latency" value={tierWeights.latency} color="#1565c0" />
                <WeightMiniBar label="Accuracy" value={tierWeights.accuracy} color="#6a1b9a" />
                <WeightMiniBar label="Cost" value={tierWeights.cost} color="#e65100" />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: "0.72rem", color: "var(--hpe-soft-ink)" }}>
                  <span>🤖 Online REINFORCE · v{rlVersion}</span>
                  <span>{rlEpisodes} episodes</span>
                </div>
              </div>
            ) : (
              <div className="empty-sidebar-card">RL controller loading…</div>
            )}
          </SidebarSection>

          {/* ── LIVE: Learned quality/latency estimator ──────────────────── */}
          <SidebarSection label="Router ML estimator">
            {estimatorData ? (() => {
              const variants = Object.entries(estimatorData.variants || {});
              const warmup = estimatorData.warmup_min_obs ?? 8;
              const trustedCount = variants.filter(([, v]) => v.trusted).length;
              return (
                <div style={{
                  padding: "10px 12px", borderRadius: 14,
                  background: estimatorData.enabled ? "rgba(255,255,255,0.72)" : "rgba(0,0,0,0.03)",
                  border: "1px solid var(--hpe-border)",
                  display: "flex", flexDirection: "column", gap: 7,
                }}>
                  {/* Status header */}
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <LiveDot color={estimatorData.enabled ? "#01a982" : "#9aa0a6"} />
                    <strong style={{ fontSize: "0.82rem", color: estimatorData.enabled ? "#01a982" : "#9aa0a6" }}>
                      {estimatorData.enabled ? "Learning active" : "Disabled"}
                    </strong>
                    <span style={{ fontSize: "0.72rem", color: "var(--hpe-soft-ink)", marginLeft: "auto" }}>
                      {estimatorData.total_updates ?? 0} updates
                    </span>
                  </div>

                  {/* One-line explainer */}
                  <div style={{ fontSize: "0.72rem", color: "var(--hpe-soft-ink)", lineHeight: 1.35 }}>
                    Refines per-prompt accuracy &amp; latency into CSS · carbon untouched · cold-start = baseline
                  </div>

                  {/* Per-variant learned state */}
                  {variants.length === 0 ? (
                    <div style={{ fontSize: "0.72rem", color: "var(--hpe-soft-ink)" }}>
                      No observations yet — routing on static baselines.
                    </div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      {variants.map(([name, v]) => {
                        const pct = Math.min(100, ((v.n_obs || 0) / warmup) * 100);
                        return (
                          <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontSize: "0.72rem", color: "var(--hpe-ink)", width: 74, flexShrink: 0, fontFamily: "monospace" }}>
                              {name}
                            </span>
                            <div style={{ flex: 1, height: 5, background: "rgba(0,0,0,0.08)", borderRadius: 4, overflow: "hidden" }}>
                              <div style={{
                                width: `${pct}%`, height: "100%",
                                background: v.trusted ? "#01a982" : "#e6a817",
                                borderRadius: 4, transition: "width 0.5s ease",
                              }} />
                            </div>
                            <span style={{
                              fontSize: "0.68rem", fontWeight: 700, minWidth: 58, textAlign: "right",
                              color: v.trusted ? "#01a982" : "#a07010",
                            }}>
                              {v.trusted ? "applied" : `${v.n_obs}/${warmup}`}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  <div style={{ fontSize: "0.7rem", color: "var(--hpe-soft-ink)", borderTop: "1px solid var(--hpe-border)", paddingTop: 5 }}>
                    🧩 {trustedCount}/{variants.length} variants applied · warm-up ≥ {warmup} obs · lr {estimatorData.learning_rate}
                  </div>
                </div>
              );
            })() : (
              <div className="empty-sidebar-card">Estimator loading…</div>
            )}
          </SidebarSection>

          {/* ── LIVE: GPU Metrics ────────────────────────────────────────── */}
          <SidebarSection label="GPU Utilisation">
            {gpuData?.gpu ? (() => {
              const g = gpuData.gpu;
              const utilColor = g.utilization_pct > 80 ? "#c94040"
                : g.utilization_pct > 50 ? "#e6a817" : "#01a982";
              const memPct = g.memory_utilization_pct ?? 0;
              return (
                <div style={{
                  padding: "10px 14px", borderRadius: 14,
                  background: g.constrained ? "rgba(201,64,64,0.06)" : "rgba(255,255,255,0.72)",
                  border: `1px solid ${g.constrained ? "rgba(201,64,64,0.25)" : "var(--hpe-border)"}`,
                  display: "flex", flexDirection: "column", gap: 7,
                }}>
                  {/* Header row */}
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <LiveDot color={utilColor} />
                    <strong style={{ fontSize: "1.05rem", color: utilColor }}>
                      {g.utilization_pct}%
                    </strong>
                    {g.utilization_source === "estimated" && (
                      <span style={{
                        fontSize: "0.7rem", padding: "1px 7px", borderRadius: 999,
                        background: "rgba(230,168,23,0.1)", color: "#a07010",
                        fontWeight: 600, border: "1px solid rgba(230,168,23,0.25)",
                      }}>~ estimated</span>
                    )}
                    <span style={{ fontSize: "0.76rem", color: "var(--hpe-soft-ink)", marginLeft: "auto" }}>
                      util · {g.power_w} W
                    </span>
                    {g.constrained && (
                      <span style={{
                        fontSize: "0.72rem", padding: "2px 7px", borderRadius: 999,
                        background: "rgba(201,64,64,0.12)", color: "#c94040", fontWeight: 700,
                      }}>⚡ Routing adjusted</span>
                    )}
                  </div>

                  {/* Utilisation bar */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.73rem", marginBottom: 3 }}>
                      <span style={{ color: "var(--hpe-soft-ink)" }}>Compute</span>
                      <span style={{ fontWeight: 700, color: utilColor }}>{g.utilization_pct}%</span>
                    </div>
                    <div style={{ height: 6, background: "rgba(0,0,0,0.08)", borderRadius: 4, overflow: "hidden" }}>
                      <div style={{
                        width: `${g.utilization_pct}%`, height: "100%",
                        background: utilColor, borderRadius: 4, transition: "width 0.6s ease",
                      }} />
                    </div>
                  </div>

                  {/* Memory bar */}
                  {g.total_memory_mb > 0 && (
                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.73rem", marginBottom: 3 }}>
                        <span style={{ color: "var(--hpe-soft-ink)" }}>VRAM</span>
                        <span style={{ fontWeight: 700 }}>
                          {g.used_memory_mb?.toFixed(0)} / {g.total_memory_mb?.toFixed(0)} MB
                        </span>
                      </div>
                      <div style={{ height: 6, background: "rgba(0,0,0,0.08)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{
                          width: `${memPct}%`, height: "100%",
                          background: memPct > 90 ? "#c94040" : "#1565c0",
                          borderRadius: 4, transition: "width 0.6s ease",
                        }} />
                      </div>
                    </div>
                  )}

                  {/* Temperature + perf state */}
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem" }}>
                    {g.temperature_c > 0 && (
                      <span style={{ color: g.temperature_c > 80 ? "#c94040" : "var(--hpe-soft-ink)" }}>
                        🌡 {g.temperature_c}°C
                      </span>
                    )}
                    <span style={{ color: "var(--hpe-soft-ink)" }}>
                      {g.performance_state !== "unknown" ? `P-state: ${g.performance_state}` : ""}
                    </span>
                  </div>

                  {/* System CO2 */}
                  {gpuData.system?.co2_emission_g > 0 && (
                    <div style={{ fontSize: "0.75rem", color: "var(--hpe-soft-ink)", borderTop: "1px solid var(--hpe-border)", paddingTop: 5 }}>
                      System CO₂:{" "}
                      <strong style={{ color: "var(--hpe-deep-green)" }}>
                        {carbonLabel(gpuData.system.co2_emission_g)}
                      </strong>
                      {" "}· total power{" "}
                      <strong>{gpuData.system.total_power_w} W</strong>
                    </div>
                  )}
                </div>
              );
            })() : (
              <div className="empty-sidebar-card">GPU metrics loading…</div>
            )}
          </SidebarSection>

          {/* ── User feedback (fine-tuning dataset signal) ─────────────── */}
          <SidebarSection label="User Feedback">
            {feedbackStats ? (() => {
              const up = feedbackStats.up || 0;
              const down = feedbackStats.down || 0;
              const total = feedbackStats.total || 0;
              const upPct = total ? Math.round((up / total) * 100) : 0;
              return (
                <div style={{
                  padding: "10px 14px", borderRadius: 14,
                  background: "rgba(255,255,255,0.72)",
                  border: "1px solid var(--hpe-border)",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
                    <span style={{ fontSize: "0.74rem", color: "var(--hpe-soft-ink)" }}>
                      Fine-tuning dataset
                    </span>
                    <strong style={{ fontSize: "1.05rem", color: "var(--hpe-deep-green)" }}>
                      {total} rated
                    </strong>
                  </div>
                  {/* up/down split bar */}
                  <div style={{ display: "flex", height: 6, borderRadius: 4, overflow: "hidden", background: "rgba(0,0,0,0.08)", marginBottom: 8 }}>
                    <div style={{ width: `${upPct}%`, background: "#01a982", transition: "width 0.5s ease" }} />
                    <div style={{ width: `${100 - upPct}%`, background: "#c94040", transition: "width 0.5s ease" }} />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 600 }}>
                    <span style={{ color: "#01a982" }}>👍 {up}</span>
                    <span style={{ color: "var(--hpe-soft-ink)" }}>
                      {total ? `${upPct}% positive` : "no votes yet"}
                    </span>
                    <span style={{ color: "#c94040" }}>{down} 👎</span>
                  </div>
                </div>
              );
            })() : (
              <div className="empty-sidebar-card">Feedback loading…</div>
            )}
          </SidebarSection>

          {/* ── LIVE: EcoServe Queue (fully automatic) ─────────────────── */}
          <SidebarSection label="EcoServe Queue">

            {/* Auto-dispatch mode badge */}
            <div style={{
              display: "flex", alignItems: "center", gap: 6, marginBottom: 8,
              padding: "5px 10px", borderRadius: 20,
              background: "rgba(1,169,130,0.08)",
              border: "1px solid rgba(1,169,130,0.2)",
            }}>
              <LiveDot color="#01a982" />
              <span style={{ fontSize: "0.74rem", color: "#015c47", fontWeight: 600 }}>
                Auto-dispatch active · checks every 10 s
              </span>
            </div>

            {/* Main queue card */}
            <div style={{
              padding: "10px 14px", borderRadius: 14,
              background: (queueData?.queue_size ?? 0) > 0
                ? "rgba(230,168,23,0.08)" : "rgba(255,255,255,0.72)",
              border: `1px solid ${
                (queueData?.queue_size ?? 0) > 0
                  ? "rgba(230,168,23,0.3)" : "var(--hpe-border)"}
              `,
              display: "flex", flexDirection: "column", gap: 6,
            }}>
              {/* Counts row */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: "1.1rem" }}>
                    {(queueData?.queue_size ?? 0) > 0 ? "⏳" : "✅"}
                  </span>
                  <div>
                    <div style={{ fontSize: "0.85rem", fontWeight: 700 }}>
                      {queueData?.queue_size ?? 0} pending
                    </div>
                    <div style={{ fontSize: "0.72rem", color: "var(--hpe-soft-ink)" }}>
                      {queueData?.dispatched_total ?? 0} dispatched total
                    </div>
                  </div>
                </div>
                {(queueData?.queue_size ?? 0) === 0 && (
                  <span style={{
                    fontSize: "0.72rem", padding: "2px 8px", borderRadius: 999,
                    background: "rgba(1,169,130,0.12)", color: "#015c47", fontWeight: 700,
                  }}>Clear</span>
                )}
              </div>

              {/* Threshold row */}
              {queueData?.high_carbon_threshold != null && (
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.76rem" }}>
                  <span style={{ color: "var(--hpe-soft-ink)" }}>Dispatch below</span>
                  <span style={{ fontWeight: 700, color: ciColor(queueData.high_carbon_threshold) }}>
                    {queueData.high_carbon_threshold} gCO₂/kWh
                  </span>
                </div>
              )}

              {/* Current CI vs threshold */}
              {queueData?.current_carbon_g_per_kwh != null && primaryCI != null && (
                <div style={{ fontSize: "0.75rem", color: "var(--hpe-soft-ink)" }}>
                  Current grid: <strong style={{ color: ciColor(primaryCI) }}>
                    {primaryCI.toFixed(0)} gCO₂/kWh
                  </strong>
                  {primaryCI < queueData.high_carbon_threshold
                    ? " — dispatching now"
                    : " — waiting for lower CI"}
                </div>
              )}

              {/* Per-pending-request list */}
              {(queueData?.pending_requests || []).slice(0, 3).map(r => {
                const secLeft = Math.max(r.seconds_until_deadline ?? 0, 0);
                const minutesLeft = Math.floor(secLeft / 60);
                const deadlinePct = Math.min(
                  100 - (secLeft / Math.max(secLeft + 60, 300)) * 100, 100
                );
                return (
                  <div key={r.request_id} style={{
                    padding: "6px 8px", borderRadius: 8,
                    background: "rgba(230,168,23,0.06)",
                    border: "1px solid rgba(230,168,23,0.2)",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.74rem", marginBottom: 3 }}>
                      <span style={{ fontFamily: "monospace", color: "var(--hpe-soft-ink)" }}>
                        #{r.request_id.slice(0, 8)}
                      </span>
                      <span style={{ fontWeight: 700, color: secLeft < 120 ? "#c94040" : "#8a6000" }}>
                        ⏱ {minutesLeft}m left
                      </span>
                    </div>
                    {/* Deadline countdown bar */}
                    <div style={{ height: 3, background: "rgba(0,0,0,0.08)", borderRadius: 2 }}>
                      <div style={{
                        width: `${deadlinePct}%`, height: "100%",
                        background: secLeft < 120 ? "#c94040" : "#e6a817",
                        borderRadius: 2, transition: "width 1s linear",
                      }} />
                    </div>
                  </div>
                );
              })}
              {(queueData?.pending_requests?.length ?? 0) > 3 && (
                <div style={{ fontSize: "0.74rem", color: "var(--hpe-soft-ink)", textAlign: "center" }}>
                  +{queueData.pending_requests.length - 3} more pending…
                </div>
              )}
            </div>

            {/* Explainer — always visible */}
            <div style={{
              marginTop: 6, padding: "6px 10px", borderRadius: 10,
              background: "rgba(1,169,130,0.05)",
              fontSize: "0.73rem", color: "var(--hpe-soft-ink)", lineHeight: 1.5,
            }}>
              🤖 Requests are held when grid CI is high and released automatically
              when a clean-energy window is detected — no action required.
            </div>
          </SidebarSection>

          {/* ── Request Controls ─────────────────────────────────────────── */}
          <div className="sidebar-section">
            <div className="knowledge-card" style={{ marginBottom: 10 }}>
              <strong>Automatic semantic routing</strong>
              <span>SBERT infers priority, intent, and complexity so the router can keep simple prompts on lighter models and escalate only when the task or evidence really needs it.</span>
            </div>
            <div className="checkbox-row" style={{ marginTop: 8 }}>
              <input id="persist-attachments" type="checkbox" checked={persistAttachments} onChange={e => setPersistAttachments(e.target.checked)} />
              <label htmlFor="persist-attachments">Index attachments to RAG</label>
            </div>
          </div>

          {/* ── Conversations ─────────────────────────────────────────────── */}
          <div className="sidebar-section grow">
            <div className="section-header">
              <span>Conversations</span>
              <button className="text-button" onClick={loadConversations}>↻</button>
            </div>
            <div className="conversation-list">
              {conversations.length === 0 && (
                <div className="empty-sidebar-card">No conversations yet</div>
              )}
              {conversations.map(conv => (
                <div key={conv.id} className={`conversation-item${conv.id === activeConvId ? " active" : ""}`}>
                  <button className="conversation-button" onClick={() => handleSelectConversation(conv.id)}>
                    <strong>{conv.title || "Untitled"}</strong>
                    <span>{conv.message_count ?? 0} messages</span>
                  </button>
                  <button className="icon-button" onClick={e => handleDeleteConversation(e, conv.id)} aria-label="Delete" style={{ borderRadius: 10 }}>
                    🗑
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* ── Knowledge Base ───────────────────────────────────────────── */}
          <div className="sidebar-section">
            <div className="section-header"><span>Knowledge base</span></div>
            <div className="knowledge-card">
              <strong>{ragStatus?.document_count ?? 0} docs · {ragStatus?.chunk_count ?? 0} chunks</strong>
              <span>{ragStatus?.embedding_backend || "fallback"} backend</span>
            </div>
            {ragDocuments.slice(0, 3).map(doc => (
              <div key={doc.id} className="knowledge-item">
                <span>{doc.name || doc.document_name}</span>
                <span>{doc.chunk_count} chunks</span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── Main ─────────────────────────────────────────────────────── */}
        <main className="chat-main">
          <header className="chat-header">
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button className="icon-button mobile-only" onClick={() => setSidebarOpen(v => !v)} style={{ borderRadius: 12 }}>☰</button>
              <h1>Green AI Chat</h1>
            </div>

            <div className="header-actions">
              {TABS.map((tab, i) => (
                <button key={tab} className={`header-tab${activeTab === i ? " active" : ""}`} onClick={() => setActiveTab(i)}>{tab}</button>
              ))}
            </div>

            <div className="header-status">
              <span>Auto semantic routing</span>
              <span style={{ color: ciColor(primaryCI) }}>
                <LiveDot color={ciColor(primaryCI)} /> {primaryCI != null ? `${primaryCI.toFixed(0)} gCO₂/kWh` : "Grid…"}
              </span>
              {bestZone && <span>Best zone: {bestZone}</span>}
              {ragStatus?.document_count > 0 && <span>RAG: {ragStatus.document_count} docs</span>}
            </div>
          </header>

          {activeTab === 1 && <div className="chat-stage" style={{ overflowY: "auto", height: "100%" }}><CarbonDashboard /></div>}
          {activeTab === 2 && <div className="chat-stage" style={{ overflowY: "auto", height: "100%" }}><ObservabilityPanel /></div>}
          {activeTab === 3 && <div className="chat-stage" style={{ overflowY: "auto", height: "100%" }}><AgentPanel /></div>}
          {activeTab === 4 && <div className="chat-stage" style={{ overflowY: "auto", height: "100%" }}><BenchmarkPanel /></div>}

          {activeTab === 0 && (
            <>
              <div className="chat-stage">
                <div className="messages-scroll">
                  {error && <div className="error-banner">{error}</div>}

                  {showHero && (
                    <div className="hero-panel">
                      <div className="hero-copy">
                        <p className="eyebrow">Adaptive Green AI</p>
                        <h2>Carbon-aware answers, grounded in your data</h2>
                        <p>Every request is routed to the most sustainable model using real-time grid carbon signals, LLMCarbon accounting, and an online RL policy that adapts automatically.</p>
                      </div>
                      <div className="starter-grid">
                        {STARTERS.map(s => (
                          <button key={s.label} className="starter-card" onClick={() => handleStarterClick(s.prompt)}>
                            <strong>{s.label}</strong>
                            <p style={{ margin: "6px 0 0", fontSize: "0.9rem", color: "var(--hpe-soft-ink)" }}>{s.prompt}</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {messages.map(msg => <MessageCard key={msg.id} msg={msg} />)}
                  {sending && <TypingIndicator modelVariant={typingVariant} />}
                  <div ref={messagesEndRef} />
                </div>
              </div>

              <div className="composer-panel">
                {files.length > 0 && (
                  <div className="pending-files" style={{ marginBottom: 10 }}>
                    {files.map((f, i) => (
                      <AttachmentChip key={i} file={f} onRemove={() => setFiles(prev => prev.filter((_, j) => j !== i))} />
                    ))}
                  </div>
                )}
                <div
                  className="composer-box"
                  onDragOver={e => e.preventDefault()}
                  onDrop={handleDrop}
                >
                  <textarea ref={textareaRef} id="chat-input" className="composer-input"
                    placeholder={files.length > 0
                      ? `${files.length} file${files.length > 1 ? 's' : ''} attached — add a question or send now…`
                      : "Ask anything — the router picks the greenest model…"}
                    value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
                    rows={2} disabled={sending}
                  />
                  <div className="composer-actions">
                    <input ref={fileInputRef} type="file" multiple id="file-upload-input"
                      accept=".pdf,.txt,.md,.csv,.json,.py,.js,.jsx,.ts,.tsx,.yaml,.yml,.log,.png,.jpg,.jpeg,.webp,.gif,.bmp,image/*"
                      style={{ display: "none" }} onChange={handleFileChange}
                    />
                    <button
                      className="icon-button"
                      style={{
                        borderRadius: 12,
                        background: files.length > 0 ? "rgba(1,169,130,0.12)" : undefined,
                        color: files.length > 0 ? "var(--hpe-deep-green)" : undefined,
                        position: "relative",
                      }}
                      onClick={() => fileInputRef.current?.click()}
                      disabled={sending}
                      title={`Attach files${files.length > 0 ? ` (${files.length} selected)` : ""}`}
                    >
                      📎
                      {files.length > 0 && (
                        <span style={{
                          position: "absolute", top: -4, right: -4,
                          background: "var(--hpe-deep-green)", color: "#fff",
                          fontSize: "0.65rem", fontWeight: 800, borderRadius: 999,
                          minWidth: 16, height: 16, display: "flex",
                          alignItems: "center", justifyContent: "center", padding: "0 3px",
                        }}>{files.length}</span>
                      )}
                    </button>
                    <button id="send-button" className="primary-button" style={{ borderRadius: 14 }}
                      onClick={handleSend}
                      disabled={sending || (input.trim() === "" && files.length === 0)}>
                      {sending ? "Thinking…" : "Send"}
                    </button>
                  </div>
                </div>
                {sending && <div className="inline-status">Routing to greenest available model with online RL policy…</div>}
              </div>
            </>
          )}
        </main>
      </div>
    </>
  );
}

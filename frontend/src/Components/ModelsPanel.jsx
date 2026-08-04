/**
 * ModelsPanel — Adaptive Green AI
 *
 * Browse Hugging Face, size a quantization plan against the headroom that
 * actually exists right now, download, serve, and register.
 *
 * This screen exists because of a measured negative result. A three-arm routing
 * comparison (1350 requests, frozen router) showed CSS losing to always-full on
 * *both* carbon and quality, and the cause was the menu rather than the policy: four zoo candidates all dispatch to the same
 * TinyLlama container with hand-written differing TDPs and accuracies, and
 * TinyLlama is both more verbose and less accurate than Qwen2.5-1.5B. A router
 * can only be as good as the rungs it ranks. Quantization builds real rungs.
 *
 * Three things are deliberate in this UI:
 *
 * 1. **An onboarded model is shown as not routable until it is measured.** CSS
 *    ranks on accuracy_baseline, and the shipped zoo declares `full` at 0.92
 *    while it measured 0.793. A pipeline that let a downloaded model
 *    declare its own accuracy would automate that gap, so the registry column
 *    that matters is "routable", not "downloaded".
 *
 * 2. **The plan shows what it rejected and why.** An operator seeing "4-bit"
 *    should be able to see that fp16 was skipped for VRAM rather than silently
 *    ignored, and how far short the box was.
 *
 * 3. **Capabilities that are off say why.** Dynamic serving needs the Docker
 *    socket mounted, which is root-equivalent on the host; local quantization
 *    needs an image that carries a calibration toolchain. Both are off by
 *    default, and a disabled button with no reason is a support ticket.
 */

import React, { useCallback, useEffect, useState } from "react";
import {
  fetchModelCapability,
  searchHuggingFaceModels,
  previewModelPlan,
  onboardModel,
  fetchOnboardJobs,
  fetchModelRegistry,
  serveOnboardedModel,
  unserveOnboardedModel,
  quantizeModel,
  fetchQuantizedArtifacts,
  artifactDownloadUrl,
  deleteQuantizedArtifact,
} from "../lib/api";
import { FineTunePanel } from "./FineTunePanel";

const PALETTE = {
  primary: "#01a982",
  deep:    "#0f5f59",
  ink:     "#0b1f1f",
  soft:    "#5f7272",
  border:  "#d4e3df",
  warn:    "#e6a817",
  danger:  "#c94040",
  blue:    "#1565c0",
  bg:      "#f4f8f7",
};

const SOURCE_BLURB = {
  prequantized: "Already quantized upstream — the calibration carbon was paid by someone else.",
  native: "Served as published. No quantization, so no quality loss.",
  inflight: "vLLM quantizes to 4-bit at load time. Costs throughput, not carbon.",
  local_quantize: "A local calibration pass. The most carbon-expensive option here — metered.",
};

const fmtGb = (x) => (x == null ? "—" : `${Number(x).toFixed(1)} GB`);
const fmtMb = (x) => (x == null ? "—" : `${Math.round(Number(x)).toLocaleString()} MB`);
const fmtInt = (x) => (x == null ? "—" : Number(x).toLocaleString());

function Pill({ tone = "soft", children }) {
  const bg = { ok: "#e6f6f1", warn: "#fdf4e0", bad: "#fbeaea", soft: "#eef3f2" }[tone];
  const fg = { ok: PALETTE.primary, warn: "#8a6400", bad: PALETTE.danger, soft: PALETTE.soft }[tone];
  return (
    <span style={{ background: bg, color: fg, borderRadius: 999, padding: "2px 10px", fontSize: 12, fontWeight: 600 }}>
      {children}
    </span>
  );
}

function Section({ title, subtitle, children, right }) {
  return (
    <section style={{ background: "#fff", border: `1px solid ${PALETTE.border}`, borderRadius: 14, padding: 18, marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 12 }}>
        <div>
          <h3 style={{ margin: 0, color: PALETTE.ink, fontSize: 16 }}>{title}</h3>
          {subtitle && <p style={{ margin: "4px 0 0", color: PALETTE.soft, fontSize: 13 }}>{subtitle}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

/** Host headroom. The MIG basis is on screen because it changes the number. */
function Resources({ resources }) {
  if (!resources) return null;
  const vramPct = resources.vram_total_mb ? (resources.vram_used_mb / resources.vram_total_mb) * 100 : 0;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
      <div>
        <div style={{ fontSize: 12, color: PALETTE.soft }}>VRAM free</div>
        <div style={{ fontSize: 22, fontWeight: 700, color: PALETTE.ink }}>{fmtMb(resources.vram_free_mb)}</div>
        <div style={{ height: 6, background: "#eef3f2", borderRadius: 4, marginTop: 6 }}>
          <div style={{ width: `${Math.min(100, vramPct)}%`, height: "100%", background: vramPct > 85 ? PALETTE.danger : PALETTE.primary, borderRadius: 4 }} />
        </div>
        <div style={{ fontSize: 11, color: PALETTE.soft, marginTop: 4 }}>
          of {fmtMb(resources.vram_total_mb)}
          {resources.vram_basis === "mig_instance" && " · MIG instance, not the whole board"}
          {resources.vram_basis === "gpu_board_mig_unparsed" && " · ⚠ board figure; MIG table unreadable, may overstate"}
        </div>
      </div>
      <div>
        <div style={{ fontSize: 12, color: PALETTE.soft }}>Admittable now</div>
        <div style={{ fontSize: 22, fontWeight: 700, color: PALETTE.ink }}>{fmtMb(resources.vram_budget_mb)}</div>
        <div style={{ fontSize: 11, color: PALETTE.soft, marginTop: 4 }}>after a {fmtMb(resources.vram_reserve_mb)} reserve</div>
      </div>
      <div>
        <div style={{ fontSize: 12, color: PALETTE.soft }}>Disk free</div>
        <div style={{ fontSize: 22, fontWeight: 700, color: PALETTE.ink }}>{fmtGb(resources.disk_free_gb)}</div>
        <div style={{ fontSize: 11, color: PALETTE.soft, marginTop: 4 }}>{fmtGb(resources.disk_budget_gb)} usable after reserve</div>
      </div>
    </div>
  );
}

/** The plan, including what it turned down. */
function PlanCard({ preview, onOnboard, onQuantizeOnly, busy, capability }) {
  if (!preview) return null;
  const plan = preview.plan || {};
  const admission = preview.admission || {};
  const canServe = capability?.serving?.enabled;

  return (
    <div style={{ border: `1px solid ${PALETTE.border}`, borderRadius: 12, padding: 14, background: PALETTE.bg }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontWeight: 700, color: PALETTE.ink }}>{preview.model?.repo_id}</div>
          <div style={{ fontSize: 12, color: PALETTE.soft }}>
            {plan.parameter_count_b ? `${plan.parameter_count_b}B params` : "size unknown"}
            {plan.parameter_basis && ` · ${plan.parameter_basis.replace(/_/g, " ")}`}
          </div>
        </div>
        <Pill tone={plan.fits ? "ok" : "bad"}>{plan.fits ? `${plan.quant_format} · ${plan.source}` : "does not fit"}</Pill>
      </div>

      <p style={{ margin: "10px 0", fontSize: 13, color: PALETTE.ink }}>{plan.reason}</p>
      {plan.fits && SOURCE_BLURB[plan.source] && (
        <p style={{ margin: "0 0 10px", fontSize: 12, color: PALETTE.soft }}>{SOURCE_BLURB[plan.source]}</p>
      )}

      {plan.fits && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10, fontSize: 13 }}>
          <div><span style={{ color: PALETTE.soft }}>VRAM</span><br /><strong>{fmtMb(plan.est_vram_mb)}</strong></div>
          <div><span style={{ color: PALETTE.soft }}>weights</span><br /><strong>{fmtMb(plan.est_weights_mb)}</strong></div>
          <div><span style={{ color: PALETTE.soft }}>KV cache</span><br /><strong>{fmtMb(plan.est_kv_cache_mb)}</strong></div>
          <div><span style={{ color: PALETTE.soft }}>download</span><br /><strong>{fmtGb(plan.est_disk_gb)}</strong></div>
          <div><span style={{ color: PALETTE.soft }}>gpu-mem-util</span><br /><strong>{plan.gpu_memory_utilization}</strong></div>
        </div>
      )}

      {Array.isArray(plan.rejected) && plan.rejected.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ cursor: "pointer", fontSize: 12, color: PALETTE.soft }}>
            {plan.rejected.length} option{plan.rejected.length === 1 ? "" : "s"} rejected — see why
          </summary>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12, color: PALETTE.soft }}>
            {plan.rejected.map((r, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                <strong>{r.quant_format}</strong>{r.source ? ` (${r.source})` : ""} — {r.reason}
              </li>
            ))}
          </ul>
        </details>
      )}

      {preview.prequantized_candidates?.length > 0 && (
        <details style={{ marginTop: 8 }}>
          <summary style={{ cursor: "pointer", fontSize: 12, color: PALETTE.soft }}>
            {preview.prequantized_candidates.length} pre-quantized build(s) found upstream
          </summary>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12 }}>
            {preview.prequantized_candidates.map((s) => (
              <li key={s.repo_id}>{s.repo_id} <Pill tone="ok">{s.quant_format}</Pill></li>
            ))}
          </ul>
        </details>
      )}

      {!admission.ok && plan.fits && (
        <p style={{ marginTop: 10, fontSize: 12, color: PALETTE.danger }}>Admission: {admission.detail}</p>
      )}

      <div style={{ display: "flex", gap: 10, marginTop: 14, alignItems: "center", flexWrap: "wrap" }}>
        <button
          disabled={!plan.fits || busy}
          onClick={() => onOnboard(false)}
          style={{
            background: plan.fits ? PALETTE.primary : "#c9d6d2", color: "#fff", border: 0,
            borderRadius: 8, padding: "8px 16px", fontWeight: 600, cursor: plan.fits && !busy ? "pointer" : "not-allowed",
          }}
        >
          {busy ? "Starting…" : "Download & register"}
        </button>
        <button
          disabled={!plan.fits || busy || !canServe || !admission.ok}
          onClick={() => onOnboard(true)}
          title={!canServe ? capability?.serving?.reason : undefined}
          style={{
            background: "transparent", color: PALETTE.deep, border: `1px solid ${PALETTE.border}`,
            borderRadius: 8, padding: "8px 16px", fontWeight: 600,
            cursor: plan.fits && canServe && admission.ok && !busy ? "pointer" : "not-allowed",
          }}
        >
          Download, register & serve
        </button>
        <button
          disabled={busy || !capability?.local_quantization?.enabled}
          onClick={onQuantizeOnly}
          title={
            capability?.local_quantization?.enabled
              ? "Runs the calibration pass and stops. Nothing enters the model zoo."
              : capability?.local_quantization?.reason
          }
          style={{
            background: "transparent", color: PALETTE.deep, border: `1px dashed ${PALETTE.border}`,
            borderRadius: 8, padding: "8px 16px", fontWeight: 600,
            cursor: capability?.local_quantization?.enabled && !busy ? "pointer" : "not-allowed",
          }}
        >
          Quantize only (downloadable)
        </button>
        <span style={{ fontSize: 12, color: PALETTE.soft }}>
          The first two register as <strong>unavailable</strong> — CSS cannot route to them until measured.
          Quantize-only registers nothing: it produces a checkpoint you can download and run elsewhere.
        </span>
      </div>
    </div>
  );
}

export function ModelsPanel() {
  const [capability, setCapability] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [registry, setRegistry] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [error, setError] = useState(null);
  const [maxModelLen, setMaxModelLen] = useState(2048);
  const [allowLocalQuantize, setAllowLocalQuantize] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [cap, j, reg, art] = await Promise.all([
        fetchModelCapability(),
        fetchOnboardJobs(20).catch(() => ({ jobs: [] })),
        fetchModelRegistry().catch(() => ({ models: [] })),
        fetchQuantizedArtifacts().catch(() => ({ artifacts: [] })),
      ]);
      setCapability(cap);
      setJobs(j.jobs || []);
      setRegistry(reg.models || []);
      setArtifacts(art.artifacts || []);
    } catch (e) {
      setError(String(e.message || e));
    }
  }, []);

  useEffect(() => {
    refresh();
    // Onboarding jobs are long (a multi-GB download, then a model load), so this
    // polls rather than assuming the operator will refresh by hand.
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const runSearch = async (e) => {
    e?.preventDefault();
    setSearching(true);
    setError(null);
    try {
      const r = await searchHuggingFaceModels({ q: query, limit: 20 });
      setResults(r.results || []);
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setSearching(false);
    }
  };

  const openPreview = async (repoId) => {
    setPreviewing(true);
    setError(null);
    setPreview(null);
    try {
      setPreview(await previewModelPlan(repoId, { maxModelLen, allowLocalQuantize }));
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setPreviewing(false);
    }
  };

  const startOnboard = async (autoServe) => {
    if (!preview?.model?.repo_id) return;
    setBusy(true);
    setError(null);
    try {
      await onboardModel({
        repo_id: preview.model.repo_id,
        max_model_len: maxModelLen,
        allow_local_quantize: allowLocalQuantize,
        auto_serve: autoServe,
      });
      await refresh();
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  };

  /**
   * Quantize and stop. No zoo entry, so CSS never sees the result — the
   * deliverable is a checkpoint to download, not a routing rung.
   */
  const startQuantizeOnly = async () => {
    if (!preview?.model?.repo_id) return;
    setBusy(true);
    setError(null);
    try {
      await quantizeModel({
        repo_id: preview.model.repo_id,
        max_model_len: maxModelLen,
        allow_local_quantize: true,
      });
      await refresh();
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  };

  const removeArtifact = async (artifactId) => {
    setError(null);
    try {
      await deleteQuantizedArtifact(artifactId);
      await refresh();
    } catch (err) {
      setError(String(err.message || err));
    }
  };

  const enabled = capability?.enabled;

  return (
    <div style={{ padding: 20, maxWidth: 1100, margin: "0 auto", background: PALETTE.bg, minHeight: "100%" }}>
      <div style={{ marginBottom: 18 }}>
        <h2 style={{ margin: 0, color: PALETTE.ink }}>Model onboarding</h2>
        <p style={{ margin: "6px 0 0", color: PALETTE.soft, fontSize: 13, maxWidth: 760 }}>
          Measurement showed the router losing to always-full on carbon <em>and</em> quality — not because
          carbon-aware routing is wrong, but because the ladder has no genuinely cheaper rung. This is where
          real rungs get added: browse Hugging Face, size a quantization plan against the VRAM actually free
          right now, and register the result as a candidate that CSS may only select once it has been measured.
        </p>
      </div>

      {error && (
        <div style={{ background: "#fbeaea", color: PALETTE.danger, border: `1px solid #f0c4c4`, borderRadius: 10, padding: 12, marginBottom: 14, fontSize: 13 }}>
          {error}
        </div>
      )}

      {capability && !enabled && (
        <div style={{ background: "#fdf4e0", border: "1px solid #f0dcae", borderRadius: 10, padding: 12, marginBottom: 14, fontSize: 13, color: "#8a6400" }}>
          Onboarding is disabled. Set <code>MODEL_ONBOARD_ENABLED=true</code> to enable it. Browsing and planning
          still work, so you can size a model before turning anything on.
        </div>
      )}

      <Section title="Host headroom" subtitle="What a new model may actually claim, measured now.">
        <Resources resources={capability?.resources} />
        <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
          <Pill tone={capability?.serving?.enabled ? "ok" : "warn"}>
            Dynamic serving {capability?.serving?.enabled ? "available" : "off"}
          </Pill>
          <Pill tone={capability?.local_quantization?.enabled ? "ok" : "warn"}>
            Local quantization {capability?.local_quantization?.enabled ? "available" : "off"}
          </Pill>
          <Pill tone={capability?.hf_token_present ? "ok" : "warn"}>
            HF token {capability?.hf_token_present ? "set" : "missing"}
          </Pill>
        </div>
        {!capability?.serving?.enabled && capability?.serving?.reason && (
          <p style={{ margin: "10px 0 0", fontSize: 12, color: PALETTE.soft }}>Serving: {capability.serving.reason}</p>
        )}
        {!capability?.local_quantization?.enabled && capability?.local_quantization?.reason && (
          <p style={{ margin: "6px 0 0", fontSize: 12, color: PALETTE.soft }}>Quantization: {capability.local_quantization.reason}</p>
        )}
      </Section>

      <Section
        title="Browse Hugging Face"
        subtitle="Search the hub, then size a candidate against this host before committing to a download."
      >
        <form onSubmit={runSearch} style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. Qwen2.5 1.5B, Phi-3-mini, Llama-3.2-1B"
            style={{ flex: "1 1 260px", padding: "9px 12px", borderRadius: 8, border: `1px solid ${PALETTE.border}` }}
          />
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: PALETTE.soft }}>
            ctx
            <input
              type="number"
              value={maxModelLen}
              min={512}
              step={512}
              onChange={(e) => setMaxModelLen(Number(e.target.value) || 2048)}
              style={{ width: 90, padding: "9px 8px", borderRadius: 8, border: `1px solid ${PALETTE.border}` }}
            />
          </label>
          <button type="submit" disabled={searching} style={{ background: PALETTE.primary, color: "#fff", border: 0, borderRadius: 8, padding: "9px 18px", fontWeight: 600, cursor: "pointer" }}>
            {searching ? "Searching…" : "Search"}
          </button>
        </form>

        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: PALETTE.soft, marginBottom: 12 }}>
          <input type="checkbox" checked={allowLocalQuantize} onChange={(e) => setAllowLocalQuantize(e.target.checked)} />
          Consider a local calibration pass — the most carbon-expensive option, and it competes with live traffic for the GPU.
        </label>

        {results.length > 0 && (
          <div style={{ maxHeight: 280, overflowY: "auto", border: `1px solid ${PALETTE.border}`, borderRadius: 10 }}>
            {results.map((r) => (
              <div
                key={r.repo_id}
                onClick={() => openPreview(r.repo_id)}
                style={{
                  padding: "10px 12px", borderBottom: `1px solid ${PALETTE.border}`, cursor: "pointer",
                  display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10,
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600, color: PALETTE.ink, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis" }}>{r.repo_id}</div>
                  <div style={{ fontSize: 11, color: PALETTE.soft }}>
                    {fmtInt(r.downloads)} downloads · {r.likes} likes
                    {r.gated && " · gated"}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                  {r.quant_format && <Pill tone="ok">{r.quant_format}</Pill>}
                  <Pill>plan →</Pill>
                </div>
              </div>
            ))}
          </div>
        )}

        {previewing && <p style={{ fontSize: 13, color: PALETTE.soft }}>Sizing against this host…</p>}
        {preview && (
          <div style={{ marginTop: 14 }}>
            <PlanCard
              preview={preview}
              onOnboard={startOnboard}
              onQuantizeOnly={startQuantizeOnly}
              busy={busy}
              capability={capability}
            />
          </div>
        )}
      </Section>

      {jobs.length > 0 && (
        <Section title="Onboarding jobs" subtitle="Downloads are multi-GB and model loads are slow; this refreshes itself.">
          {jobs.map((j) => (
            <div key={j.job_id} style={{ borderBottom: `1px solid ${PALETTE.border}`, padding: "10px 0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <div>
                  <strong style={{ fontSize: 13, color: PALETTE.ink }}>{j.repo_id}</strong>
                  <span style={{ fontSize: 12, color: PALETTE.soft }}> → {j.model_id}</span>
                </div>
                <Pill tone={j.state === "succeeded" ? "ok" : j.state === "failed" ? "bad" : "warn"}>{j.state}</Pill>
              </div>
              {j.error && <div style={{ fontSize: 12, color: PALETTE.danger, marginTop: 4 }}>{j.error}</div>}
              {j.steps?.length > 0 && (
                <div style={{ fontSize: 12, color: PALETTE.soft, marginTop: 4 }}>
                  {j.steps[j.steps.length - 1].step}: {j.steps[j.steps.length - 1].detail}
                </div>
              )}
              {j.quantization_carbon_g > 0 && (
                <div style={{ fontSize: 12, color: PALETTE.warn, marginTop: 4 }}>
                  quantization cost {j.quantization_carbon_g} gCO₂e (one-off)
                </div>
              )}
            </div>
          ))}
        </Section>
      )}

      {artifacts.length > 0 && (
        <Section
          title="Quantized checkpoints"
          subtitle="Produced here, yours to take. Downloading one does not involve the router."
        >
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left", color: PALETTE.soft, fontSize: 12 }}>
                  <th style={{ padding: "6px 8px" }}>artifact</th>
                  <th style={{ padding: "6px 8px" }}>format</th>
                  <th style={{ padding: "6px 8px" }}>size</th>
                  <th style={{ padding: "6px 8px" }}>cost to make</th>
                  <th style={{ padding: "6px 8px" }}>in zoo</th>
                  <th style={{ padding: "6px 8px" }}></th>
                </tr>
              </thead>
              <tbody>
                {artifacts.map((a) => (
                  <tr key={a.artifact_id} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                    <td style={{ padding: "8px" }}>
                      <div style={{ fontWeight: 600, color: PALETTE.ink }}>{a.artifact_id}</div>
                      <div style={{ fontSize: 11, color: PALETTE.soft }}>
                        from {a.source_repo_id || "unknown"}
                        {a.calibration_source && ` · calibrated on ${a.calibration_source}`}
                      </div>
                    </td>
                    <td style={{ padding: "8px" }}>
                      <Pill tone={a.complete ? "ok" : "bad"}>
                        {a.complete ? `${a.quant_method || "?"}${a.bits ? ` ${a.bits}-bit` : ""}` : "incomplete"}
                      </Pill>
                    </td>
                    <td style={{ padding: "8px" }}>{fmtGb(a.size_gb)}</td>
                    <td style={{ padding: "8px" }}>
                      {a.quantization_carbon_g != null ? (
                        <span title="Measured wall-clock against spec TDP — an upper bound.">
                          {a.quantization_carbon_g} gCO₂e
                        </span>
                      ) : (
                        <span style={{ color: PALETTE.soft }}>—</span>
                      )}
                    </td>
                    <td style={{ padding: "8px" }}>
                      <Pill tone={a.in_zoo ? "ok" : "soft"}>{a.in_zoo ? "registered" : "no"}</Pill>
                    </td>
                    <td style={{ padding: "8px", textAlign: "right", whiteSpace: "nowrap" }}>
                      <a
                        href={a.complete ? artifactDownloadUrl(a.artifact_id) : undefined}
                        download
                        title={a.complete ? "Streams as an uncompressed tar" : "A failed pass left this behind; it will not load"}
                        style={{
                          display: "inline-block", border: `1px solid ${PALETTE.border}`, borderRadius: 6,
                          padding: "4px 10px", fontSize: 12, marginRight: 6, textDecoration: "none",
                          color: a.complete ? PALETTE.deep : "#9bb0ab",
                          pointerEvents: a.complete ? "auto" : "none",
                        }}
                      >
                        Download
                      </a>
                      <button
                        onClick={() => removeArtifact(a.artifact_id)}
                        disabled={a.in_zoo}
                        title={a.in_zoo ? "Registered in the zoo — deregister it first" : "Delete the weights from disk"}
                        style={{
                          background: "transparent", border: `1px solid ${PALETTE.border}`, borderRadius: 6,
                          padding: "4px 10px", fontSize: 12,
                          color: a.in_zoo ? "#9bb0ab" : PALETTE.danger,
                          cursor: a.in_zoo ? "not-allowed" : "pointer",
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 12, color: PALETTE.soft, marginTop: 12, marginBottom: 0 }}>
              Downloads stream as an uncompressed tar — 4-bit safetensors do not compress, and staging a
              zipped copy would need the disk twice over. The tarball carries a{" "}
              <code>quantization_manifest.json</code> recording what it came from, what it was calibrated on,
              and what the pass cost. Upstream licence terms carry over to the quantized weights.
            </p>
          </div>
        </Section>
      )}

      <Section
        title="Onboarded models"
        subtitle="Routable means measured AND served. Either one alone is not enough."
      >
        {registry.length === 0 ? (
          <p style={{ fontSize: 13, color: PALETTE.soft, margin: 0 }}>
            Nothing onboarded yet. The models CSS currently ranks are the statically declared ones in{" "}
            <code>config/model_zoo.json</code>.
          </p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left", color: PALETTE.soft, fontSize: 12 }}>
                  <th style={{ padding: "6px 8px" }}>model</th>
                  <th style={{ padding: "6px 8px" }}>format</th>
                  <th style={{ padding: "6px 8px" }}>VRAM</th>
                  <th style={{ padding: "6px 8px" }}>accuracy</th>
                  <th style={{ padding: "6px 8px" }}>routable</th>
                  <th style={{ padding: "6px 8px" }}></th>
                </tr>
              </thead>
              <tbody>
                {registry.map((m) => (
                  <tr key={m.id} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                    <td style={{ padding: "8px" }}>
                      <div style={{ fontWeight: 600, color: PALETTE.ink }}>{m.id}</div>
                      <div style={{ fontSize: 11, color: PALETTE.soft }}>{m.serve_repo_id}</div>
                    </td>
                    <td style={{ padding: "8px" }}>
                      <Pill>{m.quantization}</Pill>
                    </td>
                    <td style={{ padding: "8px" }}>{fmtMb(m.est_vram_mb)}</td>
                    <td style={{ padding: "8px" }}>
                      {m.accuracy_basis === "unmeasured" ? (
                        <span style={{ color: PALETTE.warn }}>unmeasured</span>
                      ) : (
                        <span>{m.accuracy_baseline}<br /><span style={{ fontSize: 11, color: PALETTE.soft }}>{m.accuracy_basis}</span></span>
                      )}
                    </td>
                    <td style={{ padding: "8px" }}>
                      <Pill tone={m.routable ? "ok" : "warn"}>{m.routable ? "yes" : "no"}</Pill>
                    </td>
                    <td style={{ padding: "8px", textAlign: "right" }}>
                      {m.endpoint_url ? (
                        <button
                          onClick={() => unserveOnboardedModel(m.id).then(refresh)}
                          style={{ background: "transparent", border: `1px solid ${PALETTE.border}`, borderRadius: 6, padding: "4px 10px", cursor: "pointer", fontSize: 12 }}
                        >
                          Stop
                        </button>
                      ) : (
                        <button
                          disabled={!capability?.serving?.enabled}
                          title={!capability?.serving?.enabled ? capability?.serving?.reason : undefined}
                          onClick={() => serveOnboardedModel(m.id).then(refresh)}
                          style={{
                            background: "transparent", border: `1px solid ${PALETTE.border}`, borderRadius: 6,
                            padding: "4px 10px", fontSize: 12,
                            cursor: capability?.serving?.enabled ? "pointer" : "not-allowed",
                          }}
                        >
                          Serve
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 12, color: PALETTE.soft, marginTop: 12, marginBottom: 0 }}>
              A model shows <strong>unmeasured</strong> until measured accuracy and latency are posted to{" "}
              <code>/api/models/&#123;id&#125;/measure</code> against a live endpoint. That is deliberate: CSS ranks on
              accuracy, and this zoo already declares one model at 0.92 that measured 0.793. Onboarding must
              not automate that gap.
            </p>
          </div>
        )}
      </Section>

      {/* The other half of the same problem. Onboarding imports a better rung;
          fine-tuning makes the one already here good enough to pick. */}
      <FineTunePanel palette={PALETTE} Section={Section} Pill={Pill} />
    </div>
  );
}

export default ModelsPanel;

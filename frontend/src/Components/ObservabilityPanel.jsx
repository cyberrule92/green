/**
 * ObservabilityPanel — Adaptive Green AI
 * LLM observability dashboard inspired by Datadog / Elastic Observability.
 * Aggregates the HMAC-signed audit log into KPIs, latency percentiles,
 * model/tier/intent distributions, time-series charts, latency histograms,
 * top conversations, anomalies, and a searchable trace list with detail view.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fetchObservabilitySummary } from "../lib/api";

const WINDOW_PRESETS = [
  { label: "5m",  windowMinutes: 5,    bucketSeconds: 15 },
  { label: "15m", windowMinutes: 15,   bucketSeconds: 30 },
  { label: "1h",  windowMinutes: 60,   bucketSeconds: 60 },
  { label: "6h",  windowMinutes: 360,  bucketSeconds: 300 },
  { label: "24h", windowMinutes: 1440, bucketSeconds: 900 },
];

const REFRESH_MS = 20_000;
const LIVE_TAIL_MS = 5_000;
const NEW_TRACE_FLASH_MS = 4_000;

const PALETTE = {
  primary: "#01a982",
  deep:    "#0f5f59",
  ink:     "#0b1f1f",
  soft:    "#5f7272",
  border:  "#d4e3df",
  warn:    "#e6a817",
  danger:  "#c94040",
  blue:    "#1565c0",
  purple:  "#6a1b9a",
  orange:  "#e65100",
  bg:      "#f4f8f7",
};

const MODEL_COLORS = {
  "ultra-light": "#1565c0",
  "medium":      "#01a982",
  "full":        "#6a1b9a",
  "moe":         "#e65100",
  "cpu-fallback":"#5f7272",
  "unknown":     "#9aa6a6",
};

// ─── Formatters ──────────────────────────────────────────────────────────────
const fmtMs = (v) => {
  if (v == null) return "–";
  if (v < 1) return `${(v * 1000).toFixed(0)}µs`;
  if (v < 1000) return `${v.toFixed(0)}ms`;
  return `${(v / 1000).toFixed(2)}s`;
};
const fmtNum = (v) => {
  if (v == null) return "–";
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(2)}k`;
  return Number(v).toLocaleString();
};
const fmtCo2 = (g) => {
  if (g == null) return "–";
  if (g < 0.001) return `${(g * 1e6).toFixed(2)} µg`;
  if (g < 1) return `${(g * 1000).toFixed(2)} mg`;
  if (g < 1000) return `${g.toFixed(3)} g`;
  return `${(g / 1000).toFixed(3)} kg`;
};
const fmtPct = (v) => `${(v * 100).toFixed(2)}%`;
const fmtTime = (iso) => {
  if (!iso) return "–";
  try { return new Date(iso).toLocaleString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
  catch { return iso; }
};
const fmtTimeShort = (ts) => {
  try {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch { return ""; }
};

// ─── Building blocks ─────────────────────────────────────────────────────────
function Card({ title, action, children, padded = true }) {
  return (
    <div style={{
      background: "white",
      border: `1px solid ${PALETTE.border}`,
      borderRadius: 12,
      boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "10px 14px", borderBottom: `1px solid ${PALETTE.border}`,
      }}>
        <strong style={{
          fontSize: "0.78rem", textTransform: "uppercase", letterSpacing: "0.06em",
          color: PALETTE.soft,
        }}>{title}</strong>
        {action}
      </div>
      <div style={{ padding: padded ? 14 : 0, flex: 1 }}>{children}</div>
    </div>
  );
}

function Kpi({ label, value, sub, color = PALETTE.ink, accent, delta, deltaInverse = false }) {
  return (
    <div style={{
      background: "white",
      border: `1px solid ${PALETTE.border}`,
      borderRadius: 12,
      padding: "12px 14px",
      display: "flex", flexDirection: "column", gap: 4,
      borderLeft: accent ? `3px solid ${accent}` : `1px solid ${PALETTE.border}`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{
          fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em",
          color: PALETTE.soft, fontWeight: 700,
        }}>{label}</span>
        {delta && <DeltaPill delta={delta} inverse={deltaInverse} />}
      </div>
      <span style={{ fontSize: "1.45rem", fontWeight: 700, color, lineHeight: 1.1 }}>
        {value}
      </span>
      {sub && <span style={{ fontSize: "0.74rem", color: PALETTE.soft }}>{sub}</span>}
    </div>
  );
}

function Pill({ children, color = PALETTE.primary, bg }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "1px 8px", borderRadius: 999,
      fontSize: "0.72rem", fontWeight: 700,
      color, background: bg || `${color}1f`, border: `1px solid ${color}33`,
    }}>{children}</span>
  );
}

// ─── Delta pill (period-over-period change) ─────────────────────────────────
// `inverse` = true means a decrease is *good* (e.g. latency, errors, CO₂).
function DeltaPill({ delta, inverse = false, format = (v) => v.toFixed(1) }) {
  if (!delta || delta.pct == null) {
    return <span style={{ fontSize: "0.68rem", color: PALETTE.soft }}>—</span>;
  }
  const pct = delta.pct;
  if (Math.abs(pct) < 0.5) {
    return <span style={{ fontSize: "0.68rem", color: PALETTE.soft, fontWeight: 600 }}>·flat</span>;
  }
  const isUp = pct > 0;
  const isGood = inverse ? !isUp : isUp;
  const color = isGood ? PALETTE.primary : PALETTE.danger;
  const arrow = isUp ? "▲" : "▼";
  return (
    <span style={{ fontSize: "0.68rem", fontWeight: 700, color }}>
      {arrow} {format(Math.abs(pct))}%
    </span>
  );
}

// ─── Horizontal gauge bar (used by SLO + cost) ──────────────────────────────
function GaugeBar({ value, max = 100, color = PALETTE.primary, height = 8 }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div style={{ height, background: "rgba(0,0,0,0.07)", borderRadius: height / 2, overflow: "hidden" }}>
      <div style={{
        width: `${pct}%`, height: "100%", background: color, borderRadius: height / 2,
        transition: "width 0.6s",
      }} />
    </div>
  );
}

// ─── SLO + error-budget card ────────────────────────────────────────────────
function SloCard({ slo }) {
  if (!slo) return null;
  const statusColor = slo.status === "breach" ? PALETTE.danger
                    : slo.status === "warn"   ? PALETTE.warn
                    : PALETTE.primary;
  const burnColor = slo.error_budget_burned_pct > 75 ? PALETTE.danger
                  : slo.error_budget_burned_pct > 50 ? PALETTE.warn
                  : PALETTE.primary;
  return (
    <Card title="Service health & SLO" action={
      <Pill color={statusColor}>{slo.status}</Pill>
    }>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: 4 }}>
            <span style={{ color: PALETTE.soft, fontWeight: 600 }}>P95 latency vs target</span>
            <span style={{ fontWeight: 700, color: PALETTE.ink }}>
              {fmtMs(slo.p95_actual_ms)} / {fmtMs(slo.p95_target_ms)}
            </span>
          </div>
          <GaugeBar
            value={Math.min(slo.p95_actual_ms, slo.p95_target_ms * 1.5)}
            max={slo.p95_target_ms * 1.5}
            color={slo.p95_actual_ms > slo.p95_target_ms ? PALETTE.danger : PALETTE.primary}
          />
          <div style={{ fontSize: "0.7rem", color: PALETTE.soft, marginTop: 4 }}>
            <strong style={{ color: PALETTE.ink }}>{slo.p95_compliance_pct}%</strong> of requests under target
            · {slo.p95_breach_count} breaches
          </div>
        </div>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: 4 }}>
            <span style={{ color: PALETTE.soft, fontWeight: 600 }}>Error budget burned</span>
            <span style={{ fontWeight: 700, color: burnColor }}>
              {slo.error_budget_burned_pct}%
            </span>
          </div>
          <GaugeBar value={slo.error_budget_burned_pct} max={100} color={burnColor} />
          <div style={{ fontSize: "0.7rem", color: PALETTE.soft, marginTop: 4 }}>
            actual {fmtPct(slo.error_actual_rate)} / target {fmtPct(slo.error_target_rate)}
            · <strong style={{ color: PALETTE.ink }}>{slo.error_budget_remaining_pct}%</strong> remaining
          </div>
        </div>
      </div>
    </Card>
  );
}

// ─── Latency heatmap (time × latency-bin) ───────────────────────────────────
function LatencyHeatmap({ heatmap }) {
  if (!heatmap || !heatmap.cells || heatmap.cells.length === 0) {
    return <div style={{ color: PALETTE.soft, fontSize: "0.85rem" }}>No data</div>;
  }
  const { cells, lat_bin_edges_ms, col_starts, max_count } = heatmap;
  const cols = cells[0]?.length || 0;
  if (cols === 0) return <div style={{ color: PALETTE.soft }}>No data</div>;
  const cellW = 100 / cols;
  const rows = cells.length;
  const cellH = 18;
  const W = 800;
  const labelW = 70;
  const H = rows * cellH;
  const colorFor = (count) => {
    if (count === 0 || max_count === 0) return "rgba(15,95,89,0.04)";
    const t = count / max_count;
    // green→yellow→red gradient via HSL (140° → 0°)
    const hue = 140 - 140 * Math.min(1, t);
    const light = 92 - 50 * Math.min(1, t);
    return `hsl(${hue}, 75%, ${light}%)`;
  };
  const rowLabel = (i) => {
    const upper = lat_bin_edges_ms[i];
    if (upper == null) return "30s+";
    const prev = i === 0 ? 0 : lat_bin_edges_ms[i - 1];
    return `${prev}–${upper}ms`;
  };
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H + 22}`} preserveAspectRatio="none"
           style={{ width: "100%", height: H + 22, display: "block" }}>
        {cells.map((row, ri) => row.map((c, ci) => (
          <rect
            key={`${ri}-${ci}`}
            x={labelW + ci * ((W - labelW) / cols)}
            y={(rows - 1 - ri) * cellH}
            width={(W - labelW) / cols + 0.5}
            height={cellH - 1}
            fill={colorFor(c)}
          >
            <title>{`${rowLabel(ri)} @ ${fmtTimeShort(col_starts[ci])}: ${c}`}</title>
          </rect>
        )))}
        {cells.map((_, ri) => (
          <text
            key={`lbl-${ri}`}
            x={labelW - 4}
            y={(rows - 1 - ri) * cellH + cellH / 2 + 3}
            textAnchor="end"
            fontSize="9"
            fill={PALETTE.soft}
          >{rowLabel(ri)}</text>
        ))}
        <text x={labelW} y={H + 12} fontSize="9" fill={PALETTE.soft}>{fmtTimeShort(col_starts[0])}</text>
        <text x={W - 4} y={H + 12} fontSize="9" fill={PALETTE.soft} textAnchor="end">
          {fmtTimeShort(col_starts[col_starts.length - 1])}
        </text>
      </svg>
      <div style={{
        display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 6,
        marginTop: 4, fontSize: "0.7rem", color: PALETTE.soft,
      }}>
        <span>density</span>
        <span style={{ width: 12, height: 8, background: "hsl(140,75%,92%)", border: `1px solid ${PALETTE.border}` }} />
        <span style={{ width: 12, height: 8, background: "hsl(70,75%,67%)", border: `1px solid ${PALETTE.border}` }} />
        <span style={{ width: 12, height: 8, background: "hsl(0,75%,42%)", border: `1px solid ${PALETTE.border}` }} />
        <span>peak {max_count}</span>
      </div>
    </div>
  );
}

// ─── Cost & efficiency card ────────────────────────────────────────────────
function CostCard({ cost }) {
  if (!cost) return null;
  const savingsColor = (cost.savings_usd ?? 0) >= 0 ? PALETTE.primary : PALETTE.danger;
  const fmtUsd = (v) => v == null ? "–" : `$${(v).toFixed(v < 1 ? 4 : 2)}`;
  return (
    <Card title="Cost & efficiency">
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 12,
      }}>
        <div>
          <div style={kpiSubLabel}>Self-host energy</div>
          <div style={kpiSubValue}>{fmtUsd(cost.energy_usd)}</div>
          <div style={{ fontSize: "0.68rem", color: PALETTE.soft }}>
            {cost.energy_kwh.toFixed(4)} kWh @ ${cost.energy_price_usd_kwh}/kWh
          </div>
        </div>
        <div>
          <div style={kpiSubLabel}>Cloud equivalent</div>
          <div style={kpiSubValue}>{fmtUsd(cost.cloud_equivalent_usd)}</div>
          <div style={{ fontSize: "0.68rem", color: PALETTE.soft }}>
            in {fmtUsd(cost.cloud_input_usd)} · out {fmtUsd(cost.cloud_output_usd)}
          </div>
        </div>
        <div>
          <div style={kpiSubLabel}>Savings vs cloud</div>
          <div style={{ ...kpiSubValue, color: savingsColor }}>
            {fmtUsd(cost.savings_usd)}
            {cost.savings_pct != null && (
              <span style={{ fontSize: "0.78rem", marginLeft: 6, color: savingsColor }}>
                ({cost.savings_pct}%)
              </span>
            )}
          </div>
          <div style={{ fontSize: "0.68rem", color: PALETTE.soft }}>
            running locally vs OpenAI-class API
          </div>
        </div>
        <div>
          <div style={kpiSubLabel}>Tokens / request</div>
          <div style={kpiSubValue}>{fmtNum(cost.tokens_per_request)}</div>
          <div style={{ fontSize: "0.68rem", color: PALETTE.soft }}>
            CO₂ {fmtCo2(cost.co2_per_1k_tokens_g)}/1k tok
          </div>
        </div>
      </div>
      {cost.by_model && cost.by_model.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["Model", "Tokens", "Energy (kWh)", "Energy $", "Cloud-equiv $"].map(h => (
                  <th key={h} style={{
                    textAlign: "left", padding: "6px 8px", fontSize: "0.7rem",
                    textTransform: "uppercase", letterSpacing: "0.04em",
                    color: PALETTE.soft, fontWeight: 700,
                    borderBottom: `1px solid ${PALETTE.border}`,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cost.by_model.map(r => (
                <tr key={r.model} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                  <td style={{ ...tdStyle, fontSize: "0.78rem" }}>
                    <Pill color={MODEL_COLORS[r.model] || PALETTE.primary}>{r.model}</Pill>
                  </td>
                  <td style={{ ...tdStyle, fontSize: "0.78rem" }}>{fmtNum(r.tokens)}</td>
                  <td style={{ ...tdStyle, fontSize: "0.78rem" }}>{r.energy_kwh.toFixed(5)}</td>
                  <td style={{ ...tdStyle, fontSize: "0.78rem" }}>{fmtUsd(r.energy_usd)}</td>
                  <td style={{ ...tdStyle, fontSize: "0.78rem" }}>{fmtUsd(r.cloud_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

const kpiSubLabel = {
  fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em",
  color: PALETTE.soft, fontWeight: 700, marginBottom: 2,
};
const kpiSubValue = {
  fontSize: "1.15rem", fontWeight: 700, color: PALETTE.ink, lineHeight: 1.1,
};

// ─── Time series line chart (SVG) ────────────────────────────────────────────
function TimeSeriesChart({ data, accessor, color, label, height = 90, format = fmtNum }) {
  if (!data || data.length === 0) {
    return <div style={{ color: PALETTE.soft, fontSize: "0.85rem" }}>No data</div>;
  }
  const values = data.map(accessor).map(v => v ?? 0);
  const max = Math.max(1, ...values);
  const min = 0;
  const W = 600, H = height, P = 4;
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(1, data.length - 1);
  const y = (v) => H - P - ((v - min) / (max - min || 1)) * (H - 2 * P);
  const path = values.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  const area = `${path} L ${x(values.length - 1).toFixed(1)} ${H - P} L ${x(0).toFixed(1)} ${H - P} Z`;
  const lastVal = values[values.length - 1];
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: "0.78rem", color: PALETTE.soft, fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: "0.85rem", fontWeight: 700, color }}>
          {format(lastVal)} <span style={{ fontSize: "0.7rem", color: PALETTE.soft, fontWeight: 500 }}>now</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
        <defs>
          <linearGradient id={`g-${label.replace(/\W/g,"")}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.35" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={area} fill={`url(#g-${label.replace(/\W/g,"")})`} />
        <path d={path} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <div style={{
        display: "flex", justifyContent: "space-between",
        fontSize: "0.68rem", color: PALETTE.soft, marginTop: 2,
      }}>
        <span>{fmtTimeShort(data[0]?.bucket_start)}</span>
        <span>peak {format(max)}</span>
        <span>{fmtTimeShort(data[data.length - 1]?.bucket_start)}</span>
      </div>
    </div>
  );
}

// ─── Histogram (latency buckets) ─────────────────────────────────────────────
function Histogram({ buckets }) {
  if (!buckets || buckets.length === 0) return <div style={{ color: PALETTE.soft }}>No data</div>;
  const max = Math.max(1, ...buckets.map(b => b.count));
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 100 }}>
      {buckets.map((b, i) => {
        const h = (b.count / max) * 100;
        const isOver = b.le_ms == null;
        return (
          <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
            <div style={{ flex: 1, display: "flex", alignItems: "flex-end", width: "100%" }}>
              <div title={`${b.count} requests`} style={{
                width: "100%",
                height: `${Math.max(h, b.count > 0 ? 4 : 0)}%`,
                background: isOver ? PALETTE.danger : PALETTE.primary,
                borderRadius: "4px 4px 0 0",
                transition: "height 0.4s",
              }} />
            </div>
            <span style={{ fontSize: "0.62rem", color: PALETTE.soft, whiteSpace: "nowrap" }}>
              {isOver ? "30s+" : `≤${b.le_ms}ms`}
            </span>
            <span style={{ fontSize: "0.65rem", fontWeight: 700 }}>{b.count}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Distribution bars ──────────────────────────────────────────────────────
function DistributionList({ map, palette = MODEL_COLORS, total }) {
  const entries = Object.entries(map || {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return <div style={{ color: PALETTE.soft, fontSize: "0.85rem" }}>No data</div>;
  const sum = total || entries.reduce((s, [, v]) => s + v, 0) || 1;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
      {entries.map(([k, v]) => {
        const pct = (v / sum) * 100;
        const color = palette[k] || PALETTE.primary;
        return (
          <div key={k}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: 2 }}>
              <span style={{ color: PALETTE.ink, fontWeight: 600 }}>{k}</span>
              <span style={{ color: PALETTE.soft }}>
                {v} <span style={{ fontWeight: 700, color }}>· {pct.toFixed(1)}%</span>
              </span>
            </div>
            <div style={{ height: 6, background: "rgba(0,0,0,0.06)", borderRadius: 4, overflow: "hidden" }}>
              <div style={{
                width: `${pct}%`, height: "100%", background: color,
                borderRadius: 4, transition: "width 0.5s",
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Trace row + detail ─────────────────────────────────────────────────────
function statusFor(t) {
  if (t.grounding_supported === false && t.grounding_reason && t.grounding_reason !== "not-grounded-request") {
    return { color: PALETTE.danger, label: "grounding" };
  }
  if (t.deferred) return { color: PALETTE.warn, label: "deferred" };
  if (t.latency_ms > 5000) return { color: PALETTE.warn, label: "slow" };
  return { color: PALETTE.primary, label: "ok" };
}

function TraceRow({ t, expanded, onToggle, isFresh }) {
  const s = statusFor(t);
  const color = MODEL_COLORS[t.model] || PALETTE.soft;
  return (
    <>
      <tr
        onClick={onToggle}
        className={isFresh ? "obs-fresh" : undefined}
        style={{
          cursor: "pointer",
          background: expanded ? "rgba(1,169,130,0.05)" : "transparent",
          borderTop: `1px solid ${PALETTE.border}`,
        }}
      >
        <td style={tdStyle}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
            <span style={{ fontFamily: "monospace", fontSize: "0.72rem" }}>
              {fmtTime(t.timestamp)}
            </span>
          </div>
        </td>
        <td style={tdStyle}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "1px 7px", borderRadius: 6,
            background: `${color}1a`, color, fontSize: "0.72rem", fontWeight: 700,
          }}>
            {t.model_name || t.model || "–"}
            <span style={{ opacity: 0.7, fontWeight: 500 }}>[{t.model}]</span>
          </div>
        </td>
        <td style={tdStyle}>
          <div style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {t.query_preview || "—"}
          </div>
        </td>
        <td style={{ ...tdStyle, fontWeight: 700, color: t.latency_ms > 3000 ? PALETTE.warn : PALETTE.ink }}>
          {fmtMs(t.latency_ms)}
        </td>
        <td style={tdStyle}>
          <span style={{ color: PALETTE.purple, fontWeight: 600 }}>
            {t.tokens_in}↑ {t.tokens_out}↓
          </span>
        </td>
        <td style={tdStyle}>{fmtCo2(t.co2_g)}</td>
        <td style={tdStyle}>
          <Pill color={s.color}>{s.label}</Pill>
        </td>
        <td style={{ ...tdStyle, textAlign: "right" }}>
          <span style={{ color: PALETTE.soft, fontSize: "0.85rem" }}>{expanded ? "▲" : "▼"}</span>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={8} style={{ background: PALETTE.bg, padding: 14, borderTop: `1px solid ${PALETTE.border}` }}>
            <TraceDetail t={t} />
          </td>
        </tr>
      )}
    </>
  );
}

function TraceDetail({ t }) {
  const items = [
    ["Request ID", t.request_id || "–"],
    ["Conversation", t.conversation_id || "–"],
    ["Tier", t.tier || "–"],
    ["Priority", t.priority || "–"],
    ["Intent", t.intent || "–"],
    ["Complexity", t.complexity || "–"],
    ["Latency", fmtMs(t.latency_ms)],
    ["Tokens (in/out/total)", `${t.tokens_in} / ${t.tokens_out} / ${t.tokens_total}`],
    ["Model CO₂", fmtCo2(t.co2_g)],
    ["Grid CI", `${(t.grid_carbon || 0).toFixed(0)} gCO₂/kWh`],
    ["CSS score", (t.css_score || 0).toFixed(3)],
    ["GPU util", `${(t.gpu_utilization_pct || 0).toFixed(1)}%`],
    ["RAG retrieved", t.rag_retrieved],
    ["Deferred", t.deferred ? "yes" : "no"],
    ["Grounding", t.grounding_supported === true ? "supported"
                  : t.grounding_supported === false ? `unsupported · ${t.grounding_reason || "—"}`
                  : "n/a"],
    ["Accuracy outcome", (t.accuracy_outcome || 0).toFixed(2)],
    ["RL policy", `v${t.rl_policy_version ?? "–"}`],
  ];
  // pipeline stages — simulated breakdown so the trace has a Datadog-flavored
  // span tree. Backend records aggregate latency only; we proportion it.
  const stages = [
    { name: "guardrails-input", weight: 0.04, color: PALETTE.purple },
    { name: "rag-retrieval",    weight: t.rag_retrieved > 0 ? 0.18 : 0.02, color: PALETTE.blue },
    { name: "routing-css",      weight: 0.05, color: PALETTE.deep },
    { name: "vllm-inference",   weight: 0.62, color: PALETTE.primary },
    { name: "guardrails-output",weight: 0.04, color: PALETTE.purple },
    { name: "audit+persist",    weight: 0.05, color: PALETTE.soft },
  ];
  const sumW = stages.reduce((s, x) => s + x.weight, 0);
  const total = t.latency_ms || 0;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 14 }}>
      <div>
        <div style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em",
          color: PALETTE.soft, fontWeight: 700, marginBottom: 6 }}>
          Span timeline
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {stages.map((s, i) => {
            const ms = (s.weight / sumW) * total;
            const pct = (ms / Math.max(total, 1)) * 100;
            return (
              <div key={i}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.74rem", marginBottom: 2 }}>
                  <span style={{ color: PALETTE.ink, fontWeight: 600 }}>{s.name}</span>
                  <span style={{ color: PALETTE.soft, fontFamily: "monospace" }}>{fmtMs(ms)}</span>
                </div>
                <div style={{ height: 6, background: "rgba(0,0,0,0.06)", borderRadius: 3 }}>
                  <div style={{ width: `${pct}%`, height: "100%", background: s.color, borderRadius: 3 }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div style={{
        display: "grid", gridTemplateColumns: "max-content 1fr",
        gap: "4px 14px", fontSize: "0.78rem", alignContent: "start",
      }}>
        {items.map(([k, v]) => (
          <React.Fragment key={k}>
            <div style={{ color: PALETTE.soft }}>{k}</div>
            <div style={{ color: PALETTE.ink, fontWeight: 600, wordBreak: "break-word" }}>{v}</div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

const tdStyle = { padding: "8px 10px", fontSize: "0.82rem", verticalAlign: "middle" };

// ─── CSV export helper ──────────────────────────────────────────────────────
function exportTracesAsCsv(traces) {
  if (!traces || traces.length === 0) return;
  const cols = [
    "timestamp", "request_id", "conversation_id", "model", "model_name",
    "tier", "priority", "intent", "complexity", "latency_ms",
    "tokens_in", "tokens_out", "tokens_total", "co2_g", "grid_carbon",
    "css_score", "gpu_utilization_pct", "rag_retrieved", "deferred",
    "grounding_supported", "grounding_reason", "accuracy_outcome", "rl_policy_version",
  ];
  const escape = (v) => {
    if (v == null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  for (const t of traces) {
    lines.push(cols.map(c => escape(t[c])).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `traces_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ─── Main panel ─────────────────────────────────────────────────────────────
export function ObservabilityPanel() {
  const [preset, setPreset] = useState(WINDOW_PRESETS[2]); // 1h
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [expandedTraceId, setExpandedTraceId] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [liveTail, setLiveTail] = useState(false);
  const [sloP95Ms, setSloP95Ms] = useState(3000);
  const [sloErrorRate, setSloErrorRate] = useState(0.01);
  const [freshTraceIds, setFreshTraceIds] = useState(new Set());

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchObservabilitySummary({
        windowMinutes: preset.windowMinutes,
        bucketSeconds: preset.bucketSeconds,
        sloP95Ms,
        sloErrorRate,
      });
      setData(prev => {
        // Detect newly-arrived traces for live-tail flash
        if (prev && res?.traces) {
          const oldIds = new Set((prev.traces || []).map(t => t.request_id || t.timestamp));
          const fresh = new Set();
          for (const t of res.traces) {
            const id = t.request_id || t.timestamp;
            if (!oldIds.has(id)) fresh.add(id);
          }
          if (fresh.size > 0) {
            setFreshTraceIds(fresh);
            setTimeout(() => setFreshTraceIds(new Set()), NEW_TRACE_FLASH_MS);
          }
        }
        return res;
      });
      setError(null);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [preset, sloP95Ms, sloErrorRate]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!autoRefresh && !liveTail) return;
    const interval = liveTail ? LIVE_TAIL_MS : REFRESH_MS;
    const t = setInterval(refresh, interval);
    return () => clearInterval(t);
  }, [refresh, autoRefresh, liveTail]);

  const k = data?.kpis || {};
  const deltas = data?.deltas || {};
  const dist = data?.distributions || {};
  const series = data?.time_series || [];
  const traces = data?.traces || [];

  const filteredTraces = useMemo(() => {
    return traces.filter(t => {
      if (modelFilter && t.model !== modelFilter) return false;
      if (statusFilter) {
        const s = statusFor(t).label;
        if (s !== statusFilter) return false;
      }
      if (search) {
        const q = search.toLowerCase();
        const hay = `${t.query_preview || ""} ${t.model || ""} ${t.model_name || ""} ${t.intent || ""} ${t.conversation_id || ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [traces, search, modelFilter, statusFilter]);

  const errorRate = k.grounding_failure_rate || 0;
  const errorColor = errorRate > 0.1 ? PALETTE.danger : errorRate > 0.02 ? PALETTE.warn : PALETTE.primary;

  return (
    <div style={{
      padding: 18,
      display: "flex",
      flexDirection: "column",
      gap: 14,
      background: PALETTE.bg,
      minHeight: "100%",
    }}>
      <style>{`
        @keyframes obs-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
        @keyframes obs-fresh-flash {
          0% { background: rgba(1,169,130,0.35); }
          100% { background: transparent; }
        }
        .obs-fresh { animation: obs-fresh-flash 4s ease-out 1; }
      `}</style>
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12,
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.4rem", color: PALETTE.ink }}>
            LLM Observability
          </h2>
          <div style={{ fontSize: "0.85rem", color: PALETTE.soft, marginTop: 2 }}>
            Traces · metrics · SLO · heatmap · cost · token analytics · anomalies
            {data?.window?.from_iso && (
              <span style={{ marginLeft: 8 }}>
                · {fmtTime(data.window.from_iso)} → {fmtTime(data.window.to_iso)}
              </span>
            )}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <div style={{
            display: "inline-flex", background: "white", border: `1px solid ${PALETTE.border}`,
            borderRadius: 10, padding: 2,
          }}>
            {WINDOW_PRESETS.map(p => (
              <button
                key={p.label}
                onClick={() => setPreset(p)}
                style={{
                  padding: "5px 12px", border: "none", background: preset.label === p.label ? PALETTE.primary : "transparent",
                  color: preset.label === p.label ? "white" : PALETTE.ink,
                  fontWeight: 700, fontSize: "0.78rem", borderRadius: 8, cursor: "pointer",
                }}
              >{p.label}</button>
            ))}
          </div>
          <label style={{ fontSize: "0.78rem", color: PALETTE.soft, display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
            Auto-refresh
          </label>
          <button
            onClick={() => setLiveTail(v => !v)}
            title="Live-tail mode polls every 5s and flashes new traces"
            style={{
              padding: "6px 12px", borderRadius: 10,
              border: `1px solid ${liveTail ? PALETTE.danger : PALETTE.border}`,
              background: liveTail ? PALETTE.danger : "white",
              color: liveTail ? "white" : PALETTE.ink,
              cursor: "pointer", fontSize: "0.82rem", fontWeight: 700,
              display: "inline-flex", alignItems: "center", gap: 6,
            }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: liveTail ? "white" : PALETTE.danger,
              animation: liveTail ? "obs-pulse 1.2s infinite" : "none",
            }} />
            {liveTail ? "Live" : "Live tail"}
          </button>
          <button onClick={refresh} style={{
            padding: "6px 12px", borderRadius: 10, border: `1px solid ${PALETTE.border}`,
            background: "white", cursor: "pointer", fontSize: "0.82rem", fontWeight: 600,
          }}>
            ↻ Refresh
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          padding: "10px 14px", borderRadius: 10, background: "rgba(201,64,64,0.08)",
          color: PALETTE.danger, border: `1px solid ${PALETTE.danger}33`, fontSize: "0.85rem",
        }}>
          Failed to load observability: {error}
        </div>
      )}

      {loading && !data && (
        <div style={{ color: PALETTE.soft, padding: 30, textAlign: "center" }}>Loading…</div>
      )}

      {data && (
        <>
          {/* ── KPIs (with prior-period deltas) ─────────────────────────── */}
          <div style={{
            display: "grid", gap: 10,
            gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
          }}>
            <Kpi label="Requests" value={fmtNum(k.total_requests)}
                 sub={`${k.requests_per_min} req/min`} accent={PALETTE.primary}
                 delta={deltas.total_requests} />
            <Kpi label="P50 latency" value={fmtMs(k.latency_p50_ms)} accent={PALETTE.deep}
                 delta={deltas.latency_p50_ms} deltaInverse />
            <Kpi label="P95 latency" value={fmtMs(k.latency_p95_ms)}
                 sub={`avg ${fmtMs(k.latency_avg_ms)}`} accent={PALETTE.blue}
                 delta={deltas.latency_p95_ms} deltaInverse />
            <Kpi label="P99 latency" value={fmtMs(k.latency_p99_ms)} accent={PALETTE.purple}
                 delta={deltas.latency_p99_ms} deltaInverse />
            <Kpi label="Tokens" value={fmtNum(k.tokens_total)}
                 sub={`${fmtNum(k.tokens_input)}↑ ${fmtNum(k.tokens_output)}↓`} accent={PALETTE.purple}
                 delta={deltas.tokens_total} />
            <Kpi label="Total CO₂" value={fmtCo2(k.co2_total_g)}
                 sub={`avg ${fmtCo2(k.co2_avg_g)}/req`} accent={PALETTE.primary}
                 delta={deltas.co2_total_g} deltaInverse />
            <Kpi label="Grid CI avg" value={`${(k.grid_ci_avg || 0).toFixed(0)}`}
                 sub="gCO₂/kWh" accent={PALETTE.warn}
                 delta={deltas.grid_ci_avg} deltaInverse />
            <Kpi label="CSS avg" value={(k.css_avg || 0).toFixed(3)}
                 sub="composite sustainability" accent={PALETTE.deep}
                 delta={deltas.css_avg} />
            <Kpi label="Error rate" value={fmtPct(errorRate)}
                 color={errorColor}
                 sub={`${k.grounding_failures || 0} grounding fail`} accent={errorColor}
                 delta={deltas.grounding_failure_rate} deltaInverse />
            <Kpi label="Deferred" value={fmtNum(k.deferred_count)}
                 sub={`${fmtPct(k.deferred_rate || 0)} of traffic`} accent={PALETTE.warn}
                 delta={deltas.deferred_rate} />
            <Kpi label="RAG used" value={fmtNum(k.rag_used_count)}
                 sub={`${fmtPct(k.rag_use_rate || 0)} of traffic`} accent={PALETTE.blue}
                 delta={deltas.rag_use_rate} />
          </div>

          {/* ── SLO + Cost row ──────────────────────────────────────────── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <SloCard slo={data.slo} />
            <Card title="SLO settings">
              <div style={{ display: "flex", flexDirection: "column", gap: 12, fontSize: "0.82rem" }}>
                <label>
                  <div style={{ color: PALETTE.soft, marginBottom: 4 }}>P95 latency target (ms)</div>
                  <input
                    type="number"
                    min="50" max="60000" step="100"
                    value={sloP95Ms}
                    onChange={e => setSloP95Ms(Number(e.target.value) || 3000)}
                    style={{
                      width: "100%", padding: "6px 10px", border: `1px solid ${PALETTE.border}`,
                      borderRadius: 8, fontSize: "0.9rem",
                    }}
                  />
                </label>
                <label>
                  <div style={{ color: PALETTE.soft, marginBottom: 4 }}>Error rate target (0..1)</div>
                  <input
                    type="number"
                    min="0" max="1" step="0.001"
                    value={sloErrorRate}
                    onChange={e => setSloErrorRate(Number(e.target.value) || 0)}
                    style={{
                      width: "100%", padding: "6px 10px", border: `1px solid ${PALETTE.border}`,
                      borderRadius: 8, fontSize: "0.9rem",
                    }}
                  />
                </label>
                <div style={{ fontSize: "0.72rem", color: PALETTE.soft }}>
                  Adjust to recompute compliance and error-budget burn.
                </div>
              </div>
            </Card>
          </div>

          <CostCard cost={data.cost} />

          {/* ── Time series row ────────────────────────────────────────── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Card title="Request rate">
              <TimeSeriesChart data={series} accessor={d => d.requests}
                color={PALETTE.primary} label="requests / bucket" />
            </Card>
            <Card title="Latency (avg vs p95)">
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <TimeSeriesChart data={series} accessor={d => d.latency_ms_avg}
                  color={PALETTE.blue} label="avg latency" format={fmtMs} height={70} />
                <TimeSeriesChart data={series} accessor={d => d.latency_ms_p95}
                  color={PALETTE.purple} label="p95 latency" format={fmtMs} height={70} />
              </div>
            </Card>
            <Card title="Carbon footprint">
              <TimeSeriesChart data={series} accessor={d => d.co2_g}
                color={PALETTE.primary} label="CO₂ / bucket" format={fmtCo2} />
            </Card>
            <Card title="Tokens & grid">
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <TimeSeriesChart data={series} accessor={d => d.tokens}
                  color={PALETTE.purple} label="tokens / bucket" height={70} />
                <TimeSeriesChart data={series} accessor={d => d.grid_ci}
                  color={PALETTE.warn} label="grid CI gCO₂/kWh" height={70} />
              </div>
            </Card>
          </div>

          {/* ── Latency heatmap (Datadog-style) ─────────────────────────── */}
          <Card title="Latency heatmap (time × latency-bin)">
            <LatencyHeatmap heatmap={data.heatmap} />
          </Card>

          {/* ── Latency histogram + distributions ─────────────────────── */}
          <div style={{
            display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr", gap: 14,
          }}>
            <Card title="Latency distribution">
              <Histogram buckets={data.latency_histogram} />
              <div style={{
                marginTop: 10, display: "flex", justifyContent: "space-around",
                fontSize: "0.78rem", color: PALETTE.soft, borderTop: `1px solid ${PALETTE.border}`, paddingTop: 8,
              }}>
                <span><strong style={{ color: PALETTE.ink }}>P50</strong> {fmtMs(k.latency_p50_ms)}</span>
                <span><strong style={{ color: PALETTE.ink }}>P95</strong> {fmtMs(k.latency_p95_ms)}</span>
                <span><strong style={{ color: PALETTE.ink }}>P99</strong> {fmtMs(k.latency_p99_ms)}</span>
              </div>
            </Card>
            <Card title="By model">
              <DistributionList map={dist.by_model} total={k.total_requests} />
            </Card>
            <Card title="By tenant tier">
              <DistributionList map={dist.by_tier} total={k.total_requests}
                palette={{ standard: PALETTE.primary, premium: PALETTE.purple, esg: PALETTE.deep, batch: PALETTE.warn }}
              />
            </Card>
          </div>

          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14,
          }}>
            <Card title="By intent">
              <DistributionList map={dist.by_intent} total={k.total_requests}
                palette={{ chitchat: PALETTE.soft, qa: PALETTE.primary, code: PALETTE.purple, math: PALETTE.deep, summary: PALETTE.blue }}
              />
            </Card>
            <Card title="By priority">
              <DistributionList map={dist.by_priority} total={k.total_requests}
                palette={{ low: PALETTE.soft, normal: PALETTE.primary, high: PALETTE.warn, critical: PALETTE.danger }}
              />
            </Card>
            <Card title="By region">
              <DistributionList map={dist.by_region} total={k.total_requests} />
            </Card>
          </div>

          {/* ── Per-model rollup table ─────────────────────────────────── */}
          <Card title="Per-model rollup" padded={false}>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead style={{ background: PALETTE.bg }}>
                  <tr>
                    {["Model", "Requests", "Share", "Avg latency", "P95 latency", "Total CO₂", "Total tokens"].map(h => (
                      <th key={h} style={{
                        textAlign: "left", padding: "8px 10px", fontSize: "0.74rem",
                        textTransform: "uppercase", letterSpacing: "0.04em", color: PALETTE.soft, fontWeight: 700,
                        borderBottom: `1px solid ${PALETTE.border}`,
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(data.model_rollup || []).map(r => (
                    <tr key={r.model} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                      <td style={tdStyle}>
                        <Pill color={MODEL_COLORS[r.model] || PALETTE.primary}>{r.model}</Pill>
                      </td>
                      <td style={tdStyle}>{r.requests}</td>
                      <td style={tdStyle}>{r.share_pct.toFixed(1)}%</td>
                      <td style={tdStyle}>{fmtMs(r.avg_latency_ms)}</td>
                      <td style={tdStyle}>{fmtMs(r.p95_latency_ms)}</td>
                      <td style={tdStyle}>{fmtCo2(r.total_co2_g)}</td>
                      <td style={tdStyle}>{fmtNum(r.total_tokens)}</td>
                    </tr>
                  ))}
                  {(!data.model_rollup || data.model_rollup.length === 0) && (
                    <tr><td colSpan={7} style={{ ...tdStyle, color: PALETTE.soft, textAlign: "center" }}>No data</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* ── Top conversations & anomalies ─────────────────────────── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <Card title="Top conversations" padded={false}>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead style={{ background: PALETTE.bg }}>
                    <tr>
                      {["Conversation", "Requests", "Tokens", "CO₂", "Last seen"].map(h => (
                        <th key={h} style={{
                          textAlign: "left", padding: "8px 10px", fontSize: "0.72rem",
                          textTransform: "uppercase", letterSpacing: "0.04em", color: PALETTE.soft, fontWeight: 700,
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(data.top_conversations || []).map(c => (
                      <tr key={c.conversation_id} style={{ borderTop: `1px solid ${PALETTE.border}` }}>
                        <td style={{ ...tdStyle, fontFamily: "monospace", fontSize: "0.75rem" }}>
                          {c.conversation_id?.slice(0, 16) || "—"}
                        </td>
                        <td style={tdStyle}>{c.requests}</td>
                        <td style={tdStyle}>{fmtNum(c.tokens)}</td>
                        <td style={tdStyle}>{fmtCo2(c.co2_g)}</td>
                        <td style={{ ...tdStyle, color: PALETTE.soft, fontSize: "0.78rem" }}>{fmtTime(c.last_ts)}</td>
                      </tr>
                    ))}
                    {(!data.top_conversations || data.top_conversations.length === 0) && (
                      <tr><td colSpan={5} style={{ ...tdStyle, color: PALETTE.soft, textAlign: "center" }}>No data</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card title="Anomalies (latency outliers)" padded={false}>
              <div style={{ padding: "8px 0" }}>
                {(data.anomalies || []).length === 0 ? (
                  <div style={{ padding: 20, textAlign: "center", color: PALETTE.soft, fontSize: "0.85rem" }}>
                    No outliers detected
                  </div>
                ) : (
                  data.anomalies.map((a, i) => (
                    <div key={i} style={{
                      display: "grid",
                      gridTemplateColumns: "auto 1fr auto auto",
                      alignItems: "center", gap: 10,
                      padding: "8px 14px", borderTop: i ? `1px solid ${PALETTE.border}` : "none",
                    }}>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: PALETTE.danger }} />
                      <span style={{ fontSize: "0.82rem" }}>
                        <strong>{a.model || "—"}</strong>
                        <span style={{ color: PALETTE.soft, marginLeft: 6 }}>{fmtTime(a.timestamp)}</span>
                      </span>
                      <span style={{ fontWeight: 700, color: PALETTE.danger }}>{fmtMs(a.latency_ms)}</span>
                      <Pill color={PALETTE.danger}>z={a.z_score}</Pill>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </div>

          {/* ── Trace explorer ─────────────────────────────────────────── */}
          <Card
            title={`Trace explorer · ${filteredTraces.length} of ${traces.length}`}
            padded={false}
            action={
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <input
                  placeholder="Search traces…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  style={{
                    padding: "5px 10px", border: `1px solid ${PALETTE.border}`,
                    borderRadius: 8, fontSize: "0.78rem", minWidth: 180,
                  }}
                />
                <select value={modelFilter} onChange={e => setModelFilter(e.target.value)}
                  style={{ padding: "5px 8px", borderRadius: 8, border: `1px solid ${PALETTE.border}`, fontSize: "0.78rem" }}>
                  <option value="">All models</option>
                  {Object.keys(dist.by_model || {}).map(m => <option key={m} value={m}>{m}</option>)}
                </select>
                <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}
                  style={{ padding: "5px 8px", borderRadius: 8, border: `1px solid ${PALETTE.border}`, fontSize: "0.78rem" }}>
                  <option value="">Any status</option>
                  <option value="ok">ok</option>
                  <option value="slow">slow</option>
                  <option value="deferred">deferred</option>
                  <option value="grounding">grounding</option>
                </select>
                <button
                  onClick={() => exportTracesAsCsv(filteredTraces)}
                  disabled={filteredTraces.length === 0}
                  title="Download filtered traces as CSV"
                  style={{
                    padding: "5px 10px", borderRadius: 8,
                    border: `1px solid ${PALETTE.border}`,
                    background: filteredTraces.length === 0 ? "#eee" : "white",
                    cursor: filteredTraces.length === 0 ? "not-allowed" : "pointer",
                    fontSize: "0.78rem", fontWeight: 600,
                    color: PALETTE.deep,
                  }}>
                  ↓ CSV
                </button>
              </div>
            }
          >
            <div style={{ overflowX: "auto", maxHeight: 600, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead style={{ background: PALETTE.bg, position: "sticky", top: 0, zIndex: 1 }}>
                  <tr>
                    {["Time", "Model", "Query", "Latency", "Tokens", "CO₂", "Status", ""].map(h => (
                      <th key={h} style={{
                        textAlign: "left", padding: "8px 10px", fontSize: "0.72rem",
                        textTransform: "uppercase", letterSpacing: "0.04em", color: PALETTE.soft, fontWeight: 700,
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredTraces.length === 0 && (
                    <tr><td colSpan={8} style={{ ...tdStyle, color: PALETTE.soft, textAlign: "center", padding: 30 }}>
                      No traces match filters
                    </td></tr>
                  )}
                  {filteredTraces.map(t => {
                    const id = t.request_id || t.timestamp;
                    return (
                      <TraceRow
                        key={t.request_id || `${t.timestamp}-${t.conversation_id}`}
                        t={t}
                        expanded={expandedTraceId === id}
                        isFresh={liveTail && freshTraceIds.has(id)}
                        onToggle={() => setExpandedTraceId(prev => prev === id ? null : id)}
                      />
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

/**
 * BenchmarkPanel — Adaptive Green AI
 *
 * The three-arm routing benchmark: always-full vs a static heuristic vs CSS.
 * This is the one screen that states the claim the whole system rests on —
 * gCO2e per request, at what quality, at what latency — as a measurement rather
 * than a projection.
 *
 * Two things are deliberate. First, nothing here can *start* a run: the harness
 * runs offline against an isolated API with the router frozen and the grid
 * pinned, and a benchmark the system under test can trigger on demand is not a
 * measurement. This panel only renders the summary that run published.
 *
 * Second, the exclusions and the limitations are on screen next to the headline,
 * not in a footnote. Carbon here is modelled (spec TDP x *measured* duration —
 * the vGPU reports no power draw at all), and a reader who cannot see that is
 * being sold a number, not shown one.
 */

import React, { useCallback, useEffect, useState } from "react";
import { fetchBenchmark } from "../lib/api";

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

const ARM_LABEL = {
  "always-full": "Always full",
  "static-heuristic": "Static heuristic",
  css: "CSS router",
};

const ARM_BLURB = {
  "always-full": "Every prompt to the largest model. The “just use the big one” baseline.",
  "static-heuristic": "Carbon-blind length + keyword rule. The baseline that actually matters.",
  css: "The system under test: carbon-aware composite scoring.",
};

const ARM_COLOR = {
  "always-full": PALETTE.danger,
  "static-heuristic": PALETTE.warn,
  css: PALETTE.primary,
};

const fmtPct = (x) => (x == null ? "—" : `${(100 * x).toFixed(1)}%`);
const fmtDelta = (x, digits = 1) =>
  x == null ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(digits)}%`;

function Card({ children, style }) {
  return (
    <div
      style={{
        background: "#fff",
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 10,
        padding: "1.1rem 1.25rem",
        marginBottom: "1rem",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function SectionTitle({ children, hint }) {
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <h3 style={{ margin: 0, fontSize: "1rem", color: PALETTE.deep }}>{children}</h3>
      {hint && (
        <p style={{ margin: "0.3rem 0 0", fontSize: "0.78rem", color: PALETTE.soft, lineHeight: 1.5 }}>
          {hint}
        </p>
      )}
    </div>
  );
}

const th = {
  textAlign: "left",
  padding: "0.5rem 0.6rem",
  fontSize: "0.72rem",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  color: PALETTE.soft,
  borderBottom: `1px solid ${PALETTE.border}`,
  whiteSpace: "nowrap",
};

const td = {
  padding: "0.55rem 0.6rem",
  fontSize: "0.85rem",
  color: PALETTE.ink,
  borderBottom: `1px solid ${PALETTE.bg}`,
  whiteSpace: "nowrap",
};

/** Carbon per request, as a bar against the worst arm. The delta column is the
 *  headline number; the bar is only there so the shape is readable at a glance. */
function CarbonBars({ arms }) {
  const max = Math.max(...arms.map((a) => a.carbon_g_mean || 0), 1e-9);
  return (
    <div style={{ display: "grid", gap: "0.6rem" }}>
      {arms.map((a) => (
        <div key={a.arm}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.8rem",
              marginBottom: "0.25rem",
            }}
          >
            <span style={{ color: PALETTE.ink, fontWeight: a.arm === "css" ? 600 : 400 }}>
              {ARM_LABEL[a.arm] || a.arm}
            </span>
            <span style={{ color: PALETTE.soft, fontVariantNumeric: "tabular-nums" }}>
              {(a.carbon_g_mean ?? 0).toFixed(4)} g
              {a.carbon_delta_pct != null && (
                <strong
                  style={{
                    marginLeft: "0.6rem",
                    color: a.carbon_delta_pct < 0 ? PALETTE.primary : PALETTE.danger,
                  }}
                >
                  {fmtDelta(a.carbon_delta_pct)}
                </strong>
              )}
            </span>
          </div>
          <div style={{ background: PALETTE.bg, borderRadius: 4, height: 10 }}>
            <div
              style={{
                width: `${Math.max(2, (100 * (a.carbon_g_mean || 0)) / max)}%`,
                height: "100%",
                borderRadius: 4,
                background: ARM_COLOR[a.arm] || PALETTE.blue,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function Headline({ data }) {
  const arms = data.arms || [];
  return (
    <Card>
      <SectionTitle
        hint="Carbon is ex-post: spec TDP × measured wall-clock, billed to the model that actually served, summed over every inference leg. Quality is the objective-checker pass rate — no LLM judge."
      >
        Headline
      </SectionTitle>

      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 640 }}>
          <thead>
            <tr>
              <th style={th}>Arm</th>
              <th style={th}>gCO₂e / request</th>
              <th style={th}>vs always-full</th>
              <th style={th}>Quality</th>
              <th style={th}>Δ quality</th>
              <th style={th}>p50</th>
              <th style={th}>p95</th>
              <th style={th}>Clean n</th>
            </tr>
          </thead>
          <tbody>
            {arms.map((a) => {
              const isCss = a.arm === "css";
              return (
                <tr key={a.arm} style={isCss ? { background: "#f2fbf8" } : undefined}>
                  <td style={{ ...td, fontWeight: isCss ? 600 : 400 }}>
                    {ARM_LABEL[a.arm] || a.arm}
                    <div style={{ fontSize: "0.72rem", color: PALETTE.soft, whiteSpace: "normal" }}>
                      {ARM_BLURB[a.arm]}
                    </div>
                  </td>
                  <td style={{ ...td, fontVariantNumeric: "tabular-nums" }}>
                    {(a.carbon_g_mean ?? 0).toFixed(4)}
                  </td>
                  <td
                    style={{
                      ...td,
                      fontWeight: 600,
                      color:
                        a.carbon_delta_pct == null
                          ? PALETTE.soft
                          : a.carbon_delta_pct < 0
                            ? PALETTE.primary
                            : PALETTE.danger,
                    }}
                  >
                    {fmtDelta(a.carbon_delta_pct)}
                  </td>
                  <td style={{ ...td, fontVariantNumeric: "tabular-nums" }}>{fmtPct(a.quality)}</td>
                  <td
                    style={{
                      ...td,
                      color:
                        a.quality_delta_pp == null
                          ? PALETTE.soft
                          : a.quality_delta_pp < 0
                            ? PALETTE.danger
                            : PALETTE.primary,
                    }}
                  >
                    {a.quality_delta_pp == null
                      ? "—"
                      : `${a.quality_delta_pp >= 0 ? "+" : ""}${a.quality_delta_pp.toFixed(1)} pp`}
                  </td>
                  <td style={td}>{Math.round(a.latency_p50_ms)} ms</td>
                  <td style={td}>{Math.round(a.latency_p95_ms)} ms</td>
                  <td style={{ ...td, color: PALETTE.soft }}>
                    {a.n_clean} / {a.n_total}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: "1.1rem" }}>
        <CarbonBars arms={arms} />
      </div>
    </Card>
  );
}

function QualityByCategory({ data }) {
  const arms = (data.arms || []).map((a) => a.arm);
  const rows = data.quality_by_category || [];
  if (!rows.length) return null;
  return (
    <Card>
      <SectionTitle hint="Where a cheaper model is good enough — and where it is not. A router that saves carbon by being wrong on code is not saving anything.">
        Quality by category
      </SectionTitle>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 480 }}>
          <thead>
            <tr>
              <th style={th}>Category</th>
              {arms.map((a) => (
                <th key={a} style={th}>
                  {ARM_LABEL[a] || a}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.category}>
                <td style={td}>
                  {r.category}{" "}
                  <span style={{ color: PALETTE.soft, fontSize: "0.75rem" }}>({r.n_prompts})</span>
                </td>
                {arms.map((a) => (
                  <td key={a} style={{ ...td, fontVariantNumeric: "tabular-nums" }}>
                    {fmtPct(r.quality?.[a])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function ModelMix({ data }) {
  return (
    <Card>
      <SectionTitle hint="What actually ran, by served model — several zoo candidates share one backend, so a delta between two candidate labels of the same model would be a config artifact, not a saving.">
        Model mix
      </SectionTitle>
      {(data.arms || []).map((a) => {
        const mix = Object.entries(a.model_mix || {});
        const total = mix.reduce((s, [, v]) => s + v, 0) || 1;
        return (
          <div key={a.arm} style={{ marginBottom: "0.9rem" }}>
            <div style={{ fontSize: "0.82rem", fontWeight: 600, color: PALETTE.ink, marginBottom: "0.35rem" }}>
              {ARM_LABEL[a.arm] || a.arm}
            </div>
            {mix.length === 0 && <div style={{ fontSize: "0.8rem", color: PALETTE.soft }}>—</div>}
            {mix.map(([name, count]) => (
              <div
                key={name}
                style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.25rem" }}
              >
                <div style={{ flex: "0 0 55%", fontSize: "0.78rem", color: PALETTE.soft, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {name}
                </div>
                <div style={{ flex: 1, background: PALETTE.bg, borderRadius: 3, height: 8 }}>
                  <div
                    style={{
                      width: `${(100 * count) / total}%`,
                      height: "100%",
                      borderRadius: 3,
                      background: ARM_COLOR[a.arm] || PALETTE.blue,
                      opacity: 0.75,
                    }}
                  />
                </div>
                <div style={{ flex: "0 0 3.5rem", fontSize: "0.78rem", color: PALETTE.ink, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {((100 * count) / total).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </Card>
  );
}

function Exclusions({ data }) {
  return (
    <Card>
      <SectionTitle hint="A sample is excluded when it did not measure the arm's routing policy: the dispatcher escalated, a quality retry fired a second inference, the cache answered, no model ran, or a guardrail blocked it. A pinned arm that silently re-routes is not a control arm — so the count is shown, not buried.">
        Exclusions
      </SectionTitle>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 520 }}>
          <thead>
            <tr>
              <th style={th}>Arm</th>
              <th style={th}>Excluded</th>
              <th style={th}>Reasons</th>
              <th style={th}>Pin violations</th>
            </tr>
          </thead>
          <tbody>
            {(data.arms || []).map((a) => (
              <tr key={a.arm}>
                <td style={td}>{ARM_LABEL[a.arm] || a.arm}</td>
                <td style={td}>
                  {a.n_excluded} <span style={{ color: PALETTE.soft }}>/ {a.n_total}</span>
                </td>
                <td style={{ ...td, whiteSpace: "normal", color: PALETTE.soft, fontSize: "0.78rem" }}>
                  {Object.entries(a.exclusions || {})
                    .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
                    .join(", ") || "—"}
                </td>
                <td style={{ ...td, color: a.pin_violations ? PALETTE.warn : PALETTE.soft }}>
                  {a.pin_violations}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Limitations({ data }) {
  const items = data.limitations || [];
  if (!items.length) return null;
  return (
    <Card style={{ background: "#fffdf5", borderColor: "#f0e2b8" }}>
      <SectionTitle hint="Read these before quoting the number above. They are the parts of the measurement a reviewer should attack first.">
        Methods and limitations
      </SectionTitle>
      <ul style={{ margin: 0, paddingLeft: "1.1rem", color: PALETTE.soft, fontSize: "0.8rem", lineHeight: 1.6 }}>
        {items.map((t) => (
          <li key={t} style={{ marginBottom: "0.4rem" }}>
            {t}
          </li>
        ))}
      </ul>
    </Card>
  );
}

export function BenchmarkPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    fetchBenchmark()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const runMeta = data?.available
    ? [
        `run ${data.run_id}`,
        `${data.n_requests} requests`,
        `${data.n_prompts} prompts × ${data.repeats} repeats`,
        data.grid_carbon_g_per_kwh != null ? `grid pinned at ${data.grid_carbon_g_per_kwh} gCO₂/kWh` : null,
      ].filter(Boolean)
    : [];

  return (
    <div style={{ padding: "1.5rem", maxWidth: 1100, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: "1rem", marginBottom: "0.35rem" }}>
        <h2 style={{ margin: 0, fontSize: "1.4rem", color: PALETTE.ink }}>Routing Benchmark</h2>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          style={{
            border: `1px solid ${PALETTE.border}`,
            background: "#fff",
            color: PALETTE.deep,
            borderRadius: 6,
            padding: "0.35rem 0.8rem",
            fontSize: "0.78rem",
            cursor: loading ? "default" : "pointer",
          }}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>
      <p style={{ margin: "0 0 1.2rem", fontSize: "0.82rem", color: PALETTE.soft, lineHeight: 1.6 }}>
        Three arms, one prompt set, one frozen router. The claim this system makes is that
        carbon-aware routing costs less carbon than always reaching for the big model —{" "}
        <em>and</em> less than the heuristic a competent engineer writes in an afternoon. This is
        where that claim is either supported or it is not.
      </p>

      {runMeta.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
          {runMeta.map((m) => (
            <span
              key={m}
              style={{
                fontSize: "0.72rem",
                color: PALETTE.deep,
                background: "#eaf6f2",
                border: `1px solid ${PALETTE.border}`,
                borderRadius: 999,
                padding: "0.2rem 0.6rem",
              }}
            >
              {m}
            </span>
          ))}
        </div>
      )}

      {error && (
        <Card style={{ borderColor: PALETTE.danger }}>
          <span style={{ color: PALETTE.danger, fontSize: "0.85rem" }}>{error}</span>
        </Card>
      )}

      {data && !data.available && (
        <Card>
          <SectionTitle>No run published yet</SectionTitle>
          <p style={{ fontSize: "0.82rem", color: PALETTE.soft, lineHeight: 1.6, margin: 0 }}>
            {data.reason}
          </p>
          <pre
            style={{
              marginTop: "0.9rem",
              marginBottom: 0,
              background: PALETTE.bg,
              border: `1px solid ${PALETTE.border}`,
              borderRadius: 6,
              padding: "0.8rem",
              fontSize: "0.75rem",
              color: PALETTE.ink,
              overflowX: "auto",
            }}
          >
{`docker compose -f docker-compose.ubuntu-vgpu.yml \\
               -f benchmark/docker-compose.bench.yml \\
               --env-file .env up -d --build api-bench

python3 benchmark/run_bench.py --repeats 3 --run-id full-01
python3 benchmark/report.py full-01`}
          </pre>
        </Card>
      )}

      {data?.available && (
        <>
          <Headline data={data} />
          <QualityByCategory data={data} />
          <ModelMix data={data} />
          <Exclusions data={data} />
          <Limitations data={data} />
          <p style={{ fontSize: "0.72rem", color: PALETTE.soft, textAlign: "right", margin: 0 }}>
            Published {new Date(data.generated_at).toLocaleString()} · basis {data.carbon_basis || "—"}
          </p>
        </>
      )}
    </div>
  );
}

export default BenchmarkPanel;

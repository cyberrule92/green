/**
 * AgentPanel — Adaptive Green AI
 *
 * UI for the agentic coding harness (coding_agent.py, LangGraph).
 *
 * This panel is deliberately not a chat. The harness is off the CSS path: CSS
 * scores carbon per *request*, but an agent is a loop, so its cost is
 * tokens x steps x attempts. The number this screen exists to show is therefore
 * carbon per *successful completion* — and, when a task fails, the carbon that
 * bought nothing at all ("wasted"). Those two numbers are the whole argument for
 * starting the ladder at the greenest code-capable model rather than the
 * greenest model.
 *
 * A task submitted on a dirty grid comes back "queued", not "done": it is held
 * on the EcoServe deferred queue and dispatched in the greenest window the
 * forecast offers. We poll it to completion here.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchAgentStatus,
  fetchAgentTask,
  fetchAgentTasks,
  submitAgentTask,
} from "../lib/api";

const POLL_MS = 4_000;

// The harness ends a task as completed | failed | harness_error | budget_exceeded |
// escalation_unavailable, and may add more. Defining "done" as the complement of the
// two in-flight states means a new terminal status can never leave the poller
// spinning forever on a task that finished minutes ago.
const IN_FLIGHT = new Set(["queued", "running"]);
const isTerminal = (status) => Boolean(status) && !IN_FLIGHT.has(status);

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
  bg:      "#f4f8f7",
};

const STATUS_COLOR = {
  completed:              PALETTE.primary,
  running:                PALETTE.blue,
  queued:                 PALETTE.warn,
  failed:                 PALETTE.danger,
  harness_error:          PALETTE.purple,   // infrastructure, not the model — read it differently
  budget_exceeded:        PALETTE.warn,
  escalation_unavailable: PALETTE.warn,
};

const STATUS_LABEL = {
  harness_error:          "harness error",
  budget_exceeded:        "carbon budget exceeded",
  escalation_unavailable: "escalation rung down",
};

const fmtCo2 = (g) => {
  if (g == null) return "–";
  if (g < 1) return `${(g * 1000).toFixed(1)} mg`;
  if (g < 1000) return `${g.toFixed(2)} g`;
  return `${(g / 1000).toFixed(3)} kg`;
};
const fmtTime = (iso) => {
  if (!iso) return "–";
  try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
  catch { return iso; }
};

// ─── Building blocks (mirrors ObservabilityPanel) ────────────────────────────
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

function Kpi({ label, value, sub, color = PALETTE.ink, accent }) {
  return (
    <div style={{
      background: "white",
      border: `1px solid ${PALETTE.border}`,
      borderLeft: accent ? `3px solid ${accent}` : `1px solid ${PALETTE.border}`,
      borderRadius: 12,
      padding: "12px 14px",
      display: "flex", flexDirection: "column", gap: 4,
    }}>
      <span style={{
        fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em",
        color: PALETTE.soft, fontWeight: 700,
      }}>{label}</span>
      <span style={{ fontSize: "1.45rem", fontWeight: 700, color, lineHeight: 1.1 }}>{value}</span>
      {sub && <span style={{ fontSize: "0.74rem", color: PALETTE.soft }}>{sub}</span>}
    </div>
  );
}

function Pill({ children, color = PALETTE.primary }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "1px 8px", borderRadius: 999,
      fontSize: "0.72rem", fontWeight: 700,
      color, background: `${color}1f`, border: `1px solid ${color}33`,
    }}>{children}</span>
  );
}

function Field({ label, hint, children }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{
        fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.05em",
        color: PALETTE.soft, fontWeight: 700,
      }}>{label}</span>
      {children}
      {hint && <span style={{ fontSize: "0.72rem", color: PALETTE.soft }}>{hint}</span>}
    </label>
  );
}

const inputStyle = {
  border: `1px solid ${PALETTE.border}`,
  borderRadius: 8,
  padding: "8px 10px",
  fontSize: "0.85rem",
  color: PALETTE.ink,
  background: "white",
  fontFamily: "inherit",
  width: "100%",
  boxSizing: "border-box",
};

// ─── Escalation ladder ──────────────────────────────────────────────────────
// Rung 0 is the greenest *code-capable* model, not the greenest model. Shown
// with liveness because escalating into a rung whose container is down is worse
// than not escalating at all.
function Ladder({ ladder = [], activeLabel }) {
  if (!ladder.length) return <span style={{ color: PALETTE.soft }}>No rungs configured.</span>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {ladder.map((rung, i) => {
        const isActive = activeLabel && rung.label === activeLabel;
        return (
          <div key={rung.target_id || i} style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "8px 10px", borderRadius: 8,
            border: `1px solid ${isActive ? PALETTE.primary : PALETTE.border}`,
            background: isActive ? `${PALETTE.primary}0f` : "transparent",
          }}>
            <span style={{
              fontSize: "0.72rem", fontWeight: 700, color: PALETTE.soft,
              minWidth: 44,
            }}>rung {i}</span>
            <span style={{ flex: 1, fontSize: "0.85rem", fontWeight: 600, color: PALETTE.ink }}>
              {rung.label}
            </span>
            {isActive && <Pill color={PALETTE.primary}>served</Pill>}
            <Pill color={rung.live ? PALETTE.primary : PALETTE.danger}>
              {rung.live ? "live" : "down"}
            </Pill>
          </div>
        );
      })}
    </div>
  );
}

// ─── Event trace ────────────────────────────────────────────────────────────
function Trace({ events = [] }) {
  if (!events.length) return <span style={{ color: PALETTE.soft }}>No events.</span>;
  const t0 = events[0].t;
  return (
    <div style={{ display: "flex", flexDirection: "column", maxHeight: 280, overflowY: "auto" }}>
      {events.map((ev, i) => {
        const { t, event, ...rest } = ev;
        const color = event === "escalate" ? PALETTE.purple
                    : event === "verify_fail" || event === "crash" ? PALETTE.danger
                    : event === "verify_pass" ? PALETTE.primary
                    : PALETTE.soft;
        return (
          <div key={i} style={{
            display: "flex", gap: 10, alignItems: "baseline",
            padding: "6px 4px",
            borderBottom: i < events.length - 1 ? `1px solid ${PALETTE.border}` : "none",
            fontSize: "0.78rem",
          }}>
            <span style={{ color: PALETTE.soft, fontVariantNumeric: "tabular-nums", minWidth: 48 }}>
              +{(t - t0).toFixed(1)}s
            </span>
            <span style={{ fontWeight: 700, color, minWidth: 110 }}>{event}</span>
            <span style={{ color: PALETTE.soft, wordBreak: "break-word", flex: 1 }}>
              {Object.entries(rest)
                .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
                .join("  ")}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Result ─────────────────────────────────────────────────────────────────
function Result({ record }) {
  const status = record.status;
  const result = record.result;

  if (status === "queued") {
    return (
      <Card title="Deferred to a low-carbon window">
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Pill color={PALETTE.warn}>queued</Pill>
            <span style={{ fontSize: "0.85rem", color: PALETTE.ink }}>{record.reason}</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 10 }}>
            <Kpi
              label="Grid at submit"
              value={`${Number(record.grid_ci_at_submit ?? 0).toFixed(0)}`}
              sub="gCO₂/kWh"
              accent={PALETTE.warn}
            />
            <Kpi
              label="Target dispatch"
              value={fmtTime(record.target_dispatch)}
              sub="greenest point in the forecast"
              accent={PALETTE.blue}
            />
          </div>
          <span style={{ fontSize: "0.78rem", color: PALETTE.soft }}>
            The task is held on the EcoServe queue, not dropped — it runs when the grid is cleanest,
            or when its deferral budget expires. This view polls until it finishes.
          </span>
        </div>
      </Card>
    );
  }

  if (status === "running" || !result) {
    return (
      <Card title="Running">
        <span style={{ fontSize: "0.85rem", color: PALETTE.soft }}>
          Agent is working… generating, running the tests in the sandbox, and escalating only if the
          verifier says so.
        </span>
      </Card>
    );
  }

  const completed = result.status === "completed";
  const files = Object.entries(result.files || {});

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 10 }}>
        {/* The metric the whole harness exists to optimise — and its shadow. A
            failed task has no carbon per completion, only waste. */}
        <Kpi
          label={completed ? "Carbon / completion" : "Wasted carbon"}
          value={fmtCo2(completed ? result.carbon_per_completion_g : result.wasted_carbon_g)}
          sub={completed ? "bought a working solution" : "bought nothing"}
          color={completed ? PALETTE.primary : PALETTE.danger}
          accent={completed ? PALETTE.primary : PALETTE.danger}
        />
        <Kpi
          label="Final rung"
          value={result.final_tier || "–"}
          sub={result.escalated ? "escalated on verifier evidence" : "solved on the greenest rung"}
          accent={result.escalated ? PALETTE.purple : PALETTE.primary}
        />
        <Kpi label="LLM calls" value={result.total_llm_calls ?? "–"} sub={`${result.duration_s ?? "–"}s wall clock`} />
        <Kpi
          label="Grid"
          value={`${Number(result.grid_ci ?? 0).toFixed(0)}`}
          sub={
            result.deferral_ci_saved != null
              ? `gCO₂/kWh · ${result.deferral_ci_saved.toFixed(0)} lower than at submit`
              : "gCO₂/kWh at execution"
          }
          accent={result.deferral_ci_saved > 0 ? PALETTE.primary : undefined}
        />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <Pill color={STATUS_COLOR[result.status] || PALETTE.soft}>
          {STATUS_LABEL[result.status] || result.status}
        </Pill>
        <Pill color={PALETTE.blue}>{result.orchestrator}</Pill>
        {result.deferred && <Pill color={PALETTE.warn}>deferred</Pill>}
        {/* Which is being claimed: "it passed your tests" or "it passed its own"? */}
        <Pill color={result.spec_source === "caller" ? PALETTE.primary : PALETTE.soft}>
          spec: {result.spec_source === "caller" ? "caller-supplied" : "model-authored"}
        </Pill>
        {result.reason && (
          <span style={{ fontSize: "0.8rem", color: PALETTE.soft }}>{result.reason}</span>
        )}
      </div>

      <Card title="Trace"><Trace events={result.events} /></Card>

      {files.length > 0 && (
        <Card title={`Files (${files.length})`} padded={false}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            {files.map(([path, content]) => (
              <details key={path} style={{ borderBottom: `1px solid ${PALETTE.border}` }}>
                <summary style={{
                  cursor: "pointer", padding: "8px 14px",
                  fontSize: "0.82rem", fontWeight: 600, color: PALETTE.ink,
                }}>{path}</summary>
                <pre style={{
                  margin: 0, padding: "10px 14px", overflowX: "auto",
                  background: PALETTE.bg, fontSize: "0.78rem", lineHeight: 1.5,
                }}>{content}</pre>
              </details>
            ))}
          </div>
        </Card>
      )}

      {result.test_output && (
        <Card title="Test output">
          <pre style={{
            margin: 0, overflowX: "auto", maxHeight: 240,
            fontSize: "0.78rem", lineHeight: 1.5, color: PALETTE.ink,
          }}>{result.test_output}</pre>
        </Card>
      )}
    </div>
  );
}

// ─── Panel ──────────────────────────────────────────────────────────────────
export function AgentPanel() {
  const [status, setStatus] = useState(null);
  const [task, setTask] = useState("");
  const [testCommand, setTestCommand] = useState("python -m pytest -q");
  const [budget, setBudget] = useState("");
  const [allowDefer, setAllowDefer] = useState(true);
  const [tests, setTests] = useState("");

  const [record, setRecord] = useState(null);
  const [recent, setRecent] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const pollRef = useRef(null);

  const loadRecent = useCallback(() => {
    fetchAgentTasks({ limit: 10 })
      .then((d) => setRecent(d.tasks || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchAgentStatus().then(setStatus).catch((e) => setError(e.message));
    loadRecent();
  }, [loadRecent]);

  // A queued or running task outlives its POST — follow it to a terminal state.
  useEffect(() => {
    if (!record || isTerminal(record.status)) {
      clearInterval(pollRef.current);
      return undefined;
    }
    pollRef.current = setInterval(() => {
      fetchAgentTask(record.task_id)
        .then((next) => {
          setRecord(next);
          if (isTerminal(next.status)) loadRecent();
        })
        .catch(() => {});
    }, POLL_MS);
    return () => clearInterval(pollRef.current);
  }, [record, loadRecent]);

  const submit = async () => {
    if (!task.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    setRecord(null);
    try {
      const parsedBudget = budget.trim() === "" ? null : Number(budget);
      const res = await submitAgentTask({
        task: task.trim(),
        testCommand,
        // Explicit null, not a falsy check: a 0 g budget is a meaningful request
        // (run nothing), and coercing it to "unset" would hand back the default.
        carbonBudgetG: Number.isFinite(parsedBudget) ? parsedBudget : null,
        allowDefer,
        tests,
      });
      // The POST returns the *record*, which for a deferred task is only the
      // queue receipt — the polling effect takes it from here.
      setRecord(res.status ? res : { ...res, status: "running" });
      loadRecent();
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const activeLabel = record?.result?.final_tier;

  return (
    <div style={{
      padding: 18,
      display: "flex",
      flexDirection: "column",
      gap: 14,
      background: PALETTE.bg,
      minHeight: "100%",
    }}>
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.4rem", color: PALETTE.ink }}>Coding Arena</h2>
          <div style={{ fontSize: "0.85rem", color: PALETTE.soft, marginTop: 2 }}>
            Optimises carbon per <em>successful completion</em>, not per token · verifier-gated escalation
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {status && (
            <>
              <Pill color={status.enabled ? PALETTE.primary : PALETTE.danger}>
                {status.enabled ? "enabled" : "disabled"}
              </Pill>
              <Pill color={PALETTE.blue}>{status.orchestrator}</Pill>
              <Pill color={PALETTE.warn}>defer &gt; {status.defer_above_ci} gCO₂/kWh</Pill>
            </>
          )}
        </div>
      </div>

      {error && (
        <div style={{
          border: `1px solid ${PALETTE.danger}33`, background: `${PALETTE.danger}12`,
          color: PALETTE.danger, borderRadius: 10, padding: "10px 14px", fontSize: "0.85rem",
        }}>{error}</div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1fr) minmax(280px, 360px)", gap: 14, alignItems: "start" }}>
        {/* ── Submit ──────────────────────────────────────────────────────── */}
        <Card title="New task">
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <Field label="Task" hint="Describe what to build. The agent writes complete files and runs the tests itself.">
              <textarea
                rows={5}
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder="Write fizzbuzz(n) in fizzbuzz.py plus a pytest suite that covers it."
                style={{ ...inputStyle, resize: "vertical" }}
              />
            </Field>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 160px", gap: 10 }}>
              <Field label="Test command">
                <input
                  value={testCommand}
                  onChange={(e) => setTestCommand(e.target.value)}
                  style={inputStyle}
                />
              </Field>
              <Field label="Carbon budget" hint={`default ${status?.carbon_budget_g ?? "–"} g`}>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={budget}
                  onChange={(e) => setBudget(e.target.value)}
                  placeholder="gCO₂eq"
                  style={inputStyle}
                />
              </Field>
            </div>

            <Field
              label="Tests (optional — the frozen spec)"
              hint="Leave empty and the agent writes its own tests, then is judged against them. Paste a pytest suite and it becomes the spec: the model only writes the implementation and may not touch the tests."
            >
              <textarea
                rows={tests ? 8 : 3}
                value={tests}
                onChange={(e) => setTests(e.target.value)}
                placeholder={"from solution import word_count\n\ndef test_counts_words():\n    assert word_count(\"a b a\") == {\"a\": 2, \"b\": 1}"}
                style={{ ...inputStyle, resize: "vertical", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: "0.78rem" }}
              />
            </Field>

            <label style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: "0.82rem", color: PALETTE.ink }}>
              <input
                type="checkbox"
                checked={!allowDefer}
                onChange={(e) => setAllowDefer(!e.target.checked)}
                style={{ marginTop: 3 }}
              />
              <span>
                Run now, ignoring grid intensity
                <span style={{ display: "block", color: PALETTE.soft, fontSize: "0.75rem" }}>
                  Off by default: on a dirty grid the task is queued for the greenest window. Tick this
                  to force an inline run.
                </span>
              </span>
            </label>

            <button
              onClick={submit}
              disabled={submitting || !task.trim() || (status && !status.enabled)}
              style={{
                alignSelf: "flex-start",
                padding: "9px 18px", borderRadius: 8, border: "none",
                background: submitting || !task.trim() ? PALETTE.soft : PALETTE.primary,
                color: "white", fontWeight: 700, fontSize: "0.85rem",
                cursor: submitting || !task.trim() ? "not-allowed" : "pointer",
              }}
            >
              {submitting ? "Submitting…" : "Run task"}
            </button>
          </div>
        </Card>

        {/* ── Ladder ──────────────────────────────────────────────────────── */}
        <Card title="Escalation ladder">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <Ladder ladder={status?.ladder} activeLabel={activeLabel} />
            <span style={{ fontSize: "0.75rem", color: PALETTE.soft }}>
              Starts at the greenest <strong>code-capable</strong> model — not the greenest model. A
              model that cannot finish the task is not green: it burns the whole step budget and
              delivers nothing. Escalation happens only on verifier evidence.
            </span>
          </div>
        </Card>
      </div>

      {record && <Result record={record} />}

      {/* ── History ───────────────────────────────────────────────────────── */}
      {recent.length > 0 && (
        <Card title="Recent tasks" padded={false}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            {recent.map((t) => (
              <button
                key={t.task_id}
                onClick={() => fetchAgentTask(t.task_id).then(setRecord).catch((e) => setError(e.message))}
                style={{
                  display: "flex", alignItems: "center", gap: 10, textAlign: "left",
                  padding: "9px 14px", border: "none", background: "transparent",
                  borderBottom: `1px solid ${PALETTE.border}`, cursor: "pointer",
                  fontSize: "0.8rem", color: PALETTE.ink, fontFamily: "inherit",
                }}
              >
                <Pill color={STATUS_COLOR[t.status] || PALETTE.soft}>
                  {STATUS_LABEL[t.status] || t.status}
                </Pill>
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {t.task}
                </span>
                {t.deferred && <Pill color={PALETTE.warn}>deferred</Pill>}
                {t.final_tier && <span style={{ color: PALETTE.soft }}>{t.final_tier}</span>}
                <span style={{ color: PALETTE.soft, fontVariantNumeric: "tabular-nums", minWidth: 70, textAlign: "right" }}>
                  {t.carbon_g != null ? fmtCo2(t.carbon_g) : "–"}
                </span>
              </button>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

export default AgentPanel;

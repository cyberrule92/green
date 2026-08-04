/**
 * FineTunePanel — carbon-aware LoRA/QLoRA fine-tuning.
 *
 * The other half of the problem the Models tab exists for. Onboarding imports a
 * better rung; this makes the small rung already installed good enough at this
 * deployment's traffic that the router stops escalating to the big one.
 *
 * Four things this screen refuses to hide, because each is a way the feature
 * could quietly become greenwashing:
 *
 * 1. **Training spends carbon.** It is the largest single compute event the
 *    system performs. The estimate is shown before the button, and the measured
 *    cost is shown after — never just the saving.
 *
 * 2. **The dataset is shown with its rejections.** "We trained on 40 of your 900
 *    rows" is something an operator has to see before a GPU spins up, not after.
 *    Down-votes are discarded on purpose: supervised tuning cannot use a signal
 *    that says what *not* to say.
 *
 * 3. **A job that is waiting is not a job that is stuck.** Above the deferral
 *    threshold the run waits for the cleanest window in the forecast, which can
 *    be hours. The state says so, and says when.
 *
 * 4. **An adapter is not routable until measured, and it does not inherit the
 *    base model's accuracy.** The entire premise is that quality changed, so
 *    assuming its direction is the one thing this must not do.
 */

import React, { useCallback, useEffect, useState } from "react";
import {
  fetchFinetuneCapability,
  previewFinetunePlan,
  submitFinetuneJob,
  fetchFinetuneJobs,
  cancelFinetuneJob,
  serveFinetuneAdapter,
} from "../lib/api";

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

const STATE_TONE = {
  succeeded: "ok",
  failed: "bad",
  cancelled: "soft",
  waiting_for_window: "warn",
  training: "warn",
  registering: "warn",
  queued: "soft",
};

const STATE_BLURB = {
  waiting_for_window:
    "Waiting for the cleanest window in the 48-hour forecast. Nobody is blocked on this, which is what makes the wait free.",
  training: "On the GPU now. Grid intensity is sampled throughout, so the carbon figure tracks the hours actually used.",
  registering: "Writing the adapter into the zoo as unavailable — it cannot be routed to until measured.",
};

const fmtG = (x) => (x == null ? "—" : `${Number(x).toFixed(2)} gCO₂e`);
const fmtH = (x) => (x == null ? "—" : x < 1 ? `${Math.round(x * 60)} min` : `${Number(x).toFixed(1)} h`);

export function FineTunePanel({ palette, Section, Pill }) {
  const [capability, setCapability] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [baseModelId, setBaseModelId] = useState("");
  const [method, setMethod] = useState("qlora");
  const [loraRank, setLoraRank] = useState(16);
  const [epochs, setEpochs] = useState(3);

  const refresh = useCallback(async () => {
    try {
      const [cap, j] = await Promise.all([
        fetchFinetuneCapability(),
        fetchFinetuneJobs(20).catch(() => ({ jobs: [] })),
      ]);
      setCapability(cap);
      setJobs(j.jobs || []);
    } catch (e) {
      setError(String(e.message || e));
    }
  }, []);

  useEffect(() => {
    refresh();
    // A run can sit in waiting_for_window for hours and then train for more.
    const t = setInterval(refresh, 10000);
    return () => clearInterval(t);
  }, [refresh]);

  const runPreview = async () => {
    if (!baseModelId) return;
    setBusy(true);
    setError(null);
    setPlan(null);
    try {
      setPlan(await previewFinetunePlan({ baseModelId, method, loraRank, epochs }));
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const submit = async (forceNow) => {
    if (!baseModelId) return;
    setBusy(true);
    setError(null);
    try {
      await submitFinetuneJob({
        base_model_id: baseModelId,
        method,
        lora_rank: loraRank,
        epochs,
        force_now: forceNow,
      });
      setPlan(null);
      await refresh();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const dataset = capability?.dataset;
  const trainer = capability?.trainer;
  const enabled = capability?.enabled && trainer?.enabled;
  const usable = dataset?.usable;
  const planPlan = plan?.plan;
  const canSubmit = enabled && usable && !busy && baseModelId;

  const inputStyle = {
    padding: "8px 10px",
    borderRadius: 8,
    border: `1px solid ${palette.border}`,
    fontSize: 13,
  };

  return (
    <>
      <Section
        title="Fine-tuning"
        subtitle="Make the cheap rung good enough to pick, instead of importing a better one."
        right={
          <Pill tone={enabled ? "ok" : "warn"}>{enabled ? "available" : "off"}</Pill>
        }
      >
        <p style={{ margin: "0 0 14px", fontSize: 13, color: palette.soft, maxWidth: 780 }}>
          Trains a LoRA adapter on the up-voted answers this deployment has collected. Training{" "}
          <strong>spends</strong> carbon — it is the largest single compute event here — and only pays back if
          the adapter lets the router serve on a model it would otherwise have escalated past. The job waits
          for the cleanest window in the forecast, meters itself against sampled grid intensity, and registers
          the result as <strong>not routable</strong> until someone posts a real measurement.
        </p>

        {error && (
          <div style={{ background: "#fbeaea", color: palette.danger, border: "1px solid #f0c4c4", borderRadius: 10, padding: 12, marginBottom: 14, fontSize: 13 }}>
            {error}
          </div>
        )}

        {capability && !capability.enabled && (
          <div style={{ background: "#fdf4e0", border: "1px solid #f0dcae", borderRadius: 10, padding: 12, marginBottom: 14, fontSize: 13, color: "#8a6400" }}>
            Fine-tuning is disabled. Set <code>FINETUNE_ENABLED=true</code>. Dataset stats and plan previews
            still work, so you can see whether a run would be worth starting.
          </div>
        )}
        {capability?.enabled && trainer && !trainer.enabled && (
          <div style={{ background: "#fdf4e0", border: "1px solid #f0dcae", borderRadius: 10, padding: 12, marginBottom: 14, fontSize: 13, color: "#8a6400" }}>
            Trainer unavailable: {trainer.reason}
          </div>
        )}

        {/* ── Dataset ─────────────────────────────────────────────────────── */}
        <div style={{ border: `1px solid ${palette.border}`, borderRadius: 12, padding: 14, background: palette.bg, marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
            <strong style={{ fontSize: 14, color: palette.ink }}>Training data</strong>
            <Pill tone={usable ? "ok" : "warn"}>
              {dataset ? `${dataset.samples} usable` : "—"}
              {dataset?.min_required ? ` / ${dataset.min_required} needed` : ""}
            </Pill>
          </div>
          <p style={{ margin: "8px 0 0", fontSize: 12, color: palette.soft }}>
            Up-voted (prompt, answer) pairs from <code>/api/feedback</code>. Down-votes are discarded: they say
            what not to answer, which supervised tuning cannot consume.
          </p>
          {dataset?.rejected > 0 && (
            <details style={{ marginTop: 10 }}>
              <summary style={{ cursor: "pointer", fontSize: 12, color: palette.soft }}>
                {dataset.rejected} row{dataset.rejected === 1 ? "" : "s"} rejected — see why
              </summary>
              <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12, color: palette.soft }}>
                {Object.entries(dataset.reasons || {}).map(([why, n]) => (
                  <li key={why}>
                    <strong>{n}</strong> — {why.replace(/_/g, " ")}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {dataset && !usable && (
            <p style={{ margin: "10px 0 0", fontSize: 12, color: "#8a6400" }}>
              Below the floor. A run on this little data burns GPU-hours to memorise the set and get worse at
              everything else — and the loss curve looks fine while it happens.
            </p>
          )}
        </div>

        {/* ── Plan a run ──────────────────────────────────────────────────── */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
          <input
            value={baseModelId}
            onChange={(e) => setBaseModelId(e.target.value)}
            placeholder="base model id from the zoo, e.g. local-vgpu-small"
            style={{ ...inputStyle, flex: "1 1 260px" }}
          />
          <select value={method} onChange={(e) => setMethod(e.target.value)} style={inputStyle}>
            {(capability?.methods || ["qlora", "lora"]).map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: palette.soft }}>
            rank
            <input
              type="number" min={1} max={capability?.max_lora_rank || 64} value={loraRank}
              onChange={(e) => setLoraRank(Number(e.target.value) || 16)}
              style={{ ...inputStyle, width: 76 }}
            />
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: palette.soft }}>
            epochs
            <input
              type="number" min={1} max={10} value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value) || 3)}
              style={{ ...inputStyle, width: 70 }}
            />
          </label>
          <button
            onClick={runPreview}
            disabled={busy || !baseModelId}
            style={{
              background: palette.primary, color: "#fff", border: 0, borderRadius: 8,
              padding: "9px 18px", fontWeight: 600, cursor: busy || !baseModelId ? "not-allowed" : "pointer",
            }}
          >
            {busy ? "Sizing…" : "Estimate"}
          </button>
        </div>

        {plan && (
          <div style={{ border: `1px solid ${palette.border}`, borderRadius: 12, padding: 14, background: palette.bg }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
              <strong style={{ color: palette.ink }}>{planPlan?.base_model_id}</strong>
              <Pill tone={planPlan?.fits ? "ok" : "bad"}>
                {planPlan?.fits ? `${planPlan.method} r=${planPlan.lora_rank}` : "does not fit"}
              </Pill>
            </div>
            <p style={{ margin: "10px 0", fontSize: 13, color: palette.ink }}>{planPlan?.reason}</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10, fontSize: 13 }}>
              <div><span style={{ color: palette.soft }}>samples</span><br /><strong>{planPlan?.samples ?? "—"}</strong></div>
              <div><span style={{ color: palette.soft }}>VRAM</span><br /><strong>{planPlan?.est_vram_mb ? `${Math.round(planPlan.est_vram_mb)} MB` : "—"}</strong></div>
              <div><span style={{ color: palette.soft }}>duration</span><br /><strong>{fmtH(planPlan?.est_duration_s ? planPlan.est_duration_s / 3600 : null)}</strong></div>
              <div><span style={{ color: palette.soft }}>carbon</span><br /><strong>{fmtG(planPlan?.est_carbon_g)}</strong></div>
            </div>
            {plan?.schedule && (
              <p style={{ margin: "10px 0 0", fontSize: 12, color: palette.soft }}>
                Schedule: {plan.schedule.reason}
              </p>
            )}
            <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap", alignItems: "center" }}>
              <button
                onClick={() => submit(false)}
                disabled={!canSubmit || !planPlan?.fits}
                style={{
                  background: canSubmit && planPlan?.fits ? palette.primary : "#c9d6d2", color: "#fff",
                  border: 0, borderRadius: 8, padding: "8px 16px", fontWeight: 600,
                  cursor: canSubmit && planPlan?.fits ? "pointer" : "not-allowed",
                }}
              >
                Queue for the cleanest window
              </button>
              <button
                onClick={() => submit(true)}
                disabled={!canSubmit || !planPlan?.fits}
                title="Skips the wait and starts on the current grid, whatever its intensity."
                style={{
                  background: "transparent", color: palette.deep, border: `1px solid ${palette.border}`,
                  borderRadius: 8, padding: "8px 16px", fontWeight: 600,
                  cursor: canSubmit && planPlan?.fits ? "pointer" : "not-allowed",
                }}
              >
                Start now
              </button>
              <span style={{ fontSize: 12, color: palette.soft }}>
                Grid is {Math.round(capability?.current_ci || 0)} gCO₂e/kWh; the deferral threshold is{" "}
                {Math.round(capability?.defer_above_ci || 0)}.
              </span>
            </div>
          </div>
        )}
      </Section>

      {jobs.length > 0 && (
        <Section title="Fine-tuning jobs" subtitle="A run can wait hours for a clean window, then train for hours more.">
          {jobs.map((j) => (
            <div key={j.job_id} style={{ borderBottom: `1px solid ${palette.border}`, padding: "10px 0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <div>
                  <strong style={{ fontSize: 13, color: palette.ink }}>{j.base_model_id}</strong>
                  <span style={{ fontSize: 12, color: palette.soft }}> → {j.adapter_id}</span>
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <Pill tone={STATE_TONE[j.state] || "soft"}>{String(j.state).replace(/_/g, " ")}</Pill>
                  {!TERMINAL.has(j.state) && (
                    <button
                      onClick={() => cancelFinetuneJob(j.job_id).then(refresh).catch((e) => setError(String(e.message || e)))}
                      style={{ background: "transparent", border: `1px solid ${palette.border}`, borderRadius: 6, padding: "3px 10px", fontSize: 12, cursor: "pointer" }}
                    >
                      Cancel
                    </button>
                  )}
                  {j.state === "succeeded" && j.adapter_id && (
                    <button
                      onClick={() => serveFinetuneAdapter(j.adapter_id).then(refresh).catch((e) => setError(String(e.message || e)))}
                      title="Serves base + adapter through one vLLM container (--enable-lora)."
                      style={{ background: "transparent", border: `1px solid ${palette.border}`, borderRadius: 6, padding: "3px 10px", fontSize: 12, cursor: "pointer" }}
                    >
                      Serve
                    </button>
                  )}
                </div>
              </div>
              {STATE_BLURB[j.state] && (
                <div style={{ fontSize: 12, color: palette.soft, marginTop: 4 }}>{STATE_BLURB[j.state]}</div>
              )}
              {j.error && <div style={{ fontSize: 12, color: palette.danger, marginTop: 4 }}>{j.error}</div>}
              {j.steps?.length > 0 && (
                <div style={{ fontSize: 12, color: palette.soft, marginTop: 4 }}>
                  {j.steps[j.steps.length - 1].step}: {j.steps[j.steps.length - 1].detail}
                </div>
              )}
              <div style={{ fontSize: 12, color: palette.soft, marginTop: 4, display: "flex", gap: 14, flexWrap: "wrap" }}>
                {j.training_carbon_g > 0 && <span style={{ color: palette.warn }}>cost {fmtG(j.training_carbon_g)}</span>}
                {j.duration_s > 0 && <span>ran {fmtH(j.duration_s / 3600)}</span>}
                {j.dataset?.samples != null && <span>{j.dataset.samples} samples</span>}
                {j.ci_sample_count > 0 && <span>{j.ci_sample_count} grid samples</span>}
              </div>
              {j.state === "succeeded" && (
                <div style={{ fontSize: 12, color: "#8a6400", marginTop: 4 }}>
                  Registered as <strong>unavailable</strong>. It does not inherit the base model's accuracy —
                  the point of tuning is that quality changed — so post measured figures to{" "}
                  <code>/api/finetune/adapters/{j.adapter_id}/measure</code> to make it routable and to record
                  whether the {fmtG(j.training_carbon_g)} spent here ever pays back.
                </div>
              )}
            </div>
          ))}
        </Section>
      )}
    </>
  );
}

export default FineTunePanel;

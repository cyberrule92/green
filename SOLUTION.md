# Adaptive Green AI — Solution Document

A carbon-aware LLM orchestration platform. A single FastAPI control plane turns every chat request into a sustainability-optimised routing decision across multiple vLLM backends, applies safety rails, retrieves grounded evidence, defers high-carbon work, learns from every outcome, and writes an HMAC-signed audit trail. A React/Grommet front-end drives a chat UI plus four operational dashboards (Architecture, Carbon, RL Policy, **Observability**).

---

## 1. Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Frontend (React 19 + Vite + Grommet)            http://host:8080 / 5200 │
│  GreenAIChat → Chat | Architecture | Carbon | RL Policy | Observability  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │  /api/*
┌──────────────────────────────────▼───────────────────────────────────────┐
│  decision_engine.py  (FastAPI, port 8100)         30 endpoints           │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ pipeline:  guardrails-in → RAG → profiler → CSS rank → EcoServe →    │ │
│  │            vLLM dispatch → guardrails-out → audit → RL update        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────┬─────────┬────────┬─────────┬────────┬─────────┬─────────────┬───────┘
     │         │        │         │        │         │             │
     ▼         ▼        ▼         ▼        ▼         ▼             ▼
 routing_   advanced_  rl_      monitoring_ deferred_ model_zoo  conversation_
 policies   rag        controller layer      queue              store
   │         │          │         │          │
   │         │          │         │          ▼
   │         │          │         ▼     data/decision_logs.jsonl (HMAC-signed)
   │         │          ▼   Electricity Maps API (live + 48 h forecast)
   │         ▼    rl_state.json  (online REINFORCE persisted weights)
   ▼   rag_store.json  (chunks + embeddings)

host_metrics_service.py  (port 9000) → system_metrics.sh → nvidia-smi / top
vLLM containers          (8001 medium · 8002 full · 8004 stem-math · 8006 stem-coding)
```

Containers (production stack `docker-compose.ubuntu-vgpu.yml`):

| Service | Image | Port | Role |
|---|---|---|---|
| `green-api` | `green-api` (built) | 8100 | Decision engine + 30 endpoints |
| `green-metrics` | `green-metrics` (built) | 9000 | Host GPU/CPU sidecar |
| `green-frontend` | `green-frontend` (built nginx) | 8080 | Static React bundle + `/api` reverse proxy |
| `green-vllm-medium` / `-full` | `vllm/vllm-openai` | 8001 / 8002 | Dense LLM endpoints |
| `green-vllm-stem-math` / `-coding` | `vllm/vllm-openai` | 8004 / 8006 | Domain-specialist LLMs |

---

## 2. Request lifecycle (`POST /api/chat`)

`process_chat_request()` in `decision_engine.py:2149` orchestrates **every stage** in order. Failure paths are not optional bypass routes — each layer either passes the work forward or short-circuits with a deterministic, auditable substitute.

| # | Stage | Module | Outputs added to context |
|---|---|---|---|
| 1 | Conversation lookup / create | `conversation_store.py` | `conversation_id`, history |
| 2 | Attachment parsing | inline in `decision_engine.py` | `parsed_attachments` |
| 3 | Semantic prompt profiling | `routing_policies.infer_prompt_profile` | intent, complexity, recommended variant, SLA, accuracy floor, STEM domain |
| 4 | Tier-policy resolution + RL override | `rl_controller.get_policy` | per-tier `(w_carbon, w_latency, w_accuracy, w_cost)` |
| 5 | Live carbon + GPU metrics | `monitoring_layer` + sidecar | `grid_carbon`, `system_power_w`, GPU util, 48h forecast |
| 6 | CSS candidate ranking | `routing_policies.rank_routing_candidates` | ranked list with op + embodied carbon |
| 7 | GPU-aware re-rank | `apply_gpu_routing_adjustment` | demote candidates when GPU constrained |
| 8 | STEM domain steer | inline | promote stem-math / stem-coding / full |
| 9 | EcoServe action eval | `routing_policies.evaluate_ecoserve_actions` | `deferral_recommended`, `regional_reroute`, `best_low_carbon_window` |
| 10 | MoE expert placement + SLA guard | `model_zoo.plan_expert_placement` | placement plan, dense fallback if SLA blowout |
| 11 | RAG retrieval | `advanced_rag.AdvancedRAGService.retrieve` | hybrid-fused, cross-encoder reranked chunks |
| 12 | Evidence sufficiency assessment | inline | `grounded_request`, `coverage_ratio` |
| 13 | Prompt assembly + cache | inline | system + history-summary + RAG context + user prompt |
| 14 | Token counting + overflow escalation | inline | escalates to next variant if `tokens > _VARIANT_CAPS[v]` |
| 15 | Deterministic intercept | inline | math tables / arithmetic answered without vLLM |
| 16 | Input guardrails (`apply_guardrails(phase="input")`) | `nemo_guardrails.py` | hard-block jailbreaks / harmful content |
| 17 | vLLM inference | `run_vllm_inference` | model response + actual variant used |
| 18 | GPU CO₂ attribution | `compute_gpu_co2` | per-request gpu_co2_g |
| 19 | Output guardrails | `nemo_guardrails.py` | hard-block PII / unsafe content |
| 20 | Grounding verification + multi-tier fallbacks | inline | extractive-evidence fallback, off-topic detection, full-model retry, rule-based final safety net |
| 21 | Persistence | `conversation_store.save_message` | user + assistant rows in SQLite |
| 22 | Audit log entry | `log_decision` | HMAC-SHA256-signed JSONL row |
| 23 | RL outcome update (background thread) | `rl_controller.record_outcome` | new tier weights, baseline EMA, episode |

### 2.1 Where every value comes from

The chat response carries **everything** computed during the pipeline so the front-end can render fully:

```
sustainability_score   ← selected_candidate.css_score
model_variant          ← selected_candidate.model_variant
grid_carbon            ← Electricity Maps live signal
system_power_w         ← host metrics sidecar (nvidia-smi + top)
system_co2_g           ← system_power_w × grid_carbon × infer_duration_s
estimated_request_co2_g← op + embodied carbon (LLMCarbon)
input_understanding    ← semantic prompt profile
routing                ← {policy_version, tier, candidate rankings, eco_actions}
retrieval              ← {sources, evidence_assessment, grounding_verification}
guardrails             ← {input: {...}, output: {...}}
rl_policy              ← {tier, version, weights, exploration_applied}
tokens                 ← {input, output, total, co2_per_token_ug}
gpu                    ← {utilization_pct, power_w, co2_g, inference_duration_s}
```

---

## 3. Module-by-module logic

### 3.1 `routing_policies.py` — Composite Sustainability Score (CSS)

The heart of the router. Implements four mathematical layers from the paper.

#### (a) LLMCarbon operational carbon — `compute_operational_carbon_llmcarbon`

```
C_op (gCO₂) = (TDP × t × PUE) / (HE_eff × 3.6e6)  ×  CI  ×  region_mult

  TDP        = candidate.power_tdp_w           (e.g. 145 W medium, 225 W full)
  t          = inference_duration_s            (estimated_latency_ms / 1000)
  PUE        = candidate.pue                   (1.3 default for vGPU rack)
  HE_eff     = HE × (1 − all_to_all_overhead)  if MoE else HE
  CI         = grid carbon intensity (gCO₂/kWh) — live from Electricity Maps
  region_mult= candidate.region_carbon_multiplier
```

#### (b) LLMCarbon embodied carbon — `compute_embodied_carbon_llmcarbon`

```
C_emb_request = (mfg_carbon_kg × 1000) / (lifetime_years × annual_volume × avg_s) × t
  mfg_carbon_kg     ≈ 143 kg per A100-class device
  lifetime_years    = 5
  annual_volume     = 100 000 inferences
  avg_s             = device average inference duration
```

`carbon_total = C_op + C_emb`. Both are stored on every candidate so the audit log can prove the breakdown.

#### (c) MoE all-to-all overhead — `compute_moe_all_to_all_latency_ms`

```
T_comm_ms = (k × token_count × d_model × element_bytes) / bandwidth_bps × 1000
  k             = active_experts_k     (e.g. 2)
  d_model       = 4096
  element_bytes = 2 (fp16)
  bandwidth     = expert_bandwidth_gbps × 1e9
```

If a candidate is MoE the dispatch cost is added to its `estimated_latency_ms` *and* its hardware-efficiency is reduced by the all-to-all overhead ratio in the carbon term. SLA blowout (>1.5× SLA) triggers a dense fallback in stage #10 of the pipeline.

#### (d) CSS scoring — `rank_routing_candidates`

For each candidate, normalise each dimension across the candidate set with min-max:

```
carbon_score   = 1 − norm(total_carbon_g, min_carbon_g, max_carbon_g)
latency_score  = 1 − norm(latency_eff, 40 ms, max(latency)+moe + 1.5×SLA)
accuracy_score = norm(accuracy, 0.45, 1.0)
cost_score     = 1 − norm(cost_units, 0.05, max_cost)
region_score   = 1 − norm(region_w_c·CI/600 + region_w_l·net_ms/300, …)

CSS = w_c·carbon + w_l·latency + w_a·accuracy + w_cost·cost + w_region·region
```

Penalties / bonuses then adjust `CSS`:

| Adjustment | Trigger | Δ CSS |
|---|---|---|
| SLA penalty | `latency_eff > sla_ms` | `−min(0.12 + 0.05×overshoot, 0.25)` |
| Accuracy floor penalty | `accuracy < accuracy_floor` | `−0.18` |
| Semantic alignment | `|VARIANT_RANK(c) − preferred|` distance | up to ±0.14 |
| MoE complexity bonus | `variant=moe ∧ complexity_score > 0.7` | `+0.04` |
| High-carbon period | `CI ≥ 450 ∧ heavy variant not requested` | `−0.05` |
| Urgent priority | `priority∈{urgent,high} ∧ ultra-light` | `−0.04` |

The candidate with the highest adjusted CSS wins. The top-5 are persisted in the audit log so any decision can be replayed.

#### (e) EcoServe — `evaluate_ecoserve_actions`

Three sustainability primitives:

1. **Deferral** (Section 3.5.2): when `CI ≥ 450 g` AND `deferral_tolerance_ms > 0` AND the selected target supports batching AND the forecast contains a window ≥15% lower in CI within budget → enqueue.
2. **Regional reroute** (Section 3.5.3): if multi-region signals are available and a different zone has lower CI, set `regional_reroute=True`.
3. **Load shaping**: when high-carbon but not deferrable, surface a hint for batch workloads.

#### (f) Semantic prompt profiler — `infer_prompt_profile`

A SentenceTransformer encodes both the prompt and 4 prototype banks (`SEMANTIC_ROUTE_PROTOTYPES`, `SEMANTIC_PRIORITY_PROTOTYPES`, `SEMANTIC_INTENT_PROTOTYPES`, STEM keyword sets). When the model is unavailable a hashed-vector fallback keeps the contract identical.

The profile drives:
- **Recommended model variant** (`ultra-light | medium | full | moe | stem-math | stem-science | stem-coding`)
- **Priority** (`low | medium | high | urgent`)
- **Intent** (`explanation | analysis | troubleshooting | implementation | summarization`)
- **Complexity score** [0,1] from token count, attachments, conversation depth, reasoning keywords
- **SLA + accuracy floor** lookup tables keyed by recommended variant
- **Deferral tolerance**: 30 min for `urgent`, 5 min for `high`, 15 min for `medium`, 30 min for `low`

### 3.2 `rl_controller.py` — Online REINFORCE policy controller

Adapts `(w_carbon, w_latency, w_accuracy, w_cost)` per tenant tier on **every** request outcome — no offline training, no UI controls.

**Reward** (∈ [0, 1]):
```
R = λ_sla·r_sla + λ_carbon·r_carbon + λ_acc·r_acc + λ_cost·r_cost
λ defaults: 0.35 / 0.30 / 0.25 / 0.10  (env: RL_REWARD_LAMBDA_*)

r_sla     = max(0, 1 − over/sla_ms)            over = max(0, actual − sla)
r_carbon  = max(0, 1 − actual_g / 0.05)
r_acc     = {clean:1.0, quality_retry:0.65, fallback:0.30, timeout:0.05}
r_cost    = max(0, 1 − actual_cost_units / 1.0)
```

**Update rule**:
```
∇_w_i log π(a|s) = score_i(selected) − E_π[score_i]      (softmax over CSS)
advantage  = R − baseline_ema
α_t        = α_0 / (1 + √t)                              (decaying lr)
w_i ← w_i + α_t · advantage · ∇_w_i log π
project onto simplex with floor w_min = 0.05
baseline_ema = β·baseline_ema + (1−β)·R                  (β=0.95)
```

**Exploration**: Dirichlet noise mixed at ε=0.15 with α=0.3 (sparse perturbation) so the controller occasionally explores around the current policy without a discrete ε-greedy switch.

**Convergence**: when 50 consecutive episodes have reward variance < 0.005, `policy_version` is bumped — the front-end shows this transition on the RL Policy panel.

State is persisted to `data/rl_state.json` by a background saver thread (every 15 s when dirty); each tier carries `(weights, episode_count, baseline_ema, reward_history, policy_version, last_updated)`.

### 3.3 `advanced_rag.py` — Hybrid RAG

```
documents → chunk (size=900, overlap=180) → 256-dim embedding (or hashed fallback)
                                                                    ↓
query → tokenise + embed → dense top-14 (cosine)
                          ↘
                            sparse top-14 (BM25-like Counter scoring)
                          ↗
                         RRF fusion (1/(60+rank))
                                ↓
                cross-encoder rerank (ms-marco-MiniLM, fallback heuristic)
                                ↓
                top_k chunks → context_char_limit (≤5200 chars) → prompt
```

- Persists to `data/rag_store.json` (chunks + embeddings).
- Supports **ephemeral** chunks: per-request attachments not committed to the store unless `persist_attachments=true`.
- `index_documents` returns chunk counts; `delete_document` removes both the document row and its chunks.

### 3.4 `monitoring_layer.py` — Carbon + system metrics

- **Electricity Maps v3** integration for both live carbon-intensity AND 48-hour forecast (at 15-min granularity).
- Authentication picks `Bearer` vs `auth-token` header by token shape.
- Per-zone cache + global cache (`CACHE_DURATION` env, default 60 s) — protects the free-tier 60s rate limit.
- `find_low_carbon_window(forecast, deferral_budget_ms)` finds the lowest-CI point ≤ deferral budget that is also ≤ 25th-percentile of the forecast window.
- `fetch_all_zone_signals()` fetches all configured zones in parallel with a `ThreadPoolExecutor` for multi-region scoring.
- `fetch_system_metrics()` calls the host-metrics sidecar at `127.0.0.1:9000/metrics`. The sidecar invokes `system_metrics.sh` (which runs `nvidia-smi` + `top`) and caches for 60 s.

### 3.5 `deferred_queue.py` — Carbon-aware batching

- Min-heap keyed by `priority_ts` (= the best low-carbon dispatch time within the budget).
- `enqueue()` rejects if the queue is at `MAX_QUEUE_SIZE` (500) — caller must dispatch immediately (back-pressure).
- A daemon thread runs every `DISPATCH_INTERVAL_S` (10 s):
  - Pops items where `now ≥ priority_ts` (low-carbon window arrived) or `now ≥ deadline_ts` (deadline expired).
  - Calls each item's `dispatch_fn(payload)`, optionally invoking `callback(result)`.
- `update_carbon(ci)` triggers an immediate dispatch pass when CI drops below threshold.
- `status()` exposes queue size, pending requests with target-dispatch / deadline timestamps, and counters (used by `/api/queue/status`).

### 3.6 `model_zoo.py` — Versioned registry + MoE placement

- Loads `config/model_zoo.json` with: per-model `flop_count_per_token`, `power_tdp_w`, `pue`, `hardware_efficiency`, `mfg_carbon_kg`, `lifetime_years`, `hardware_affinity`, `num_experts`, `active_experts_k`, `all_to_all_overhead_ratio`, `expert_bandwidth_gbps`, etc.
- `available_targets()` returns whatever the routing engine should consider live.
- `plan_expert_placement(model_id, device_topology, token_routing)` runs **load- and capacity-aware round-robin** assignment of experts to devices:
  1. Filter devices by `hardware_affinity` and remaining `expert_slots`.
  2. Sort experts heaviest-first by token-routing load (uniform fallback).
  3. Greedily place each expert on the device with smallest `load/capacity` ratio.
  4. Compute **skew** = `min(per-device load) / max(per-device load)` and **estimated comm overhead** scaling with cross-device fraction.
- Returns `{placement, device_loads, skew, estimated_comm_ms, fallback}` — when `fallback=True`, the pipeline downgrades to dense and records `moe_fallback_reason`.

### 3.7 `nemo_guardrails.py` — Programmable rails

Pure-pattern programmable rails (no external LLM dependency):

- **Input rail**: ~30 regex blocked-patterns covering jailbreaks (DAN, instruction overrides), violence/CSAM/WMD/CBRN, self-harm promotion, sexually explicit instruction generation. Match → `blocked=True`, with safe replacement message.
- **Sensitive rail**: lighter-weight patterns that produce non-blocking warnings logged in the trace.
- **Output rail**: blocks model echoing of jailbreak strings, leaked credentials/PII (`password=…`, `BEGIN RSA`, `ssh-rsa`, `api_key=…`), and step-by-step violence instructions that may have slipped through.

Every check returns a structured trace with `blocked`, `reason`, `safe_replacement`, `warnings`, `checks`, `latency_ms`, `phase`. Both phases are written to the audit log under `guardrail_trace`.

### 3.8 `conversation_store.py` — SQLite persistence

- WAL mode for crash safety + reader/writer concurrency.
- Two tables: `conversations(id PK, title, created_at, updated_at)` and `messages(id PK, conversation_id FK, role, content, created_at, metadata_json)`.
- `metadata_json` carries the **entire** decision blob (sustainability, routing, retrieval, grounding, RL trace, tokens, GPU) so any message in the UI can be replayed from a single row.
- `ensure_conversation` will auto-rename a placeholder title (`"New chat"`) on the first user message.

### 3.9 `host_metrics_service.py` — Sidecar

A separate FastAPI on port 9000 that wraps `system_metrics.sh`. Exposes:
- `GET /metrics` → JSON snapshot of GPU utilisation, memory, clocks, temperature, power-draw, CPU utilisation/power, derived `system_total_power`, `system_energy`, `system_co2_emission`, plus the `timestamp`.
- `GET /health` → script existence + last-error + cache age.

Isolation lets us pin the heavy `subprocess.run` (which can take >1 s) into its own process so the API container's event loop never blocks.

### 3.10 Audit log — HMAC-signed JSONL (`data/decision_logs.jsonl`)

Every decision becomes one JSON line carrying:

```
timestamp · request_id · conversation_id · user_message_id · assistant_message_id
selected_model · resolved_model_name · user_tier · priority · mode
input_understanding (full semantic profile)
policy_coefficients (with rl_version & rl_exploration flags)
candidate_rankings[:5] (full CSS breakdown per candidate)
selected_candidate (carbon_total, op_g, emb_g, scores, MoE placement)
eco_actions (deferral, reroute, low-carbon window)
rag_retrieved_count · rag_sources[:5]
evidence_assessment · grounding_verification
guardrail_trace.input · guardrail_trace.output
system_metrics · system_power_w · grid_signal · grid_carbon · system_co2_g
actual_latency_ms · accuracy_outcome
rl_policy_version · rl_episode
tokens.{input,output,total,co2_per_token_ug}
gpu_utilization_pct · gpu_power_w · gpu_co2_g · infer_duration_s
_hmac (SHA-256 over the canonical-JSON body)
```

The signing key comes from `AUDIT_HMAC_KEY` (production rejects the placeholder default). `read_audit_log()` supports filters by from/to timestamp, model, tenant, and minimum carbon — these power the `/api/audit` endpoint and the new `/api/observability/summary`.

---

## 4. API surface (30 endpoints)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Main pipeline (form-encoded, multipart for attachments) |
| `POST` | `/decision` | Legacy JSON variant of /api/chat |
| `GET` | `/api/conversations` | List all conversations |
| `GET` | `/api/conversations/{id}` | Fetch one conversation + all messages |
| `DELETE` | `/api/conversations/{id}` | Cascade-delete |
| `GET` | `/api/rag/status` | Index size + backends in use |
| `GET` | `/api/rag/documents` | List indexed documents |
| `POST` | `/api/rag/index` | Index new documents |
| `DELETE` | `/api/rag/documents/{id}` | Remove a document + its chunks |
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | Readiness — waits for sidecar reachable |
| `GET` | `/api/audit` | Filtered audit-log query |
| `GET` | `/api/queue/status` | Deferred queue contents + counters |
| `POST` | `/api/queue/dispatch-now` | Force-dispatch all pending items |
| `GET` | `/api/model-zoo` | Versioned model registry |
| `GET` | `/api/model-zoo/{id}/carbon` | Per-model LLMCarbon breakdown for given duration & tokens |
| `GET` | `/api/model-zoo/{id}/expert-health` | MoE expert availability |
| `POST` | `/api/model-zoo/{id}/expert-health` | Update expert availability |
| `POST` | `/api/model-zoo/reconcile` | Force a placement reconcile pass |
| `GET` | `/api/model-zoo/{id}/expert-placement` | Run the placement planner |
| `GET` | `/api/grid/zones` | Multi-region carbon snapshot |
| `GET` | `/api/grid/forecast` | 48-h forecast for a zone |
| `GET` | `/api/policy/suggest` | Suggested coefficient changes from recent history |
| `GET` | `/api/rl/status` | Per-tier weights, baseline EMA, reward trend |
| `GET` | `/api/rl/history` | Reward time-series per tier |
| `POST` | `/api/rl/reset/{tier}` | Reset one tier's RL state to initial policy |
| `GET` | `/api/system/metrics` | Latest sidecar snapshot |
| `GET` | `/api/observability/summary` | **Datadog-style observability rollup** (see §5) |

---

## 5. Observability subsystem (the new module)

### 5.1 Backend — `GET /api/observability/summary`

Query parameters (all validated by Pydantic):

| Param | Default | Range | Purpose |
|---|---|---|---|
| `window_minutes` | 60 | 1..1440 | Time window for KPIs + traces |
| `bucket_seconds` | 60 | 10..3600 | Time-series bucket size |
| `slo_p95_ms` | 3000 | 50..60000 | P95 latency SLO target |
| `slo_error_rate` | 0.01 | 0..1 | Acceptable grounding-failure rate |
| `energy_price_usd_kwh` | 0.12 | 0..10 | Local electricity price |
| `cloud_input_usd_per_1k` | 0.0015 | ≥0 | Comparator for tokens-in cost |
| `cloud_output_usd_per_1k` | 0.0020 | ≥0 | Comparator for tokens-out cost |

The handler reads **two** equal windows (current + prior) in one audit-log pass, then computes:

#### KPIs (`kpis`)
`total_requests`, `requests_per_min`, `tokens_total/input/output`, `co2_total_g/avg_g`, `css_avg`, `grid_ci_avg`, `latency_avg/p50/p95/p99`, `grounding_failure_rate/_failures`, `deferred_count/_rate`, `rag_used_count/_rate`.

#### Period-over-period deltas (`prior_kpis` + `deltas`)
For each of 14 keyed metrics: `{abs, pct}`. The frontend renders ▲/▼ pills; `inverse=true` keys (latency, CO₂, errors, grid CI) flip the colour so a *decrease* is shown green.

#### SLO + error budget (`slo`)
```
p95_compliance_pct = 100 × #{requests with latency ≤ slo_p95_ms} / total
p95_breach_count   = total − above
error_budget_remaining_pct = max(0, slo_error_rate − actual) / slo_error_rate × 100
status = "breach" if p95_actual > target OR error_actual > target
       | "warn"   if p95_actual > 0.9·target OR burn > 75 %
       | "healthy"
```

#### Latency heatmap (`heatmap`)
2D Datadog-style density grid: 60 time-columns × 11 latency-bins (`50..30000ms` + overflow). Each cell holds the request count; the frontend renders a green→yellow→red SVG gradient by `count / max_count`.

#### Cost & efficiency (`cost`)
```
energy_kwh_total = Σ (gpu_power_w × infer_duration_s) / 3.6e6
energy_usd       = energy_kwh_total × energy_price_usd_kwh

cloud_equivalent_usd = Σ (in_tokens/1000 × $0.0015 + out_tokens/1000 × $0.002)
savings_usd = cloud_equivalent_usd − energy_usd
savings_pct = savings_usd / cloud_equivalent_usd × 100

tokens_per_request = tokens_total / total
energy_kwh_per_1k_tokens = energy_kwh_total / (tokens_total/1000)
co2_per_1k_tokens_g = co2_total_g / (tokens_total/1000)
cost.by_model[]     = same metrics, grouped by selected_model
```

#### Time series (`time_series`)
`bucket_count = window_seconds / bucket_seconds` buckets. Each bucket aggregates `requests`, `latency_ms_avg`, `latency_ms_p95`, `co2_g`, `tokens`, `grid_ci`.

#### Latency histogram (`latency_histogram`)
10 log-spaced bins from `≤50ms` to `≤30000ms`, plus a `>30s` overflow bin.

#### Distributions (`distributions`)
`by_model`, `by_tier`, `by_intent`, `by_priority`, `by_region` — `name → count`.

#### Per-model rollup (`model_rollup`)
Per model: `requests`, `share_pct`, `avg_latency_ms`, `p95_latency_ms`, `total_co2_g`, `total_tokens`.

#### Anomalies (`anomalies`)
Per-window z-score on `actual_latency_ms`. Any request with `(value − mean) / std > 2.5` is reported (top 10 by z-score). Requires ≥10 latency samples to fire.

#### Top conversations (`top_conversations`)
The 10 chattiest conversation IDs by request count, with `tokens`, `co2_g`, `last_ts`.

#### Traces (`traces`)
Newest 50 audit entries, flattened into a single trace row each — every field needed to drive the trace explorer (model, latency, tokens, CO₂, grid CI, CSS score, GPU util, RAG retrieved, deferred, grounding result, accuracy outcome, RL policy version).

### 5.2 Frontend — `ObservabilityPanel.jsx`

Built with no chart library — every visualisation is hand-rendered SVG. Components:

| Component | Role |
|---|---|
| `Card` / `Kpi` / `Pill` / `DeltaPill` / `GaugeBar` | Reusable building blocks (Grommet-themed) |
| `TimeSeriesChart` | SVG line chart with gradient area fill, axis labels, peak callout |
| `Histogram` | Vertical bars for latency bins; overflow bar coloured red |
| `LatencyHeatmap` | 11×60 SVG grid, `hsl(140 → 0)` density gradient, hover tooltips |
| `DistributionList` | Horizontal-bar breakdown per category |
| `SloCard` | Two gauges (P95 vs target; error-budget burned), live status pill |
| `CostCard` | KPI strip (energy $, cloud-equiv $, savings, tokens/req) + per-model rollup table |
| `TraceRow` / `TraceDetail` | Trace-explorer row with expand-to-detail. Detail shows a synthetic span timeline (guardrails-input / RAG / routing / vLLM / guardrails-out / audit-persist) plus all 17 metadata fields |
| `exportTracesAsCsv()` | Builds an in-browser CSV blob with 23 columns and triggers download |

#### Top-bar controls

- **Window presets** (5m / 15m / 1h / 6h / 24h) — change `windowMinutes` + `bucketSeconds`.
- **Auto-refresh** checkbox (20 s polling).
- **Live-tail** toggle: red pulsing dot, polls every 5 s, **flashes new traces green** by diffing trace IDs across snapshots and applying a 4 s `obs-fresh-flash` keyframe animation.
- **Refresh** button.

#### SLO settings card

Side-by-side with `SloCard`, contains two number inputs that round-trip the `slo_p95_ms` / `slo_error_rate` query params to the backend so the user can see the SLO state recompute in real time.

#### Trace explorer

- Search box (matches model, intent, query preview, conversation ID).
- Model filter dropdown (built from `distributions.by_model`).
- Status filter (`ok`, `slow`, `deferred`, `grounding`).
- **CSV export** button — disabled when no rows match.
- Sticky header, scroll-locked body.

---

## 6. Frontend architecture (`frontend/src/Components`)

| Component | Responsibility |
|---|---|
| `App.jsx` | Mounts `<Grommet>` with HPE-green theme |
| `GreenAIChat.jsx` | Top-level shell: 4 tabs (Chat / Carbon / Observability / Models) + chat composer + sidebar polling |
| `ArchitecturePanel.jsx` | Static component diagram + RAG document manager |
| `CarbonDashboard.jsx` | Real-time grid CI, system power, CO₂ trend, regional zones, queue status |
| `RLPanel.jsx` | Per-tier weights, reward history sparkline, baseline EMA, policy-version transitions |
| `ObservabilityPanel.jsx` | This document's §5.2 |
| `lib/api.js` | Centralised `fetch` wrapper; one helper per endpoint (incl. `fetchObservabilitySummary` with all 7 tunables) |

The chat composer triggers `POST /api/chat`. While inference is in flight, the right sidebar polls the lightweight read-only endpoints (`/api/system/metrics`, `/api/grid/zones`, `/api/queue/status`, `/api/rl/status`) so the UI stays live without round-tripping the full pipeline.

---

## 7. Configuration

### 7.1 Files

| File | Contains |
|---|---|
| `.env` | All runtime env vars (see `.env.example`) |
| `config/policies.json` | Per-tier CSS coefficients (initial weights for RL) |
| `config/routing_targets.json` | Active routing candidates with full LLMCarbon parameters |
| `config/model_zoo.json` | Versioned model registry (richer view of routing targets) |
| `guardrails/config.yml` | NemoGuardrails action registry (programmable rails only) |

### 7.2 Key env vars

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_MEDIUM_URL` / `VLLM_FULL_URL` / `VLLM_MOE_URL` / `VLLM_STEM_*_URL` | localhost ports 8001-8006 | Per-variant vLLM endpoints |
| `VLLM_TIMEOUT_SECONDS` | 45 | Hard timeout per vLLM call |
| `EMAP_TOKEN` | (empty) | Electricity Maps API key |
| `EMAP_ZONE` | (none) | Primary zone (e.g. `IN`, `DE`, `US-CAL-CISO`) |
| `MULTI_REGION_ENABLED` | false | Toggle parallel zone fetches |
| `GUARDRAILS_ENABLED` | true | Toggle input + output rails |
| `GPU_TDP` / `GPU_VRAM_GB` | per `.env` | Hardware constants for LLMCarbon |
| `RL_ALPHA_0` | 0.06 | Initial learning rate |
| `RL_REWARD_LAMBDA_*` | 0.35/0.30/0.25/0.10 | Reward composition weights |
| `RL_W_MIN` | 0.05 | Simplex floor |
| `RL_DIRICHLET_EPSILON` | 0.15 | Exploration mix-in |
| `RL_CONVERGENCE_WINDOW` | 50 | Episodes to detect convergence |
| `AUDIT_HMAC_KEY` | (must set in prod) | HMAC signing key (rejected if placeholder) |
| `MAX_CONTEXT_CHARS` | 15000 | RAG context cap pre-prompt |
| `MOE_RECONCILER_ENABLED` | true | Background MoE expert reconciliation |

### 7.3 Data layout

```
data/
  green_ai.db         SQLite (WAL) — conversations + messages
  rag_store.json      Indexed RAG chunks + embeddings
  rl_state.json       Persisted RL weights, episode counts, baselines
  decision_logs.jsonl HMAC-signed audit trail (append-only)
  hf-cache/           Hugging Face model cache
```

---

## 8. Operations

### Local dev

```bash
# Backend
pip install -r requirements.txt
uvicorn decision_engine:app --reload --host 0.0.0.0 --port 8100
python host_metrics_service.py              # second terminal, port 9000

# Frontend (Vite dev server with /api proxy → 8100)
cd frontend
npm install
npm run dev
# open http://localhost:5200
```

### Production stack

```bash
docker compose -f docker-compose.ubuntu-vgpu.yml --env-file .env up --build -d
# open http://localhost:8080  →  proxied by nginx to green-api-1:8100
```

With HTTPS edge:
```bash
docker compose -f docker-compose.ubuntu-vgpu.yml -f docker-compose.https.yml \
  --env-file .env up --build -d
```

Hot-rebuilding only the moving parts after a code change:
```bash
docker compose -f docker-compose.ubuntu-vgpu.yml --env-file .env up -d \
  --build api frontend
```

### Health checks

```bash
curl http://localhost:8100/health/ready   # API readiness (waits for sidecar)
curl http://localhost:9000/health         # Sidecar
curl http://localhost:8001/health         # vLLM medium
curl http://localhost:8002/health         # vLLM full
```

### Audit log forensics

```bash
# Latest decision (signed)
tail -1 data/decision_logs.jsonl | python -m json.tool

# Filter by model + carbon threshold via API
curl "http://localhost:8100/api/audit?model=full&min_carbon_g=0.05&limit=20"
```

### RL safety valve

```bash
# Reset one tier back to initial policy
curl -X POST http://localhost:8100/api/rl/reset/standard
```

---

## 9. End-to-end example (annotated)

A premium-tier user sends `"Compare quicksort vs mergesort and recommend one for low-carbon batch jobs"` (no attachments) at 14:32 UTC, when grid CI is **480 gCO₂/kWh** (high).

1. **Profiler**: SBERT classifies → intent=`analysis`, complexity=0.78, recommended=`full`, priority=`high`, accuracy_floor=0.88, sla_ms=180.
2. **RL policy** for `premium`: `(carbon=0.20, latency=0.34, accuracy=0.34, cost=0.12)`, ε-mixed with Dirichlet noise.
3. **Candidates ranked**: with CI=480 the heavy variants take a bigger carbon hit — `local-vgpu-medium` wins by CSS=0.79, `local-vgpu-full` second at 0.74.
4. **GPU re-rank**: GPU util at 62 % → no demotion.
5. **EcoServe**: `deferral_recommended=False` (priority=`high` → tolerance=300 s, but the request can be served now and the question is interactive, not batch).
6. **MoE**: not selected, skipped.
7. **RAG**: no documents in store relevant → `rag_retrieved_count=0`.
8. **Input guardrails**: clean.
9. **Inference** on `vllm-medium` → 1 050 ms wall-clock.
10. **Output guardrails**: clean. Grounding skipped (not a grounded request).
11. **Audit row** signed and appended; `system_co2_g=0.0382`, `tokens={input:42, output:218, total:260}`.
12. **RL update** (background): `R=0.81`, advantage `+0.06`, weights drift toward higher `accuracy` and lower `cost` for premium.

The Observability tab will show this request in:
- KPIs (request count +1, P95 nudge up if it was a large value),
- Time-series request-rate bucket,
- `by_model.medium` + 1, `by_intent.analysis` + 1, `by_tier.premium` + 1,
- Latency heatmap cell at the current 1-min bucket × `≤1500ms` row +1,
- Per-model rollup row updates,
- Trace explorer (newest row, status `ok`).

If the same user sends 20 more requests and grid CI rises to 540, EcoServe will start enqueuing batchable ones; the SLO card will show error budget burned if more than 1 % of requests trigger grounding failures; the cost card will report the running self-host energy bill vs the cloud-equivalent.

---

## 10. Why the design holds together

- **Single source of truth**: every observable metric is derived from `data/decision_logs.jsonl`. Even `/api/observability/summary` doesn't keep its own state — it scans the audit log per-request. There is no metrics drift between dashboards.
- **HMAC integrity**: every audit row is signed by a server-only key. Any post-hoc tampering invalidates the line. The audit query API surfaces all the routing reasoning per decision.
- **Closed-loop learning**: every chat improves the next chat. RL weights, RAG corpus, and MoE expert health all adapt online without any manual training step.
- **Graceful degradation**: every external dependency has a fallback (`SentenceTransformer` → hashed embeddings; `Electricity Maps` → cached + 475 g default; `nvidia-smi` → estimated GPU util; cross-encoder → heuristic; vLLM → rule-based extractive).
- **Carbon as a first-class objective**: not just measured but **acted on** — `grid_carbon` participates in CSS, SLA penalties, MoE go/no-go, deferral decisions, and the RL reward.
- **Observability for an LLM stack, not just an HTTP service**: token analytics, per-model carbon, grounding failures, RL policy versions, MoE fallback reasons, and the deferred-queue backlog are all cross-correlated in the Observability tab — every signal a Datadog or Elastic-style review would want, plus the green-ops signals nobody else surfaces.

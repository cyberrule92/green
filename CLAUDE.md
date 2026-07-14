# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (Python / FastAPI)

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server (port 8100)
uvicorn decision_engine:app --reload --host 0.0.0.0 --port 8100

# Run host metrics sidecar (port 9000, separate terminal)
python host_metrics_service.py
```

### Frontend (React / Vite)

```bash
cd frontend
npm install
npm run dev      # dev server on port 5200, proxies /api/* → http://127.0.0.1:8100
npm run build    # production build → ./dist
npm run lint     # ESLint
```

### Docker (production)

```bash
# Default vLLM stack
docker compose -f docker-compose.ubuntu-vgpu.yml --env-file .env up --build -d

# With HTTPS (Caddy edge)
docker compose -f docker-compose.ubuntu-vgpu.yml -f docker-compose.https.yml --env-file .env up --build -d
```

### Health checks

```bash
curl http://localhost:8100/health/ready  # API (waits for metrics sidecar)
curl http://localhost:9000/health        # metrics sidecar
curl http://localhost:8001/health        # vLLM medium
curl http://localhost:8002/health        # vLLM full
```

There are no automated tests. All configuration comes from `.env` (see `.env.example` for all options).

---

## Architecture

### Request lifecycle (`/api/chat`)

The decision engine is the central control plane. Every chat request follows this pipeline:

1. **Guardrails** (`nemo_guardrails.py`) — input safety check (jailbreaks, injection, harmful content). Toggle via `GUARDRAILS_ENABLED`.
2. **RAG** (`advanced_rag.py`) — hybrid dense (sentence-transformers) + sparse retrieval, then cross-encoder reranking, if the knowledge base has indexed documents.
3. **Prompt profiling** (`routing_policies.py`) — semantic classification to estimate accuracy/latency requirements, and **modality** detection (`text` / `vision` image-attachment / `image-gen` request). CSS filters candidates to the request's modality axis.
4. **CSS routing** (`routing_policies.py`) — Composite Sustainability Score ranks the four candidates (small/medium/full/cpu-fallback) against carbon, latency, accuracy, and cost dimensions; RL-learned weights per tenant tier are applied. The per-candidate accuracy/latency inputs are refined per-prompt by a learned estimator (`quality_latency_estimator.py`); carbon is not adjusted.
5. **EcoServe deferral** (`deferred_queue.py`) — if grid carbon exceeds threshold, request is queued and dispatched during the next low-carbon window (up to 48-hour forecast horizon).
6. **Dispatch** — text → selected vLLM container endpoint; `vision` → VLM (`multimodal.run_vlm_inference`); `image-gen` → diffusion (`multimodal.run_image_generation`, carbon-capped steps). Image endpoints are pluggable NIM URLs with graceful fallback.
7. **Output guardrails** — response safety check.
8. **Audit logging** — HMAC-signed entry written to `data/decision_logs.jsonl`.
9. **RL reward** (`rl_controller.py`) — observed outcome updates per-tier policy weights via online REINFORCE + EMA baseline.
10. **Persistence** (`conversation_store.py`) — message saved to SQLite (`data/green_ai.db`).

### Component responsibilities

| File | Role |
|---|---|
| `decision_engine.py` | FastAPI app; 26 endpoints; orchestrates all of the above |
| `routing_policies.py` | CSS scoring, tier policies, semantic prompt profiler, MoE FLOP accounting, multi-region reroute |
| `advanced_rag.py` | Hybrid retrieval, BM25-like sparse fallback, cross-encoder reranking, chunked JSON store |
| `rl_controller.py` | Online REINFORCE; persists learned weights to `data/rl_state.json` |
| `quality_latency_estimator.py` | Online-learned per-prompt accuracy/latency correction feeding CSS's accuracy/latency scores (carbon untouched); cold-start = identity; persists to `data/ql_estimator_state.json` |
| `multimodal.py` | Image analysis (VLM) + generation (diffusion) dispatch via pluggable NVIDIA NIM endpoints; graceful fallback (SVG placeholder / metadata description) when no endpoint; dependency-free |
| `monitoring_layer.py` | Electricity Maps API (real-time + 48 h forecast), GPU/CPU metrics via sidecar |
| `model_zoo.py` | Versioned model registry backed by `config/model_zoo.json`; LLMCarbon operational + embodied carbon |
| `coding_agent.py` | Agentic coding harness (LangGraph, **no LangChain**). Off the CSS path: CSS scores carbon per *request*, but an agent is a loop (tokens × steps × attempts), so it optimises **carbon per successful completion**. Ladder starts at the greenest *code-capable* model (Qwen2.5-Coder-1.5B) — never TinyLlama/DialoGPT, which can't finish a coding task and only burn the step budget — and escalates to the 30B MoE only on verifier evidence. Sandboxed workspace, per-task gCO2 budget. Above `AGENT_DEFER_CI` the task is **queued** on `deferred_queue` (the system's first real `enqueue` caller — chat's deferral is advisory only) and runs in the next low-carbon window; `POST /api/agent/task` returns `queued`, `GET /api/agent/task/{id}` follows it to completion. The **tests are the spec and are frozen** once written (an unfrozen verifier gets reward-hacked), so `invalid_test_reason` gates what may become one: it must parse, contain `test_*` functions, import what it tests, and not repeat itself. Model calls use `AGENT_LLM_TIMEOUT_S` (180 s), *not* the 45 s chat timeout. An empty backend response is infrastructure, not evidence: it aborts rather than escalating. **Who authors the spec decides the carbon:** pass `tests` (a pytest source string, or `{path: source}`) to `POST /api/agent/task` and the caller's suite becomes the frozen spec — validated at submit (400 before any carbon is spent), written to the workspace at step 0, and the model may not emit a test file at all (`spec_locked`). Left unset, the *weakest* rung authors the spec it is then judged against, and a well-formed but wrong one is unsatisfiable by design: measured on the same fizzbuzz task, a model-authored spec asserting `fizzbuzz(0) == "0"` (0 is divisible by 3 and 5, so it must be `"FizzBuzz"`) escalated and wasted **2.98 gCO2 failing**, while the caller-supplied spec **completed on rung 1 in one call for 0.028 g**. `spec_source` (`caller`/`model`) is on every result and audit entry, because "it passed your tests" and "it passed its own" are different claims |
| `deferred_queue.py` | Priority queue; holds requests during high-carbon windows; max 500 items |
| `conversation_store.py` | SQLite WAL; conversations + messages schema |
| `nemo_guardrails.py` | Action-based safety rails (no external LLM dependency) |
| `host_metrics_service.py` | Separate FastAPI on port 9000; calls `system_metrics.sh` → nvidia-smi / top |

### Configuration files

- `config/policies.json` — per-tier CSS weight coefficients (standard / premium / esg / batch)
- `config/routing_targets.json` — four routing candidates with baseline accuracy, latency, and power figures
- `config/model_zoo.json` — LLMCarbon parameters (FLOPs, HE, PUE, mfg carbon) for each registered model
- `guardrails/config.yml` — NemoGuardrails action mapping (no ColBERT / external LLM)

### Frontend

React 19 + Grommet (HPE design system). `GreenAIChat.jsx` is the primary component: chat panel plus real-time sidebars that poll backend endpoints for grid carbon, RL weights, queue status, and system metrics. `src/lib/api.js` centralises all fetch calls.

### Data layout

```
data/
  green_ai.db          SQLite — conversations, messages
  rag_store.json       RAG chunks + embeddings (persisted)
  rl_state.json        RL policy weights per tier (persisted)
  ql_estimator_state.json  Learned per-variant accuracy/latency weights feeding CSS (persisted)
  decision_logs.jsonl  HMAC-signed audit trail
  hf-cache/            Hugging Face model cache
```

### Key environment variables

| Variable | Purpose |
|---|---|
| `TRITON_MEDIUM_MODEL`, `TRITON_FULL_MODEL` | vLLM endpoint URLs |
| `EMAP_TOKEN`, `EMAP_ZONE` | Electricity Maps API credentials + primary grid zone |
| `GPU_TDP`, `GPU_VRAM_GB` | Hardware spec for LLMCarbon calculations |
| `RL_ALPHA_0`, `RL_REWARD_LAMBDA_*` | RL learning rate and reward component weights |
| `QL_ESTIMATOR_ENABLED`, `QL_ESTIMATOR_LR`, `QL_ESTIMATOR_WARMUP` | Learned quality/latency estimator toggle, learning rate, warm-up threshold (cold-start = identity) |
| `NIM_VLM_URL`, `NIM_SDXL_URL`, `NIM_FLUX_URL` | NVIDIA NIM endpoints for image analysis (VLM) + generation (diffusion); unset → graceful placeholder fallback |
| `DIFFUSION_HIGH_CARBON_CI` | Grid carbon threshold above which image generation trims its denoising-step budget |
| `GUARDRAILS_ENABLED` | Toggle NemoGuardrails |
| `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL` | Sentence-transformer model IDs |

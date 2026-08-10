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

# With quantization + fine-tuning (needs the Docker socket — root-equivalent)
docker compose -f docker-compose.ubuntu-vgpu.yml -f docker-compose.onboarding.yml \
               -f docker-compose.finetune.yml --env-file .env up --build -d
```

### GPU runner images

Quantization and training run in their own containers, built from
`vllm/vllm-openai:latest`. That base is deliberate: it is already on this host,
and **its torch is the one proven to open a CUDA context against this 12.8-era
vGPU driver** (verified with `torch.zeros(8, device="cuda")`), unlike the API
image's CUDA-13 build. `peft`/`trl`/`datasets` and `autoawq` all resolve on top
without pulling a second torch; `bitsandbytes` 0.49 and `accelerate` ship in the
base already and 4-bit quantize was verified working on this GPU. Compose does
not manage them — they are one-shot jobs, not services:

```bash
docker build -f docker/quantize/Dockerfile -t green-quantize:latest .
docker build -f docker/finetune/Dockerfile -t green-finetune:latest .
```

Both Dockerfiles **clear `ENTRYPOINT`**. The base image's entrypoint starts the
vLLM API server, and both runners are launched with `Cmd=["sh","-lc", …]` and no
entrypoint override — left in place, every job silently starts a web server and
times out instead of doing the work.

### Health checks

```bash
curl http://localhost:8100/health/ready  # API (waits for metrics sidecar)
curl http://localhost:9000/health        # metrics sidecar
```

**Do not trust `/health` on a vLLM container.** It is answered by the API server
process and stays 200 while the EngineCore behind it is wedged — on 2026-07-29 all
three containers here reported healthy for ~21 hours while every completion hung
until the client timed out. The compose healthchecks therefore run
`scripts/vllm_healthcheck.py <port>`, which generates one token (falling back to
`/v1/chat/completions` on a 400 for chat-only models). To check by hand:

```bash
docker exec green-vllm-full-1 python3 /healthcheck.py 8002 && echo generating
```

### vGPU licensing — the failure mode to check first

This host is an H100L-2-24C **vGPU** running NVIDIA Virtual Compute Server. When
the licence lapses the driver keeps *existing* CUDA contexts alive but refuses
**new** ones, so already-running containers look fine while anything restarted
dies with `CUDA error: operation not supported` during engine init. A bare
`torch.zeros(8, device="cuda")` in the vLLM image reproduces it in seconds.

Before restarting or recreating any vLLM container, check:

```bash
nvidia-smi -q | grep -A2 "vGPU Software Licensed Product"   # want: Licensed
journalctl -u nvidia-gridd -n 20 --no-pager
```

If it reports `Unlicensed`, restarting a container **destroys the only thing
keeping it working**. Observed remedy for a corrupted trusted store (symptom:
`Failed to update local trusted store - Maximum buffer size exceeded`): stop
`nvidia-gridd`, clear `/var/lib/nvidia/vGPULicensing/*`, start it again so it
re-registers from the token in `/etc/nvidia/ClientConfigToken/`.

### Tests

```bash
python3 -m pytest          # whole suite (config in pyproject.toml)
```

`tests/` covers the carbon arithmetic (golden values against
`tests/fixtures/model_zoo_min.json`, plus invariants and bounds on the shipped
zoo) and the RL weight projection. It deliberately does **not** import
`decision_engine` — that needs fastapi and the full backend stack — so tests
target the dependency-light modules directly and construct isolated instances
via explicit paths rather than patching import-time constants.

All configuration comes from `.env` (see `.env.example` for all options).

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
| `decision_engine.py` | FastAPI app; 71 endpoints; orchestrates all of the above |
| `routing_policies.py` | CSS scoring, tier policies, semantic prompt profiler, MoE FLOP accounting, multi-region reroute |
| `advanced_rag.py` | Hybrid retrieval, BM25-like sparse fallback, cross-encoder reranking, chunked JSON store |
| `rl_controller.py` | Online REINFORCE; persists learned weights to `data/rl_state.json` |
| `quality_latency_estimator.py` | Online-learned per-prompt accuracy/latency correction feeding CSS's accuracy/latency scores (carbon untouched); cold-start = identity; persists to `data/ql_estimator_state.json` |
| `multimodal.py` | Image analysis (VLM) + generation (diffusion) dispatch via pluggable NVIDIA NIM endpoints; graceful fallback (SVG placeholder / metadata description) when no endpoint; dependency-free |
| `monitoring_layer.py` | Electricity Maps API (real-time + 48 h forecast), GPU/CPU metrics via sidecar |
| `model_zoo.py` | Versioned model registry backed by `config/model_zoo.json`; operational + embodied carbon. **Operational carbon is `TDP × duration × PUE / HE_eff × CI`** — power × time against a *measured* duration. FLOP counts and token counts are reporting metadata and do **not** affect any carbon number (using both would double-count the same work); the LLMCarbon FLOP form is the alternative for when no measured duration exists, which is never the case here. Two consequences: equal-duration responses cost the same regardless of length, and the MoE penalty is carried by `all_to_all_overhead_ratio` via `HE_eff`, not by the sparse FLOP count. `compute_request_carbon` (ex-post, the source of truth for what a request cost) and `compute_operational_carbon` (ex-ante, used for CSS ranking) share the identical energy term and must not diverge |
| `model_onboarding.py` | Browse Hugging Face → size a quantization plan → download → serve → register, as one pipeline (`/api/models`, "Models" tab). Exists because a measured negative result is a *menu* problem, not a policy one: `ultra-light`/`small`/`medium`/`local-cpu-fallback` all dispatch to the same TinyLlama container with hand-written differing TDPs and accuracies, and TinyLlama is both more verbose (108 vs 94 tokens) and less accurate than Qwen2.5-1.5B — so the "cheap" rung costs more for worse answers. Quantization builds *real* rungs. Dependency-light: no `decision_engine` import, no new dependency (Docker control is stdlib HTTP over the unix socket; grid CI is an injected callable). Three rules are load-bearing. **(1) Measurement gates availability.** A new model registers `available: false` / `accuracy_basis: "unmeasured"`, and `routing_policies` filters candidates on `available`, so CSS cannot select it. Only `apply_measurement` (posted by the caller, against a live endpoint) flips it, stamping a `basis` — the shipped zoo declares `full` at 0.92 and it measured 0.793 in practice, and onboarding must not automate that gap. Device-level fields (TDP, embodied carbon, lifetime, `device_share`) are *inherited* from a same-hardware donor and labelled `device_fields_basis`, because they genuinely describe the same board; model-level fields are never guessed. **(2) VRAM is the binding constraint, and MIG makes the obvious query wrong** — `--query-gpu=memory.total` reports the whole board (24576 MiB) while the usable instance is 21547 MiB, so the probe parses the MIG table and records `vram_basis`. **(3) Prefer a pre-quantized checkpoint.** Plan order: upstream AWQ/GPTQ → fp16 when it fits with slack → vLLM in-flight `bitsandbytes` (load-time, so it costs throughput not carbon) → local AWQ pass, opt-in only, run in its own container, metered from measured wall-clock into `quantization_carbon_g`, with `estimate_payback` turning that into requests-to-break-even *after* both rungs are measured. Every refusal carries `rejected[]` explaining what lost and why. Routability rides on `vllm_endpoint_env`, which `resolve_vllm_endpoint` re-reads via `os.getenv` at lookup time; the URL is persisted and re-exported at startup by `restore_endpoints`, since an in-process env var does not survive a restart. Readiness is gated on a **real one-token completion, not `/health`** — on 2026-07-29 all three vLLM containers here returned 200 from `/health` for hours while their EngineCore processes were wedged and every completion hung. Dynamic serving needs the Docker socket bind-mounted (`docker-compose.onboarding.yml`, opt-in: it is root-equivalent on the host); `trust_remote_code` is arbitrary code execution and needs its own separate opt-in. **(4) Registration is optional.** `POST /api/models/quantize` (`register=False`) runs the same pipeline and stops after the pass: the checkpoint is produced and downloadable from `/api/models/artifacts/{id}/download`, but nothing enters the zoo, because quantizing a model and adding a rung to *this* deployment's ladder are separate decisions — forcing the first to imply the second leaves unavailable entries behind for models nobody here intends to serve. `auto_serve` with `register=False` is refused rather than silently ignored. Downloads stream as an **uncompressed** tar (4-bit safetensors do not compress, and staging a copy would need the disk twice on a box at 65% full); an artifact whose directory has no `config.json` or no shards is the residue of a failed pass and is flagged `complete: false` rather than handed out. **Path discipline:** the runner and the vLLM container both see the shared cache at `CONTAINER_HF_CACHE` (`/root/.cache/huggingface`) while the API sees the same bytes at its own `HF_HOME`. Anything handed to a launched container must use the container path — the quantized `out_dir` was computed from `self.cache_dir` before 2026-07-30, so the pass wrote into its own throwaway filesystem and the serve step then pointed vLLM at nothing. `artifact_local_path` is the only place that conversion happens |
| `secret_box.py` | Stdlib authenticated encryption (HMAC-SHA256 keystream + encrypt-then-MAC) for secrets at rest — currently `HF_TOKEN_ENC`. Extracted from the removed workflow engine; the `wf-secret-v1` domain tag is unchanged so previously encrypted values still decrypt. Key from `SECRET_KEY`, falling back to legacy `WF_SECRET_KEY`, then the audit key |
| `finetuning.py` | Carbon-aware **LoRA/QLoRA fine-tuning** (`/api/finetune`). The counterpart to `model_onboarding`: onboarding imports a better rung, this makes the *existing* small one good enough at real traffic that CSS stops escalating. Training data is up-voted pairs from `/api/feedback/export` — down-votes are discarded because supervised tuning cannot use them (that needs DPO), and the rule-based degradation notice is filtered out so the model never learns to apologise for being unavailable. Five load-bearing rules. **(1) PEFT only** — a full fine-tune does not fit a shared 21.5 GB MIG slice beside live inference and costs orders of magnitude more energy for what a rank-16 adapter captures. **(2) Training defers, for real.** Nobody waits on it, so above `FINETUNE_DEFER_CI` the job waits for the cleanest window in the 48 h forecast; deferral is refused when no forecast exists, because that is an open-ended wait rather than a saving. **(3) Carbon is integrated, not snapshotted** — CI is sampled every `FINETUNE_CI_SAMPLE_S` and energy×CI accumulated per interval, so a job that waited for a clean window is credited for the hours it actually ran in. **(4) The adapter is not routable until measured** (`available: false`, `accuracy_basis: unmeasured`); it deliberately does *not* inherit the base's accuracy, since the entire point is that quality changed. `apply_measurement` records `training_payback` — a fine-tune that never pays back is one that should not have run. **(5) Runs in its own container** (`MODEL_FINETUNE_IMAGE`): the API image's torch targets CUDA 13 against a 12.8 driver and cannot init CUDA, and peft/trl/datasets are absent. Serving uses vLLM's `--enable-lora`/`--lora-modules` (verified on 0.21), so an adapter costs a few MB on top of a loaded base — expensive to make, nearly free to keep. `MIN_TRAINING_SAMPLES=200` floors the dataset: below it a run burns GPU-hours to overfit |
| `scripts/finetune_train.py`, `scripts/quantize_awq.py` | The two GPU runners, invoked as `MODEL_FINETUNE_COMMAND` / `MODEL_QUANT_COMMAND` inside the images built from `docker/*/Dockerfile`. Heavy imports (torch, peft, awq) are **lazy, inside `main()`**, so the parsing and verification functions stay importable for `tests/test_runner_scripts.py` without a GPU or a toolchain. Neither script computes a carbon figure: the caller times the container against sampled grid intensity, and a second number here would compete with it as a source of truth. **Trainer:** writes an adapter only, never a merged model — a merged 1.5B checkpoint is 3 GB on disk and a second full set of weights in VRAM, which is the whole carbon argument gone. TRL prompt-completion format masks the prompt out of the loss (the task is producing the response, not predicting user text). A held-out split is always evaluated *and* the base model is evaluated on it first, so `holdout_loss_delta` in `training_manifest.json` says whether the adapter got better or worse on data it never saw — a falling training loss on a few hundred rows is what overfitting looks like. `supported_kwargs` filters `SFTConfig` kwargs by signature because TRL renames fields between releases (`max_seq_length` → `max_length`, gone in 1.9.2) and crashing three hours into a metered job is worse than dropping a kwarg. **Quantizer:** `--calib-file` takes the same JSONL the trainer consumes, because AWQ fits its scales to whatever it calibrates on — the Pile fallback makes the model generically good, this deployment's own traffic makes it good *here*, and which was used is recorded in the manifest. `verify_output` re-reads `config.json` and fails if `quant_method != "awq"`: an exit-0 pass that wrote a full-precision directory spends the calibration carbon and buys nothing |
| `deferred_queue.py` | Priority queue; holds requests during high-carbon windows; max 500 items |
| `conversation_store.py` | SQLite WAL; conversations + messages schema |
| `nemo_guardrails.py` | Action-based safety rails (no external LLM dependency) |
| `host_metrics_service.py` | Separate FastAPI on port 9000; calls `system_metrics.sh` → nvidia-smi / top |

### Configuration files

- `config/policies.json` — per-tier CSS weight coefficients (standard / premium / esg / batch)
- `config/routing_targets.json` — **not present in the repo**, and its absence does *not* fall through to the module defaults. `load_routing_targets` (`routing_policies.py`) walks three tiers: the given path, then **`model_zoo.json` in the same directory**, then `DEFAULT_ROUTING_TARGETS`. Since `config/model_zoo.json` exists, tier two is what actually runs — the live candidate set is its 14-entry `models` list. `DEFAULT_ROUTING_TARGETS` only fires if the zoo is missing or unparseable too, so **editing it will not change routing on a working checkout**. Override the whole chain with `ROUTING_TARGETS_PATH`
- `config/model_zoo.json` — the routing candidate list *and* the LLMCarbon parameters (FLOPs, HE, PUE, mfg carbon) for each registered model, in one file, which is why the candidates CSS ranks and the carbon model pricing them cannot drift apart. Routing reads accuracy/latency/power via the `accuracy_baseline` / `latency_ms_p50` / `power_tdp_w` keys (`rank_routing_candidates` accepts either those or the flat `accuracy` / `latency_ms` / `power_w` names `DEFAULT_ROUTING_TARGETS` uses). Candidates are filtered on modality, on `available`, and on the request's `accuracy_floor` before scoring; that floor filter is permissive on empty — if nothing clears it, the unconstrained set is scored instead, which is the only case where the `-0.18` accuracy penalty is not redundant with the filter
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
  model_onboarding.json    HF onboarding job history (mid-flight jobs load as `failed`/interrupted)
  finetuning.json      LoRA/QLoRA job history + measured training carbon
  decision_logs.jsonl  HMAC-signed audit trail
  hf-cache/            Hugging Face model cache
    quantized/<id>/    Locally quantized checkpoints. Downloadable whole; may or may not be in the zoo
  finetune/<job>/      train.jsonl + adapter/. Bind-mounted into the trainer at the *same* path
```

### Key environment variables

| Variable | Purpose |
|---|---|
| `VLLM_MEDIUM_URL`, `VLLM_FULL_URL` | The two required vLLM endpoints. `ultra-light` and `medium` both dispatch to `VLLM_MEDIUM_URL` — that shared backend is the measured negative result `model_onboarding.py` exists to fix |
| `VLLM_MOE_URL`, `VLLM_STEM_MATH_URL`, `VLLM_STEM_SCIENCE_URL`, `VLLM_STEM_CODING_URL`, `VLLM_FALLBACK_URL` | Optional dedicated endpoints; each **defaults to `VLLM_FULL_URL`**, so a single-container deploy still routes. A STEM steer that looks like it did nothing is usually this default, not the steer |
| `VLLM_TIMEOUT_SECONDS` | Per-request inference timeout (default 45 s), tuned for chat where a user is waiting |
| ~~`TRITON_*`~~ | **Dead — removed from `.env.example` 2026-08-10.** Triton Inference Server is not used anywhere; serving is vLLM's OpenAI-compatible server end to end. Older `.env` files may still carry `TRITON_TIMEOUT_SECONDS`, `TRITON_READY_WAIT_SECONDS`, `TRITON_{ULTRA_LIGHT,MEDIUM,FULL,MOE}_MODEL` — **no Python ever read them**, and they held model *names*, never URLs. Don't reintroduce them when reconciling an old env file |
| `EMAP_TOKEN`, `EMAP_ZONE` | Electricity Maps API credentials + primary grid zone |
| `GPU_TDP`, `GPU_VRAM_GB` | Hardware spec for LLMCarbon calculations |
| `RL_ALPHA_0`, `RL_REWARD_LAMBDA_*` | RL learning rate and reward component weights |
| `QL_ESTIMATOR_ENABLED`, `QL_ESTIMATOR_LR`, `QL_ESTIMATOR_WARMUP` | Learned quality/latency estimator toggle, learning rate, warm-up threshold (cold-start = identity) |
| `NIM_VLM_URL`, `NIM_SDXL_URL`, `NIM_FLUX_URL` | NVIDIA NIM endpoints for image analysis (VLM) + generation (diffusion); unset → graceful placeholder fallback |
| `DIFFUSION_HIGH_CARBON_CI` | Grid carbon threshold above which image generation trims its denoising-step budget |
| `MODEL_ONBOARD_ENABLED` | Master switch for HF onboarding. Browse/preview work while false; nothing downloads, serves or registers |
| `HF_TOKEN_ENC`, `MODEL_ONBOARD_VRAM_RESERVE_MB`, `MODEL_ONBOARD_DISK_RESERVE_GB` | Encrypted HF token (secret box, preferred over plaintext `HF_TOKEN`) and the headroom the planner must leave free |
| `MODEL_SERVE_IMAGE`, `MODEL_SERVE_NETWORK`, `MODEL_SERVE_PORT_BASE`, `HF_CACHE_HOST_PATH`, `DOCKER_SOCKET` | Dynamic serving of onboarded models; needs `docker-compose.onboarding.yml` to mount the socket |
| `MODEL_QUANT_IMAGE`, `MODEL_QUANT_COMMAND` | Opt-in local AWQ pass. Defaults to `green-quantize:latest` running `scripts/quantize_awq.py` (build it first). Unset → capability reports unavailable rather than pretending to work |
| `QUANT_CALIB_FILE` | Calibration corpus for the AWQ pass. Point it at a `/api/feedback/export` JSONL to fit the scales on this deployment's traffic; unset falls back to the generic Pile slice, and the manifest records which |
| `FINETUNE_ENABLED`, `MODEL_FINETUNE_IMAGE`, `MODEL_FINETUNE_COMMAND` | Master switch and trainer runner. Defaults to `green-finetune:latest` running `scripts/finetune_train.py` (build it first); needs `docker-compose.finetune.yml` on top of the onboarding overlay |
| `FINETUNE_DEFER_CI`, `FINETUNE_LOOKAHEAD_H`, `FINETUNE_MAX_WAIT_S`, `FINETUNE_CI_SAMPLE_S` | When a training job waits for a clean window, how far ahead it looks, when it gives up waiting, and how often CI is sampled for integrated carbon |
| `FINETUNE_WORK_DIR`, `FINETUNE_WORK_DIR_HOST` | Dataset + adapter location as the API sees it and as the Docker daemon sees it. They must be the same directory, or the trainer is handed a dataset path that does not exist |
| `MODEL_ONBOARD_ALLOW_TRUST_REMOTE_CODE` | Second gate on `trust_remote_code` (arbitrary code execution from the model repo) |
| `GUARDRAILS_ENABLED` | Toggle NemoGuardrails |
| `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL` | Sentence-transformer model IDs |

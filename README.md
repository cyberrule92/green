# Adaptive Green AI

This project now implements the architecture from the attached Adaptive Green AI design in a practical form:

- Persistent chat conversations and audit logs
- Hybrid RAG with chunking, metadata enrichment, dense + sparse retrieval, reranking, and context fusion
- Carbon-aware routing that ranks model, hardware, and region candidates with tenant-tier policy coefficients
- HPE-themed frontend with an embedded Advanced RAG architecture view inspired by the uploaded diagram
- Ubuntu vGPU deployment path around Triton, the API, the metrics sidecar, and the frontend
- Optional HTTPS edge for real browser-safe production access behind a DNS hostname
- Model onboarding: browse Hugging Face, size a quantization plan against the VRAM actually free, download, serve and register new routing candidates — which stay unavailable to the router until measured
- Carbon-aware fine-tuning: LoRA/QLoRA runs scheduled into the cleanest window in the 48-hour grid forecast, metered against sampled carbon intensity, with a requests-to-break-even payback figure
- Quantize without onboarding: run an AWQ pass and download the resulting checkpoint as a tarball, with no model-zoo entry and nothing exposed to the router

## Main services

- `decision_engine.py`: FastAPI control plane for chat, routing, RAG, and audit logging
- `advanced_rag.py`: document indexing, hybrid retrieval, reranking, and context fusion
- `routing_policies.py`: tier policies, candidate ranking, and EcoServe-style routing signals
- `conversation_store.py`: persistent conversation storage in SQLite
- `model_onboarding.py`: Hugging Face browse/quantize/serve/register pipeline behind `/api/models`
- `finetuning.py`: carbon-scheduled LoRA/QLoRA fine-tuning from collected feedback, behind `/api/finetune`
- `frontend/`: HPE chat UI with architecture and knowledge-base panels

## Ubuntu vGPU deployment

Assumptions:

- Ubuntu VM already has NVIDIA drivers and Docker GPU runtime available
- Triton-compatible model repository is present in `model_repository/`
- Required binaries and network access for model pulls are already available on the VM
- The compose stack grants GPU visibility only to Triton, persists Hugging Face model caches under `./data/hf-cache`, and keeps the metrics sidecar CPU-safe so CDI/runtime hiccups there do not block the app

### 1. Prepare env

Copy `.env.example` to `.env` and set `EMAP_TOKEN` if you want live grid-carbon signals.

### 2. Start the stack

```bash
docker compose -f docker-compose.ubuntu-vgpu.yml --env-file .env up --build -d
```

### 2b. Quantization and fine-tuning (opt-in)

Both run in their own GPU containers — the API image cannot do either, since its
torch targets CUDA 13 against this host's 12.8 driver. Build the two runner
images first; each derives from `vllm/vllm-openai:latest`, which is already
pulled, so they add roughly a gigabyte rather than a fresh multi-GB torch:

```bash
docker build -f docker/quantize/Dockerfile -t green-quantize:latest .
docker build -f docker/finetune/Dockerfile -t green-finetune:latest .
```

Then bring the stack up with both overlays. The onboarding overlay is required
for either — both need the Docker socket to launch their runner, which is
root-equivalent on the host, hence the separate opt-in file:

```bash
docker compose -f docker-compose.ubuntu-vgpu.yml \
               -f docker-compose.onboarding.yml \
               -f docker-compose.finetune.yml \
               --env-file .env up --build -d
```

Both surface on the **Models** tab: quantize a model and download the checkpoint,
or train a LoRA adapter on collected up-votes and watch it wait for a clean grid
window. Neither result is routable until measured figures are posted for it.

### 3. Endpoints

- Frontend: `http://localhost:8080`
- API health: `http://localhost:8100/health`
- API readiness: `http://localhost:8100/health/ready`
- Triton health: `http://localhost:8000/v2/health/ready`
- Metrics health: `http://localhost:9000/health`

The frontend container now proxies `/api/*` to the backend container, so remote browsers can use the UI without rebuilding it against the VM IP. If you open the app at `http://<vm-ip>:8080`, chat requests stay on that same origin and are forwarded internally to the API service.

The stack now waits for Triton and green-metrics to become healthy before declaring the API ready, which avoids the earlier startup race where the frontend could load before the model server had finished pulling and initializing models.

### 4. HTTPS for production

If you access the app over `http://<vm-ip>:8080`, the browser will correctly show `Not secure` because that is plain HTTP on an IP address. For a real production setup, point a DNS hostname at the VM, set `PUBLIC_HOSTNAME` in `.env`, and run:

```bash
docker compose -f docker-compose.ubuntu-vgpu.yml -f docker-compose.https.yml --env-file .env up --build -d
```

That starts a Caddy edge on ports `80` and `443` and terminates TLS in front of the existing frontend container.

## API additions

- `POST /api/chat`: chat request with optional file uploads and knowledge-base indexing
- `GET /api/rag/status`: indexed document and chunk counts
- `GET /api/rag/documents`: indexed document inventory
- `POST /api/rag/index`: direct file indexing into the persistent RAG store
- `DELETE /api/rag/documents/{document_id}`: remove an indexed document

## Notes

- File uploads can be used ephemerally for one answer or persisted into the knowledge base.
- If sentence-transformer models are unavailable, the RAG layer falls back to hashed dense embeddings and heuristic reranking instead of failing closed.
- Triton remains the generation backend; the API builds retrieved context and routing decisions before dispatch.
- Document-heavy and summary-like prompts are now guardrailed toward the `full` model path, degraded short/gibberish responses automatically retry on the higher-fidelity model, and uploaded-document summaries fall back to an extractive summary if the model output is still low quality.
- `HF_TOKEN` is optional but strongly recommended in `.env` for faster, more reliable Triton cold starts on Hugging Face-hosted models.
- `DISABLE_GPU_METRICS=1` is the safe default so the green-metrics sidecar stays healthy even when Docker GPU/CDI integration is imperfect; Triton still gets the GPU.
- `TRITON_READY_WAIT_SECONDS` lets the API wait for Triton warm-up instead of failing the first request during model initialization.

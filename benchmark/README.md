# Routing benchmark

Three arms, one prompt set, one frozen router. The system claims that carbon-aware
routing costs less carbon than always reaching for the big model — *and* less than
the heuristic a competent engineer writes in an afternoon. This harness is where
that claim is either supported or it is not.

| arm | mechanism |
|---|---|
| `always-full` | every prompt pinned to `local-vgpu-full`. The "just use the big one" baseline. |
| `static-heuristic` | carbon-blind client-side rule (prompt length + keywords) → small / medium / full. Beating `always-full` is easy; beating this is the actual claim. |
| `css` | no pin. The real router: CSS scoring, RL weights, quality/latency estimator, deferral. |

## Run it

```bash
# 1. isolated bench API on :8101 — own data dir, frozen router, pinned grid.
#    Reuses the running vLLM containers; the live API on :8100 is never touched.
docker compose -f docker-compose.ubuntu-vgpu.yml \
               -f benchmark/docker-compose.bench.yml \
               --env-file .env up -d --build api-bench

# data/bench must be writable by the container's uid, or the engine silently
# falls back to /tmp *inside* the container and the run vanishes on restart.
mkdir -p data/bench && chown -R 1000:1000 data/bench

# 2. replay. Resumable: same --run-id skips (arm, prompt, repeat) triples already done.
python3 benchmark/run_bench.py --repeats 3 --run-id full-01 --sleep 0.3

# 3. report → report.md, summary.csv, summary.json, and data/benchmark_summary.json
python3 benchmark/report.py full-01
```

`report.py` publishes `data/benchmark_summary.json`, which the live API serves at
`GET /api/benchmark` and the frontend renders on the **Benchmark** tab. `data/` is
the only host directory mounted into the API container, which is why the summary
lands there and not in `benchmark/results/`.

Nothing in the running system can *start* a run. A benchmark the system under test
can trigger on demand is not a measurement.

## What the numbers mean

**Carbon is ex-post.** Not the CSS estimate — that is a spec constant, and it is what
the router used to *choose*. The reported number is the bill: spec TDP × **measured**
wall-clock, billed to the model that **actually served**, summed over every inference
leg the request burned (a quality retry runs two inferences and both burn GPU time).
Both numbers are recorded per request, so the report can show how far the forecast was
off.

**Power is modelled, not measured, and this is not optional.** The GPU is a vGPU slice
(`H100L-2-24C`); `nvidia-smi` returns `[N/A]` for `power.draw`. No per-request power
reading exists on this hardware. TDP is a spec **upper bound** on real draw. What *is*
measured: duration, tokens, and which model served.

**Quality is a lower bound.** Objective checkers only — numeric with tolerance,
substring, regex, and executed code with asserts. No LLM judge, so no judge variance
and no API key. A correct answer phrased unexpectedly is marked wrong. The same checker
runs against every arm, so the *comparison* is fair even where the absolute level is
pessimistic.

## Confounders, and why each is frozen

Every variable in `docker-compose.bench.yml` is a confirmed confounder. Removing any
one of them makes the three arms incomparable.

| var | why |
|---|---|
| `SEMANTIC_CACHE_ENABLED=false` | The cache is only conversation-scoped for *followups*; standalone prompts match by embedding cosine **across arms**. A hit short-circuits routing, returns `system_co2_g: 0.0`, and fabricates a `model_variant`. Arm 1's answers would be served to arms 2 and 3. |
| `RL_EXPLORATION=false` | Unseeded Dirichlet noise on **every** request — the same prompt routes differently twice in a row. |
| `RL_ALPHA_0=0` | RL reads *and writes* on every request. Unfrozen, arm 1 trains the router that arm 2 then runs against. |
| `QL_ESTIMATOR_ENABLED=false` | `LR=0` does **not** freeze it: `update()` still increments `n_obs`, and `adjust()` gates trust on `n_obs >= warmup`. A variant crossing the warm-up threshold mid-run flips from identity to trusted between two arms. |
| `EMAP_TOKEN=""` + `GRID_CARBON_FALLBACK` | Live grid CI drifts between arms, scales every carbon number, and straddles the hard CI≥450 threshold that penalises `full`. Pinned, it scales all arms identically and cancels out. Sweep `BENCH_GRID_CI` for the sensitivity curve. |
| `MOE_RECONCILER_ENABLED=false` | Background loop (10 s) can flip `available` and change the candidate set mid-run. |
| `GUARDRAILS_LLM_CLASSIFIER_ENABLED=false` | Each classifier adds an LLM call with a 6 s timeout that fails open — pure latency variance, independent of whether anything is blocked. The action-based rails stay **on**; flagged samples are excluded instead. |

Per request: no `conversation_id` (history depth feeds the profiler, so a shared thread
would make prompt N's routing depend on prompt N-1), `deferral_tolerance_ms=0` (the
default is 900,000 ms, which lets EcoServe defer a request out of the run entirely), and
one `X-Tenant-Id` per arm so a missed freeze flag surfaces as a tenant mismatch rather
than silent cross-arm contamination. Requests are serialised with a short pause: GPU
utilisation above 80% triggers a candidate **re-sort** that has no env toggle.

## Exclusions — the harness is worthless without them

A sample is excluded from the headline when it did not measure the arm's routing policy:

- `auto_escalated` — the dispatcher escalated inside `run_vllm_inference`; the model that ran is not the model CSS ranked
- `overflow_escalated` — history + retrieval overflowed the context window and forced a bigger model
- `quality_retry` — the first answer looked broken and a second inference ran on `full`
- `from_semantic_cache` — no model ran; carbon is 0.0 and the variant label is fabricated
- `direct_response` — arithmetic/multiplication-table shortcut answered without a model
- `blocked` / `redacted` — a guardrail intervened

A pinned arm that silently re-routes is not a control arm. The exclusion counts are
printed next to the table, not buried: **if an arm is excluding a lot, that is itself
the finding.**

A **pin violation** is a pinned arm re-routed anyway — the engine's quality guardrail
raises the accuracy floor for some intents, and a pinned model below that floor is
silently dropped. It applies to all three arms equally and the rate is published per arm.

## Two gotchas worth knowing

**Pin by zoo `id`, never by `model_variant`.** `target_matches` matches a preference
against id **or** variant **or** hardware **or** region, and variants are not unique:
`full` also admits `local-cpu-llama2-7b-fallback`, and `ultra-light` also admits
`local-cpu-fallback`, which always wins the carbon dimension. Pinning a variant does not
pin a model.

**Pass an explicit `accuracy_floor`.** `active_targets = constrained_targets or
filtered_targets` silently discards the pin whenever the floor empties the constrained
set.

## The bench model zoo

`benchmark/config/` shadows `config/` (mounted read-only) and differs in exactly one way:
candidates with no distinct live backend are marked `available: false`, so the router can
only pick a model that exists.

Not every zoo candidate is a distinct model. `ultra-light` and `medium` both dispatch to
`vllm-medium` serving TinyLlama-1.1B, and `local-cpu-fallback` has no endpoint of its own,
so it resolves there too — while being billed at 70 W as a "CPU device". Their differing
TDPs (70 / 95 / 145 W) and accuracies (0.60 / 0.66 / 0.81) are hand-written config, so a
carbon delta *between those labels* would be an artifact, not a saving. **Read the model
mix by `served_model`, not by candidate id.**

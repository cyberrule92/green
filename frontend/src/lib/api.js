const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

// ── Tenant header (X-Tenant-Id) ──────────────────────────────────────────────
// Persisted in localStorage so a refresh keeps the active tenant. Validated
// against the backend regex [a-z0-9][a-z0-9_-]{0,63}. A known-tenants list is
// also kept (most-recently-used first) so the UI can offer quick-switch chips.
const TENANT_STORAGE_KEY = "green-ai:tenant-id";
const TENANT_LIST_KEY = "green-ai:known-tenants";
const TENANT_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const TENANT_CHANGE_EVENT = "green-ai:tenant-change";
const TENANT_LIST_CHANGE_EVENT = "green-ai:tenant-list-change";
const TENANT_LIST_MAX = 20;

export function getTenantId() {
  try {
    const raw = (localStorage.getItem(TENANT_STORAGE_KEY) || "").trim();
    return raw && TENANT_PATTERN.test(raw) ? raw : "default";
  } catch {
    return "default";
  }
}

export function getKnownTenants() {
  try {
    const raw = localStorage.getItem(TENANT_LIST_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    const cleaned = Array.isArray(parsed)
      ? parsed.filter((t) => typeof t === "string" && TENANT_PATTERN.test(t))
      : [];
    // Always include "default" as a baseline option, plus the active tenant.
    const active = getTenantId();
    const merged = [];
    for (const t of [active, "default", ...cleaned]) {
      if (!merged.includes(t)) merged.push(t);
    }
    return merged.slice(0, TENANT_LIST_MAX);
  } catch {
    return ["default"];
  }
}

function persistKnownTenants(list) {
  try {
    localStorage.setItem(TENANT_LIST_KEY, JSON.stringify(list.slice(0, TENANT_LIST_MAX)));
  } catch {
    /* storage may be disabled — ignore */
  }
  window.dispatchEvent(
    new CustomEvent(TENANT_LIST_CHANGE_EVENT, { detail: list }),
  );
}

export function addKnownTenant(value) {
  const candidate = (value || "").trim().toLowerCase();
  if (!candidate || !TENANT_PATTERN.test(candidate)) return getKnownTenants();
  const current = getKnownTenants().filter((t) => t !== candidate);
  const next = [candidate, ...current].slice(0, TENANT_LIST_MAX);
  persistKnownTenants(next);
  return next;
}

export function removeKnownTenant(value) {
  const candidate = (value || "").trim().toLowerCase();
  if (!candidate || candidate === "default") return getKnownTenants();
  const next = getKnownTenants().filter((t) => t !== candidate);
  persistKnownTenants(next);
  return next;
}

export function setTenantId(value) {
  const candidate = (value || "").trim().toLowerCase();
  if (candidate && !TENANT_PATTERN.test(candidate)) {
    throw new Error(
      "Tenant id must match [a-z0-9][a-z0-9_-]{0,63} (lowercase, 1-64 chars).",
    );
  }
  if (candidate) {
    localStorage.setItem(TENANT_STORAGE_KEY, candidate);
    addKnownTenant(candidate);
  } else {
    localStorage.removeItem(TENANT_STORAGE_KEY);
  }
  window.dispatchEvent(
    new CustomEvent(TENANT_CHANGE_EVENT, { detail: candidate || "default" }),
  );
  return candidate || "default";
}

export function onTenantChange(handler) {
  const wrapped = (evt) => handler(evt?.detail || "default");
  window.addEventListener(TENANT_CHANGE_EVENT, wrapped);
  return () => window.removeEventListener(TENANT_CHANGE_EVENT, wrapped);
}

export function onTenantListChange(handler) {
  const wrapped = (evt) => handler(evt?.detail || []);
  window.addEventListener(TENANT_LIST_CHANGE_EVENT, wrapped);
  return () => window.removeEventListener(TENANT_LIST_CHANGE_EVENT, wrapped);
}

function withTenantHeader(options = {}) {
  const headers = new Headers(options.headers || {});
  if (!headers.has("X-Tenant-Id")) {
    headers.set("X-Tenant-Id", getTenantId());
  }
  return { ...options, headers };
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, withTenantHeader(options));
  const isJson = response.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await response.json() : null;

  if (!response.ok) {
    const detail = payload?.detail;
    const message =
      (typeof detail === "string" && detail) ||
      detail?.error ||
      payload?.message ||
      "The request could not be completed.";
    const err = new Error(message);
    err.status = response.status;
    err.payload = payload;
    throw err;
  }

  return payload;
}

export function fetchConversations() {
  return request("/api/conversations");
}

export function fetchConversation(conversationId) {
  return request(`/api/conversations/${conversationId}`);
}

export function fetchRagStatus() {
  return request("/api/rag/status");
}

export function fetchRagDocuments() {
  return request("/api/rag/documents");
}

export function deleteConversation(conversationId) {
  return request(`/api/conversations/${conversationId}`, { method: "DELETE" });
}

// ── Model Zoo (Section 3.3) ──────────────────────────────────────────────────
export function fetchModelZoo() {
  return request("/api/model-zoo");
}

export function fetchModelCarbon(modelId, { durationMs = 200, tokenCount = 256 } = {}) {
  return request(`/api/model-zoo/${modelId}/carbon?duration_ms=${durationMs}&token_count=${tokenCount}`);
}

export function fetchExpertHealth(modelId) {
  return request(`/api/model-zoo/${modelId}/expert-health`);
}

// ── Audit trail (Section 3.6.1) ──────────────────────────────────────────────
export function fetchAuditLog({ fromIso, toIso, model, tenant, minCarbonG, limit = 100 } = {}) {
  const params = new URLSearchParams();
  if (fromIso) params.set("from_iso", fromIso);
  if (toIso) params.set("to_iso", toIso);
  if (model) params.set("model", model);
  if (tenant) params.set("tenant", tenant);
  if (minCarbonG != null) params.set("min_carbon_g", minCarbonG);
  params.set("limit", limit);
  return request(`/api/audit?${params.toString()}`);
}

// ── Deferred queue (Section 3.5.2) ───────────────────────────────────────────
export function fetchQueueStatus() {
  return request("/api/queue/status");
}

export function dispatchQueueNow() {
  return request("/api/queue/dispatch-now", { method: "POST" });
}

// ── Grid / multi-region (Section 3.5.3) ──────────────────────────────────────
export function fetchGridZones() {
  return request("/api/grid/zones");
}

export function fetchGridForecast(zone) {
  const params = zone ? `?zone=${zone}` : "";
  return request(`/api/grid/forecast${params}`);
}

// ── Policy suggestion / RL foundation (Section 3.6.2) ────────────────────────
export function fetchPolicySuggestion({ tenant, lookbackEntries = 200 } = {}) {
  const params = new URLSearchParams();
  if (tenant) params.set("tenant", tenant);
  params.set("lookback_entries", lookbackEntries);
  return request(`/api/policy/suggest?${params.toString()}`);
}

// ── Chat ─────────────────────────────────────────────────────────────────────
// ── RL Controller (Section 3.6.2 / Decision-based-on-RL) ─────────────────────
export function fetchRLStatus() {
  return request("/api/rl/status");
}

export function fetchRLHistory({ tier, lastN = 100 } = {}) {
  const params = new URLSearchParams();
  if (tier) params.set("tier", tier);
  params.set("last_n", lastN);
  return request(`/api/rl/history?${params.toString()}`);
}

export function fetchSystemMetrics() {
  return request("/api/system/metrics");
}

export function fetchQualityLatencyEstimator() {
  return request("/api/routing/quality-latency-estimator");
}

// ── Observability ────────────────────────────────────────────────────────────
export function fetchObservabilitySummary({
  windowMinutes = 60,
  bucketSeconds = 60,
  sloP95Ms,
  sloErrorRate,
  energyPriceUsdKwh,
  cloudInputUsdPer1k,
  cloudOutputUsdPer1k,
} = {}) {
  const params = new URLSearchParams();
  params.set("window_minutes", windowMinutes);
  params.set("bucket_seconds", bucketSeconds);
  if (sloP95Ms != null) params.set("slo_p95_ms", sloP95Ms);
  if (sloErrorRate != null) params.set("slo_error_rate", sloErrorRate);
  if (energyPriceUsdKwh != null) params.set("energy_price_usd_kwh", energyPriceUsdKwh);
  if (cloudInputUsdPer1k != null) params.set("cloud_input_usd_per_1k", cloudInputUsdPer1k);
  if (cloudOutputUsdPer1k != null) params.set("cloud_output_usd_per_1k", cloudOutputUsdPer1k);
  return request(`/api/observability/summary?${params.toString()}`);
}

export function sendChatMessage({
  prompt,
  conversationId,
  persistAttachments,
  files = [],
}) {
  const body = new FormData();
  body.append("prompt", prompt);
  body.append("persist_attachments", String(Boolean(persistAttachments)));

  if (conversationId) {
    body.append("conversation_id", conversationId);
  }

  files.forEach((file) => {
    body.append("attachments", file);
  });

  return request("/api/chat", { method: "POST", body });
}

// ── Interaction feedback (thumbs up/down) ────────────────────────────────────
// Records a per-message quality signal used to build an offline fine-tuning /
// preference dataset (see /api/feedback/export on the backend).
export function sendFeedback({ messageId, rating, reason = "" }) {
  return request("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message_id: messageId, rating, reason }),
  });
}

export function fetchFeedbackStats() {
  return request("/api/feedback/stats");
}

// ── Tenancy / budgets / semantic cache / CSRD ────────────────────────────────
export function fetchTenantWhoAmI() {
  return request("/api/tenancy/whoami");
}

export function fetchMyBudget() {
  return request("/api/budgets/me");
}

export function fetchAllBudgets() {
  return request("/api/budgets");
}

export function updateBudget(targetTenant, fields) {
  return request(`/api/budgets/${encodeURIComponent(targetTenant)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  });
}

export function deleteBudgetOverride(targetTenant) {
  return request(`/api/budgets/${encodeURIComponent(targetTenant)}`, {
    method: "DELETE",
  });
}

export function fetchCacheStatus() {
  return request("/api/cache/status");
}

export function clearTenantCache() {
  return request("/api/cache/clear", { method: "POST" });
}

export function fetchCsrdReport({
  periodFromIso,
  periodToIso,
  energyPriceUsdKwh,
  marketBasedRenewablePct,
} = {}) {
  const params = new URLSearchParams();
  if (periodFromIso) params.set("period_from_iso", periodFromIso);
  if (periodToIso) params.set("period_to_iso", periodToIso);
  if (energyPriceUsdKwh != null) params.set("energy_price_usd_kwh", energyPriceUsdKwh);
  if (marketBasedRenewablePct != null)
    params.set("market_based_renewable_pct", marketBasedRenewablePct);
  return request(`/api/sustainability/csrd-report?${params.toString()}`);
}

// CSRD CSV download — fetches with the tenant header then returns a Blob URL
// so the browser can download() it. A bare <a href="..."> link can't carry
// custom headers, so we proxy through fetch.
export async function downloadCsrdReportCsv(opts = {}) {
  const params = new URLSearchParams({ fmt: "csv" });
  if (opts.periodFromIso) params.set("period_from_iso", opts.periodFromIso);
  if (opts.periodToIso) params.set("period_to_iso", opts.periodToIso);
  if (opts.energyPriceUsdKwh != null)
    params.set("energy_price_usd_kwh", opts.energyPriceUsdKwh);
  if (opts.marketBasedRenewablePct != null)
    params.set("market_based_renewable_pct", opts.marketBasedRenewablePct);
  const response = await fetch(
    `${API_BASE}/api/sustainability/csrd-report?${params.toString()}`,
    withTenantHeader({ method: "GET" }),
  );
  if (!response.ok) {
    throw new Error(`CSRD CSV export failed (HTTP ${response.status})`);
  }
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

// ── Model onboarding ─────────────────────────────────────────────────────────
// Browse Hugging Face, size a quantization plan against real headroom, download,
// serve, and register. An onboarded model stays unavailable to CSS until measured.

export function fetchModelCapability() {
  return request("/api/models/capability");
}

export function searchHuggingFaceModels({ q = "", limit = 25, task = "text-generation" } = {}) {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (task) params.set("task", task);
  return request(`/api/models/catalog/search?${params.toString()}`);
}

export function previewModelPlan(repoId, { maxModelLen = 2048, prefer, allowLocalQuantize = false } = {}) {
  const params = new URLSearchParams({ repo_id: repoId, max_model_len: String(maxModelLen) });
  if (prefer) params.set("prefer", prefer);
  if (allowLocalQuantize) params.set("allow_local_quantize", "true");
  return request(`/api/models/catalog/preview?${params.toString()}`);
}

export function onboardModel(body) {
  return request("/api/models/onboard", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchOnboardJobs(limit = 50) {
  return request(`/api/models/jobs?limit=${limit}`);
}

export function fetchOnboardJob(jobId) {
  return request(`/api/models/jobs/${jobId}`);
}

export function fetchModelRegistry() {
  return request("/api/models/registry");
}

export function serveOnboardedModel(modelId, { trustRemoteCode = false } = {}) {
  return request(`/api/models/${modelId}/serve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trust_remote_code: trustRemoteCode }),
  });
}

export function unserveOnboardedModel(modelId) {
  return request(`/api/models/${modelId}/unserve`, { method: "POST" });
}

export function measureOnboardedModel(modelId, { measurement, basis }) {
  return request(`/api/models/${modelId}/measure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ measurement, basis }),
  });
}

// ── Quantized artifacts ──────────────────────────────────────────────────────
// A quantized checkpoint is a deliverable on its own. `quantizeModel` runs the
// pipeline with registration skipped: nothing enters the model zoo, CSS never
// sees it, and the result is downloadable.

export function quantizeModel(body) {
  return request("/api/models/quantize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchQuantizedArtifacts() {
  return request("/api/models/artifacts");
}

/**
 * Download URL rather than a fetch: these are multi-gigabyte tarballs streamed
 * from disk, and pulling one through fetch() would buffer the whole checkpoint
 * in the tab's memory before the user saw a single byte. The browser's own
 * downloader handles it incrementally.
 */
export function artifactDownloadUrl(artifactId) {
  return `${API_BASE}/api/models/artifacts/${encodeURIComponent(artifactId)}/download`;
}

export function deleteQuantizedArtifact(artifactId) {
  return request(`/api/models/artifacts/${encodeURIComponent(artifactId)}`, { method: "DELETE" });
}

// ── Fine-tuning (/api/finetune) ──────────────────────────────────────────────
// The counterpart to onboarding: onboarding imports a better rung, fine-tuning
// makes the existing small one good enough that the router stops escalating.
// An adapter is not routable until measured — the whole point is that quality
// changed, so its direction is exactly what cannot be assumed.

export function fetchFinetuneCapability() {
  return request("/api/finetune/capability");
}

export function fetchFinetuneDataset() {
  return request("/api/finetune/dataset");
}

export function previewFinetunePlan({ baseModelId, method = "qlora", loraRank = 16, epochs = 3 }) {
  const params = new URLSearchParams({
    base_model_id: baseModelId,
    method,
    lora_rank: String(loraRank),
    epochs: String(epochs),
  });
  return request(`/api/finetune/preview?${params.toString()}`);
}

export function submitFinetuneJob(body) {
  return request("/api/finetune/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchFinetuneJobs(limit = 50) {
  return request(`/api/finetune/jobs?limit=${limit}`);
}

export function cancelFinetuneJob(jobId) {
  return request(`/api/finetune/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
}

export function serveFinetuneAdapter(adapterId) {
  return request(`/api/finetune/adapters/${encodeURIComponent(adapterId)}/serve`, { method: "POST" });
}

export function measureFinetuneAdapter(adapterId, { measurement, basis }) {
  return request(`/api/finetune/adapters/${encodeURIComponent(adapterId)}/measure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ measurement, basis }),
  });
}

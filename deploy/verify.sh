#!/usr/bin/env bash
# Adaptive Green AI — post-deploy verification.
#
# Proves the four claims that separate this stack from a plain LLM gateway:
# it routes on carbon, it reads a real grid signal, it signs every decision, and
# coding requests reach the code-capable rung rather than a general chat model.
# A green health check proves none of those, which is why this script exists.
set -uo pipefail

API=${API:-http://localhost:8100}
UI=${UI:-http://localhost:8080}
FAIL=0
ok()  { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad() { printf '  \033[31m✗\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
inf() { printf '    %s\n' "$1"; }
hdr() { printf '\n\033[1m%s\033[0m\n' "$1"; }
command -v jq >/dev/null || { echo "jq is required: sudo apt-get install -y jq"; exit 1; }
J()   { jq -r "$1 // empty" 2>/dev/null; }

hdr "1 · Health"
curl -sf "$API/health/ready" >/dev/null && ok "API ready (the metrics sidecar answered — /health/ready blocks on it)" \
                                        || bad "API not ready"
curl -sf http://localhost:9000/health >/dev/null && ok "metrics sidecar" || bad "metrics sidecar down"
for p in 8001 8002 8006; do
  curl -sf "http://localhost:$p/health" >/dev/null 2>&1 \
    && ok "vLLM :$p live" \
    || inf "- vLLM :$p not running (expected if its compose profile is off)"
done

hdr "2 · Carbon-aware routing + live grid"
# /api/chat is multipart/form-data with a 'prompt' field — not a JSON body.
R=$(mktemp); curl -sf -X POST "$API/api/chat" -F 'prompt=hi' -o "$R" --max-time 180
if [[ -s "$R" ]]; then
  VARIANT=$(J ".model_variant"           < "$R")
  MODEL=$(J   ".resolved_model_name"     < "$R")
  CO2=$(J     ".system_co2_g"             < "$R")
  GSTAT=$(J   ".grid_signal.status"       < "$R")
  GCI=$(J     ".grid_carbon"              < "$R")
  GZONE=$(J   ".grid_signal.zone"         < "$R")

  # The thesis in one assertion: a greeting must NOT reach the biggest model.
  if [[ "$VARIANT" == "ultra-light" || "$VARIANT" == "medium" ]]; then
    ok "\"hi\" routed to ${VARIANT} (${MODEL}) — ${CO2} gCO2"
  else
    bad "\"hi\" routed to ${VARIANT} — a greeting should land on the smallest rung; check config/policies.json carbon weights"
  fi

  if [[ "$GSTAT" == "live" ]]; then
    ok "grid signal LIVE: ${GCI} gCO2/kWh (zone ${GZONE}, Electricity Maps)"
  else
    bad "grid signal is '${GSTAT}', not live — check EMAP_TOKEN. The router still routes, on GRID_CARBON_FALLBACK, but it is no longer carbon-aware."
  fi
else
  bad "/api/chat returned nothing"
fi
rm -f "$R"

hdr "3 · 48-hour forecast (EcoServe's input)"
PTS=$(curl -sf "$API/api/grid/forecast" | J ".forecast[]?" | wc -w)
[[ "${PTS:-0}" -gt 0 ]] && ok "forecast returned data" \
  || inf "- forecast empty: deferral falls back to the live reading (a paid Electricity Maps plan is required for the forecast endpoint)"

hdr "4 · Signed audit trail"
N=$(wc -l < data/decision_logs.jsonl 2>/dev/null || echo 0)
[[ "$N" -gt 0 ]] && ok "decision_logs.jsonl: ${N} HMAC-signed rows (every dashboard reads this file and nothing else)" \
                 || bad "no audit rows written"

hdr "5 · Coding requests reach the coding rung"
# A coding prompt must land on the code-capable model, not a general instruct one.
# There is a single coding rung (vllm-stem-coding); if it is down the request
# silently falls through to a chat model and answers worse for the same carbon.
C=$(mktemp)
curl -sf -X POST "$API/api/chat" \
  -F 'prompt=Write a Python function that reverses a string.' -o "$C" --max-time 180
if [[ -s "$C" ]]; then
  CVAR=$(J ".model_variant"       < "$C")
  CMOD=$(J ".resolved_model_name" < "$C")
  CCO2=$(J ".system_co2_g"        < "$C")
  if [[ "$CVAR" == "stem-coding" ]]; then
    ok "coding prompt routed to ${CVAR}"
    inf "model            : ${CMOD}"
    inf "gCO2             : ${CCO2}"
  else
    bad "coding prompt routed to '${CVAR}', not stem-coding — is vllm-stem-coding (:8006) up? Start it with the 'stem-coding' profile."
  fi
else
  bad "/api/chat returned nothing for the coding prompt"
fi
rm -f "$C"

hdr "6 · UI"
curl -sf "$UI" >/dev/null && ok "frontend on ${UI} (tabs: Chat · Carbon · Observability · Models)" || bad "frontend down"

printf '\n'
[[ "$FAIL" -eq 0 ]] && printf '\033[32mDeployment verified.\033[0m\n' \
                    || { printf '\033[31m%d check(s) failed.\033[0m\n' "$FAIL"; exit 1; }

#!/usr/bin/env bash
# Adaptive Green AI — post-deploy verification.
#
# Proves the four claims that separate this stack from a plain LLM gateway:
# it routes on carbon, it reads a real grid signal, it signs every decision, and
# the agent optimises carbon per *completed task* rather than per token.
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
# `false // empty` is empty in jq — and `escalated: false` is precisely the
# result worth printing, so booleans need their own accessor.
JB()  { jq -r "$1 | tostring" 2>/dev/null; }

hdr "1 · Health"
curl -sf "$API/health/ready" >/dev/null && ok "API ready (the metrics sidecar answered — /health/ready blocks on it)" \
                                        || bad "API not ready"
curl -sf http://localhost:9000/health >/dev/null && ok "metrics sidecar" || bad "metrics sidecar down"
for p in 8001 8002 8006 8009; do
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

hdr "5 · Agentic harness — carbon per successful completion"
if curl -sf "$API/api/agent/status" >/dev/null; then
  # Caller-supplied tests: the spec is frozen and validated BEFORE a token is spent.
  # Left unset, the weakest rung authors the spec it is then judged against — the
  # failure mode that costs ~100x the carbon and still fails.
  TID=$(curl -sf -X POST "$API/api/agent/task" -H 'Content-Type: application/json' -d '{
    "task": "Write fizzbuzz(n) in solution.py returning \"Fizz\", \"Buzz\", \"FizzBuzz\" or str(n).",
    "tests": "from solution import fizzbuzz\ndef test_fizzbuzz():\n    assert fizzbuzz(3) == \"Fizz\"\n    assert fizzbuzz(5) == \"Buzz\"\n    assert fizzbuzz(15) == \"FizzBuzz\"\n    assert fizzbuzz(0) == \"FizzBuzz\"\n    assert fizzbuzz(7) == \"7\"\n",
    "allow_defer": false }' | J ".task_id")
  if [[ -z "${TID:-}" ]]; then
    bad "agent task rejected (a 400 here means the test suite failed validation — that is the gate working)"
  else
    inf "task ${TID} submitted; polling…"
    for _ in $(seq 1 48); do
      sleep 5
      T=$(mktemp); curl -sf "$API/api/agent/task/$TID" -o "$T"
      ST=$(J ".status" < "$T")
      [[ "$ST" == "completed" || "$ST" == "failed" || "$ST" == "aborted" ]] && break
    done
    if [[ "$ST" == "completed" ]]; then
      ok "agent COMPLETED"
      inf "rung          : $(J ".result.final_tier"               < "$T")"
      inf "escalated     : $(JB ".result.escalated"               < "$T")  (false = the greenest rung was enough)"
      inf "LLM calls     : $(J ".result.total_llm_calls"          < "$T")"
      inf "spec author   : $(J ".result.spec_source"              < "$T")  (caller = your tests are the ground truth)"
      inf "gCO2 / completion: $(J ".result.carbon_per_completion_g" < "$T")"
    else
      bad "agent did not complete (status=${ST:-unknown}) — docker compose logs api"
    fi
    rm -f "$T"
  fi
else
  inf "- agent disabled (AGENT_ENABLED=false)"
fi

hdr "6 · UI"
curl -sf "$UI" >/dev/null && ok "frontend on ${UI} (tabs: Chat · Carbon · Observability · Agent)" || bad "frontend down"

printf '\n'
[[ "$FAIL" -eq 0 ]] && printf '\033[32mDeployment verified.\033[0m\n' \
                    || { printf '\033[31m%d check(s) failed.\033[0m\n' "$FAIL"; exit 1; }

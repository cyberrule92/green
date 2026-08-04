#!/usr/bin/env bash
# Adaptive Green AI — bootstrap a new environment.
#
#   deploy/bootstrap.sh                   base stack (chat routing, no coding rung)
#   deploy/bootstrap.sh stem-coding       + Qwen2.5-Coder-1.5B, the coding rung
#   deploy/bootstrap.sh stem-coding stem  + the STEM math rung
#
# Profiles are additive. The VRAM budget is the hard constraint, not the CPU or
# the disk: on a 24 GB card the base stack + coding rung uses ~13 GB. See the
# deploy guide's VRAM table.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
COMPOSE="docker compose -f docker-compose.ubuntu-vgpu.yml --env-file .env"

PROFILES=()
for p in "$@"; do PROFILES+=(--profile "$p"); done

echo "==> 1/5  .env"
if [[ -f .env ]]; then
  echo "    .env exists — keeping it (delete it to re-seed from deploy/env.template)"
else
  cp deploy/env.template .env
  chmod 600 .env
  echo "    seeded .env from deploy/env.template (mode 600)"
fi
# The template ships with the four secrets scrubbed — deliberately, so a real
# credential never sits in the repo. They live in Appendix A of the deploy guide.
if grep -q '__PASTE_FROM_DEPLOY_GUIDE_APPENDIX_A__' .env; then
  echo
  echo "    STOP. .env still has placeholder credentials:"
  grep -n '__PASTE_FROM_DEPLOY_GUIDE_APPENDIX_A__' .env | sed 's/^/      /'
  echo
  echo "    Fill them in from Appendix A of Adaptive_Green_AI_Deployment.pdf, then re-run."
  echo "    (HF_TOKEN is the only one that is strictly required — every vLLM"
  echo "     container pulls its weights with it. Without EMAP_TOKEN the router"
  echo "     falls back to GRID_CARBON_FALLBACK; without AUDIT_HMAC_KEY the audit"
  echo "     trail is unsigned; without ADMIN_API_KEY the admin endpoints are open.)"
  exit 1
fi

echo "==> 2/5  data directories"
# hf-cache is bind-mounted into every vLLM container. If it does not exist,
# Docker creates it root-owned and the weights re-download on every recreate.
mkdir -p data/hf-cache
echo "    data/ and data/hf-cache ready"

echo "==> 3/5  tuning .env to this host's GPU"
# GPU_TDP and GPU_VRAM_GB feed the LLMCarbon operational-carbon formula directly.
# Wrong values do not crash anything — they silently produce wrong carbon numbers,
# which is the one failure this system cannot tolerate.
if command -v nvidia-smi >/dev/null; then
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  VRAM_GB=$(( VRAM_MIB / 1024 ))
  sed -i -E "s/^GPU_VRAM_GB=.*/GPU_VRAM_GB=${VRAM_GB}/" .env
  echo "    GPU_VRAM_GB=${VRAM_GB} (detected)"
  echo "    GPU_TDP=$(grep -E '^GPU_TDP=' .env | cut -d= -f2) — VERIFY THIS BY HAND against your card's spec sheet"
else
  echo "    no nvidia-smi; leaving GPU_TDP / GPU_VRAM_GB as shipped"
fi

echo "==> 4/5  build + start"
# First run pulls ~19 GB of weights into data/hf-cache. Expect 20-40 min.
$COMPOSE "${PROFILES[@]}" up --build -d

echo "==> 5/5  waiting for health"
for i in $(seq 1 60); do
  if curl -sf http://localhost:8100/health/ready >/dev/null 2>&1; then
    echo "    API ready"
    break
  fi
  [[ $i -eq 60 ]] && { echo "    API did not become ready in 10 min — $COMPOSE logs api"; exit 1; }
  sleep 10
done

echo
echo "Up. UI: http://localhost:8080   API: http://localhost:8100"
echo "Verify with: deploy/verify.sh"

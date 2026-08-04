#!/usr/bin/env bash
# Adaptive Green AI — preflight. Checks every prerequisite BEFORE anything is
# built, because each failure below costs 20+ minutes to discover the slow way
# (a 19 GB model pull that dies on a gated repo, an OOM 40 minutes into a build).
# Read-only: this script changes nothing.
set -uo pipefail

PASS=0; WARN=0; FAIL=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; WARN=$((WARN+1)); }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

hdr "Host"
. /etc/os-release 2>/dev/null || true
[[ "${ID:-}" == "ubuntu" ]] && ok "OS: ${PRETTY_NAME}" || warn "OS: ${PRETTY_NAME:-unknown} (tested on Ubuntu 22.04/24.04)"

hdr "Docker"
if command -v docker >/dev/null; then
  ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
  docker info >/dev/null 2>&1 && ok "docker daemon reachable" \
    || bad "docker daemon not reachable — is the user in the 'docker' group?"
else
  bad "docker not installed"
fi
if docker compose version >/dev/null 2>&1; then
  ok "compose plugin $(docker compose version --short)"
else
  bad "'docker compose' plugin missing (the old docker-compose binary will not work: this stack uses profiles + service_healthy)"
fi

hdr "GPU"
if command -v nvidia-smi >/dev/null; then
  # Comma-separated, not whitespace: the GPU name itself contains spaces
  # ("NVIDIA H100L-2-24C"), which silently shifts every field if you split on IFS.
  IFS=',' read -r GNAME GDRV GMEM < <(nvidia-smi --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader,nounits | head -1)
  GNAME="${GNAME# }"; GDRV="${GDRV# }"; GMEM="${GMEM# }"
  ok "GPU: ${GNAME} (driver ${GDRV})"
  if [[ "${GMEM:-0}" -ge 23000 ]]; then
    ok "VRAM: ${GMEM} MiB"
  elif [[ "${GMEM:-0}" -ge 15000 ]]; then
    warn "VRAM: ${GMEM} MiB — enough for the base stack, tight with the coding rung (see the VRAM budget in the deploy guide)"
  else
    bad "VRAM: ${GMEM} MiB — below the 16 GB floor; only the CPU fallback profile will run"
  fi
  # The toolkit is the single most common failure: driver present, containers blind.
  if docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1; then
    ok "nvidia-container-toolkit wired into docker (containers can see the GPU)"
  else
    bad "containers cannot see the GPU — install nvidia-container-toolkit and run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  fi
else
  bad "nvidia-smi not found — install the NVIDIA driver first"
fi

hdr "Disk"
# Measured on the reference box: vllm/vllm-openai is 32.9 GB, the api/metrics
# image 8.7 GB, and the six model repos 19 GB. ~61 GB before any build cache.
AVAIL_G=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
if   [[ "$AVAIL_G" -ge 90 ]]; then ok "free space: ${AVAIL_G} GB"
elif [[ "$AVAIL_G" -ge 65 ]]; then warn "free space: ${AVAIL_G} GB — tight. The stack needs ~61 GB (33 GB vLLM image + 9 GB API image + 19 GB weights) and build cache grows on top (docker builder prune)"
else bad "free space: ${AVAIL_G} GB — need ≥65 GB (33 GB vLLM image + 9 GB API image + 19 GB weights)"; fi

hdr "Ports"
# A port held by a green-* container is this stack already running, not a clash.
MINE=$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep '^green-' || true)
for p in 8001 8002 8006 8080 8100 9000; do
  if ! ss -lntH "sport = :$p" 2>/dev/null | grep -q .; then
    ok "port $p free"
  elif grep -q ":$p->" <<<"$MINE"; then
    warn "port $p held by $(grep -o "^green-[a-z0-9-]*" <<<"$(grep ":$p->" <<<"$MINE")") — this stack is already up"
  else
    bad "port $p in use by something else"
  fi
done

hdr "Network / credentials"
if [[ -f "$REPO/.env" ]]; then
  ok ".env present"
  set -a; . "$REPO/.env" 2>/dev/null; set +a
else
  warn ".env missing — bootstrap.sh will create it from deploy/env.template"
  set -a; . "$REPO/deploy/env.template" 2>/dev/null; set +a
fi
if [[ -n "${HF_TOKEN:-}" ]] && curl -sf -H "Authorization: Bearer ${HF_TOKEN}" \
     https://huggingface.co/api/whoami-v2 >/dev/null; then
  ok "HF_TOKEN valid"
  # Gated repos 401 on download, not on build — so check them here, not later.
  for m in meta-llama/Llama-Guard-3-1B meta-llama/Llama-2-7b-chat-hf; do
    if curl -sf -H "Authorization: Bearer ${HF_TOKEN}" "https://huggingface.co/api/models/$m" >/dev/null; then
      ok "gated model accessible: $m"
    else
      warn "no access to gated repo $m — accept its licence at https://huggingface.co/$m (only needed for the 'guard' / 'fallback' profiles)"
    fi
  done
else
  bad "HF_TOKEN missing or rejected — every vLLM container pulls its weights with it"
fi
if [[ -n "${EMAP_TOKEN:-}" ]] && curl -sf -H "auth-token: ${EMAP_TOKEN}" \
     "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=${EMAP_ZONE:-IN-WE}" >/dev/null; then
  ok "EMAP_TOKEN valid for zone ${EMAP_ZONE:-IN-WE}"
else
  warn "Electricity Maps unreachable — the router still routes, it just falls back to GRID_CARBON_FALLBACK and stops being clever about the grid"
fi

printf '\n\033[1m%d passed, %d warnings, %d failures\033[0m\n' "$PASS" "$WARN" "$FAIL"
[[ "$FAIL" -eq 0 ]] || { echo "Fix the failures above before running bootstrap.sh."; exit 1; }
echo "Preflight clean — run: deploy/bootstrap.sh"

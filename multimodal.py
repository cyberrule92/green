"""
multimodal.py — image analysis (VLM) and image generation (diffusion) dispatch.

This is the dispatch layer for the two multimodal modalities the router now
supports, built on NVIDIA reference architectures:

  * **Image analysis (vision):** NVIDIA NIM for VLMs — e.g. Nemotron Nano V2 VL,
    served OpenAI-compatibly (`/chat/completions` with image_url content parts).
  * **Image generation (diffusion):** NVIDIA NIM for Visual Generative AI —
    SDXL / FLUX, served via a text-to-image `infer` endpoint.

Everything here is **pluggable with graceful fallback**: endpoints are resolved
from env vars (like `resolve_vllm_endpoint` for chat). When a NIM endpoint is not
configured or is unreachable, the dispatcher returns a deterministic,
dependency-free placeholder (an SVG for generation, a metadata description for
analysis) so the *whole* carbon-aware pipeline — routing, CSS, carbon
accounting, EcoServe deferral, audit — runs and is demonstrable today, and swaps
to real NIM inference the moment the containers are stood up. No image bytes are
ever fabricated as if real: fallbacks are clearly labelled placeholders.

Kept deliberately dependency-free (stdlib + requests, already a project dep). No
Pillow / torch: image dimensions are parsed from raw headers.
"""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger("green_ai.multimodal")

# Short timeouts: a slow NIM must not stall the request path — we fall back.
_CONNECT_TIMEOUT = float(os.getenv("NIM_CONNECT_TIMEOUT", "3"))
_READ_TIMEOUT = float(os.getenv("NIM_READ_TIMEOUT", "60"))

# ── Hugging Face Inference fallback (real images when no local NIM) ───────────
# When no NIM diffusion endpoint is configured we generate a *real* image with a
# hosted Hugging Face text-to-image model via the HF Inference router, using the
# HF_TOKEN already present for the model cache. Lightweight (no torch in the API
# container), returns genuine PNG bytes. Only if this also fails do we fall back
# to the labelled SVG placeholder.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN") or ""
HF_IMAGE_FALLBACK_ENABLED = os.getenv("HF_IMAGE_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes"}
HF_INFERENCE_BASE = os.getenv(
    "HF_INFERENCE_BASE", "https://router.huggingface.co/hf-inference/models"
).rstrip("/")
# FLUX.1-schnell is Apache-2.0, fast (few-step distilled) and reliably served on
# the hf-inference provider. Overridable per diffusion variant.
_HF_DEFAULT_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
_HF_IMAGE_MODEL_BY_VARIANT = {
    "diffusion-sdxl": os.getenv("HF_SDXL_MODEL", _HF_DEFAULT_IMAGE_MODEL),
    "diffusion-flux": os.getenv("HF_FLUX_MODEL", "black-forest-labs/FLUX.1-schnell"),
}
# Distilled few-step models ignore large step counts; clamp what we pass so the
# carbon-capped step budget still modulates the request within the useful range.
HF_IMAGE_MAX_STEPS = int(os.getenv("HF_IMAGE_MAX_STEPS", "8"))
HF_READ_TIMEOUT = float(os.getenv("HF_READ_TIMEOUT", "120"))  # cold starts can be slow


def _hf_generate_image(
    variant: str, prompt: str, steps: int, width: int, height: int
) -> tuple[bytes, str] | None:
    """Return (png_bytes, hf_model_id) from the HF Inference router, or None."""
    if not (HF_IMAGE_FALLBACK_ENABLED and HF_TOKEN):
        return None
    model = _HF_IMAGE_MODEL_BY_VARIANT.get((variant or "").lower(), _HF_DEFAULT_IMAGE_MODEL)
    payload = {
        "inputs": prompt or "an image",
        "parameters": {
            "num_inference_steps": max(1, min(int(steps or 4), HF_IMAGE_MAX_STEPS)),
            "width": width,
            "height": height,
        },
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "image/png",
        # Block until the model is warm rather than getting a 503 on cold start.
        "x-wait-for-model": "true",
    }
    try:
        r = requests.post(f"{HF_INFERENCE_BASE}/{model}", json=payload, headers=headers,
                          timeout=(_CONNECT_TIMEOUT + 2, HF_READ_TIMEOUT))
        ctype = r.headers.get("content-type", "")
        if r.status_code == 200 and ctype.startswith("image"):
            return r.content, model
        logger.warning("HF image-gen non-image (%s, %s): %s",
                       r.status_code, ctype, r.text[:160] if not ctype.startswith("image") else "")
    except Exception as exc:
        logger.warning("HF image-gen call failed: %s", exc)
    return None


# ── Endpoint resolution ───────────────────────────────────────────────────────
def resolve_nim_endpoint(candidate: dict[str, Any]) -> tuple[str | None, str]:
    """
    Resolve (base_url, model_name) for a multimodal candidate.

    Reads the candidate's ``vllm_endpoint_env`` (e.g. NIM_VLM_URL / NIM_SDXL_URL
    / NIM_FLUX_URL) at call time so a deployment can rotate URLs without a
    restart. Returns (None, model_name) when the env is unset — the caller then
    uses the graceful fallback.
    """
    env_name = candidate.get("vllm_endpoint_env")
    model_name = candidate.get("resolved_model_name") or candidate.get("model_variant") or "unknown"
    url = os.getenv(env_name) if env_name else None
    if url:
        url = url.rstrip("/")
    return url, model_name


# ── Lightweight image header parsing (no Pillow) ──────────────────────────────
def _image_dimensions(raw: bytes) -> tuple[int, int] | None:
    """Best-effort (width, height) from PNG or JPEG bytes; None if unknown."""
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
            return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
        if raw[:2] == b"\xff\xd8":  # JPEG
            i, n = 2, len(raw)
            while i + 9 < n:
                if raw[i] != 0xFF:
                    i += 1
                    continue
                marker = raw[i + 1]
                # SOF0..SOF15 (excluding DHT/DAC/RST) carry frame dimensions.
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h = int.from_bytes(raw[i + 5:i + 7], "big")
                    w = int.from_bytes(raw[i + 7:i + 9], "big")
                    return w, h
                seg_len = int.from_bytes(raw[i + 2:i + 4], "big")
                i += 2 + seg_len
    except Exception:
        return None
    return None


def _decode_data_uri(data_uri: str) -> bytes | None:
    try:
        if "," in data_uri:
            return base64.b64decode(data_uri.split(",", 1)[1])
        return base64.b64decode(data_uri)
    except Exception:
        return None


# ── Image generation ──────────────────────────────────────────────────────────
def _palette_from_prompt(prompt: str) -> tuple[str, str]:
    """Deterministic two-colour gradient seeded by the prompt (placeholder art)."""
    h = hashlib.sha256((prompt or "green").encode("utf-8")).digest()
    c1 = f"#{h[0]:02x}{h[1]:02x}{h[2]:02x}"
    c2 = f"#{h[3]:02x}{h[4]:02x}{h[5]:02x}"
    return c1, c2


def _placeholder_image_data_uri(
    prompt: str, model_name: str, steps: int, width: int, height: int, reason: str
) -> str:
    """A labelled SVG placeholder returned when no diffusion endpoint is live."""
    c1, c2 = _palette_from_prompt(prompt)
    safe_prompt = html.escape((prompt or "")[:120])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/>
  </linearGradient></defs>
  <rect width="{width}" height="{height}" fill="url(#g)"/>
  <rect x="16" y="16" width="{width-32}" height="{height-32}" fill="none" stroke="#ffffff" stroke-opacity="0.35" stroke-width="2" rx="14"/>
  <text x="50%" y="42%" fill="#ffffff" font-family="sans-serif" font-size="{max(width//22,14)}" font-weight="700" text-anchor="middle">Image placeholder</text>
  <text x="50%" y="52%" fill="#ffffff" fill-opacity="0.92" font-family="sans-serif" font-size="{max(width//34,11)}" text-anchor="middle">{safe_prompt}</text>
  <text x="50%" y="88%" fill="#ffffff" fill-opacity="0.8" font-family="monospace" font-size="{max(width//40,10)}" text-anchor="middle">{html.escape(model_name)} · {steps} steps · {reason}</text>
</svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def run_image_generation(
    candidate: dict[str, Any],
    prompt: str,
    steps: int,
    width: int = 1024,
    height: int = 1024,
) -> dict[str, Any]:
    """
    Generate one image for ``prompt`` using the candidate diffusion model.

    Returns a dict:
      {image_data_uri, backend: "nim"|"fallback", model, steps, width, height,
       actual_latency_ms, note}
    Never raises — falls back to a labelled placeholder on any failure.
    """
    start = time.monotonic()
    url, model_name = resolve_nim_endpoint(candidate)
    steps = max(int(steps or 1), 1)

    if url:
        try:
            # NVIDIA NIM for Visual GenAI text-to-image (tolerant to response shape).
            payload = {
                "prompt": prompt,
                "model": model_name,
                "steps": steps,
                "width": width,
                "height": height,
                "cfg_scale": 5.0,
                "samples": 1,
            }
            r = requests.post(f"{url}/v1/infer", json=payload,
                              timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            if r.status_code == 200:
                data = r.json()
                b64 = _extract_generated_b64(data)
                if b64:
                    return {
                        "image_data_uri": b64 if b64.startswith("data:") else f"data:image/png;base64,{b64}",
                        "backend": "nim", "model": model_name, "steps": steps,
                        "width": width, "height": height,
                        "actual_latency_ms": (time.monotonic() - start) * 1000.0,
                        "note": "NVIDIA NIM diffusion",
                    }
            logger.warning("NIM image-gen non-200/parse (%s); trying HF", getattr(r, "status_code", "?"))
        except Exception as exc:
            logger.warning("NIM image-gen call failed (%s); trying HF", exc)

    # ── Real image via Hugging Face Inference (no local NIM) ──
    hf = _hf_generate_image(candidate.get("model_variant") or "", prompt, steps, width, height)
    if hf:
        png_bytes, hf_model = hf
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {
            "image_data_uri": f"data:image/png;base64,{b64}",
            "backend": "huggingface", "model": hf_model, "steps": min(steps, HF_IMAGE_MAX_STEPS),
            "width": width, "height": height,
            "actual_latency_ms": (time.monotonic() - start) * 1000.0,
            "note": f"Hugging Face Inference · {hf_model}",
        }

    reason = ("no NIM endpoint, HF unavailable → placeholder" if not url
              else "NIM error, HF unavailable → placeholder")
    return {
        "image_data_uri": _placeholder_image_data_uri(prompt, model_name, steps, width, height, reason),
        "backend": "fallback", "model": model_name, "steps": steps,
        "width": width, "height": height,
        "actual_latency_ms": (time.monotonic() - start) * 1000.0,
        "note": reason,
    }


def _extract_generated_b64(data: Any) -> str | None:
    """Pull a base64 image out of the various shapes NIM/diffusion APIs return."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("artifacts"), list) and data["artifacts"]:
        art = data["artifacts"][0]
        if isinstance(art, dict):
            return art.get("base64") or art.get("b64_json")
    if isinstance(data.get("data"), list) and data["data"]:
        d0 = data["data"][0]
        if isinstance(d0, dict):
            return d0.get("b64_json") or d0.get("image")
    return data.get("image") or data.get("b64_json")


# ── Image analysis (VLM) ──────────────────────────────────────────────────────
def run_vlm_inference(
    candidate: dict[str, Any],
    prompt: str,
    image_data_uris: list[str],
) -> dict[str, Any]:
    """
    Analyse the attached image(s) with the candidate VLM.

    Returns {text, backend: "nim"|"fallback", model, actual_latency_ms}.
    Never raises — falls back to a metadata-grounded description.
    """
    start = time.monotonic()
    url, model_name = resolve_nim_endpoint(candidate)
    question = (prompt or "").strip() or "Describe this image."

    if url and image_data_uris:
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": question}]
            for uri in image_data_uris[:4]:
                content.append({"type": "image_url", "image_url": {"url": uri}})
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 640,
                "temperature": 0.2,
            }
            r = requests.post(f"{url}/chat/completions", json=payload,
                              timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            if r.status_code == 200:
                choices = r.json().get("choices") or []
                if choices:
                    text = (choices[0].get("message") or {}).get("content", "").strip()
                    if text:
                        return {"text": text, "backend": "nim", "model": model_name,
                                "actual_latency_ms": (time.monotonic() - start) * 1000.0}
            logger.warning("NIM VLM non-200 (%s); using metadata fallback", getattr(r, "status_code", "?"))
        except Exception as exc:
            logger.warning("NIM VLM call failed (%s); using metadata fallback", exc)

    # ── Fallback: describe what we *can* determine without a vision model ──
    lines = []
    for i, uri in enumerate(image_data_uris, 1):
        raw = _decode_data_uri(uri)
        dims = _image_dimensions(raw) if raw else None
        size_kb = round(len(raw) / 1024.0, 1) if raw else 0.0
        dim_str = f"{dims[0]}×{dims[1]}px" if dims else "unknown dimensions"
        lines.append(f"  • image {i}: {dim_str}, ~{size_kb} KB")
    detail = "\n".join(lines) if lines else "  • (no decodable image data)"
    text = (
        f"[Vision model offline — metadata-only response]\n\n"
        f"I received {len(image_data_uris)} image(s) for the request "
        f"\"{question}\", but the VLM endpoint (NIM_VLM_URL) is not currently "
        f"configured, so I can't analyse pixel content. What I can confirm:\n"
        f"{detail}\n\n"
        f"Configure a NVIDIA NIM VLM endpoint (e.g. {model_name}) to enable full "
        f"image understanding."
    )
    return {"text": text, "backend": "fallback", "model": model_name,
            "actual_latency_ms": (time.monotonic() - start) * 1000.0}

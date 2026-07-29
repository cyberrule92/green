#!/usr/bin/env python3
"""Docker healthcheck for a vLLM container: does it actually *generate*?

Why this exists rather than `curl -f /health`
---------------------------------------------
On 2026-07-29 all three vLLM containers on this host reported HTTP 200 from
`/health` — and were marked `healthy` by Docker — for roughly 21 hours while
their EngineCore processes were wedged and every `/v1/completions` request hung
until the client gave up. The production API was down that whole time and
nothing flagged it, because `/health` is answered by the API server process and
says nothing about the engine behind it. `/v1/models` is no better: it reads
static config.

A healthcheck that a wedged container passes is worse than no healthcheck,
because it actively certifies a dead backend as live and `depends_on:
service_healthy` gates on it.

So this probe generates one token. That is the smallest thing that cannot
succeed unless the engine loop is running.

Cost
----
One token per container per interval. At a 60 s interval across the vLLM
services here that is on the order of a few gCO2e per day — set against an
outage that ran for 21 hours undetected, which is the trade being made. The
interval is deliberately longer than the old 20 s `/health` poll for this
reason; a wedge lasts hours, so detecting it within a minute is ample.

Usage: vllm_healthcheck.py <port>
Exit 0 = generating, 1 = not.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

MODELS_TIMEOUT_S = 5
GENERATE_TIMEOUT_S = 25


def fail(reason: str) -> int:
    # Docker surfaces healthcheck output in `docker inspect .State.Health.Log`,
    # so the reason is worth printing even though nothing reads stdout live.
    print(f"unhealthy: {reason}", file=sys.stderr)
    return 1


def main() -> int:
    if len(sys.argv) < 2:
        return fail("usage: vllm_healthcheck.py <port>")
    port = sys.argv[1]
    base = f"http://127.0.0.1:{port}"

    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=MODELS_TIMEOUT_S) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return fail(f"/v1/models unreachable: {exc}")

    data = payload.get("data") or []
    if not data:
        return fail("/v1/models returned no model")
    model = data[0].get("id")
    if not model:
        return fail("/v1/models returned a model with no id")

    # Try text completions, then chat. Some models here are served with a chat
    # template only (Llama Guard is the case in this stack) and reject /v1/completions
    # with a 400 — which means "wrong endpoint", not "engine wedged". A wedge shows
    # up as a timeout on either endpoint, so falling through on a 400 keeps one
    # script correct for every service rather than special-casing by port.
    attempts = (
        ("/v1/completions", {"model": model, "prompt": "ping", "max_tokens": 1, "temperature": 0}),
        (
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
            },
        ),
    )

    last = "no attempt made"
    for path, payload in attempts:
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=GENERATE_TIMEOUT_S) as resp:
                out = json.load(resp)
            # A 200 with no choices means the request was accepted but nothing was
            # produced — the wedge presenting differently.
            if out.get("choices"):
                return 0
            last = f"{path} returned no choices"
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                last = f"{path} rejected the request (400); trying the next endpoint"
                continue
            last = f"{path} returned {exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # The wedged case: the socket accepts, the request never completes,
            # and the read times out. No point trying the other endpoint — the
            # engine, not the route, is the problem.
            return fail(f"generation hung on {path} for {model}: {exc}")

    return fail(f"could not generate with {model}: {last}")


if __name__ == "__main__":
    raise SystemExit(main())

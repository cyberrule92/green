"""The three arms under comparison.

An arm is a function from a prompt row to the request parameters sent to
/api/chat. Everything else about the request is identical across arms.

    always-full       every prompt to the largest available model. The "just use
                      the big one" baseline the router has to beat.
    static-heuristic  a carbon-blind client-side rule (prompt length + keywords)
                      picks small / medium / full. This is what a competent
                      engineer writes in an afternoon, and it is the baseline
                      that matters: beating always-full is easy, beating a
                      sensible heuristic is the actual claim.
    css               no preference. The real router: CSS scoring, RL weights,
                      quality/latency estimator, deferral.
    always-coder7b    every prompt pinned to the 4-bit AWQ 7B. Not a routing
                      strategy -- a measurement. CSS ranks this candidate last
                      because its DECLARED accuracy (0.93) is only +0.01 over
                      `full` (0.92) while its spec carbon is 2.37x higher, so a
                      carbon-dominant score will never choose it. But declared
                      accuracy is a spec constant and the prior run measured
                      `full` delivering 0.793 actual against its declared 0.92 --
                      so the declared figures compress the real gap by an unknown
                      amount. This arm answers the question those constants
                      cannot: does a genuinely larger model deliver enough extra
                      quality to justify 2.37x the carbon? If yes, the zoo's
                      accuracy_baseline is wrong and should be corrected from
                      measurement, which would change what CSS picks. If no, the
                      ladder has no better rung and CSS is right to ignore it.

Two mechanics matter and both are load-bearing:

1. **Pin by zoo `id`, never by `model_variant`.** `target_matches`
   (routing_policies.py) matches a preference against id OR variant OR hardware
   OR region, and variants are not unique — `full` also admits
   `local-cpu-llama2-7b-fallback`, and `ultra-light` also admits
   `local-cpu-fallback`, which always wins the carbon dimension. Pinning a
   variant does not pin a model.

2. **Pass an explicit `accuracy_floor`.** `active_targets = constrained_targets
   or filtered_targets` silently discards the pin whenever the floor empties the
   constrained set. We pass 0.0 to keep our own floor out of it — but note the
   engine's quality guardrail can still *raise* the floor above a pinned model's
   accuracy (`local-vgpu-small` is 0.66; an "implementation" intent raises the
   floor to 0.78), which unpins the arm with no signal in the response. The
   harness detects that after the fact by comparing the requested pin against
   `served_target_id`, and reports it as `pin_violated`. It is not hidden: the
   quality guardrail applies to all three arms equally, and the adherence rate
   is published per arm.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Zoo ids, not variants. See docstring.
FULL = "local-vgpu-full"
MEDIUM = "local-vgpu-medium"
SMALL = "local-vgpu-small"
CODER7B = "local-vgpu-coder-7b"

# Keywords a carbon-blind engineer would reach for: "this looks hard, use the
# big model". Deliberately naive — that is the point of the baseline.
_HARD_KEYWORDS = re.compile(
    r"\b(analy[sz]e|explain|why|reason|prove|derive|design|architect|compare|"
    r"trade[- ]?off|summari[sz]e|refactor|debug|implement|write a|code|function)\b",
    re.IGNORECASE,
)
_EASY_KEYWORDS = re.compile(
    r"\b(what is|who is|when did|where is|which|name|list|capital|how many)\b",
    re.IGNORECASE,
)


def _static_heuristic_pin(prompt: str) -> str:
    """Length + keyword rule. No carbon signal, no grid signal, no learning."""
    n = len(prompt)
    if n > 320 or (_HARD_KEYWORDS.search(prompt) and n > 120):
        return FULL
    if _EASY_KEYWORDS.search(prompt) and n <= 120:
        return SMALL
    if _HARD_KEYWORDS.search(prompt):
        return MEDIUM
    return MEDIUM if n > 120 else SMALL


def arm_always_full(row: dict[str, Any]) -> dict[str, Any]:
    return {"model_preference": FULL, "accuracy_floor": 0.0}


def arm_static_heuristic(row: dict[str, Any]) -> dict[str, Any]:
    return {"model_preference": _static_heuristic_pin(row["prompt"]), "accuracy_floor": 0.0}


def arm_always_coder7b(row: dict[str, Any]) -> dict[str, Any]:
    return {"model_preference": CODER7B, "accuracy_floor": 0.0}


def arm_css(row: dict[str, Any]) -> dict[str, Any]:
    # No pin, no floor. The router decides. This is the system under test.
    return {}


ARMS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "always-full": arm_always_full,
    "static-heuristic": arm_static_heuristic,
    "css": arm_css,
    "always-coder7b": arm_always_coder7b,
}

# Tenant per arm keeps RL/cache/budget state from mixing even if a freeze flag
# is ever missed. A leak becomes a visible tenant mismatch instead of silent
# cross-arm contamination.
TENANTS = {name: f"bench-{name}" for name in ARMS}


def expected_pin(arm: str, row: dict[str, Any]) -> str | None:
    """The zoo id this arm asked for, or None when the arm does not pin."""
    return ARMS[arm](row).get("model_preference")

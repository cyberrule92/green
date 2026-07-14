"""Objective answer checkers for the routing benchmark.

Deterministic only: no LLM judge, no API key, no judge variance. A checker that
cannot decide returns False — never a guess. Quality is therefore a *lower*
bound on correctness, applied identically to every arm, which is what a
between-arm comparison needs.

Check types (the `check` object on each prompts.jsonl row):

    numeric       {"value": 21, "tol": 0}         any number in the response matches
    contains_all  {"value": ["a", "b"]}           every string present
    contains_any  {"value": ["a", "b"]}           at least one string present
    regex         {"value": "\\bAu\\b"}            pattern matches
    code_exec     {"value": "assert add(1,2)==3"} extracted code block + asserts run clean

All string matching is case-insensitive unless the check sets
`"case_sensitive": true`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CODE_EXEC_TIMEOUT_S = 15

# 1,240  |  -3.5  |  4e6  |  .5
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def _check_numeric(response: str, check: dict[str, Any]) -> bool:
    target = float(check["value"])
    tol = float(check.get("tol", 0.0))
    return any(abs(n - target) <= tol for n in _numbers(response))


def _check_contains(response: str, check: dict[str, Any], mode: str) -> bool:
    needles = check["value"]
    if isinstance(needles, str):
        needles = [needles]
    hay = response if check.get("case_sensitive") else response.lower()
    hits = [
        (n if check.get("case_sensitive") else n.lower()) in hay
        for n in needles
    ]
    return all(hits) if mode == "all" else any(hits)


def _check_regex(response: str, check: dict[str, Any]) -> bool:
    flags = 0 if check.get("case_sensitive") else re.IGNORECASE
    return re.search(check["value"], response or "", flags) is not None


def _extract_code(response: str) -> str:
    """Prefer a fenced block; fall back to the raw text if the model skipped fences."""
    blocks = _CODE_BLOCK_RE.findall(response or "")
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    return (response or "").strip()


def _check_code_exec(response: str, check: dict[str, Any]) -> bool:
    code = _extract_code(response)
    if not code:
        return False
    program = code + "\n\n" + check["value"] + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                capture_output=True,
                timeout=CODE_EXEC_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
    return proc.returncode == 0


_CHECKERS = {
    "numeric": _check_numeric,
    "contains_all": lambda r, c: _check_contains(r, c, "all"),
    "contains_any": lambda r, c: _check_contains(r, c, "any"),
    "regex": _check_regex,
    "code_exec": _check_code_exec,
}


def score_response(response: str, check: dict[str, Any]) -> bool:
    """Return True when `response` satisfies `check`. Unknown check type → False."""
    checker = _CHECKERS.get((check or {}).get("type", ""))
    if checker is None:
        return False
    try:
        return bool(checker(response or "", check))
    except Exception:
        return False


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


if __name__ == "__main__":
    # Self-test: every check must accept a known-good answer. A checker that
    # cannot pass its own reference answer would silently zero out an arm.
    here = Path(__file__).parent
    prompts = load_prompts(here / "prompts.jsonl")
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate prompt ids"
    print(f"{len(prompts)} prompts, {len({p['category'] for p in prompts})} categories")
    for cat in sorted({p["category"] for p in prompts}):
        n = sum(1 for p in prompts if p["category"] == cat)
        print(f"  {cat:<14} {n:>3}")

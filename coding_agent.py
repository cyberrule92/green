"""
Carbon-aware agentic coding harness (LangGraph).

Why this module exists as a separate path from ``/api/chat``
-----------------------------------------------------------
CSS (``routing_policies.rank_routing_candidates``) optimises carbon *per
request*. An agent is not a request — it is a loop. Its real cost is

    tokens/step  x  steps  x  attempts

which means the greenest-per-token candidate is routinely the *dirtiest* per
completed task: a model that cannot hold a coding task emits broken code, fails
the verifier, and burns the whole step budget without ever converging. The
greenest-feasible invariant still holds, but "feasible" has to be evaluated
against task completion, not against a single forward pass.

So this harness optimises **carbon per successful task completion**:

  * The ladder starts at the greenest *code-capable* candidate
    (Qwen2.5-Coder-1.5B) and never at ultra-light/medium. DialoGPT and TinyLlama
    are not green on this path — they are cheap failures that loop. (This is the
    same trap as the 2026-07-10 code->TinyLlama mis-route, amplified by a loop.)
  * Escalation is **verifier-gated**: we move up the ladder only on hard
    evidence from the sandboxed test run, never speculatively.
  * The task carries a **gCO2eq budget**, not a token budget. Every model call
    debits it against live grid intensity. Exhausting it aborts (or defers via
    EcoServe) rather than silently looping.

Tool protocol
-------------
Small models (1.5B) do not emit well-formed tool-call JSON reliably enough to
drive a graph; a malformed call costs a full retry, which is carbon we spent to
learn nothing. We therefore use a fenced-block protocol (```python path=...```)
which is inside the distribution these models were actually trained on, and
parse it deterministically. Escalated tiers use the same protocol so a task can
cross tiers without changing representation.

LangGraph is the orchestrator. LangChain is deliberately NOT a dependency —
see requirements.txt: pulling it drags in langchain/openai and breaks the
metrics image build. If ``langgraph`` is unavailable at import time we fall back
to an internal executor with identical node semantics, so the image always boots.
"""

from __future__ import annotations

import ast
import builtins
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# LangGraph (optional at runtime — never break the image on a missing dep)
# --------------------------------------------------------------------------

# langgraph pulls langchain-core, which pulls langsmith. Left alone, langsmith
# will happily POST traces (prompts included) to an external endpoint. Off by
# default here: it is unreviewed egress from an on-prem box, and network I/O we
# never asked for is carbon we never accounted for. Set explicitly rather than
# relying on the library default, which has changed between releases.
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

try:  # pragma: no cover - import guard
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    StateGraph = None  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

AGENT_ENABLED = os.getenv("AGENT_ENABLED", "true").lower() == "true"

# Escalation ladder: (model_variant, model_zoo target_id, label).
#
# Ordered greenest-code-capable first. ultra-light/medium are deliberately absent:
# they cannot complete a coding task, so they are not "green" here — they are
# cheap failures that burn the whole step budget and deliver nothing.
#
# Rung 2 is a 4-bit AWQ 7B from the same Qwen-Coder family, NOT the MoE. The MoE
# (Qwen3-30B-A3B) is ~60 GB of fp16 weights and cannot load on this 24 GB vGPU
# slice at any --gpu-memory-utilization; keeping it here would configure a rung
# that can never come up. Override AGENT_LADDER_ENV on a bigger box to restore it.
#
# Format: "variant:zoo_target_id:label,variant:zoo_target_id:label"
_DEFAULT_LADDER = (
    "stem-coding:local-vgpu-stem-coding:Qwen2.5-Coder-1.5B,"
    "coder-7b:local-vgpu-coder-7b:Qwen2.5-Coder-7B-AWQ"
)


def _parse_ladder(spec: str) -> list[tuple[str, str, str]]:
    rungs: list[tuple[str, str, str]] = []
    for chunk in spec.split(","):
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) == 3 and all(parts):
            rungs.append((parts[0], parts[1], parts[2]))
        elif chunk.strip():
            logger.warning("agent: ignoring malformed ladder rung %r", chunk)
    return rungs


# `or _DEFAULT_LADDER`, not just a getenv default: docker-compose passes
# AGENT_LADDER through as an empty string when the operator hasn't set it, and
# getenv would hand back "" — parsing to an empty ladder and taking the harness
# down. An empty or fully-malformed spec falls back to the default.
AGENT_LADDER: list[tuple[str, str, str]] = (
    _parse_ladder(os.getenv("AGENT_LADDER", "").strip() or _DEFAULT_LADDER)
    or _parse_ladder(_DEFAULT_LADDER)
)

# Attempts allowed on each rung before escalating. One retry on the green rung
# is worth it (a failed test often just needs the traceback fed back); a second
# retry on a model that has already failed twice is carbon spent on a coin flip.
AGENT_ATTEMPTS_PER_TIER = int(os.getenv("AGENT_ATTEMPTS_PER_TIER", "2"))

# Total gCO2eq a single task may emit before we abort. Sized so a runaway loop is
# bounded, but a legitimate full ladder traversal is not.
#
# Calibrated against a MEASURED run, not guessed. On the first real end-to-end task
# (fizzbuzz, IN-WE grid @ 502 gCO2/kWh) a full escalating traversal cost 42.95 g:
#   2 x Qwen2.5-Coder-1.5B  ->  5.04 g + 3.10 g   (2.9 s, 1.8 s)
#   1 x Qwen2.5-Coder-7B    -> 34.81 g            (38.7 s)
# model_zoo's carbon is dominated by *embodied* carbon, which scales with
# wall-clock duration — and agent calls run for tens of seconds, orders of
# magnitude longer than the zoo's latency_ms_p50 assumes. So a 30 g budget would
# have falsely aborted that perfectly legitimate task. 60 g permits one full
# traversal with headroom while still catching a genuine runaway.
#
# Corollary worth keeping: since duration dominates, a model that loops is
# penalised by construction — exactly the property this harness wants.
AGENT_CARBON_BUDGET_G = float(os.getenv("AGENT_CARBON_BUDGET_G", "60.0"))

# Expected wall-clock seconds for one call on an un-sampled rung, used to
# pre-cost an escalation. The budget is only ever checked BETWEEN calls, so
# without a forward estimate a single 38-second call on the big rung can sail
# straight past the cap. Seeded from the measured 7B call above; once a rung has
# been sampled in this task we use its observed duration instead.
AGENT_EST_CALL_S = float(os.getenv("AGENT_EST_CALL_S", "40.0"))

# Grid intensity above which we do not *start* a new agent task inline. Agent
# tasks are long, escalation-prone, and rarely latency-critical — they are the
# ideal EcoServe deferral candidate. Above this the task is queued, not dropped.
AGENT_DEFER_CI = float(os.getenv("AGENT_DEFER_CI", "400.0"))

# How long a deferred task may wait for a greener window. The queue dispatches at
# the lowest-carbon point the forecast offers inside this budget, and dispatches
# regardless once it expires — deferral must not become starvation. Six hours
# comfortably spans a solar/wind swing without leaving a task pending overnight.
AGENT_DEFERRAL_MS = int(os.getenv("AGENT_DEFERRAL_MS", str(6 * 60 * 60 * 1000)))

# Completed/queued task records kept in memory for status polling.
AGENT_TASK_HISTORY = int(os.getenv("AGENT_TASK_HISTORY", "100"))

AGENT_TEST_TIMEOUT_S = int(os.getenv("AGENT_TEST_TIMEOUT_S", "60"))
AGENT_MAX_FILE_BYTES = int(os.getenv("AGENT_MAX_FILE_BYTES", "131072"))

# Read timeout for the agent's own model calls, overriding VLLM_TIMEOUT_SECONDS (45 s,
# tuned for chat, where a human is waiting). Nobody is waiting on an agent call: the
# task already runs for minutes and may have been deferred for hours. Measured: the 7B
# coder needs longer than 45 s to write a complete file, so it kept tripping the chat
# timeout — and a timed-out call still burns the GPU, so the agent was billed ~16 gCO2
# for an empty response and then escalated on the "evidence". Waiting is strictly
# cheaper than paying for the tokens twice.
AGENT_LLM_TIMEOUT_S = int(os.getenv("AGENT_LLM_TIMEOUT_S", "180"))

# Sandboxes live in the system tmpdir, NOT under data/. Two reasons: the API
# container runs as an unprivileged user that cannot write to the mounted data
# volume, and model-authored code has no business landing in the same directory
# as the audit trail and the SQLite store. These workspaces are ephemeral by
# design — nothing here is meant to survive the task.
_WORKSPACE_ROOT = Path(
    os.getenv("AGENT_WORKSPACE_ROOT", str(Path(tempfile.gettempdir()) / "green-agent-workspaces"))
)


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    task_id: str
    task: str
    workspace: str

    plan: str
    files: dict[str, str]          # relative path -> source
    test_command: str

    test_output: str
    test_passed: bool
    harness_ok: bool               # False = verifier itself broke; never escalate on this
    backend_failed: bool           # True = the model endpoint returned nothing; also never escalate
    spec_feedback: str             # why the last test file was rejected; survives node_verify
    spec_locked: bool              # True = the caller supplied the tests; the model may not write any

    tier_idx: int                  # index into AGENT_LADDER
    attempts_on_tier: int
    total_llm_calls: int

    carbon_g: float
    budget_g: float
    grid_ci: float
    observed_duration_s: dict[str, float]   # zoo target_id -> last measured call duration

    status: Literal[
        "running", "completed", "failed", "budget_exceeded", "deferred",
        "harness_error", "escalation_unavailable",
    ]
    events: list[dict[str, Any]]


# --------------------------------------------------------------------------
# Sandbox tools
# --------------------------------------------------------------------------

def _safe_join(workspace: str, rel_path: str) -> Path:
    """Resolve rel_path inside workspace, refusing traversal escapes."""
    root = Path(workspace).resolve()
    target = (root / rel_path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path escapes workspace: {rel_path}")
    return target


def tool_write_file(workspace: str, rel_path: str, content: str) -> str:
    if len(content.encode("utf-8")) > AGENT_MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {AGENT_MAX_FILE_BYTES} bytes: {rel_path}")
    target = _safe_join(workspace, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def tool_read_file(workspace: str, rel_path: str) -> str:
    return _safe_join(workspace, rel_path).read_text(encoding="utf-8")


def tool_list_files(workspace: str) -> list[str]:
    root = Path(workspace).resolve()
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )


# Secrets that must never reach model-authored code running in the sandbox.
_ENV_DENYLIST = {
    "EMAP_TOKEN", "AUDIT_HMAC_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "NIM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY",
}


def tool_run_tests(workspace: str, command: str) -> tuple[bool, str, bool]:
    """
    Run the verifier in the sandbox.

    Returns ``(passed, output, harness_ok)``.

    ``harness_ok`` is the distinction that makes verifier-gated escalation safe.
    A non-zero exit can mean two very different things:

      * the model's code is wrong          -> real evidence, escalate
      * pytest isn't installed / not found -> our problem, NOT the model's

    Conflating them is expensive in exactly the direction this module exists to
    avoid: a broken runner would fail every attempt on the green rung and push
    every task onto the 30B MoE, burning maximum carbon to learn nothing. So an
    infrastructure failure aborts the task instead of escalating it.

    Env is inherited (a stripped PATH is what caused false failures) minus a
    secret denylist, since the code being executed here was written by a model.
    """
    env = {k: v for k, v in os.environ.items() if k not in _ENV_DENYLIST}
    env["HOME"] = workspace
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Ensure `import solution` resolves to the model's file, not a stray module.
    env["PYTHONPATH"] = workspace

    # Bind bare `python`/`python3` to the interpreter actually running us, so the
    # verifier can't miss pytest just because PATH points at a different install.
    resolved = re.sub(r"^\s*python3?\b", sys.executable, command)

    try:
        proc = subprocess.run(
            resolved,
            shell=True,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=AGENT_TEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # A hang IS the model's fault — real evidence, harness is fine.
        return False, f"TIMEOUT after {AGENT_TEST_TIMEOUT_S}s — code did not terminate.", True
    except Exception as exc:  # pragma: no cover
        return False, f"runner error: {exc}", False

    output = ((proc.stdout or "") + (proc.stderr or ""))[-8000:]

    # rc 127 = command not found; a missing pytest module is our packaging gap,
    # not a bad solution.
    harness_broken = proc.returncode == 127 or "No module named pytest" in output
    if harness_broken:
        return False, output, False

    # pytest rc 5 = "no tests collected". The harness is fine — the model simply
    # did not produce a test file (small models like to cram both files into one
    # fenced block). Say so explicitly, or the repair loop just sees a bare exit
    # code and keeps rewriting an implementation that was never the problem.
    if proc.returncode == 5 or "no tests ran" in output.lower():
        return False, (
            "AGENT: no tests were collected. You did not emit a test file.\n"
            "Emit the tests as their OWN fenced block: ```python path=test_solution.py\n"
            "Do not put the tests inside solution.py.\n\n" + output
        ), True

    return proc.returncode == 0, output, True


# --------------------------------------------------------------------------
# Fenced-block protocol
# --------------------------------------------------------------------------

# The path header, in every shape the rungs actually emit. The comment marker and the
# `path=` marker are INDEPENDENTLY optional because the 1.5B combines them — it answered
# ```python\n# path=solution.py, which an either/or alternation rejects outright. That
# cost a whole rung: 4033 chars of correct is_palindrome went in the bin as "no parseable
# code blocks", and the ladder escalated over a punctuation mismatch in our own protocol.
_PATH_HEADER = r"(?:#\s*)?(?:path\s*[=:]\s*)?(?P<path>[\w./-]+\.[\w]+)"

_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+-]*)\s*" + _PATH_HEADER + r"?\s*\n(?P<body>.*?)```",
    re.DOTALL,
)

# Same shape, but running to end-of-string with no closing fence — a file the model
# was still writing when it hit max_tokens. The path is mandatory here: an unlabelled
# truncated block is not worth guessing at.
_UNCLOSED_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+-]*)\s*" + _PATH_HEADER + r"\s*\n(?P<body>(?:(?!```)[\s\S])*)\Z"
)

# A file header sitting INSIDE a block body, on a line of its own.
_INLINE_HEADER_RE = re.compile(
    r"^[ \t]*#[ \t]*(?:path\s*[=:]\s*)?(?P<path>[\w./-]+\.py)[ \t]*$", re.M
)


def _split_embedded_files(path: str, body: str) -> dict[str, str]:
    """
    Split one block body into the several files a model crammed into it.

    The 1.5B rung ignores "one block per file" and emits the implementation and the
    tests inside a single fence, separated by a `# test_solution.py` comment. The
    whole thing then lands in solution.py — where pytest does not look for tests — so
    the verifier reports "no tests collected" forever while the tests sit right there
    in the file. Both rungs got burned on this. The model's intent is unambiguous, so
    honour it: a lone `# <name>.py` line starts a new file.
    """
    marks = list(_INLINE_HEADER_RE.finditer(body))
    if not marks:
        return {path: body}

    files: dict[str, str] = {}
    head = body[: marks[0].start()].strip()
    if head:
        files[path] = head + "\n"
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        segment = body[mark.end():end].strip()
        if segment:
            files[mark.group("path")] = segment + "\n"
    return files


def parse_code_blocks(text: str) -> dict[str, str]:
    """
    Extract {path: source} from a model response.

    Accepts ```python path=foo.py, ```python foo.py, and ```py # foo.py.
    Unlabelled blocks are only used when they are the sole block (the model
    answered with one file and forgot the header) — otherwise we would happily
    overwrite the wrong file with high confidence, which is worse than failing.
    """
    blocks = list(_BLOCK_RE.finditer(text))
    files: dict[str, str] = {}
    for m in blocks:
        path, body = m.group("path"), m.group("body")
        if path:
            files.update(_split_embedded_files(path, body.rstrip() + "\n"))

    # Recover a trailing block that never got its closing fence.
    #
    # Small models run out of max_tokens mid-file — measured: the 1.5B degenerated
    # into fifty near-identical asserts and was cut off mid-line, so its second block
    # never closed and _BLOCK_RE dropped it whole. We had already paid the carbon for
    # those tokens; throwing the file away guarantees another rung burns more. Only a
    # block with an EXPLICIT path is recovered — a truncated body is still judged by
    # the verifier, and a truncated test file by invalid_test_reason, so admitting it
    # risks nothing that was not already checked.
    tail = text[blocks[-1].end():] if blocks else text
    unclosed = _UNCLOSED_BLOCK_RE.search(tail)
    if unclosed and unclosed.group("path") not in files:
        recovered = _split_embedded_files(
            unclosed.group("path"), unclosed.group("body").rstrip() + "\n"
        )
        for path, src in recovered.items():
            files.setdefault(path, src)

    if not files and len(blocks) == 1:
        files.update(
            _split_embedded_files("solution.py", blocks[0].group("body").rstrip() + "\n")
        )
    return files


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_CODE_PROMPT = """You are a coding agent. Solve the task by writing complete files.

TASK:
{task}

Rules:
- Output ONLY fenced code blocks, one per file. Do NOT put two files in one block.
- Every block MUST start with its path, like: ```python path=solution.py
- Write the implementation in solution.py and the tests in a SEPARATE block
  path=test_solution.py, runnable with: {test_command}
- The test file MUST import what it tests: `from solution import <name>`.
- The tests are the specification and will be FROZEN after this step — you will
  not be allowed to change them later. Write them to cover the edge cases that
  actually distinguish a correct implementation from a plausible-looking wrong
  one (boundaries, overlapping conditions, empty input), not just the easy cases.
- Keep the tests SHORT — under a dozen assertions, each one distinct. Never repeat
  near-identical asserts: the response is cut off at the token limit and a file that
  ends mid-line is thrown away.
- Code must be complete and runnable. No placeholders, no TODOs, no ellipses.
"""

# The caller supplied the tests with the task, so the model never authors the spec.
#
# This closes the design's one real hole: the ladder's WEAKEST rung was writing the
# frozen specification. Observed live on word_count — the 1.5B produced a perfectly
# well-formed test asserting `{"hello": 2, "world": 2}` for "Hello world! Hello
# again." ("world" appears once; "again" is missing entirely), it froze, and a
# correct 7B implementation was then condemned by it for the rest of the ladder. No
# static gate can catch that: the file parses, imports, and tests things — it is
# simply wrong about the task. The only fix is to not let the model define truth.
_SPEC_PROMPT = """You are a coding agent. The tests are already written and are the
SPECIFICATION. Write the implementation that makes them pass.

TASK:
{task}

THE TESTS (frozen — you may not modify them, and any test file you emit is discarded):
{tests}

TEST COMMAND: {test_command}

Rules:
- Output ONLY fenced code blocks, one per file. Do NOT put two files in one block.
- Every block MUST start with its path, like: ```python path=solution.py
- Write ONLY the implementation. Do NOT emit a test file.
- Read the tests carefully: they define the exact names, signatures and return
  values you must provide. Match them exactly — do not guess a nicer API.
- Code must be complete and runnable. No placeholders, no TODOs, no ellipses.
"""

_REPAIR_PROMPT = """You are a coding agent. Your previous solution FAILED its tests.

TASK:
{task}

YOUR CURRENT FILES:
{files}

TEST COMMAND: {test_command}
TEST OUTPUT (this is the ground truth — fix what it says):
{test_output}

Rules:
- The TESTS ARE FROZEN. They are the specification. You may NOT modify, weaken,
  delete or rewrite any test file. Any test file you emit will be discarded.
- Fix the IMPLEMENTATION so the existing tests pass. Do not change the tests to
  match your code — change your code to match the tests.
- Output ONLY fenced code blocks, one per file, for implementation files you are
  CHANGING. Every block MUST start with its path, like: ```python path=solution.py
- Fix the actual failure shown above. Do not rewrite unrelated code.
"""


# The implementation landed but no test file did, so pytest collected nothing.
#
# This case CANNOT use the repair prompt: that prompt's rules say "the tests are
# frozen, any test file you emit will be discarded", which is the exact opposite of
# what we need here. Sent that, the models dutifully obeyed the rules and rewrote
# solution.py again — twice on the 1.5B, twice on the 7B, four LLM calls and the
# whole ladder burned without anyone ever writing the tests the verifier was waiting
# for. A prompt that contradicts its own instructions gets obeyed in the wrong half.
_TESTS_PROMPT = """You are a coding agent. The implementation exists, but there are NO TESTS,
so the test command collected nothing and cannot tell us whether the code is right.
Write the tests, and nothing else.

TASK:
{task}

YOUR CURRENT FILES:
{files}

TEST COMMAND: {test_command}
{feedback}
Rules:
- Output EXACTLY ONE fenced block: ```python path=test_solution.py
- It MUST import what it tests from the implementation module, e.g.
  `from solution import fizzbuzz`. A test file that calls the function without
  importing it fails with NameError and proves nothing.
- Write test_* functions asserting the behaviour THE TASK describes — not the
  behaviour the current code happens to have. These tests become the specification.
- Keep it SHORT: a handful of assertions. Do not repeat near-identical asserts —
  the file gets cut off at the token limit and is then unusable.
- Do NOT rewrite the implementation in this response.
"""


# Prompt budget. The repair prompt is the biggest thing we send, and blowing the
# backend's max-model-len is not a soft failure: vLLM returns 400 and
# run_vllm_inference escalates to a different variant, so the repair would never
# run on the rung we chose. Keep files + test output well inside the 8192-token
# window even after tokenizer drift on code and unicode.
_MAX_FILES_CHARS = 3000
_MAX_TEST_OUTPUT_CHARS = 2000


def _render_files(files: dict[str, str], limit: int = _MAX_FILES_CHARS) -> str:
    out = []
    for path, src in files.items():
        out.append(f"--- {path} ---\n{src}")
    joined = "\n".join(out)
    return joined[:limit]


def _trim_test_output(output: str, limit: int = _MAX_TEST_OUTPUT_CHARS) -> str:
    """
    Keep the TAIL of the test output. pytest puts the banner, collection notes and
    plugin list up front and the actual assertion + traceback at the end — the head
    is the part with no diagnostic value.
    """
    text = output or ""
    if len(text) <= limit:
        return text
    return "...(truncated)...\n" + text[-limit:]


# --------------------------------------------------------------------------
# Carbon accounting
# --------------------------------------------------------------------------

def _debit_carbon(
    state: AgentState,
    target_id: str,
    duration_s: float,
    token_count: int,
) -> float:
    """Charge one model call to the task's carbon budget (LLMCarbon, live CI)."""
    try:
        from model_zoo import get_model_zoo

        breakdown = get_model_zoo().compute_total_carbon(
            model_id=target_id,
            grid_carbon_g_per_kwh=state.get("grid_ci", 400.0),
            inference_duration_s=duration_s,
            token_count=token_count,
        )
        cost = float(breakdown.get("total_carbon_g", 0.0))
    except Exception as exc:  # pragma: no cover
        logger.warning("agent: carbon accounting failed for %s: %s", target_id, exc)
        cost = 0.0

    state["carbon_g"] = round(state.get("carbon_g", 0.0) + cost, 8)
    return cost


def _target_for_variant(variant: str) -> str | None:
    """
    Map a model_variant back to its zoo target id, so carbon can be billed to
    whatever run_vllm_inference actually served rather than what we asked for.
    Read from the zoo (not hardcoded) so it can't drift from config/model_zoo.json.
    """
    try:
        from model_zoo import get_model_zoo

        for entry in get_model_zoo().list_models():
            if entry.get("model_variant") == variant and entry.get("region") == "local":
                return entry.get("id")
    except Exception:  # pragma: no cover
        logger.warning("agent: could not resolve zoo target for variant %s", variant)
    return None


_TEST_FILE_RE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test)\.py$|(^|/)conftest\.py$")


def is_test_file(path: str) -> bool:
    return bool(_TEST_FILE_RE.search(path))


def _unimported_names(tree: ast.Module, tests: list[Any]) -> list[str]:
    """
    Names the test functions read but the file never binds.

    Deliberately conservative — ANY binding anywhere in the file (import, def,
    class, assignment, argument) counts, and a star-import is treated as binding
    everything. A false positive here rejects a valid spec, which is worse than
    missing a bad one.
    """
    bound: set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return []
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)

    used: set[str] = set()
    for fn in tests:
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
    return sorted(used - bound)


def invalid_test_reason(src: str) -> Optional[str]:
    """
    Why `src` cannot serve as a spec, or None if it can.

    A test file is frozen the moment it lands (see node_generate), so it has to
    earn that status first: it must parse, it must actually contain tests, and it
    must be able to reach the code it is testing.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return f"not valid Python: {exc.msg} (line {exc.lineno})"

    tests = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    if not tests:
        return "defines no test_* function, so pytest collects nothing from it"

    # Seen live: a syntactically perfect test file that called fizzbuzz() without
    # importing it. Every assertion died on NameError, and because the tests were
    # already frozen nobody was allowed to add the missing import — a correct
    # implementation was reported as a failure. A spec that cannot reach the code
    # under test is not testing it.
    missing = _unimported_names(tree, tests)
    if missing:
        names = ", ".join(f"`{n}`" for n in missing)
        return f"calls {names} but never imports it, so every test dies on NameError"

    # A spec that repeats itself is a model that has started looping, and the loop is
    # not free: it fills the token budget, truncates the file mid-line, and — because
    # the tests are frozen the moment they land — pins the whole task to whatever
    # nonsense it emitted before it started repeating. Measured: word_count came back
    # with thirty byte-identical whitespace assertions and one wrong count, and a
    # perfectly correct implementation was condemned by a spec nobody was allowed to
    # touch. Distinct assertions are cheap; duplicated ones are worse than useless.
    asserts = [
        ast.dump(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]
    if len(asserts) >= 6 and len(set(asserts)) * 2 < len(asserts):
        return (
            f"repeats the same assertion ({len(asserts)} asserts, only "
            f"{len(set(asserts))} distinct) — write a handful of DISTINCT assertions"
        )
    return None


def normalize_caller_tests(tests: str | dict[str, str] | None) -> dict[str, str]:
    """
    Validate and normalise a caller-supplied test suite into {path: source}.

    Raises ValueError if the suite could not serve as a spec — and the caller sees
    that error *synchronously*, before a single token is spent. This is the whole
    point of the feature: a bad spec from the model costs a full ladder to discover
    (it fails every implementation, and the freeze forbids fixing it), whereas a bad
    spec from the caller is caught here for free. Same gate as the model's own tests
    (`invalid_test_reason`); no exemption for humans, who forget the import too.
    """
    if tests is None:
        return {}
    if isinstance(tests, str):
        tests = {"test_solution.py": tests} if tests.strip() else {}
    if not isinstance(tests, dict):
        raise ValueError("tests must be a string or a {path: source} object")

    normalized: dict[str, str] = {}
    for raw_path, src in tests.items():
        path = str(raw_path).strip()
        if not isinstance(src, str) or not src.strip():
            raise ValueError(f"{path}: test source is empty")
        if not is_test_file(path):
            raise ValueError(
                f"{path}: caller tests must be named so pytest collects them (test_*.py)"
            )
        # Reuse the sandbox's traversal check: these paths are written to disk.
        _safe_join(tempfile.gettempdir(), path)
        reason = invalid_test_reason(src)
        if reason is not None:
            raise ValueError(f"{path}: {reason}")
        normalized[path] = src

    return normalized


def _emit(state: AgentState, kind: str, **fields: Any) -> None:
    state.setdefault("events", []).append(
        {"t": round(time.time(), 3), "event": kind, **fields}
    )


# --------------------------------------------------------------------------
# Model call
# --------------------------------------------------------------------------

# Injected by decision_engine at wiring time to avoid a circular import.
_inference_fn: Optional[Callable[..., tuple[str, str]]] = None
_liveness_fn: Optional[Callable[..., bool]] = None


def set_inference_fn(
    fn: Callable[..., tuple[str, str]],
    liveness_fn: Callable[..., bool] | None = None,
) -> None:
    """
    Bind ``decision_engine.run_vllm_inference`` (+ ``_is_vllm_live``) into the harness.

    ``fn`` must accept ``allow_rule_based_fallback``, ``target_id`` and ``timeout_s``
    keywords — the agent overrides the chat read timeout on every call.
    """
    global _inference_fn, _liveness_fn
    _inference_fn = fn
    _liveness_fn = liveness_fn


def _rung_is_live(tier_idx: int) -> bool:
    """
    Is this rung's backend actually up?

    Without this check, escalating into a container that isn't running is worse
    than useless: run_vllm_inference would fail to connect and silently escalate
    to whatever generic model backs the next variant, so the agent would burn a
    general instruct model while reporting that it escalated to the 7B coder.
    Unknown (no liveness fn bound) is treated as live — we do not want a probe
    failure to block a rung that would have worked.
    """
    if _liveness_fn is None or tier_idx >= len(AGENT_LADDER):
        return True
    variant, target_id, _ = AGENT_LADDER[tier_idx]
    try:
        return bool(_liveness_fn(variant, target_id))
    except Exception:  # pragma: no cover
        logger.warning("agent: liveness probe failed for %s; assuming live", variant)
        return True


def _call_model(state: AgentState, prompt: str) -> str:
    if _inference_fn is None:
        raise RuntimeError("coding_agent: inference fn not bound (call set_inference_fn)")

    variant, target_id, label = AGENT_LADDER[state["tier_idx"]]
    started = time.time()
    # allow_rule_based_fallback=False: a canned rule-based string is worthless to
    # an agent loop and would be parsed as "no code blocks" -> wasted escalation.
    text, actual_variant = _inference_fn(
        variant,
        prompt,
        allow_rule_based_fallback=False,
        target_id=target_id,
        timeout_s=AGENT_LLM_TIMEOUT_S,
    )
    duration = time.time() - started

    # run_vllm_inference escalates INTERNALLY when an endpoint is dead or the
    # context overflows, and reports what it actually served via actual_variant.
    # Billing the requested target would then understate the real cost — the
    # agent would quietly burn `full`/`moe` while the audit trail said
    # `stem-coding`. Carbon must be charged against what actually ran, or the
    # budget stops bounding anything and the CSRD figures are simply wrong.
    billed_target = target_id
    if actual_variant and actual_variant != variant:
        billed_target = _target_for_variant(actual_variant) or target_id
        _emit(
            state,
            "backend_substituted",
            requested=variant,
            served=actual_variant,
            billed=billed_target,
            note="run_vllm_inference escalated internally; carbon billed to the served model",
        )

    tokens = max(1, len(text) // 4)
    cost = _debit_carbon(state, billed_target, duration, tokens)
    state["total_llm_calls"] = state.get("total_llm_calls", 0) + 1
    # Feed the real duration back so _estimate_call_carbon stops guessing once a
    # rung has actually been sampled in this task.
    state.setdefault("observed_duration_s", {})[billed_target] = duration

    _emit(
        state,
        "llm_call",
        tier=label,
        variant=actual_variant,
        billed_target=billed_target,
        duration_s=round(duration, 3),
        tokens=tokens,
        carbon_g=round(cost, 6),
        cumulative_carbon_g=state["carbon_g"],
    )
    return text or ""


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------

def node_generate(state: AgentState) -> AgentState:
    """Write (or repair) the solution files."""
    if state.get("files") and state.get("test_output"):
        if any(is_test_file(p) for p in state["files"]):
            prompt = _REPAIR_PROMPT.format(
                task=state["task"],
                files=_render_files(state["files"]),
                test_command=state["test_command"],
                test_output=_trim_test_output(state["test_output"]),
            )
            mode = "repair"
        else:
            # No spec yet — ask for the tests with a prompt that actually wants them.
            rejected = state.get("spec_feedback") or ""
            prompt = _TESTS_PROMPT.format(
                task=state["task"],
                files=_render_files(state["files"]),
                test_command=state["test_command"],
                feedback=(
                    f"\nYOUR LAST TEST FILE WAS REJECTED — {rejected}\nFix exactly that.\n"
                    if rejected else ""
                ),
            )
            mode = "write_tests"
    elif state.get("spec_locked"):
        # Caller-supplied tests: they are already on disk, so the very first call
        # asks for the implementation alone.
        prompt = _SPEC_PROMPT.format(
            task=state["task"],
            tests=_render_files(
                {p: s for p, s in (state.get("files") or {}).items() if is_test_file(p)}
            ),
            test_command=state["test_command"],
        )
        mode = "generate"
    else:
        prompt = _CODE_PROMPT.format(
            task=state["task"], test_command=state["test_command"]
        )
        mode = "generate"

    response = _call_model(state, prompt)

    if not response.strip():
        # An EMPTY response is the backend failing, not the model failing the task:
        # run_vllm_inference returns "" once its timeout and internal retries are
        # exhausted. Seen live on a deferred run — the 7B rung timed out twice at
        # 182 s, and because "no parseable blocks" was scored as verifier evidence,
        # the ladder cheerfully escalated its way through 31 gCO2 of nothing. A dead
        # endpoint tells us nothing about the model, so it gets the same treatment as
        # pytest's rc-127: abort, do not escalate, do not spend another rung on it.
        state["backend_failed"] = True
        state["test_passed"] = False
        state["test_output"] = (
            "AGENT: the model backend returned an empty response (timed out or "
            f"unavailable) on rung {AGENT_LADDER[state['tier_idx']][2]}."
        )
        _emit(
            state,
            "backend_empty",
            mode=mode,
            tier=AGENT_LADDER[state["tier_idx"]][2],
            detail="empty response from run_vllm_inference; treating as infrastructure failure",
        )
        return state

    new_files = parse_code_blocks(response)

    if not new_files:
        # Malformed output *is* a verifier failure — the model could not even
        # produce parseable files. Record it so the router escalates on evidence.
        state["test_passed"] = False
        state["test_output"] = (
            "AGENT: model produced no parseable code blocks. "
            "Expected fenced blocks with a path header."
        )
        # Keep the head of the response. Without it a parse failure is unfalsifiable —
        # you know the model said something and that we could not use it, which is
        # exactly the information needed to tell a prompt bug from a model that rambles.
        _emit(
            state,
            "parse_failed",
            mode=mode,
            response_chars=len(response),
            response_head=response[:300],
        )
        return state

    # Caller-supplied spec: the model does not get a vote on what "correct" means,
    # in any mode. Dropping its test files here (rather than in the freeze block
    # below, which only guards overwrites) also stops it inventing a *second* test
    # file that asserts the behaviour its own buggy code happens to have — pytest
    # would run that too, and a spec you can extend is a spec you can hack.
    if state.get("spec_locked"):
        rejected = sorted(p for p in new_files if is_test_file(p))
        if rejected:
            for path in rejected:
                new_files.pop(path, None)
            _emit(
                state,
                "test_write_rejected",
                files=rejected,
                reason="tests were supplied with the task and are the spec; write only the implementation",
            )
        if not new_files:
            state["test_passed"] = False
            state["test_output"] = (
                "AGENT: you emitted only test files. The tests were supplied with the "
                "task and are frozen. Emit the IMPLEMENTATION that makes them pass, as "
                "a fenced block ```python path=solution.py\n\n"
                + (state.get("test_output") or "")
            )
            return state

    # A test file is about to become immutable, so first make sure it IS a test.
    #
    # Observed live: asked for the missing tests, the 1.5B echoed the repair prompt
    # back inside a ```test_solution.py block. The freeze only blocks OVERWRITES, so
    # this junk was accepted as a new file — and from that point pytest died in
    # collection on a file no repair was permitted to touch. The 7B correctly tried
    # to replace it and was refused; the ladder burned its whole budget on a state it
    # was forbidden to fix. Junk is rejected at the gate instead, so nothing that
    # cannot serve as a spec ever earns the protection of being one.
    junk_tests = {
        p: reason
        for p, src in new_files.items()
        if is_test_file(p) and (reason := invalid_test_reason(src))
    }
    if junk_tests:
        for path in junk_tests:
            new_files.pop(path, None)
        why = "; ".join(f"{p}: {r}" for p, r in sorted(junk_tests.items()))
        _emit(state, "test_write_rejected", files=sorted(junk_tests), reason=why)

        # Tell the model WHY on its next turn. Without this the rejection is silent:
        # the implementation file in the same response still lands, so pytest runs and
        # reports its generic "no tests were collected", the model re-emits the very
        # same broken test file, and the ladder grinds through both rungs rejecting it
        # over and over. Measured: three identical rejections in one run. A gate that
        # cannot explain itself is just an expensive way to fail.
        state["spec_feedback"] = why

        if not new_files:
            # Everything it emitted was an unusable test file (an earlier bail already
            # covered "no blocks at all"). Name the fault: a bare failure would send
            # the repair loop after an implementation that was never the problem.
            state["test_passed"] = False
            state["test_output"] = (
                f"AGENT: your test file was rejected — {why}\n"
                "Emit a real pytest file: a fenced block ```python path=test_solution.py "
                "containing test_* functions that import from solution.py."
            )
            return state
    elif any(is_test_file(p) for p in new_files):
        state["spec_feedback"] = ""   # a good spec landed; stop replaying the complaint

    # THE TESTS ARE THE SPEC — freeze them once they exist.
    #
    # Observed live on the first real run: the 1.5B rung failed its tests twice,
    # we escalated, and the 7B "fixed" the task by rewriting test_solution.py to
    # assert the broken behaviour instead of repairing fizzbuzz(). Tests went
    # green, the task reported `completed`, and the shipped code still returned
    # "Fizz" for 15. That is reward hacking, and it silently invalidates the whole
    # design: escalation is only meaningful if the verifier is ground truth the
    # model cannot edit. So after the initial generate, test files are immutable —
    # a repair may only touch the implementation.
    if mode == "repair":
        existing = state.get("files") or {}
        # Freeze OVERWRITES of tests that already exist — that is the reward-hack
        # path. Creating a NEW test file is allowed: the model may still owe us one
        # (small models often cram tests into solution.py, so pytest collects
        # nothing), and an added test cannot make an existing failing test pass —
        # pytest runs them all.
        blocked = {
            p: src for p, src in new_files.items()
            if is_test_file(p) and p in existing
        }
        new_files = {p: src for p, src in new_files.items() if p not in blocked}
        if blocked:
            _emit(
                state,
                "test_write_rejected",
                files=sorted(blocked),
                reason="tests are frozen once written; repair the implementation, not the spec",
            )
        if not new_files:
            # It tried to rewrite the tests and nothing else. No progress — count it
            # as a failed attempt rather than looping on the same move.
            state["test_passed"] = False
            state["test_output"] = (
                "AGENT: your previous response only modified existing test files, which "
                "are frozen. You MUST fix the implementation so the existing tests pass.\n\n"
                + (state.get("test_output") or "")
            )
            return state

    merged = dict(state.get("files") or {})
    merged.update(new_files)
    state["files"] = merged

    for path, src in new_files.items():
        try:
            tool_write_file(state["workspace"], path, src)
        except ValueError as exc:
            _emit(state, "write_rejected", path=path, reason=str(exc))

    _emit(state, "wrote_files", mode=mode, files=sorted(new_files))
    return state


def node_verify(state: AgentState) -> AgentState:
    """Run the tests. This is ground truth and the only thing that gates escalation."""
    if not state.get("files"):
        # Nothing to run: the model emitted no parseable files. That is the
        # model's failure, not the harness's — but the attempt still has to be
        # counted, or the router keeps seeing attempts=0 and retries forever.
        state["test_passed"] = False
        state["harness_ok"] = True
        state["attempts_on_tier"] = state.get("attempts_on_tier", 0) + 1
        _emit(
            state,
            "verify",
            passed=False,
            harness_ok=True,
            tier=AGENT_LADDER[state["tier_idx"]][2],
            attempt=state["attempts_on_tier"],
            note="no files to test",
        )
        return state

    passed, output, harness_ok = tool_run_tests(state["workspace"], state["test_command"])
    state["test_passed"] = passed
    state["test_output"] = output
    state["harness_ok"] = harness_ok
    state["attempts_on_tier"] = state.get("attempts_on_tier", 0) + 1

    _emit(
        state,
        "verify",
        passed=passed,
        harness_ok=harness_ok,
        tier=AGENT_LADDER[state["tier_idx"]][2],
        attempt=state["attempts_on_tier"],
    )
    return state


def node_escalate(state: AgentState) -> AgentState:
    """Move one rung up the ladder. Only ever reached on verifier evidence."""
    prev = AGENT_LADDER[state["tier_idx"]][2]
    state["tier_idx"] += 1
    state["attempts_on_tier"] = 0
    _emit(
        state,
        "escalate",
        reason="verifier_failed",
        from_tier=prev,
        to_tier=AGENT_LADDER[state["tier_idx"]][2],
        carbon_so_far_g=state["carbon_g"],
    )
    return state


def _estimate_call_carbon(state: AgentState, tier_idx: int) -> float:
    """Forward-cost one call on ``tier_idx`` so we can refuse an escalation we cannot afford."""
    if tier_idx >= len(AGENT_LADDER):
        return 0.0
    _, target_id, _ = AGENT_LADDER[tier_idx]
    est_s = (state.get("observed_duration_s") or {}).get(target_id, AGENT_EST_CALL_S)
    try:
        from model_zoo import get_model_zoo

        breakdown = get_model_zoo().compute_total_carbon(
            model_id=target_id,
            grid_carbon_g_per_kwh=state.get("grid_ci", 400.0),
            inference_duration_s=est_s,
            token_count=600,
        )
        return float(breakdown.get("total_carbon_g", 0.0))
    except Exception:  # pragma: no cover
        return 0.0


def classify(state: AgentState) -> str:
    """
    The carbon-aware control decision. PURE — it must not mutate state.

    LangGraph conditional-edge functions are routers, not nodes: whatever they
    write to state is discarded. So this function only *decides*, and
    ``node_finalize`` is the real node that records the outcome. Keeping both on
    one classifier means the decision and the recorded status can never drift.

    Order matters:
      * a broken verifier is checked first — it is not evidence about the model
        and must never trigger an escalation;
      * budget is checked before escalation, because escalating into an
        exhausted budget is the most expensive move this graph can make.
    """
    if state.get("test_passed"):
        return "completed"

    # Checked independently of harness_ok, which node_verify overwrites with True on
    # its "no files to test" path — the exact path an empty backend response lands on.
    if state.get("backend_failed") or state.get("harness_ok") is False:
        return "harness_error"

    if state.get("carbon_g", 0.0) >= state.get("budget_g", AGENT_CARBON_BUDGET_G):
        return "budget_exceeded"

    if state.get("attempts_on_tier", 0) < AGENT_ATTEMPTS_PER_TIER:
        return "retry"

    if state["tier_idx"] + 1 < len(AGENT_LADDER):
        # Can we actually AFFORD the next rung? The budget is only ever checked
        # between calls, so without this forward estimate one 38-second call on
        # the big rung sails straight past the cap (measured: 34.8 g in a single
        # call). Pre-costing the escalation is what makes the budget a real bound
        # rather than a post-hoc observation.
        spent = state.get("carbon_g", 0.0)
        budget = state.get("budget_g", AGENT_CARBON_BUDGET_G)
        if spent + _estimate_call_carbon(state, state["tier_idx"] + 1) > budget:
            return "budget_exceeded"

        # Only escalate into a rung that is actually serving. A dead endpoint
        # would make run_vllm_inference fall through to a general model while we
        # reported an escalation to the coder — wrong answer, wrong carbon, wrong
        # audit trail. Better to stop and say the rung is unavailable.
        if _rung_is_live(state["tier_idx"] + 1):
            return "escalate"
        return "escalation_unavailable"

    return "failed"


def route_after_verify(state: AgentState) -> str:
    """Map the classification onto graph edges."""
    decision = classify(state)
    return decision if decision in ("retry", "escalate") else "done"


def node_finalize(state: AgentState) -> AgentState:
    """Terminal node — records the outcome ``classify`` decided."""
    decision = classify(state)
    state["status"] = decision  # completed | harness_error | budget_exceeded | escalation_unavailable | failed

    if decision == "escalation_unavailable":
        nxt = AGENT_LADDER[state["tier_idx"] + 1]
        _emit(
            state,
            "escalation_unavailable",
            wanted_tier=nxt[2],
            variant=nxt[0],
            detail="escalation rung is not serving; refusing to fall through to a general model",
        )
    elif decision == "harness_error":
        _emit(state, "harness_error", detail=(state.get("test_output") or "")[-300:])
    elif decision == "budget_exceeded":
        _emit(
            state,
            "budget_exceeded",
            carbon_g=state.get("carbon_g", 0.0),
            budget_g=state.get("budget_g", AGENT_CARBON_BUDGET_G),
        )
    elif decision == "failed":
        _emit(state, "exhausted", carbon_g=state.get("carbon_g", 0.0))
    else:
        _emit(
            state,
            "completed",
            tier=AGENT_LADDER[state["tier_idx"]][2],
            carbon_g=state.get("carbon_g", 0.0),
        )
    return state


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph():
    """Compile the LangGraph state machine (None when langgraph is absent)."""
    if not LANGGRAPH_AVAILABLE:
        return None

    g = StateGraph(AgentState)
    g.add_node("generate", node_generate)
    g.add_node("verify", node_verify)
    g.add_node("escalate", node_escalate)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"retry": "generate", "escalate": "escalate", "done": "finalize"},
    )
    g.add_edge("escalate", "generate")
    g.add_edge("finalize", END)

    return g.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _run_fallback(state: AgentState) -> AgentState:
    """
    Identical node semantics without langgraph, so a missing dep degrades to a
    working harness instead of a broken image (requirements.txt:4).
    """
    hard_stop = AGENT_ATTEMPTS_PER_TIER * len(AGENT_LADDER) + 2
    for _ in range(hard_stop):
        state = node_generate(state)
        state = node_verify(state)
        decision = route_after_verify(state)
        if decision == "done":
            return node_finalize(state)
        if decision == "escalate":
            state = node_escalate(state)
    state["status"] = "failed"
    return state


# --------------------------------------------------------------------------
# Task registry
#
# A deferred task outlives its HTTP request by hours, so the POST response
# cannot be the only place its result exists. Every task is registered at submit
# time and updated in place as it moves queued -> running -> completed/failed;
# GET /api/agent/task/{id} reads it back.
#
# In-memory and bounded. A restart loses queued tasks — but the DeferredQueue is
# in-memory too, so they are gone regardless; persisting the registry alone would
# only produce records of tasks nothing will ever run. Completed tasks are also
# written to the audit log, which is the durable record.
# --------------------------------------------------------------------------

_REGISTRY: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_REGISTRY_LOCK = threading.Lock()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _registry_put(record: dict[str, Any]) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY[record["task_id"]] = record
        while len(_REGISTRY) > AGENT_TASK_HISTORY:
            _REGISTRY.popitem(last=False)


def _registry_update(task_id: str, **fields: Any) -> dict[str, Any] | None:
    with _REGISTRY_LOCK:
        record = _REGISTRY.get(task_id)
        if record is None:  # aged out of a full registry mid-run
            return None
        record.update(fields)
        return dict(record)


def get_task(task_id: str) -> dict[str, Any] | None:
    with _REGISTRY_LOCK:
        record = _REGISTRY.get(task_id)
        return dict(record) if record else None


def list_tasks(limit: int = 20) -> list[dict[str, Any]]:
    """Most recent first, without the (large) result payload."""
    with _REGISTRY_LOCK:
        records = list(_REGISTRY.values())[-limit:]

    summaries: list[dict[str, Any]] = []
    for record in reversed(records):
        result = record.get("result") or {}
        summaries.append({
            "task_id": record["task_id"],
            "status": record["status"],
            "task": record["task"][:160],
            "created_at": record["created_at"],
            "deferred": record.get("deferred", False),
            "target_dispatch": record.get("target_dispatch"),
            "spec_source": record.get("spec_source", "model"),
            "final_tier": result.get("final_tier"),
            "escalated": result.get("escalated"),
            "carbon_g": result.get("carbon_g"),
            "carbon_per_completion_g": result.get("carbon_per_completion_g"),
        })
    return summaries


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------

def run_coding_task(
    task: str,
    test_command: str = "python -m pytest -q",
    carbon_budget_g: float | None = None,
    keep_workspace: bool = True,
    allow_defer: bool = True,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
    tests: str | dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Submit one agentic coding task under a carbon budget.

    When the grid is clean enough the task runs inline and the final state is
    returned, including the ``carbon_per_completion_g`` summary this harness
    exists to optimise. (It is only meaningful when ``status == "completed"``: a
    failed task has no carbon per completion, it has pure waste — which is the
    whole point of the ladder.)

    Above ``AGENT_DEFER_CI`` the task is handed to the EcoServe queue and a
    ``queued`` record is returned immediately; it will run in the greenest window
    the forecast offers within ``AGENT_DEFERRAL_MS``, and ``on_complete`` fires
    then. Poll ``get_task(task_id)`` for the outcome. Pass ``allow_defer=False``
    to force an inline run on a dirty grid.

    ``tests`` supplies the frozen spec from the caller (a pytest source string, or
    ``{path: source}``) instead of having the ladder's weakest rung author it. When
    given, the agent writes only the implementation. Raises ``ValueError`` if the
    suite cannot serve as a spec — before any carbon is spent.
    """
    from monitoring_layer import fetch_carbon_intensity

    caller_tests = normalize_caller_tests(tests)   # may raise; nothing spent yet
    task_id = uuid.uuid4().hex[:12]

    try:
        grid_ci = float(fetch_carbon_intensity())
    except Exception:
        grid_ci = 400.0

    record: dict[str, Any] = {
        "task_id": task_id,
        "task": task,
        "test_command": test_command,
        # Explicit None check: `or` would treat a 0.0 budget as "unset" and hand
        # back the default, quietly letting a zero-budget task escalate.
        "carbon_budget_g": AGENT_CARBON_BUDGET_G if carbon_budget_g is None else carbon_budget_g,
        "keep_workspace": keep_workspace,
        "tests": caller_tests,
        "spec_source": "caller" if caller_tests else "model",
        "created_at": _iso(time.time()),
        "grid_ci_at_submit": grid_ci,
        "status": "running",
        "deferred": False,
        "target_dispatch": None,
        "reason": None,
        "result": None,
    }
    _registry_put(record)

    if allow_defer and grid_ci > AGENT_DEFER_CI:
        queued = _defer_task(record, on_complete=on_complete)
        if queued is not None:
            return queued
        # Queue full. Backpressure is the queue's documented contract: the caller
        # runs it now. Dirty-grid execution beats silently dropping the task.
        logger.warning(
            "agent: deferred queue full; running task %s inline at %.0f gCO2/kWh",
            task_id, grid_ci,
        )

    return _execute_task(record, on_complete=on_complete)


def _defer_task(
    record: dict[str, Any],
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """
    Hand the task to the EcoServe queue. Returns the queued record, or None if
    the queue rejected it (full) and the caller should run it inline.
    """
    from deferred_queue import get_deferred_queue
    from monitoring_layer import fetch_grid_signal

    task_id = record["task_id"]
    grid_ci = record["grid_ci_at_submit"]
    queue = get_deferred_queue(forecast_provider=fetch_grid_signal)
    request_id = f"agent:{task_id}"

    def _dispatch(payload: dict[str, Any]) -> None:
        # The queue calls dispatch_fn from its dispatch loop *while holding its
        # lock* (deferred_queue.py:206). An agent task runs for minutes, so doing
        # the work here would stall every enqueue and status read for its whole
        # duration. Hand off to a worker and return immediately.
        threading.Thread(
            target=_execute_task,
            args=(payload,),
            kwargs={"on_complete": on_complete},
            daemon=True,
            name=f"agent-task-{task_id}",
        ).start()

    accepted = queue.enqueue(
        request_id=request_id,
        payload=record,
        dispatch_fn=_dispatch,
        deferral_ms=AGENT_DEFERRAL_MS,
    )
    if not accepted:
        return None

    target_dispatch = next(
        (
            item.get("target_dispatch")
            for item in queue.status().get("pending_requests", [])
            if item.get("request_id") == request_id
        ),
        None,
    )
    logger.info(
        "agent: task %s deferred at %.0f gCO2/kWh; target dispatch %s",
        task_id, grid_ci, target_dispatch,
    )
    return _registry_update(
        task_id,
        status="queued",
        deferred=True,
        target_dispatch=target_dispatch,
        reason=(
            f"grid intensity {grid_ci:.0f} gCO2/kWh exceeds AGENT_DEFER_CI "
            f"({AGENT_DEFER_CI:.0f}); queued for the next low-carbon window"
        ),
    )


def _execute_task(
    record: dict[str, Any],
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run the graph for one task. Blocking — called inline from ``run_coding_task``
    or, for a deferred task, on a worker thread hours later.
    """
    from monitoring_layer import fetch_carbon_intensity

    task_id = record["task_id"]

    # Re-read the grid at *execution* time, not submit time. For a deferred task
    # those are hours and possibly hundreds of gCO2/kWh apart, and every model
    # call is billed against this number — keeping the submit-time value would
    # bill the task at the dirty grid it was deferred away from, erasing the
    # saving in the very books that are supposed to show it.
    try:
        grid_ci = float(fetch_carbon_intensity())
    except Exception:
        grid_ci = float(record.get("grid_ci_at_submit", 400.0))

    _WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = tempfile.mkdtemp(prefix=f"task-{task_id}-", dir=str(_WORKSPACE_ROOT))

    # Caller-supplied tests land in the workspace before the first model call, so the
    # spec exists — and is frozen — from step 0. Validated at submit, not here: this
    # runs on a worker thread hours after the request returned, where a ValueError has
    # nobody to tell.
    caller_tests: dict[str, str] = dict(record.get("tests") or {})
    for path, src in caller_tests.items():
        tool_write_file(workspace, path, src)

    state: AgentState = {
        "task_id": task_id,
        "task": record["task"],
        "workspace": workspace,
        "test_command": record["test_command"],
        "files": dict(caller_tests),
        "spec_locked": bool(caller_tests),
        "tier_idx": 0,
        "attempts_on_tier": 0,
        "total_llm_calls": 0,
        "carbon_g": 0.0,
        "budget_g": record["carbon_budget_g"],
        "grid_ci": grid_ci,
        "status": "running",
        "events": [],
    }
    _registry_update(task_id, status="running", started_at=_iso(time.time()))
    if caller_tests:
        _emit(state, "spec_supplied", files=sorted(caller_tests), source="caller")

    started = time.time()
    graph = _get_graph()
    try:
        if graph is not None:
            # recursion_limit bounds the graph even if a router bug loops.
            final = graph.invoke(
                state,
                config={"recursion_limit": 4 * AGENT_ATTEMPTS_PER_TIER * len(AGENT_LADDER) + 8},
            )
        else:
            final = _run_fallback(state)
    except Exception as exc:
        logger.exception("agent task %s crashed", task_id)
        state["status"] = "failed"
        _emit(state, "crash", error=str(exc))
        final = state

    final = dict(final)
    final["duration_s"] = round(time.time() - started, 2)
    final["orchestrator"] = "langgraph" if graph is not None else "fallback"
    final["final_tier"] = AGENT_LADDER[final.get("tier_idx", 0)][2]
    final["escalated"] = final.get("tier_idx", 0) > 0

    completed = final.get("status") == "completed"
    final["carbon_per_completion_g"] = final.get("carbon_g", 0.0) if completed else None
    final["wasted_carbon_g"] = None if completed else final.get("carbon_g", 0.0)

    # Who authored the ground truth this result is measured against. A `completed`
    # against a model-authored spec means "it passed its own tests"; against a caller
    # spec it means "it passed yours" — different claims, so say which one.
    final["spec_source"] = record.get("spec_source", "model")

    final["deferred"] = record.get("deferred", False)
    if final["deferred"]:
        submit_ci = float(record.get("grid_ci_at_submit", grid_ci))
        final["grid_ci_at_submit"] = submit_ci
        # What the deferral actually bought, in the units the whole system is
        # scored in: the drop in grid intensity between submit and execution.
        final["deferral_ci_saved"] = round(submit_ci - grid_ci, 1)

    if not record.get("keep_workspace", True):
        shutil.rmtree(workspace, ignore_errors=True)
        final["workspace"] = None

    _registry_update(
        task_id,
        status=final.get("status", "failed"),
        finished_at=_iso(time.time()),
        result=final,
    )

    if on_complete is not None:
        try:
            on_complete(final)
        except Exception:  # a bad hook must not fail the task
            logger.warning("agent: on_complete hook failed for %s", task_id, exc_info=True)

    return final

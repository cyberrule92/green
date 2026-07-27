"""
Routing Policies — Adaptive Green AI
Implements:
- Composite Sustainability Score (CSS) per Section 3.4 of the paper
- LLMCarbon operational and embodied carbon formulas (Section 3.4.2, 4.1)
- MoE sparse FLOP accounting and all-to-all overhead (Section 5.1)
- Multi-region scoring: w1·CI_r + w2·latency_r (Section 3.5.3)
- EcoServe scheduling actions: deferral, regional reroute (Section 3.5.2-3)
- Semantic prompt profiler with MoE awareness (Section 3.4.1)
- Tenant-tier policy coefficients (Section 3.4.4)
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

# ── LLMCarbon constants ────────────────────────────────────────────────────────
_W_TO_KWH = 1.0 / (1_000.0 * 3_600.0)     # Watts × seconds → kWh
_SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
_HIGH_CARBON_THRESHOLD = 450.0               # gCO₂/kWh above which deferral is triggered

# ── Defaults ───────────────────────────────────────────────────────────────────

_CSS_DIMENSIONS = ("carbon", "latency", "accuracy", "cost", "region")

DEFAULT_POLICY_CONFIG = {
    "version": "2026-05-13",
    # Adaptive Green AI is a sustainability solution: the CSS winner must be the
    # greenest feasible candidate. Carbon weight dominates here; accuracy/SLA
    # floors are still enforced via the penalty terms in rank_routing_candidates.
    "default": {"carbon": 0.60, "latency": 0.12, "accuracy": 0.16, "cost": 0.08, "region": 0.04},
    "tiers": {
        "standard": {"carbon": 0.55, "latency": 0.14, "accuracy": 0.18, "cost": 0.08, "region": 0.05},
        "premium":  {"carbon": 0.45, "latency": 0.20, "accuracy": 0.22, "cost": 0.08, "region": 0.05},
        "esg":      {"carbon": 0.70, "latency": 0.08, "accuracy": 0.12, "cost": 0.06, "region": 0.04},
        "batch":    {"carbon": 0.65, "latency": 0.06, "accuracy": 0.13, "cost": 0.12, "region": 0.04},
    },
}

DEFAULT_ROUTING_TARGETS = [
    {
        "id": "local-vgpu-small",   "model_variant": "ultra-light", "hardware": "vgpu",
        "region": "local",          "accuracy": 0.66,               "latency_ms": 55,
        "cost_units": 0.20,         "power_w": 95,                  "region_carbon_multiplier": 1.0,
        "supports_batching": True,  "available": True,
        "hardware_efficiency": 0.68, "pue": 1.3,
        "flop_count_per_token": 690_000_000,
        "mfg_carbon_kg": 143.0,     "device_lifetime_years": 5.0,  "annual_inference_volume": 100_000,
        "moe": False,               "all_to_all_overhead_ratio": 0.0,
        "grid_zone": "local",       "network_latency_ms": 0.0,
    },
    {
        "id": "local-vgpu-medium",  "model_variant": "medium",      "hardware": "vgpu",
        "region": "local",          "accuracy": 0.81,               "latency_ms": 110,
        "cost_units": 0.44,         "power_w": 145,                 "region_carbon_multiplier": 1.0,
        "supports_batching": True,  "available": True,
        "hardware_efficiency": 0.72, "pue": 1.3,
        "flop_count_per_token": 2_200_000_000,
        "mfg_carbon_kg": 143.0,     "device_lifetime_years": 5.0,  "annual_inference_volume": 100_000,
        "moe": False,               "all_to_all_overhead_ratio": 0.0,
        "grid_zone": "local",       "network_latency_ms": 0.0,
    },
    {
        "id": "local-vgpu-full",    "model_variant": "full",        "hardware": "vgpu",
        "region": "local",          "accuracy": 0.92,               "latency_ms": 225,
        "cost_units": 0.90,         "power_w": 225,                 "region_carbon_multiplier": 1.0,
        "supports_batching": False, "available": True,
        "hardware_efficiency": 0.75, "pue": 1.3,
        "flop_count_per_token": 3_000_000_000,
        "mfg_carbon_kg": 143.0,     "device_lifetime_years": 5.0,  "annual_inference_volume": 100_000,
        "moe": False,               "all_to_all_overhead_ratio": 0.0,
        "grid_zone": "local",       "network_latency_ms": 0.0,
    },
    {
        "id": "local-cpu-fallback", "model_variant": "ultra-light", "hardware": "cpu",
        "region": "local",          "accuracy": 0.60,               "latency_ms": 320,
        "cost_units": 0.16,         "power_w": 70,                  "region_carbon_multiplier": 1.0,
        "supports_batching": True,  "available": True,
        "hardware_efficiency": 0.35, "pue": 1.3,
        "flop_count_per_token": 690_000_000,
        "mfg_carbon_kg": 30.0,      "device_lifetime_years": 5.0,  "annual_inference_volume": 100_000,
        "moe": False,               "all_to_all_overhead_ratio": 0.0,
        "grid_zone": "local",       "network_latency_ms": 0.0,
    },
]

# ── Routing intent keywords (preserved from original) ─────────────────────────
SEMANTIC_MODEL_NAME = os.getenv("ROUTING_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RESPECT_MANUAL_ROUTING_INPUTS = os.getenv("ROUTING_RESPECT_MANUAL_OVERRIDES", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
VARIANT_RANK = {
    "ultra-light": 0, "medium": 1, "full": 2, "moe": 3,
    # Multimodal variants live on a separate modality axis (isolated by the
    # modality filter in rank_routing_candidates); ranks here only affect
    # intra-modality semantic-alignment distance.
    "vlm": 2, "diffusion-sdxl": 1, "diffusion-flux": 2,
}

# Coarse capability tiers for the semantic-cache staleness guard in
# decision_engine ("is a cached answer from a weaker model than this prompt now
# routes to?"). Intentionally coarser than VARIANT_RANK above: every
# high-accuracy text variant (full / MoE / STEM specialist) shares tier 2, so a
# cached answer is not regenerated merely because it was re-tagged to a sibling
# specialist of equal capability — only a genuine step up in tier regenerates.
VARIANT_CAPABILITY_TIER = {
    "ultra-light": 0,
    "medium": 1,
    "full": 2, "moe": 2, "stem-math": 2, "stem-science": 2, "stem-coding": 2,
}


def variant_capability_tier(variant: str | None) -> int:
    """Capability tier of a text model variant (higher = more capable); 0 if unknown."""
    return VARIANT_CAPABILITY_TIER.get((variant or "").strip(), 0)

# ── Multimodal detection ──────────────────────────────────────────────────────
# Image-generation intent: an explicit request to *produce* an image. Kept
# deliberately tight (verb + image-noun, or "text to image") so that questions
# *about* images ("what is in this picture") do not trigger generation.
IMAGE_GEN_PATTERN = re.compile(
    r"\b("
    r"(?:generate|create|draw|paint|render|make|produce|design|sketch|illustrate|"
    r"imagine|visualan?ize|visualise|visualize)\s+"
    r"(?:me\s+|a\s+|an\s+|some\s+)?"
    r"(?:image|images|picture|pictures|photo|photos|photograph|artwork|art|"
    r"illustration|drawing|painting|logo|icon|render|rendering|graphic|poster|"
    r"wallpaper|portrait|scene)"
    r"|text[\s-]?to[\s-]?image"
    r"|image\s+of\s+"
    r")\b",
    re.IGNORECASE,
)
_IMAGE_CONTENT_TYPE_PREFIX = "image/"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
# Grid carbon (gCO₂/kWh) above which image generation trims its denoising-step
# budget toward the model's floor — the diffusion-specific analogue of the CSS
# high-carbon downgrade. Diffusion carbon is ~linear in steps, so this is the
# single biggest per-request carbon lever available for generation.
DIFFUSION_HIGH_CARBON_CI = float(os.getenv("DIFFUSION_HIGH_CARBON_CI", "400.0"))
DOCUMENT_KEYWORDS = {"document","documents","pdf","docx","doc","architecture","policy","compliance","root cause","design","review","summarize","summary"}
REASONING_KEYWORDS = {"compare","tradeoff","troubleshoot","debug","analysis","analyze","steps","workflow","plan","design","migration","implement","how","why"}
URGENT_KEYWORDS = {"urgent","asap","immediately","now","prod","production","incident","outage","sev1","critical","broken","failing"}
INTENT_KEYWORDS = {
    "summarization":  {"summarize","summary","tldr","brief"},
    "explanation":    {"explain","what","overview","intro"},
    "analysis":       {"analyze","analysis","review","compare","evaluate"},
    "troubleshooting":{"debug","fix","issue","error","problem","broken"},
    "implementation": {"implement","build","create","steps","workflow"},
}
SEMANTIC_ROUTE_PROTOTYPES = {
    "ultra-light": [
        "short greeting or simple factual question answered in one or two sentences",
        "basic definition, common knowledge, lightweight explanation with little reasoning",
    ],
    "medium": [
        "step by step explanation, moderate reasoning, comparison, troubleshooting, or how to answer",
        "multi sentence answer that needs some reasoning but not a full enterprise document analysis",
    ],
    "full": [
        "document grounded analysis, architecture review, compliance review, policy comparison, or uploaded file summary",
        "complex enterprise reasoning over retrieved context, long prompts, attachments, or knowledge base evidence",
    ],
    "moe": [
        "very complex multi-step reasoning requiring highest accuracy with energy efficiency via sparse computation",
        "mixture of experts model for large-scale document analysis or multi-domain enterprise knowledge fusion",
    ],
    "stem-math": [
        "solve a math problem, derive a formula, compute a result, explain calculus, statistics, or linear algebra",
        "mathematical proof, equation solving, numerical computation, or STEM mathematics reasoning",
    ],
    "stem-science": [
        "explain a physics, chemistry, or biology concept, describe a scientific process or natural phenomenon",
        "science question about thermodynamics, quantum mechanics, genetics, chemistry reactions, or ecology",
    ],
    "stem-coding": [
        "write or explain a scientific algorithm, numerical simulation, data analysis code, or machine learning model",
        "computational science, algorithm complexity, numpy or scipy usage, or scientific programming",
    ],
}
SEMANTIC_PRIORITY_PROTOTYPES = {
    "low":    ["casual informational request, background explanation, non urgent question"],
    "medium": ["normal business request requiring a useful answer soon but not a production emergency"],
    "high":   ["important troubleshooting or production-impacting request that needs a careful answer quickly"],
    "urgent": ["critical outage, emergency, or immediate production incident"],
}
SEMANTIC_INTENT_PROTOTYPES = {
    "summarization":  ["summarize a document or long text into concise bullets"],
    "explanation":    ["explain a concept or provide a straightforward factual answer"],
    "analysis":       ["compare options, review architecture, evaluate tradeoffs, or analyze content"],
    "troubleshooting":["diagnose a problem, fix an issue, or debug a broken workflow"],
    "implementation": ["provide steps, implementation guidance, or procedural instructions"],
    "creative":       ["write a poem, haiku, story, song lyrics, joke, or other creative literary content"],
    # NOTE: "stem" is intentionally absent — STEM is a domain (see stem_domain),
    # not an intent. Listing it here caused the hashed-embedding fallback to
    # score "stem" as the top intent for unrelated prompts (e.g. "write a haiku
    # on trees" scored stem > explanation).
}

# Creative-writing keyword set — short-circuits STEM detection so a "haiku on
# trees" routes to a small chat model instead of Qwen2.5-Math/Coder. Detected
# before the STEM keyword pass; sets intent="creative" and clears stem_domain.
CREATIVE_KEYWORDS = {
    "haiku", "poem", "poetry", "sonnet", "limerick", "verse", "stanza",
    "lyrics", "song", "rhyme", "ballad",
    "story", "fable", "fairy tale", "short story", "novella",
    "joke", "pun", "riddle", "anecdote",
    "screenplay", "monologue", "dialogue scene",
}

# STEM keyword sets for fast heuristic domain classification
STEM_MATH_KEYWORDS = {
    # Pure / classical math
    "equation", "integral", "derivative", "calculus", "algebra", "geometry",
    "trigonometry", "matrix", "vector", "probability", "statistics", "theorem",
    "proof", "formula", "polynomial", "quadratic", "logarithm", "factorial",
    "permutation", "combination", "fourier", "laplace", "eigenvalue", "gradient",
    "divergence", "curl", "differential", "arithmetic", "fraction", "decimal",
    "percentage", "ratio", "prime", "fibonacci", "binomial", "coefficient",
    "variance", "deviation", "regression", "distribution", "hypothesis",
    # Everyday math vocabulary (so "what is 2+2", "table of 12", "math problem"
    # actually reach the math model instead of falling through to TinyLlama)
    "math", "maths", "mathematics", "table of", "times table", "multiplication",
    "multiply", "addition", "subtract", "subtraction", "division",
    "plus", "minus", "modulo", "modulus", "remainder",
    "square root", "cube root", "exponent", "exponential", "power of",
    "sine", "cosine", "tangent", "sin(", "cos(", "tan(", "log(",
    "compute", "calculate", "simplify", "expand", "factorize", "factorise",
    # Practical engineering math (calc + linear algebra + numerical).
    # NOTE: bare "ode"/"pde" deliberately omitted — substring match collides
    # with "code"/"update". "differential" + "boundary value" + "initial value"
    # already catch ODE/PDE problems unambiguously.
    "boundary value", "initial value", "newton-raphson",
    "trapezoidal", "simpson's rule", "taylor series", "maclaurin", "convergence",
    "determinant", "transpose", "dot product", "cross product",
    "stress", "strain", "young's modulus", "torque", "moment of inertia",
    "deflection", "bending moment", "shear force", "load distribution",
    "rms", "amplitude", "phase angle", "sinusoidal", "harmonic",
    "complex number", "imaginary", "phasor", "impedance",
    "heat transfer", "thermal conductivity", "reynolds", "bernoulli",
}
STEM_SCIENCE_KEYWORDS = {
    "physics", "chemistry", "biology", "thermodynamics", "quantum", "relativity",
    "entropy", "kinetics", "photosynthesis", "dna", "rna", "protein", "atom",
    "molecule", "electron", "neutron", "proton", "orbital", "bond", "reaction",
    "energy", "force", "velocity", "acceleration", "momentum", "gravity",
    "magnetism", "electromagnetism", "optics", "wavelength", "frequency",
    "periodic", "element", "compound", "isotope", "radioactive", "cell",
    "chromosome", "evolution", "ecosystem", "genetics", "neuroscience",
    "stoichiometry", "titration", "ph", "acid", "base", "catalyst",
}
STEM_CODING_KEYWORDS = {
    # Strong, unambiguous coding signals — fire stem-coding on their own.
    "algorithm", "complexity", "bigO", "recursion", "dynamic programming",
    "sorting", "graph", "tree", "linked list", "binary search", "hash",
    "stack", "queue", "heap", "dfs", "bfs", "dijkstra", "greedy",
    "backtracking", "bitwise", "regex", "compiler", "parser", "automata",
    "turing", "numpy", "scipy", "pandas", "matplotlib", "tensorflow",
    "pytorch", "sklearn", "linear algebra", "numerical", "simulation",
    "monte carlo", "neural network", "machine learning", "deep learning",
    # Language / framework / tool names + strong coding nouns. These are
    # specific enough to route on their own — the set above was all algorithm/DS
    # jargon, so a plain "write Java code to ..." matched nothing and fell
    # through to the short-prompt→TinyLlama default, producing weak code.
    "codebase", "source code", "snippet", "compile", "debugger", "refactor",
    "boilerplate", "iterator", "closure", "namespace", "sdk",
    "java", "javascript", "typescript", "python", "kotlin", "golang", "scala",
    "php", "perl", "sql", "html", "css", "bash", "powershell", "c++", "c#",
    ".net", "django", "flask", "fastapi", "nodejs", "kubernetes", "docker",
    "terraform",
}

# Broad, ambiguous coding tokens. On their own these mis-route benign prompts —
# "software update policy", "the API is rate-limiting me", "dress code",
# "function of the liver" — to the heavy coding model, which breaks the
# greenest-feasible invariant. They count as coding evidence only when a coding
# verb OR a strong coding token (above) co-occurs (see infer_prompt_profile).
STEM_CODING_BROAD_KEYWORDS = {
    "code", "coding", "function", "api", "software", "programming",
    "runtime", "syntax",
}

# Coding-action verbs that, alongside a broad token, confirm a coding request.
# Deliberately excludes words that are themselves broad nouns (code/program/
# script) so "dress code" can't self-satisfy the gate.
CODING_VERB_KEYWORDS = {
    "write", "implement", "debug", "refactor", "compile", "build", "create",
    "generate", "parse", "deploy", "optimize", "instantiate", "initialize",
    "import", "iterate", "print", "return", "execute", "call", "fix", "run",
}

# General-knowledge domain keyword sets. These don't pin a dedicated model
# (unlike STEM math/science/coding) — they exist so the router can recognize
# the topic, log a domain label, and let CSS pick the greenest feasible
# candidate. Knowledge-recall tasks rarely need more than ultra-light unless
# the prompt is long, attachment-bearing, or explicitly reasoning-heavy.
LITERATURE_KEYWORDS = {
    "novel", "novella", "novelist", "author", "playwright", "protagonist",
    "antagonist", "character arc", "plot", "subplot", "narrator", "narrative",
    "prose", "literary", "literature", "fiction", "non-fiction", "memoir",
    "metaphor", "simile", "allegory", "symbolism", "foreshadowing", "irony",
    "shakespeare", "hamlet", "macbeth", "othello", "tolkien", "hemingway",
    "dickens", "austen", "kafka", "orwell", "dostoevsky", "tolstoy",
    "chapter", "verse of", "iliad", "odyssey", "epic poem", "literary device",
    "book review", "summarize the book", "plot of",
}
HISTORY_KEYWORDS = {
    "history", "historical", "historian", "ancient", "medieval", "renaissance",
    "revolution", "empire", "dynasty", "monarchy", "republic", "colonial",
    "colonization", "decolonization", "world war", "ww1", "ww2", "cold war",
    "civil war", "civilization", "century", "bce", "ad ", "bc ",
    "egypt", "mesopotamia", "rome", "roman empire", "byzantine", "ottoman",
    "mughal", "mongol", "viking", "crusade", "reformation", "enlightenment",
    "industrial revolution", "great depression", "holocaust",
    "napoleon", "caesar", "alexander the great", "genghis khan", "lincoln",
    "gandhi", "churchill", "hitler", "stalin",
}
PSYCHOLOGY_KEYWORDS = {
    "psychology", "psychological", "psychiatrist", "psychiatry", "cognition",
    "cognitive", "behavior", "behavioral", "behaviour", "behavioural",
    "emotion", "emotional", "mental health", "mental illness", "depression",
    "anxiety", "trauma", "ptsd", "ocd", "adhd", "bipolar", "schizophrenia",
    "therapy", "psychotherapy", "counseling", "counselling", "cbt",
    "freud", "jung", "skinner", "piaget", "maslow", "pavlov",
    "subconscious", "unconscious", "ego", "id", "superego", "archetype",
    "conditioning", "reinforcement", "neurosis", "personality", "introvert",
    "extrovert", "narcissism", "empathy", "self-esteem", "motivation",
}
ASTROLOGY_KEYWORDS = {
    "astrology", "astrological", "horoscope", "zodiac", "birth chart",
    "natal chart", "sun sign", "moon sign", "rising sign", "ascendant",
    "aries", "taurus", "gemini", "cancer sign", "leo", "virgo", "libra",
    "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
    "mercury retrograde", "venus retrograde", "saturn return",
    "tarot", "tarot card", "palmistry", "numerology", "vedic astrology",
    "kundli", "rashi", "nakshatra", "compatibility chart",
}
ASTRONOMY_KEYWORDS = {
    "astronomy", "astronomer", "astrophysics", "cosmology", "cosmologist",
    "galaxy", "galaxies", "nebula", "supernova", "black hole", "neutron star",
    "pulsar", "quasar", "white dwarf", "red giant", "main sequence",
    "milky way", "andromeda", "exoplanet", "solar system", "heliocentric",
    "mercury planet", "venus planet", "mars", "jupiter", "saturn planet",
    "uranus", "neptune", "pluto", "kuiper belt", "oort cloud", "asteroid",
    "comet", "meteor", "meteorite", "telescope", "hubble", "james webb",
    "big bang", "dark matter", "dark energy", "redshift", "light year",
    "parsec", "constellation", "orbit of", "lunar eclipse", "solar eclipse",
}

def _build_keyword_pattern(keywords: set[str]) -> re.Pattern:
    """
    Compile a word-boundary regex from a keyword set.

    Substring matching (`kw in text`) caused stem-science to fire for benign
    queries — "ph" matched "graph"/"phone", "rna" matched "internal", "base"
    matched "database", "cell" matched "excellent", "force" matched "reinforce".
    Word boundaries (`\\b`) only match keywords as standalone tokens.

    Keywords starting/ending in non-word characters (e.g. `sin(`) skip the
    boundary on that side, because `\\b` between two non-word chars never matches.
    """
    parts = []
    for kw in sorted(keywords, key=len, reverse=True):
        if not kw:
            continue
        escaped = re.escape(kw)
        prefix = r"\b" if kw[0].isalnum() else ""
        suffix = r"\b" if kw[-1].isalnum() else ""
        parts.append(prefix + escaped + suffix)
    return re.compile("|".join(parts), re.IGNORECASE)


STEM_MATH_PATTERN = _build_keyword_pattern(STEM_MATH_KEYWORDS)
STEM_SCIENCE_PATTERN = _build_keyword_pattern(STEM_SCIENCE_KEYWORDS)
STEM_CODING_PATTERN = _build_keyword_pattern(STEM_CODING_KEYWORDS)
STEM_CODING_BROAD_PATTERN = _build_keyword_pattern(STEM_CODING_BROAD_KEYWORDS)
CODING_VERB_PATTERN = _build_keyword_pattern(CODING_VERB_KEYWORDS)
CREATIVE_PATTERN = _build_keyword_pattern(CREATIVE_KEYWORDS)
LITERATURE_PATTERN = _build_keyword_pattern(LITERATURE_KEYWORDS)
HISTORY_PATTERN = _build_keyword_pattern(HISTORY_KEYWORDS)
PSYCHOLOGY_PATTERN = _build_keyword_pattern(PSYCHOLOGY_KEYWORDS)
ASTROLOGY_PATTERN = _build_keyword_pattern(ASTROLOGY_KEYWORDS)
ASTRONOMY_PATTERN = _build_keyword_pattern(ASTRONOMY_KEYWORDS)

# General-knowledge domains the router classifies but does NOT pin to a
# specific model — CSS picks the greenest feasible candidate. Used for
# logging and to keep the semantic backfill from mis-routing these to STEM.
GENERAL_DOMAIN_PATTERNS = {
    "literature": LITERATURE_PATTERN,
    "history":    HISTORY_PATTERN,
    "psychology": PSYCHOLOGY_PATTERN,
    "astrology":  ASTROLOGY_PATTERN,
    "astronomy":  ASTRONOMY_PATTERN,
}

_semantic_lock = threading.Lock()
_semantic_model = None
_semantic_backend = "heuristic"
_semantic_embeddings: dict[str, dict[str, list[list[float]]]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val - min_val <= 0:
        return 0.0
    bounded = max(min_val, min(max_val, value))
    return (bounded - min_val) / (max_val - min_val)


# ── Config loaders ─────────────────────────────────────────────────────────────

def load_policy_config(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return DEFAULT_POLICY_CONFIG
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_POLICY_CONFIG
    return payload if isinstance(payload, dict) else DEFAULT_POLICY_CONFIG


def load_routing_targets(path: str | Path) -> list[dict[str, Any]]:
    """
    Load routing targets from JSON.
    Falls back gracefully to DEFAULT_ROUTING_TARGETS.
    Also attempts to load from model_zoo.json if routing_targets.json is missing.
    """
    file_path = Path(path)
    # Try model_zoo.json next to the given path as fallback
    zoo_path = file_path.parent / "model_zoo.json"

    for candidate in (file_path, zoo_path):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # Flat list
        if isinstance(payload, list) and payload:
            return payload
        # model_zoo.json format: {"models": [...]}
        if isinstance(payload, dict) and isinstance(payload.get("models"), list):
            return payload["models"]

    return DEFAULT_ROUTING_TARGETS


def resolve_policy(user_tier: str, policy_config: dict[str, Any]) -> dict[str, Any]:
    tier_name = (user_tier or "standard").strip().lower()
    default_policy = dict(policy_config.get("default") or DEFAULT_POLICY_CONFIG["default"])
    tier_policy = dict((policy_config.get("tiers") or {}).get(tier_name) or {})
    combined = {**default_policy, **tier_policy}
    # Region coefficient is optional in older configs; absent → 0 then renormalised
    total = sum(safe_float(combined.get(k), 0.0) for k in _CSS_DIMENSIONS) or 1.0
    normalized = {k: safe_float(combined.get(k), 0.0) / total for k in _CSS_DIMENSIONS}
    normalized["tier"] = tier_name
    normalized["version"] = policy_config.get("version", "unversioned")
    return normalized


# ── LLMCarbon carbon computation ───────────────────────────────────────────────

def compute_operational_carbon_llmcarbon(
    target: dict[str, Any],
    grid_carbon_g_per_kwh: float,
    inference_duration_s: float,
    token_count: int = 256,
) -> float:
    """
    LLMCarbon operational carbon formula (Section 3.4.2, 4.1):
        C_op = TDP × PUE × t / HE_eff  [in Wh terms] × CI

    For MoE:  HE_eff = HE × (1 − all_to_all_overhead_ratio)
    Returns gCO₂ per inference.
    """
    tdp_w = safe_float(target.get("power_tdp_w") or target.get("power_w"), 100.0)
    pue = safe_float(target.get("pue"), 1.3)
    he = safe_float(target.get("hardware_efficiency"), 0.6)
    region_mult = safe_float(target.get("region_carbon_multiplier"), 1.0)

    # MoE all-to-all penalty
    if target.get("moe"):
        all_to_all = safe_float(target.get("all_to_all_overhead_ratio"), 0.0)
        he_eff = he * (1.0 - all_to_all)
    else:
        he_eff = he

    he_eff = max(he_eff, 0.05)  # numerical guard

    energy_kwh = (tdp_w * inference_duration_s * pue) / (he_eff * 1000.0 * 3600.0)
    return energy_kwh * grid_carbon_g_per_kwh * region_mult


def compute_embodied_carbon_llmcarbon(
    target: dict[str, Any],
    inference_duration_s: float,
) -> float:
    """
    LLMCarbon embodied carbon amortisation (Section 4.1):
        C_emb = mfg_carbon_g / (lifetime_years × annual_volume × avg_duration_s)
        Then per-request: C_emb_request = C_emb_rate × inference_duration_s
    """
    mfg_carbon_kg = safe_float(target.get("mfg_carbon_kg"), 143.0)
    lifetime_years = safe_float(target.get("device_lifetime_years"), 5.0)
    annual_volume = safe_float(target.get("annual_inference_volume"), 100_000.0)
    avg_s = safe_float(target.get("latency_ms_p50") or target.get("latency_ms"), 200.0) / 1000.0

    mfg_carbon_g = mfg_carbon_kg * 1000.0
    total_lifetime_inference_s = annual_volume * avg_s * lifetime_years
    if total_lifetime_inference_s <= 0:
        return 0.0
    emb_rate_g_per_s = mfg_carbon_g / total_lifetime_inference_s
    return emb_rate_g_per_s * inference_duration_s


def compute_moe_all_to_all_latency_ms(
    target: dict[str, Any],
    token_count: int = 256,
    d_model: int = 4096,
) -> float:
    """
    Estimate MoE all-to-all communication overhead in ms.
    T_comm = (k × token_count × d_model × element_bytes) / bandwidth (Section 5.1)
    """
    if not target.get("moe"):
        return 0.0
    k = int(safe_float(target.get("active_experts_k"), 2))
    bandwidth_gbps = safe_float(target.get("expert_bandwidth_gbps"), 100.0)
    bandwidth_bps = bandwidth_gbps * 1e9
    element_bytes = 2  # fp16
    comm_bytes = k * token_count * d_model * element_bytes
    return round((comm_bytes / bandwidth_bps) * 1000.0, 3)


# ── Multimodal helpers ─────────────────────────────────────────────────────────

def attachment_is_image(attachment: dict[str, Any]) -> bool:
    """True when an attachment is an image (by content-type or extension)."""
    ctype = str(attachment.get("content_type") or "").lower()
    if ctype.startswith(_IMAGE_CONTENT_TYPE_PREFIX):
        return True
    name = str(attachment.get("name") or "").lower()
    return any(name.endswith(ext) for ext in _IMAGE_EXTENSIONS)


def detect_request_modality(
    prompt: str,
    attachments: list[dict[str, Any]] | None,
) -> str:
    """
    Classify the request modality feeding CSS candidate filtering:
      * "vision"    — image attachment(s) present → needs a VLM.
      * "image-gen" — explicit request to generate an image (and no image
                      attached to reason over).
      * "text"      — everything else (unchanged text routing).

    An attached image dominates: "what's in this photo" is analysis even if the
    word "image" appears. Generation is only inferred from an explicit verb+noun
    request with no image to analyse.
    """
    atts = attachments or []
    if any(attachment_is_image(a) for a in atts):
        return "vision"
    if prompt and IMAGE_GEN_PATTERN.search(prompt):
        return "image-gen"
    return "text"


def carbon_capped_diffusion_steps(target: dict[str, Any], grid_carbon: float) -> int:
    """
    Choose a denoising-step budget for a diffusion candidate, trimmed toward the
    model's floor as grid carbon rises. Diffusion operational carbon is ~linear
    in steps, so this is the highest-leverage per-request carbon control for
    image generation (analogue of the CSS high-carbon model downgrade).

    Interpolates from default_steps (clean grid) down to min_steps at/above
    DIFFUSION_HIGH_CARBON_CI. Clean-grid requests keep full quality.
    """
    default_steps = int(safe_float(target.get("diffusion_default_steps"), 30))
    min_steps = int(safe_float(target.get("diffusion_min_steps"), max(8, default_steps // 3)))
    if grid_carbon <= 150.0:
        return default_steps
    # Linear ramp between 150 and DIFFUSION_HIGH_CARBON_CI gCO₂/kWh.
    span = max(DIFFUSION_HIGH_CARBON_CI - 150.0, 1.0)
    frac = min(max((grid_carbon - 150.0) / span, 0.0), 1.0)
    steps = round(default_steps - frac * (default_steps - min_steps))
    return int(max(min_steps, min(default_steps, steps)))


def compute_diffusion_carbon(
    target: dict[str, Any],
    grid_carbon_g_per_kwh: float,
    steps: int,
    width: int = 1024,
    height: int = 1024,
) -> tuple[float, float]:
    """
    Operational + embodied carbon for one generated image (LLMCarbon-style, but
    step-based instead of token-based).

        FLOPs_image = flops_per_step × steps × (W·H / base_res²)
        energy_kwh  = FLOPs_image / (HE · peak_FLOP/s_per_W_proxy) …

    We reuse the same TDP·duration·PUE / HE energy model as the token path for
    consistency: operational energy is anchored on the model's measured per-step
    wall time (derived from latency_ms_p50 / default_steps) scaled by the actual
    step count and resolution, so a 12-step FP4 render costs proportionally less
    than a 40-step FP16 one — exactly the lever CSS should see.

    Returns (op_carbon_g, emb_carbon_g) for the image.
    """
    default_steps = max(int(safe_float(target.get("diffusion_default_steps"), 30)), 1)
    p50_ms = safe_float(target.get("latency_ms_p50") or target.get("latency_ms"), 2600.0)
    per_step_s = (p50_ms / 1000.0) / default_steps
    res_scale = max((width * height) / float(target.get("diffusion_base_resolution", 1024) ** 2), 0.25)
    duration_s = max(per_step_s * max(int(steps), 1) * res_scale, 0.05)

    tdp_w = safe_float(target.get("power_tdp_w") or target.get("power_w"), 350.0)
    pue = safe_float(target.get("pue"), 1.3)
    he = max(safe_float(target.get("hardware_efficiency"), 0.75), 0.05)
    region_mult = safe_float(target.get("region_carbon_multiplier"), 1.0)

    energy_kwh = (tdp_w * duration_s * pue) / (he * 1000.0 * 3600.0)
    op_g = energy_kwh * grid_carbon_g_per_kwh * region_mult
    emb_g = compute_embodied_carbon_llmcarbon(target, duration_s)
    return op_g, emb_g


# ── CSS scoring ────────────────────────────────────────────────────────────────

def _ql_estimator():
    """Lazy accessor for the learned quality/latency estimator.

    Imported lazily so importing routing_policies (e.g. in tooling/tests) does
    not spin up the estimator's background saver thread, and so a missing module
    never breaks CSS — on any failure we return a null object whose adjust()
    hands back the baseline unchanged.
    """
    try:
        from quality_latency_estimator import get_quality_latency_estimator
        return get_quality_latency_estimator()
    except Exception:  # pragma: no cover — estimator is strictly additive
        class _NullEstimator:
            def adjust(self, _sp, _v, acc, lat):
                return {"accuracy": acc, "latency_ms": lat,
                        "accuracy_residual": 0.0, "latency_scale": 1.0, "applied": False}
        return _NullEstimator()


# Reference output length that each candidate's configured latency_ms_p50 is
# taken to correspond to. A single global constant rather than a per-model figure
# because the p50 values are spec numbers of unknown provenance; the per-variant
# differentiation comes from the estimator's learned length head, which is
# measured rather than declared.
CSS_REFERENCE_OUTPUT_TOKENS = int(os.getenv("CSS_REFERENCE_OUTPUT_TOKENS", "96"))


def rank_routing_candidates(
    request_context: dict[str, Any],
    routing_targets: list[dict[str, Any]],
    grid_carbon: float,
    zone_carbon_map: dict[str, float] | None = None,
    token_count: int = 256,
) -> list[dict[str, Any]]:
    """
    Compute and rank (model, hardware, region) candidates by CSS score.
    CSS = w_c·C_score + w_l·L_score + w_a·A_score + w_cost·Cost_score  (Section 3.4)

    Full LLMCarbon carbon model (Section 3.4.2):
        carbon_total = C_op + C_emb

    Regional scoring adjustments via zone_carbon_map (Section 3.5.3).
    MoE latency penalty for all-to-all communication (Section 5.1).
    """
    coefficients = request_context["policy_coefficients"]
    explicit_preference = request_context.get("model_preference") or ""
    recommended_variant = request_context.get("recommended_model_variant") or ""
    semantic_profile = request_context.get("semantic_profile") or {}

    # ── Modality gate ──
    # Candidates live on one of three modality axes: "text" (default / missing),
    # "vision" (VLM), "image-gen" (diffusion). A request may only be served by a
    # candidate on its own axis — a diffusion model cannot answer a chat prompt
    # and a chat model cannot render an image. This filter runs *before* the
    # carbon/accuracy scoring so CSS never compares across modalities.
    request_modality = str(semantic_profile.get("modality") or "text").lower()

    def target_modality(t: dict[str, Any]) -> str:
        return str(t.get("modality") or "text").lower()

    filtered_targets = [
        t for t in routing_targets
        if t.get("available", True) and target_modality(t) == request_modality
    ]
    # If a modality has no available candidate (e.g. no VLM deployed) fall back to
    # the full target set so the request still routes rather than 500ing; the
    # dispatcher's graceful-fallback path then handles the missing endpoint.
    if not filtered_targets:
        filtered_targets = [t for t in routing_targets if t.get("available", True)]

    def target_matches(target: dict[str, Any]) -> bool:
        if not explicit_preference:
            return True
        return explicit_preference in {
            str(target.get("model_variant", "")).lower(),
            str(target.get("hardware", "")).lower(),
            str(target.get("region", "")).lower(),
            str(target.get("id", "")).lower(),
        }

    constrained_targets = [
        t for t in filtered_targets
        if safe_float(t.get("accuracy") or t.get("accuracy_baseline"), 0.0) >= request_context["accuracy_floor"]
        and target_matches(t)
    ]
    active_targets = constrained_targets or filtered_targets or DEFAULT_ROUTING_TARGETS

    max_cost = max(safe_float(t.get("cost_units"), 0.1) for t in active_targets)
    max_latency = max(
        safe_float(t.get("latency_ms") or t.get("latency_ms_p50"), request_context["sla_ms"])
        for t in active_targets
    )

    # Regional carbon weights for multi-region scoring (Section 3.5.3)
    region_w_carbon = 0.6
    region_w_latency = 0.4

    # ── First pass: compute carbon/region per candidate so we can do
    # candidate-set min-max normalisation per Section 3.4 of the paper.
    pre_scored: list[dict[str, Any]] = []
    for target in active_targets:
        _variant_for_len = str(target.get("model_variant", "")).lower()
        _baseline_acc_for_len = safe_float(
            target.get("accuracy") or target.get("accuracy_baseline"), 0.5)
        zone = target.get("grid_zone", "local")
        effective_ci = grid_carbon
        if zone_carbon_map and zone in zone_carbon_map:
            effective_ci = zone_carbon_map[zone]

        diffusion_steps = 0
        _len_meta: dict[str, Any] = {}   # text-only; diffusion carbon is step-based
        if target.get("diffusion"):
            # ── Diffusion (image-gen) carbon path: step-based, not token-based ──
            # Steps are trimmed toward the model floor as the grid dirties — the
            # single biggest per-request carbon lever for generation.
            diffusion_steps = carbon_capped_diffusion_steps(target, effective_ci)
            op_g, emb_g = compute_diffusion_carbon(target, effective_ci, diffusion_steps)
            # Latency scales with the (capped) step count vs the model's default.
            default_steps = max(int(safe_float(target.get("diffusion_default_steps"), 30)), 1)
            base_lat = safe_float(target.get("latency_ms_p50") or target.get("latency_ms"), 2600.0)
            latency_ms = base_lat * (diffusion_steps / default_steps)
        else:
            latency_ms = safe_float(target.get("latency_ms") or target.get("latency_ms_p50"),
                                    request_context["sla_ms"])
            # ── Output-length-aware duration ──────────────────────────────────
            # Carbon here is power x duration, and for autoregressive decoding
            # duration is dominated by how many tokens the model chooses to emit.
            # Pricing every candidate at the same static p50 made verbosity
            # invisible to CSS, and the three-arm benchmark showed exactly what
            # that costs: TinyLlama emitted 1.6x the tokens of Qwen2.5-1.5B on the
            # same prompts (123.9 vs 75.5) and so burned MORE carbon, while CSS,
            # unable to see it, kept selecting it and lost on both carbon (+10%)
            # and quality (-14.7pp) against always-full.
            #
            # So the estimator's learned per-variant length head now scales the
            # duration that carbon is computed from. This is a deliberate reversal
            # of the previous "carbon is never touched by the estimator" rule: that
            # rule was meant to protect the greenest-feasible invariant, but a
            # carbon number blind to output length does not measure carbon.
            _ql_len = _ql_estimator().adjust(
                semantic_profile, _variant_for_len, _baseline_acc_for_len, latency_ms,
                baseline_output_tokens=float(CSS_REFERENCE_OUTPUT_TOKENS),
            )
            expected_out = safe_float(_ql_len.get("output_tokens"), CSS_REFERENCE_OUTPUT_TOKENS)
            # A per-variant cap is a hard ceiling on that expectation, because the
            # dispatcher will enforce the same cap via max_tokens.
            cap = safe_float(target.get("max_output_tokens"), 0.0)
            if cap > 0:
                expected_out = min(expected_out, cap)
            expected_out = max(expected_out, 1.0)
            length_factor = expected_out / max(float(CSS_REFERENCE_OUTPUT_TOKENS), 1.0)

            inference_duration_s = max(latency_ms * length_factor / 1000.0, 0.05)
            op_g = compute_operational_carbon_llmcarbon(
                target, effective_ci, inference_duration_s, int(expected_out)
            )
            emb_g = compute_embodied_carbon_llmcarbon(target, inference_duration_s)
            # The latency CSS scores on must reflect the same expectation, or the
            # router would price a verbose candidate's carbon honestly while still
            # believing it is fast.
            latency_ms = latency_ms * length_factor
            _len_meta = {
                "expected_output_tokens": round(expected_out, 1),
                "length_scale": _ql_len.get("length_scale", 1.0),
                "length_factor": round(length_factor, 4),
                "max_output_tokens": int(cap) if cap > 0 else None,
                "estimator_applied": bool(_ql_len.get("applied")),
            }
        total_g = op_g + emb_g

        moe_comm_ms = compute_moe_all_to_all_latency_ms(target, token_count) if target.get("moe") else 0.0

        net_latency = safe_float(target.get("network_latency_ms"), 0.0)
        if zone_carbon_map:
            norm_ci = min(effective_ci / 600.0, 1.0)
            norm_net = min(net_latency / 300.0, 1.0)
            region_raw = region_w_carbon * norm_ci + region_w_latency * norm_net
        else:
            # Without multi-region signals, region term collapses to network-only
            region_raw = min(net_latency / 300.0, 1.0)

        pre_scored.append({
            "target": target, "latency_ms": latency_ms, "moe_comm_ms": moe_comm_ms,
            "op_g": op_g, "emb_g": emb_g, "total_g": total_g,
            "effective_ci": effective_ci, "zone": zone, "region_raw": region_raw,
            "diffusion_steps": diffusion_steps, "len_meta": _len_meta,
        })

    carbon_values = [p["total_g"] for p in pre_scored]
    min_carbon = min(carbon_values) if carbon_values else 0.0
    max_carbon = max(carbon_values) if carbon_values else 0.0
    region_values = [p["region_raw"] for p in pre_scored]
    min_region = min(region_values) if region_values else 0.0
    max_region = max(region_values) if region_values else 0.0

    ranked: list[dict[str, Any]] = []
    preferred_rank = VARIANT_RANK.get(recommended_variant, 1)
    complexity_score = safe_float(semantic_profile.get("complexity_score"), 0.5)
    priority = request_context.get("priority", "medium")

    for pre in pre_scored:
        target = pre["target"]
        latency_ms = pre["latency_ms"]
        moe_comm_ms = pre["moe_comm_ms"]
        op_carbon_g = pre["op_g"]
        emb_carbon_g = pre["emb_g"]
        total_carbon_g = pre["total_g"]
        effective_ci = pre["effective_ci"]
        zone = pre["zone"]
        region_raw = pre["region_raw"]
        baseline_accuracy = safe_float(target.get("accuracy") or target.get("accuracy_baseline"), 0.5)
        cost_units = safe_float(target.get("cost_units"), 0.1)
        candidate_variant = str(target.get("model_variant", "")).lower()

        # ── Learned quality/latency/length estimator (M5-adjacent) ──
        # Refines the *static* per-model baselines with a per-prompt correction
        # learned online from observed outcomes (quality_latency_estimator.py).
        # Cold-start / warm-up returns the baselines unchanged → no-op.
        #
        # The length head also feeds carbon, via the duration scaling applied in
        # the pre-scoring loop above. That is intentional and is a change from the
        # earlier design, which held carbon untouched to protect the
        # greenest-feasible invariant: a carbon figure that cannot see how many
        # tokens a model will emit is not measuring carbon, and the benchmark
        # showed it selecting the dirtier candidate as a result.
        #
        # `latency_ms` here already carries the length factor from pre-scoring, so
        # the latency head is applied on top of a length-adjusted baseline.
        _ql = _ql_estimator().adjust(
            semantic_profile, candidate_variant, baseline_accuracy, latency_ms
        )
        accuracy = _ql["accuracy"]
        latency_ms_effective = _ql["latency_ms"] + moe_comm_ms

        # ── Normalised component scores (Section 3.4: min-max over candidate set) ──
        carbon_score = 1.0 - normalize(total_carbon_g, min_carbon, max_carbon)
        latency_score = 1.0 - normalize(
            latency_ms_effective, 40.0, max(max_latency + moe_comm_ms, request_context["sla_ms"] * 1.5)
        )
        accuracy_score = normalize(accuracy, 0.45, 1.0)
        cost_score = 1.0 - normalize(cost_units, 0.05, max(max_cost, 1.0))
        # Region: lower raw is better (less carbon + less network latency)
        region_score = 1.0 - normalize(region_raw, min_region, max_region)

        css_score = (
            coefficients["carbon"]   * carbon_score
            + coefficients["latency"]  * latency_score
            + coefficients["accuracy"] * accuracy_score
            + coefficients["cost"]     * cost_score
            + coefficients.get("region", 0.0) * region_score
        )

        # ── SLA latency penalty (Section 3.4.3) ──
        if latency_ms_effective > request_context["sla_ms"]:
            # Penalty scales with degree of violation (not just flat -0.12)
            overshoot_ratio = latency_ms_effective / max(request_context["sla_ms"], 1.0) - 1.0
            css_score -= min(0.12 + 0.05 * overshoot_ratio, 0.25)

        # ── Accuracy floor penalty ──
        if accuracy < request_context["accuracy_floor"]:
            css_score -= 0.18

        # ── Semantic alignment bonus/malus ──
        candidate_rank = VARIANT_RANK.get(candidate_variant, 1)
        semantic_alignment = max(-0.12, 0.14 - abs(candidate_rank - preferred_rank) * 0.08)
        if recommended_variant and candidate_variant == recommended_variant:
            semantic_alignment += 0.02 + (0.04 * complexity_score)
        # High-carbon period: penalise heavy models not explicitly requested
        if effective_ci >= 450 and candidate_variant in {"full", "moe"} and recommended_variant not in {"full", "moe"}:
            semantic_alignment -= 0.05
        if priority in {"urgent", "high"} and candidate_variant == "ultra-light":
            semantic_alignment -= 0.04
        # MoE bonus for complexity when MoE is available and healthy
        if candidate_variant == "moe" and complexity_score > 0.7:
            semantic_alignment += 0.04

        css_score += semantic_alignment

        net_latency = safe_float(target.get("network_latency_ms"), 0.0)

        ranked.append({
            "target_id":             target.get("id"),
            "model_variant":         target.get("model_variant"),
            "hardware":              target.get("hardware"),
            "region":                target.get("region", "local"),
            "grid_zone":             zone,
            "supports_batching":     bool(target.get("supports_batching", False)),
            "is_moe":                bool(target.get("moe", False)),
            # Multimodal: modality axis + (for diffusion) the carbon-capped
            # denoising-step budget the dispatcher must use for this request.
            "modality":              str(target.get("modality") or "text").lower(),
            "is_diffusion":          bool(target.get("diffusion", False)),
            "diffusion_steps":       int(pre.get("diffusion_steps", 0)),
            "vllm_endpoint_env":     target.get("vllm_endpoint_env"),
            "resolved_model_name":   target.get("vllm_model_id") or target.get("model_id"),
            "estimated_latency_ms":  round(latency_ms_effective, 2),
            "moe_comm_latency_ms":   round(moe_comm_ms, 2),
            "estimated_accuracy":    round(accuracy, 3),
            "estimated_cost_units":  round(cost_units, 3),
            # Learned-estimator provenance (for the audit ledger + observe()):
            # baselines are the static config figures; ql_* record the applied
            # per-prompt correction so the update step can reconstruct residuals.
            "baseline_accuracy":     round(baseline_accuracy, 3),
            "baseline_latency_ms":   round(latency_ms, 2),
            "ql_applied":            bool(_ql.get("applied")),
            "ql_accuracy_residual":  _ql.get("accuracy_residual", 0.0),
            "ql_latency_scale":      _ql.get("latency_scale", 1.0),
            # Output-length expectation that priced this candidate's carbon. On
            # the audit entry so a routing decision can be explained after the
            # fact: "it looked greener because we expected it to say less".
            "expected_output_tokens": pre.get("len_meta", {}).get("expected_output_tokens"),
            "ql_length_scale":       pre.get("len_meta", {}).get("length_scale", 1.0),
            "length_factor":         pre.get("len_meta", {}).get("length_factor", 1.0),
            "max_output_tokens":     pre.get("len_meta", {}).get("max_output_tokens"),
            # Full carbon breakdown
            "estimated_carbon_g":    round(total_carbon_g, 8),
            "op_carbon_g":           round(op_carbon_g, 8),
            "emb_carbon_g":          round(emb_carbon_g, 8),
            "effective_ci":          round(effective_ci, 2),
            "pue":                   safe_float(target.get("pue"), 1.3),
            "hardware_efficiency":   safe_float(target.get("hardware_efficiency"), 0.6),
            # Scores
            "carbon_score":          round(carbon_score, 4),
            "latency_score":         round(latency_score, 4),
            "accuracy_score":        round(accuracy_score, 4),
            "cost_score":            round(cost_score, 4),
            "semantic_alignment":    round(semantic_alignment, 4),
            "region_score":          round(region_score, 4),
            "css_score":             round(css_score, 4),
        })

    return sorted(ranked, key=lambda item: item["css_score"], reverse=True)


# ── EcoServe scheduling (Section 3.5) ──────────────────────────────────────────

def evaluate_ecoserve_actions(
    request_context: dict[str, Any],
    grid_carbon: float,
    ranked_candidates: list[dict[str, Any]],
    forecast: list[dict[str, Any]] | None = None,
    zone_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate EcoServe scheduling primitives (Section 3.5):
    1. Carbon-aware batching (deferral) — Section 3.5.2
    2. Regional routing — Section 3.5.3
    3. Load shaping signal
    """
    selected = ranked_candidates[0] if ranked_candidates else None
    high_carbon_period = grid_carbon >= _HIGH_CARBON_THRESHOLD
    deferral_budget = int(request_context.get("deferral_tolerance_ms", 0))

    # ── 1. Deferral (Section 3.5.2) ──
    best_window: dict[str, Any] | None = None
    if forecast and deferral_budget > 0 and high_carbon_period:
        from monitoring_layer import find_low_carbon_window
        best_window = find_low_carbon_window(forecast, deferral_budget)

    deferral_recommended = bool(
        deferral_budget > 0
        and high_carbon_period
        and selected
        and selected.get("supports_batching")
        # Only defer if the best window is meaningfully better
        and (best_window is None or best_window.get("carbon_intensity", grid_carbon) < grid_carbon * 0.85)
    )

    # ── 2. Regional reroute (Section 3.5.3) ──
    region_preference = request_context.get("region_preference", "local")
    regional_reroute = bool(
        selected
        and region_preference
        and selected.get("region") != region_preference
    )

    # Find lowest-carbon available region from zone signals
    best_region: str | None = None
    if zone_signals:
        best_region = min(zone_signals.items(), key=lambda kv: safe_float(kv[1].get("carbon_intensity"), 999))[0]

    # ── 3. Load shaping ──
    load_shape_recommended = high_carbon_period and not deferral_recommended

    reasoning = []
    if deferral_recommended:
        window_info = f" (best window: {best_window.get('datetime', '?')} @ {best_window.get('carbon_intensity', '?'):.0f})" if best_window else ""
        reasoning.append(f"High carbon period ({grid_carbon:.0f} gCO₂/kWh); request can be deferred into a lower-carbon window{window_info}.")
    if regional_reroute:
        reasoning.append(f"Best CSS candidate is in region '{selected.get('region')}' but preference is '{region_preference}'.")
    if best_region and best_region != "local":
        reasoning.append(f"Lowest-carbon region currently: '{best_region}'.")
    if selected and selected.get("supports_batching"):
        reasoning.append("Selected target supports batch-friendly execution.")
    if selected and selected.get("is_moe"):
        reasoning.append("MoE model selected; expert placement and all-to-all scheduling active.")
    if load_shape_recommended:
        reasoning.append("Consider spreading batch workloads across low-carbon time windows.")

    return {
        "deferral_recommended":   deferral_recommended,
        "deferral_window_ms":     min(deferral_budget, 1_800_000) if deferral_recommended else 0,
        "best_low_carbon_window": best_window,
        "high_carbon_period":     high_carbon_period,
        "regional_reroute":       regional_reroute,
        "best_carbon_region":     best_region,
        "load_shape_recommended": load_shape_recommended,
        "reasoning":              reasoning,
    }


# ── Semantic prompt profiler (preserving + MoE extension) ─────────────────────

def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _hashed_embedding(text: str, dimension: int = 256) -> list[float]:
    vector = [0.0] * dimension
    for token in tokenize(text):
        index = sum(ord(c) for c in token) % dimension
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _load_semantic_model():
    global _semantic_model, _semantic_backend
    if _semantic_model is not None:
        return _semantic_model
    if SentenceTransformer is None:
        _semantic_backend = "heuristic-fallback"
        return None
    with _semantic_lock:
        if _semantic_model is not None:
            return _semantic_model
        try:
            _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            _semantic_backend = SEMANTIC_MODEL_NAME
        except Exception:
            _semantic_model = None
            _semantic_backend = "heuristic-fallback"
        return _semantic_model


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _load_semantic_model()
    if model is not None:
        try:
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [list(map(float, v)) for v in vectors]
        except Exception:
            pass
    return [_hashed_embedding(t) for t in texts]


def _prototype_embeddings(name: str, prototypes: dict[str, list[str]]) -> dict[str, list[list[float]]]:
    cached = _semantic_embeddings.get(name)
    if cached is not None:
        return cached
    labels: dict[str, list[list[float]]] = {}
    texts, owners = [], []
    for label, pts in prototypes.items():
        for pt in pts:
            texts.append(pt)
            owners.append(label)
    embeddings = _embed_texts(texts)
    for label, emb in zip(owners, embeddings):
        labels.setdefault(label, []).append(emb)
    _semantic_embeddings[name] = labels
    return labels


def _semantic_label_scores(text: str, prototypes: dict[str, list[str]], cache_name: str) -> dict[str, float]:
    if not text.strip():
        return {label: 0.0 for label in prototypes}
    query_embedding = _embed_texts([text])[0]
    proto_embeddings = _prototype_embeddings(cache_name, prototypes)
    return {
        label: max(_dot_product(query_embedding, emb) for emb in embs) if embs else 0.0
        for label, embs in proto_embeddings.items()
    }


def _best_label(scores: dict[str, float], default: str) -> str:
    if not scores:
        return default
    return max(scores.items(), key=lambda item: item[1])[0]


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in terms)


def infer_prompt_profile(
    prompt: str,
    attachments: list[dict[str, Any]] | None = None,
    persist_attachments: bool = False,
    conversation_message_count: int = 0,
    moe_available: bool = False,
) -> dict[str, Any]:
    """
    Classify intent, complexity, recommended model variant, priority, and SLA.
    Extended with MoE awareness: prompts scoring high on 'moe' prototype and with
    high complexity can recommend the MoE variant when available.
    Preserves all original heuristics.
    """
    attachments = attachments or []
    normalized_prompt = (prompt or "").strip()
    lowered_prompt = normalized_prompt.lower()
    tokens = tokenize(normalized_prompt)

    # Modality is resolved up front; when it is not plain text the multimodal
    # override near the end of this function pins the model variant to a VLM
    # (vision) or diffusion (image-gen) candidate. Text heuristics still run so
    # the profile carries intent/complexity for the audit trail.
    modality = detect_request_modality(normalized_prompt, attachments)
    image_attachment_count = sum(1 for a in attachments if attachment_is_image(a))

    route_scores = _semantic_label_scores(normalized_prompt, SEMANTIC_ROUTE_PROTOTYPES, "route")
    priority_scores = _semantic_label_scores(normalized_prompt, SEMANTIC_PRIORITY_PROTOTYPES, "priority")
    intent_scores = _semantic_label_scores(normalized_prompt, SEMANTIC_INTENT_PROTOTYPES, "intent")

    # Intent classification: keyword-first (high confidence), then a guarded
    # semantic fallback. The hashed-embedding scorer is noisy and routinely
    # labels trivia like "give me a random fact" as "troubleshooting" or
    # "who wrote hamlet" as "implementation" — a single accidental overlap.
    # That mislabel cascades: accuracy_floor jumps to 0.78 (filtering out the
    # greenest candidates) and quality_guardrail_reasons gets populated
    # (triggering the auto-escalate-to-full retry). Both wreck green routing.
    import re
    intent: str | None = None
    intent_source = "default"
    _intent_keyword_hits: dict[str, int] = {}
    for _label, _kws in INTENT_KEYWORDS.items():
        _hits = sum(1 for kw in _kws if re.search(rf"\b{re.escape(kw)}\b", lowered_prompt))
        if _hits:
            _intent_keyword_hits[_label] = _hits
    # High-floor intents bump accuracy_floor to 0.78 (filtering out the
    # greenest candidates) and trigger quality_guardrail_reasons. They must
    # never be assigned by the noisy hashed-embedding scorer alone — too many
    # false positives ("give me a random fact" → troubleshooting=0.37,
    # explanation=0.27, clearing any reasonable margin). Require keyword
    # evidence for these three labels; the rest can use semantic-with-margin.
    _HIGH_FLOOR_INTENTS = {"analysis", "troubleshooting", "implementation"}
    if _intent_keyword_hits:
        intent = max(_intent_keyword_hits.items(), key=lambda kv: kv[1])[0]
        intent_source = "keyword"
    else:
        _sorted_intents = sorted(intent_scores.items(), key=lambda kv: kv[1], reverse=True)
        _top_label, _top_score = (_sorted_intents[0] if _sorted_intents else ("explanation", 0.0))
        _runner_score = _sorted_intents[1][1] if len(_sorted_intents) > 1 else 0.0
        if (
            _top_label not in _HIGH_FLOOR_INTENTS
            and _top_score >= 0.25
            and _top_score >= _runner_score + 0.05
        ):
            intent = _top_label
            intent_source = "semantic"
        else:
            intent = "explanation"
            intent_source = "semantic-fallback"
    if re.match(r"^(?:what|who|where|why)\s+(?:is|are|does|do)\b", lowered_prompt):
        intent = "explanation"
        intent_source = "wh-question"
    # Default to ultra-light, not medium: the greenest candidate is the right
    # zero-evidence prior. Heuristic rules below escalate when document work,
    # reasoning, STEM, or complexity demand it.
    recommended_model_variant = _best_label(route_scores, "ultra-light")
    priority = _best_label(priority_scores, "medium")
    reasoning: list[str] = []

    token_count = len(tokens)
    attachment_chars = sum(len(a.get("context_text") or "") for a in attachments)
    has_documents = bool(attachments)
    wants_document_work = has_documents or _contains_any(lowered_prompt, DOCUMENT_KEYWORDS)
    wants_reasoning = _contains_any(lowered_prompt, REASONING_KEYWORDS)
    is_urgent = _contains_any(lowered_prompt, URGENT_KEYWORDS)

    # Creative-writing fast path — a haiku/poem/lyrics/joke request must never
    # reach the STEM keyword check, because words like "tree", "ecosystem",
    # "cell", "energy" are common in nature poetry but trigger stem-coding /
    # stem-science routing. Detected first; clears stem_domain and forces the
    # intent to "creative" (which has its own prototype so the semantic
    # classifier is also pinned).
    _is_creative = bool(CREATIVE_PATTERN.search(lowered_prompt))

    # STEM domain detection (word-boundary regex; substring matching used to
    # mis-route "graph"→ph, "internal"→rna, "database"→base, "excellent"→cell)
    stem_domain: str | None = None
    # Track how stem_domain was determined: "keyword" = a STEM keyword regex
    # matched (high confidence); "semantic" = backfilled from the classifier
    # below (low confidence). decision_engine reads this to decide whether
    # the STEM candidate hoist can bypass the complexity gate.
    stem_source: str | None = None
    if _is_creative:
        _stem_math_hits = _stem_sci_hits = _stem_code_hits = 0
        intent = "creative"
    else:
        _stem_math_hits = len(STEM_MATH_PATTERN.findall(lowered_prompt))
        _stem_sci_hits  = len(STEM_SCIENCE_PATTERN.findall(lowered_prompt))
        _stem_code_hits = len(STEM_CODING_PATTERN.findall(lowered_prompt))
        # Broad coding tokens (api/software/function/code…) only count as coding
        # evidence when a coding verb or an unambiguous coding token co-occurs,
        # so "software update policy" / "dress code" don't mis-route to the heavy
        # coding model (carbon-unsafe). "write a function", "call the API", or
        # anything already carrying a strong token still counts.
        _broad_code_hits = len(STEM_CODING_BROAD_PATTERN.findall(lowered_prompt))
        if _broad_code_hits and (_stem_code_hits > 0 or CODING_VERB_PATTERN.search(lowered_prompt)):
            _stem_code_hits += _broad_code_hits
        _max_stem_hits = max(_stem_math_hits, _stem_sci_hits, _stem_code_hits)
        if _max_stem_hits >= 1:
            if _stem_math_hits >= _stem_sci_hits and _stem_math_hits >= _stem_code_hits:
                stem_domain = "math"
            elif _stem_sci_hits >= _stem_code_hits:
                stem_domain = "science"
            else:
                stem_domain = "coding"
            stem_source = "keyword"

    # General-knowledge domain detection (literature, history, psychology,
    # astrology, astronomy). These don't pin a dedicated model — the router
    # records the label and lets CSS pick the greenest feasible candidate.
    # Skipped when a STEM keyword already matched (STEM is more specific) or
    # creative-writing fast path fired.
    general_domain: str | None = None
    if not _is_creative and stem_domain is None:
        _general_hits = {
            name: len(pattern.findall(lowered_prompt))
            for name, pattern in GENERAL_DOMAIN_PATTERNS.items()
        }
        _top_general = max(_general_hits.values(), default=0)
        if _top_general >= 1:
            # Tie-break by declaration order in GENERAL_DOMAIN_PATTERNS
            for name, count in _general_hits.items():
                if count == _top_general:
                    general_domain = name
                    break

    complexity_score = min(
        1.0,
        0.12
        + min(token_count / 60.0, 0.26)
        + (0.28 if has_documents else 0.0)
        + (0.14 if persist_attachments else 0.0)
        + (0.14 if wants_reasoning else 0.0)
        + (0.08 if conversation_message_count >= 6 else 0.0)
        + min(attachment_chars / 4000.0, 0.24),
    )

    # ── Heuristic routing rules (original, preserved) ──
    if wants_document_work and (persist_attachments or attachment_chars > 900):
        recommended_model_variant = "full"
        reasoning.append("document-grounded request")
    elif has_documents:
        recommended_model_variant = "medium" if attachment_chars < 1200 else "full"
        reasoning.append("attachments-present")
    elif _is_creative:
        # Creative writing routes to the chat model, not STEM variants. Pinned
        # here so the noisy hashed-embedding fallback can't override.
        recommended_model_variant = "medium"
        reasoning.append("creative writing request")
    elif stem_domain:
        # STEM queries need higher accuracy; route to domain-specific variant
        _stem_variant_map = {"math": "stem-math", "science": "stem-science", "coding": "stem-coding"}
        recommended_model_variant = _stem_variant_map[stem_domain]
        reasoning.append(f"STEM domain detected: {stem_domain}")
    elif wants_reasoning:
        recommended_model_variant = "medium"
        reasoning.append("reasoning-oriented prompt")
    elif general_domain and token_count <= 28 and not wants_reasoning:
        # Knowledge-recall domains (literature/history/psychology/astrology/
        # astronomy): short prompts get TinyLlama, longer ones medium. Driven
        # by length, not topic — the greenest competent model wins.
        recommended_model_variant = "ultra-light" if token_count <= 14 else "medium"
        reasoning.append(f"general-knowledge domain ({general_domain}): {recommended_model_variant}")
    elif token_count <= 16 and not wants_reasoning:
        # Greenest-by-default: short generic prompts route to TinyLlama
        # regardless of intent. "give me random fact", "what is python", and
        # similar trivia don't need a 1.5B chat model.
        recommended_model_variant = "ultra-light"
        reasoning.append("short prompt: greenest viable model")
    elif token_count <= 28 and intent == "explanation":
        recommended_model_variant = "medium"
        reasoning.append("general explanation")

    if recommended_model_variant == "full" and not wants_document_work and token_count < 18:
        recommended_model_variant = "medium"
        reasoning.append("downgraded full route for short non-document prompt")

    # ── MoE upgrade: only when available and complexity warrants it (Section 5) ──
    if (
        moe_available
        and recommended_model_variant == "full"
        and complexity_score >= 0.75
        and route_scores.get("moe", 0.0) >= route_scores.get("full", 0.0) * 0.9
    ):
        recommended_model_variant = "moe"
        reasoning.append("high-complexity prompt: MoE variant recommended for sparse efficiency")

    # ── Multimodal override ──
    # An image attachment or an explicit generation request pins the model
    # variant onto the multimodal axis; CSS then picks the greenest feasible
    # candidate within that modality (e.g. SDXL vs FLUX for generation).
    if modality == "vision":
        recommended_model_variant = "vlm"
        reasoning.append(f"image attachment(s) detected ({image_attachment_count}): VLM analysis")
    elif modality == "image-gen":
        # Greenest-default: recommend SDXL; CSS may still pick FLUX if quality
        # floors demand it. Generation is latency-tolerant, so bias to low
        # priority (long deferral tolerance) unless the user flagged urgency.
        recommended_model_variant = "diffusion-sdxl"
        reasoning.append("image-generation request: diffusion candidate")

    # ── Priority determination ──
    # Multimodal variants are decided first and use *word-boundary* urgency (the
    # substring URGENT_KEYWORDS check misfires on tokens like "s-now-y" → "now",
    # which would wrongly mark a deferrable image render as urgent).
    _urgent_wb = bool(re.search(
        r"\b(?:urgent|asap|immediately|now|prod|production|incident|outage|sev1|critical|broken|failing)\b",
        lowered_prompt,
    ))
    if recommended_model_variant.startswith("diffusion-"):
        # Image generation is latency-tolerant → low priority = long deferral
        # tolerance, so EcoServe can shift it into a low-carbon window (unless
        # the user genuinely flagged urgency).
        priority = "urgent" if _urgent_wb else "low"
        reasoning.append(
            "image-generation: urgent" if _urgent_wb
            else "image-generation: deferrable (low priority)"
        )
    elif recommended_model_variant == "vlm":
        priority = "urgent" if _urgent_wb else "medium"
    elif is_urgent:
        priority = "urgent"
        reasoning.append("urgent language detected")
        if recommended_model_variant == "moe":
            recommended_model_variant = "full"  # MoE latency too high for urgent
            reasoning.append("MoE downgraded to full for urgent SLA")
    elif recommended_model_variant in {"full", "moe"}:
        priority = "high"
    elif recommended_model_variant == "ultra-light" and token_count <= 12:
        priority = "low"
    elif priority not in {"high", "urgent"}:
        priority = "medium"

    mode_map = {
        "ultra-light": "fast", "medium": "balanced", "full": "accurate", "moe": "accurate",
        "stem-math": "accurate", "stem-science": "accurate", "stem-coding": "balanced",
        "vlm": "accurate", "diffusion-sdxl": "balanced", "diffusion-flux": "accurate",
    }
    mode = mode_map.get(recommended_model_variant, "balanced")

    accuracy_floor_map = {
        "ultra-light": 0.60, "medium": 0.75, "full": 0.88, "moe": 0.90,
        "stem-math": 0.90, "stem-science": 0.88, "stem-coding": 0.85,
        # Multimodal floors: VLM analysis needs solid grounding; image-gen
        # "accuracy" is a perceptual-quality proxy, so floors are permissive
        # enough that the greener diffusion candidate stays feasible.
        "vlm": 0.82, "diffusion-sdxl": 0.70, "diffusion-flux": 0.80,
    }
    latency_sla_map = {
        "ultra-light": 140, "medium": 240, "full": 380, "moe": 500,
        "stem-math": 400, "stem-science": 380, "stem-coding": 300,
        # Multimodal is inherently slower; generation SLAs are generous because
        # image-gen is latency-tolerant (and preferentially EcoServe-deferred).
        "vlm": 1200, "diffusion-sdxl": 6000, "diffusion-flux": 12000,
    }
    deferral_map = {"low": 1_800_000, "medium": 900_000, "high": 300_000, "urgent": 0}

    accuracy_floor = accuracy_floor_map.get(recommended_model_variant, 0.75)
    if intent in {"analysis", "implementation", "troubleshooting"}:
        accuracy_floor = max(accuracy_floor, 0.78)
    if wants_document_work:
        accuracy_floor = max(accuracy_floor, 0.86)

    sla_ms = latency_sla_map.get(recommended_model_variant, 240)
    if priority == "urgent":
        sla_ms = min(sla_ms, 120)
    elif priority == "high":
        sla_ms = min(sla_ms, 180)

    complexity_label = (
        "simple" if complexity_score < 0.34
        else "moderate" if complexity_score < 0.67
        else "advanced"
    )

    # Backfill stem_domain when keyword detection missed but the semantic
    # classifier seeded recommended_model_variant to a stem-* variant.
    # Without this, decision_engine's STEM candidate hoist (which keys off
    # stem_domain) skips, and CSS picks a non-STEM target despite the
    # routing recommendation.
    #
    # Confidence margin: a short generic prompt like "give me random fact"
    # can score stem-science=0.15 with every other route at 0.0 — a single
    # hashed-embedding overlap, not a real STEM signal. Require the top stem
    # score to clear an absolute floor AND beat the top non-stem score by a
    # margin before treating it as STEM. If the margin fails, drop the
    # stem-* recommendation back to a sensible non-STEM variant.
    if stem_domain is None and recommended_model_variant.startswith("stem-"):
        _top_stem = max(
            route_scores.get("stem-math", 0.0),
            route_scores.get("stem-science", 0.0),
            route_scores.get("stem-coding", 0.0),
        )
        _top_non_stem = max(
            route_scores.get("ultra-light", 0.0),
            route_scores.get("medium", 0.0),
            route_scores.get("full", 0.0),
            route_scores.get("moe", 0.0),
        )
        if _top_stem >= 0.25 and _top_stem >= _top_non_stem + 0.05:
            stem_domain = recommended_model_variant.split("-", 1)[1]
            stem_source = "semantic"
            reasoning.append(
                f"stem_domain inferred from semantic classifier: {stem_domain} "
                f"(score={_top_stem:.2f}, margin={_top_stem - _top_non_stem:.2f})"
            )
        else:
            reasoning.append(
                f"semantic classifier suggested {recommended_model_variant} "
                f"but score={_top_stem:.2f} below confidence threshold; "
                f"falling back to non-STEM routing"
            )
            if token_count <= 14 and intent in {"explanation", "summarization"} and not wants_reasoning:
                recommended_model_variant = "ultra-light"
            else:
                recommended_model_variant = "medium"

    return {
        "priority":                   priority,
        "mode":                       mode,
        "intent":                     intent,
        "complexity_score":           round(complexity_score, 3),
        "complexity_label":           complexity_label,
        "recommended_model_variant":  recommended_model_variant,
        "accuracy_floor":             round(accuracy_floor, 3),
        "sla_ms":                     int(sla_ms),
        "deferral_tolerance_ms":      deferral_map.get(priority, 900_000),
        "reasoning":                  reasoning or ["semantic classifier"],
        "classifier_backend":         _semantic_backend,
        "route_scores":               {k: round(v, 4) for k, v in route_scores.items()},
        "priority_scores":            {k: round(v, 4) for k, v in priority_scores.items()},
        "intent_scores":              {k: round(v, 4) for k, v in intent_scores.items()},
        "has_attachments":            has_documents,
        "attachment_characters":      attachment_chars,
        "token_count":                token_count,
        "moe_considered":             moe_available,
        "stem_domain":                stem_domain,
        "stem_source":                stem_source,
        "intent_source":              intent_source,
        "general_domain":             general_domain,
        "modality":                   modality,
        "image_attachment_count":     image_attachment_count,
    }


# ── Request context builder ───────────────────────────────────────────────────

def build_request_context(payload: dict[str, Any], policy_config: dict[str, Any]) -> dict[str, Any]:
    semantic_profile = dict(payload.get("semantic_profile") or {})

    payload_priority = payload.get("priority") if RESPECT_MANUAL_ROUTING_INPUTS else ""
    payload_mode = payload.get("mode") if RESPECT_MANUAL_ROUTING_INPUTS else ""

    priority = str(payload_priority or semantic_profile.get("priority") or "medium").strip().lower()
    mode = str(payload_mode or semantic_profile.get("mode") or "balanced").strip().lower()
    user_tier = str(payload.get("user_tier", "standard")).strip().lower()

    default_sla = {"urgent": 120, "high": 180, "medium": 260, "low": 420}.get(priority, 260)
    default_sla = int(safe_float(semantic_profile.get("sla_ms"), default_sla))
    if mode == "fast":
        default_sla = min(default_sla, 160)

    default_accuracy = {"urgent": 0.84, "high": 0.82, "medium": 0.76, "low": 0.62}.get(priority, 0.76)
    default_accuracy = safe_float(semantic_profile.get("accuracy_floor"), default_accuracy)
    if mode == "accurate":
        default_accuracy = max(default_accuracy, 0.88)

    default_deferral = 0 if priority in {"urgent", "high"} else 900_000
    default_deferral = int(safe_float(semantic_profile.get("deferral_tolerance_ms"), default_deferral))
    if mode == "fast":
        default_deferral = 0

    explicit_model_preference = str(payload.get("model_preference", "")).strip().lower()
    recommended_model_variant = str(semantic_profile.get("recommended_model_variant", "")).strip().lower()

    return {
        "priority":               priority,
        "mode":                   mode,
        "user_tier":              user_tier,
        "sla_ms":                 int(safe_float(payload.get("sla_ms"), default_sla)),
        "accuracy_floor":         safe_float(payload.get("accuracy_floor"), default_accuracy),
        "deferral_tolerance_ms":  int(safe_float(payload.get("deferral_tolerance_ms"), default_deferral)),
        "region_preference":      str(payload.get("region_preference", "local")).strip().lower() or "local",
        "model_preference":       explicit_model_preference,
        "recommended_model_variant": recommended_model_variant,
        "semantic_profile":       semantic_profile,
        "policy_coefficients":    resolve_policy(user_tier, policy_config),
    }

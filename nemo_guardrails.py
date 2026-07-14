"""
NemoGuardrails integration for Adaptive Green AI.

Provides input and output safety rails with a categorized pipeline trace.
Runs in programmable-rail mode (no external LLM needed for safety checks).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

GUARDRAILS_ENABLED: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# LLM-based safety classifier (supplementary net, rail I.9).
#
# The deterministic regex rails above are precise but literal: they key on
# explicit verb phrases ("how to kill") and miss euphemistic / paraphrased
# intent ("get rid of the pope", "take out the senator", "make him disappear").
# This classifier sends the prompt to a small instruct model (the same vLLM
# backend the app already runs) for a SAFE / UNSAFE judgement, closing that gap.
#
# It runs *only* when the regex rails did not already block — so the common
# path pays no extra inference — and it fails OPEN: an unreachable or malformed
# classifier never blocks legitimate traffic, leaving the regex rails as the
# guaranteed floor.
# ---------------------------------------------------------------------------
GUARDRAILS_LLM_CLASSIFIER_ENABLED: bool = os.getenv(
    "GUARDRAILS_LLM_CLASSIFIER_ENABLED", "true"
).lower() in {"1", "true", "yes"}
# Backed by the dedicated Llama Guard 3 container (docker-compose `vllm-guard`).
GUARDRAILS_LLM_URL: str = (
    os.getenv("GUARDRAILS_LLM_URL")
    or os.getenv("VLLM_GUARD_URL", "http://127.0.0.1:8008/v1")
)
GUARDRAILS_LLM_MODEL: str = os.getenv(
    "GUARDRAILS_LLM_MODEL", "meta-llama/Llama-Guard-3-1B"
)
# Prompt/response protocol: "guard" (Llama Guard chat template + safe/unsafe\nSxx
# output) or "instruct" (a generic instruct model with our own system prompt).
# "auto" infers "guard" from a model name containing "guard".
GUARDRAILS_LLM_MODE: str = os.getenv("GUARDRAILS_LLM_MODE", "auto").lower()
GUARDRAILS_LLM_TIMEOUT: float = float(os.getenv("GUARDRAILS_LLM_TIMEOUT", "6"))

# Defense-in-depth: run a SECOND, independent classifier (a generic instruct
# model, prompted by _SAFETY_SYSTEM_PROMPT) alongside Llama Guard and block if
# EITHER flags. Covers the 1B guard's euphemism blind spots (e.g. "make the
# senator disappear") that the instruct prompt catches. The instruct layer
# defaults to the already-running vllm-full backend, so it adds a call but no
# VRAM. The layers short-circuit — the instruct call only fires when Llama Guard
# allowed — so caught requests still cost a single inference.
GUARDRAILS_DEFENSE_IN_DEPTH: bool = os.getenv(
    "GUARDRAILS_DEFENSE_IN_DEPTH", "true"
).lower() in {"1", "true", "yes"}
GUARDRAILS_INSTRUCT_URL: str = (
    os.getenv("GUARDRAILS_INSTRUCT_URL")
    or os.getenv("VLLM_FULL_URL", "http://127.0.0.1:8002/v1")
)
GUARDRAILS_INSTRUCT_MODEL: str = os.getenv(
    "GUARDRAILS_INSTRUCT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)

# Llama Guard 3 hazard taxonomy (MLCommons) — used to turn the model's terse
# "unsafe\nS9" verdict into a human-readable block reason.
_LLAMA_GUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

# Fallback system prompt for "instruct" mode (a generic chat model standing in
# for Llama Guard). Llama Guard itself ignores this — its own chat template
# supplies the moderation prompt — so it is only sent when mode == "instruct".
_SAFETY_SYSTEM_PROMPT = (
    "You are a strict content-safety classifier for a chat assistant. "
    "Judge ONLY the user message below and decide if it requests genuinely "
    "harmful assistance.\n"
    "Treat as UNSAFE: violence or physical harm toward a person or group "
    "(murder, assassination, and euphemisms such as 'get rid of', 'take out', "
    "'eliminate', 'make disappear', 'deal with' a named person); weapons, "
    "explosives, or bioweapon creation; sexual content involving minors; "
    "facilitation of suicide or self-harm; or targeted wrongdoing against a "
    "specific victim (hacking their account, doxxing them).\n"
    "Treat as SAFE: ordinary, educational, historical, political, fictional, "
    "or technical questions — including software phrasings like 'kill a "
    "process' or 'kill the server', news and analysis, and general "
    "how-things-work questions.\n"
    "Answer with EXACTLY one line: 'SAFE' or 'UNSAFE: <short reason>'. "
    "Output nothing else."
)


def _classifier_protocol() -> str:
    """Resolve the wire protocol for the configured safety model."""
    if GUARDRAILS_LLM_MODE in ("guard", "instruct"):
        return GUARDRAILS_LLM_MODE
    return "guard" if "guard" in GUARDRAILS_LLM_MODEL.lower() else "instruct"

# ---------------------------------------------------------------------------
# Helper: Luhn validation for credit-card-format runs.
# ---------------------------------------------------------------------------

def _luhn_valid(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _compile_all(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ---------------------------------------------------------------------------
# Categorized input pattern groups (rendered as I.1, I.2, ...).
# Each group runs in order; all groups run regardless of earlier blocks so
# the trace can show every verdict. The first block sets the top-level
# blocked flag and chooses the safe-replacement text.
# ---------------------------------------------------------------------------

_INPUT_PROMPT_INJECTION = _compile_all([
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b",
    r"\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?(?:evil|unrestricted|unfiltered|jailbroken)\b",
    r"\bdan\s+mode\b",
    r"\bdo\s+anything\s+now\b",
    r"\byou\s+are\s+now\s+(?:free|unrestricted|without\s+(?:rules|restrictions|guidelines))\b",
    r"\bpretend\s+(?:you\s+(?:have\s+)?no|there\s+are\s+no)\s+(?:limits|restrictions|rules|guidelines)\b",
    r"\bforget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions?|training|guidelines?|rules?)\b",
    r"<\s*(?:system|user|assistant)\s*>.*?<\s*/\s*(?:system|user|assistant)\s*>",
    r"\[\s*SYSTEM\s*\]",
    r"###\s*(?:INSTRUCTION|SYSTEM|OVERRIDE)",
])

_INPUT_VIOLENT_HARM = _compile_all([
    r"\bhow\s+(?:to\s+)?(?:to\s+)?kill\b",
    r"\bhow\s+(?:do\s+(?:i|you|we)|can\s+(?:i|you|we)|to)\s+(?:murder|assassinate|strangle|stab|shoot|poison|torture|behead|execute)\b",
    # Lethal verb + human target in ANY phrasing — catches "how do i kill the
    # pope", "easiest way to kill someone", "i want to kill him". Gated to a
    # person/people object so the very common benign technical use ("kill a
    # process", "kill the server", "kill the lights") does NOT false-positive.
    r"\b(?:kill|murder|assassinate|strangle|stab|shoot|poison|behead|execute|"
    r"lynch|slaughter)\s+(?:the|a|an|my|your|this|that|his|her|their|some)?\s*"
    r"(?:pope|president|king|queen|prime\s+minister|senator|congress(?:man|woman)|"
    r"politician|official|celebrity|someone|somebody|anyone|a\s+person|persons?|"
    r"people|humans?|him|her|them|man|men|woman|women|child|children|kids?|baby|"
    r"neighbou?r|boss|teacher|wife|husband|ex|girlfriend|boyfriend|classmate|"
    r"co[\s\-]?worker|colleague|victim|hostage|prisoner)\b",
    r"\bstep[s\-]?\s*(?:to|by\s+step|for)\s+(?:kill|murder|assassinat|poison|strang|tortur)\w*",
    r"\bprocess\s+(?:of\s+)?(?:killing|murdering|assassinating|torturing)\b",
    r"\bways?\s+to\s+(?:kill|murder|harm|hurt|injure|torture|assassinate)\b",
    r"\b(?:best|easiest|quickest|fastest)\s+way\s+to\s+(?:kill|murder|hurt|harm)\b",
    r"\bhow\s+(?:do\s+(?:i|you))\s+(?:get\s+away\s+with\s+(?:murder|killing)|commit\s+murder)\b",
    r"\b(?:make|build|create|synthesize|produce)\s+(?:a\s+)?(?:bomb|explosive|weapon|poison\s+gas|nerve\s+agent|chemical\s+weapon|bioweapon|dirty\s+bomb)\b",
    r"\b(?:instructions?|steps?|guide|tutorial)\s+(?:for|to)\s+(?:build|make|create)\s+(?:a\s+)?(?:bomb|explosive|weapon)\b",
    r"\bhow\s+to\s+(?:make|build|create)\s+(?:a\s+)?(?:gun|firearm|explosive|bomb)\b",
    r"\b(?:synthesize|manufacture)\s+(?:drugs?|meth|heroin|fentanyl|cocaine)\b",
    r"\bhow\s+to\s+(?:commit\s+suicide|kill\s+(?:my|your)self|self.?harm|overdose)\b",
    r"\bbest\s+(?:way|method)\s+to\s+(?:end\s+(?:my|your)\s+life|die|commit\s+suicide)\b",
])

_INPUT_SEXUAL_CSAM = _compile_all([
    r"\bchild\s+(?:sexual|porn|nude|naked|exploit)",
    r"\b(?:sexual|naked|nude)\s+(?:content|image|photo)s?\s+(?:of\s+)?(?:child|minor|underage|teen)\b",
    r"\bhow\s+(?:do\s+(?:i|you|we|one)\s+|to\s+|can\s+(?:i|you|we)\s+)?(?:fuck|fucking|get\s+laid|hook\s*up|have\s+sex|perform\s+oral\s+sex|give\s+(?:a\s+)?(?:blowjob|hand\s*job|rim\s*job))\b",
    r"\b(?:write|generate|create|produce|tell|give|compose|make)\s+(?:me\s+|us\s+)?(?:a\s+|some\s+|an?\s+)?(?:porn(?:o(?:graph(?:y|ic))?)?|erotica?|smut|nsfw|sexually\s+explicit|x[\s\-]?rated|hentai)\b",
    r"\b(?:explicit|graphic|detailed)\s+(?:sex(?:ual)?|erotic|pornographic)\s+(?:scene|story|description|content|narrative|fanfic)\b",
    r"\bdescribe\s+(?:in\s+(?:graphic|explicit|vivid)\s+detail\s+)?(?:a\s+)?(?:sex(?:ual)?\s+act|intercourse|orgasm|masturbat\w*|penetrat\w*)\b",
    r"\b(?:roleplay|role\s*play|act\s+out)\s+(?:a\s+)?(?:sex(?:ual)?|erotic|porn|nsfw)\s+(?:scene|scenario|encounter)\b",
])

_INPUT_NETWORK_ABUSE = _compile_all([
    r"\b(?:crack|hack(?:ing)?(?:\s+into)?|break\s+into|bypass|guess|brute[\s\-]?force|sniff|steal|deauth)\s+(?:my\s+)?(?:neighbou?r'?s?|friend'?s?|someone'?s?|their|my\s+(?:ex|boss|teacher|landlord|roommate)'?s?|the)\s+(?:wi[\s\-]?fi|wpa\d?|wep|router|password|admin\s+panel|account|email|facebook|instagram|gmail|outlook|bank|router)\b",
    r"\b(?:how\s+(?:to|do\s+(?:i|you|we)|can\s+(?:i|you|we)))\s+(?:to\s+)?(?:crack|hack(?:\s+into)?|break\s+into|bypass|brute[\s\-]?force|gain\s+unauthorized\s+access\s+to)\s+(?:a\s+|the\s+|someone'?s?\s+|my\s+\w+'?s?\s+)?(?:wi[\s\-]?fi|wpa\d?|wep|password|router|account|email|server|network|firewall|encryption|database|admin\s+panel)\b",
    r"\b(?:wi[\s\-]?fi|wpa\d?|wep)\s+(?:password\s+)?(?:cracking|cracker|hacking|attack|brute[\s\-]?force(?:ing)?)\b",
    r"\b(?:aircrack(?:[\s\-]?ng)?|reaver|hashcat|john\s+the\s+ripper|hydra|kismet|wifite|fern[\s\-]?wifi|bettercap)\b.{0,40}\b(?:tutorial|guide|how\s+to|usage|attack|crack|capture\s+handshake|wpa\d?|password)\b",
    r"\b(?:evil\s+twin|deauth(?:entication)?|pmkid|wps\s+pin|handshake\s+capture)\s+attack\b",
])

_INPUT_PII_EXTRACTION = _compile_all([
    r"\b(?:what(?:'s|\s+is)|find|tell\s+me|give\s+me|get|look\s*up|reveal|share|show\s+me)\s+(?:[A-Za-z][\w\-']+(?:\s+[A-Za-z][\w\-']+){0,3}'?s?\s+)?(?:home\s+|personal\s+|private\s+|real\s+|current\s+)?(?:ssn|social\s+security\s+(?:number|#)?|date\s+of\s+birth|dob|credit\s+card\s+(?:number|info|details)?|cc\s+number|bank\s+account|routing\s+number|home\s+address|street\s+address|cell\s+(?:number|phone)|phone\s+number|mobile\s+number|password|pin\s+(?:code|number)?|mother'?s?\s+maiden\s+name|driver'?s?\s+licen[sc]e|passport\s+(?:number|info))\b",
    r"\b(?:doxx?|doxxing|stalk)\s+(?:my\s+)?(?:neighbou?r|ex(?:[\s\-]?(?:boyfriend|girlfriend|wife|husband|partner))?|friend|coworker|colleague|classmate|teacher|boss|person|him|her|them|someone|a\s+person)\b",
    r"\b(?:find|locate|trace)\s+(?:the\s+)?(?:home\s+address|physical\s+address|real[\s\-]?world\s+location|current\s+location|whereabouts)\s+of\s+(?:my\s+|the\s+|a\s+)?(?:user|person|individual|ex|coworker|neighbou?r|stranger|someone)\b",
])

_INPUT_SENSITIVE_TOPICS = _compile_all([
    r"\b(?:weapon|explosive|poison|hack|malware|ransomware|virus|exploit)\b",
    r"\b(?:suicide|self.?harm|overdose)\b",
    r"\b(?:illegal|illicit|drug\s+synth|synthesize\s+drugs?)\b",
])

# (id, label, patterns, mode)  — mode: "structural" | "block" | "warn"
_INPUT_GROUPS: list[tuple[str, str, list[re.Pattern] | None, str]] = [
    ("I.1", "Structural Scanner",        None,                       "structural"),
    ("I.2", "Prompt Injection (DeBERTa)", _INPUT_PROMPT_INJECTION,   "block"),
    ("I.3", "Violent / Harmful Content", _INPUT_VIOLENT_HARM,        "block"),
    ("I.4", "Sexual / CSAM Content",     _INPUT_SEXUAL_CSAM,         "block"),
    ("I.5", "Network / Credential Abuse",_INPUT_NETWORK_ABUSE,       "block"),
    ("I.6", "PII Extraction & Doxxing",  _INPUT_PII_EXTRACTION,      "block"),
    ("I.7", "Intent Classifier",         _INPUT_SENSITIVE_TOPICS,    "warn"),
]


# ---------------------------------------------------------------------------
# Image-generation prompt groups (rendered as G.1, G.2, ...).
#
# A text-to-image prompt *is* a description of the pixels to synthesize, so the
# harmful content is stated declaratively ("photorealistic nude of ...", "a
# beheaded body") rather than as the instructional verb phrases the chat rails
# (I.3/I.4) key on ("how to kill", "write me porn"). These description-oriented
# blocklists close that gap and run only when modality == "image-gen".
# ---------------------------------------------------------------------------

_IMGGEN_CSAM = _compile_all([
    r"\bcsam\b",
    r"\bchild\s+(?:porn|sexual|abuse|exploitation)\b",
    r"\b(?:loli(?:con)?|shota(?:con)?)\b",
    # nudity/sexualization co-occurring with a minor descriptor (either order)
    r"\b(?:child|children|minor|underage|under[\s\-]?age|preteen|pre[\s\-]?teen|"
    r"toddler|infant|kid|little\s+(?:girl|boy)|teen(?:age)?(?:r|d)?|school\s*girl|"
    r"school\s*boy)\b[^.?!]{0,50}\b(?:nude|naked|topless|bottomless|lingerie|"
    r"underwear|panties|sexual|sexualized|sexualised|erotic|provocative|"
    r"seductive|suggestive|fondl\w*)\b",
    r"\b(?:nude|naked|topless|bottomless|lingerie|underwear|panties|sexual|"
    r"sexualized|sexualised|erotic|provocative|seductive|suggestive)\b"
    r"[^.?!]{0,50}\b(?:child|children|minor|underage|under[\s\-]?age|preteen|"
    r"pre[\s\-]?teen|toddler|infant|kid|little\s+(?:girl|boy)|teen(?:age)?(?:r|d)?|"
    r"school\s*girl|school\s*boy)\b",
])

_IMGGEN_SEXUAL = _compile_all([
    # "naked"/"nude" as a bodily descriptor, excluding common idioms
    # ("naked truth/eye/ambition", "nude colour/tone/lipstick").
    r"\b(?:fully\s+|semi[\s\-]?|partially\s+)?naked\b(?!\s+(?:truth|eyes?|"
    r"ambition|aggression|greed|singularity|mole[\s\-]?rat|dna|city|"
    r"and\s+afraid))",
    r"\b(?:fully\s+|semi[\s\-]?|partially\s+)?nude\b(?!\s+(?:colou?r|tone|"
    r"toned|palette|lipstick|shade|beige|pink|nail|heels?))",
    r"\btopless\b|\bbottomless\b",
    r"\b(?:porn(?:o(?:graph(?:y|ic))?)?|hardcore|softcore|xxx|nsfw|hentai|"
    r"rule\s*34|ecchi)\b",
    r"\b(?:genitals?|genitalia|penis|erect(?:ion|\s+penis)|vagina|vulva|"
    r"clitoris|nipples?|areola|exposed\s+breasts?|bare\s+breasts?)\b",
    r"\b(?:having\s+sex|sexual\s+intercourse|sex\s+act|masturbat\w*|"
    r"oral\s+sex|blow\s*job|penetrat\w*|cum\s*shot|ejaculat\w*|orgasm|"
    r"gangbang|bukkake)\b",
    r"\b(?:in|wearing)\s+(?:sexy\s+|skimpy\s+|revealing\s+)?(?:lingerie|"
    r"underwear|thong|panties)\b[^.?!]{0,40}\b(?:seductive|provocative|"
    r"spread|posing|bent\s+over)\b",
])

_IMGGEN_GORE = _compile_all([
    r"\b(?:beheading|decapitat\w*|dismember\w*|disembowel\w*|eviscerat\w*)\b",
    r"\b(?:gore|gory)\b",
    r"\bgraphic\s+(?:violence|gore|death|injury)\b",
    r"\b(?:mutilated|dismembered|bloodied|bloody|severed|charred|burnt)\s+"
    r"(?:body|bodies|corpse|corpses|child|children|victim|victims|head|"
    r"limb|limbs|remains)\b",
    r"\b(?:torturing|executing|execution\s+of|lynching|impaling)\s+(?:a\s+)?"
    r"(?:person|people|child|children|prisoner|hostage|victim|man|woman)\b",
    r"\b(?:realistic|graphic|photorealistic|photo[\s\-]?real)\b"
    r"[^.?!]{0,30}\b(?:murder|massacre|killing|suicide|self[\s\-]?harm|"
    r"mass\s+shooting|genocide)\b",
])

_IMGGEN_SEXUAL_SLANG = _compile_all([
    # Sex-worker / degrading slang used as a subject descriptor.
    r"\b(?:whores?|slut(?:s|ty)?|hookers?|prostitutes?|call\s*girls?|"
    r"cam\s*girls?)\b",
    # Explicit act slang — verb forms of "fuck" and synonyms in a sexual frame.
    r"\b(?:getting|being|gets|while\s+(?:being|getting))\s+"
    r"(?:fuck|bang|rail|pound|screw|finger|penetrat|smash)\w*",
    r"\bfuck(?:ing|ed|s)?\s+(?:her|him|them|herself|himself|each\s+other|"
    r"a\s+(?:girl|guy|man|woman|boy|slut|whore|hooker)|the\s+\w+)\b",
    # "fucking" as an act (subject before, or preposition/context after) —
    # distinguished from the intensifier ("fucking gorgeous"), which is
    # followed by an adjective, not a subject/preposition.
    r"\b(?:people|couple|lovers|them|man|woman|girl|guy|boy|they|two|"
    r"threesome)\s+fuck(?:ing|ed|s)?\b",
    r"\bfuck(?:ing|ed)\s+(?:on|in|against|behind|over|from\s+behind|hard|"
    r"doggy|each\s+other|senseless)\b",
    r"\b(?:blow\s*job|hand\s*job|rim\s*job|deep\s*throat|creampie|"
    r"money\s*shot|titty\s*fuck|face\s*fuck|doggy\s*style|reverse\s+cowgirl)\b",
    # Explicit genital / sexual-object slang (idiom-guarded where ambiguous).
    r"\bpussy\b(?!\s*(?:cat|willow|foot|paws?))",
    r"\b(?:cunt|dildo|vibrator|butt\s*plug|strap[\s\-]?on|titties|boobs|"
    r"big\s+tits|bare\s+tits)\b",
    # Sexual qualifiers only (not "big/huge", which false-positive on a rooster).
    r"\b(?:erect|throbbing|veiny|sucking|riding|his|her)\s+"
    r"(?:cock|dick|shaft)\b",
])

_IMGGEN_NONCONSENSUAL = _compile_all([
    r"\bdeep\s*fake\b",
    r"\bnon[\s\-]?consensual\b",
    r"\b(?:undress|strip|remove\s+(?:the\s+|her\s+|his\s+)?clothes|nudify)\b"
    r"[^.?!]{0,25}\b(?:from|of|this|the\s+(?:person|woman|man|photo|image))\b",
    r"\b(?:nude|naked|topless|sexual|explicit|in\s+lingerie)\b[^.?!]{0,40}\b"
    r"(?:of\s+)?(?:real\s+person|celebrity|politician|actress|actor|"
    r"my\s+(?:ex|classmate|co[\s\-]?worker|colleague|teacher|neighbou?r|"
    r"friend|boss)|a\s+specific\s+(?:person|woman|man))\b",
])

# (id, label, patterns, mode)  — image-generation prompt rail groups.
_IMGGEN_GROUPS: list[tuple[str, str, list[re.Pattern] | None, str]] = [
    ("G.1", "CSAM / Minor Sexualization", _IMGGEN_CSAM,           "block"),
    ("G.2", "Explicit Sexual Imagery",    _IMGGEN_SEXUAL,         "block"),
    ("G.3", "Graphic Violence / Gore",    _IMGGEN_GORE,           "block"),
    ("G.4", "Non-Consensual / Deepfake",  _IMGGEN_NONCONSENSUAL,  "block"),
    ("G.5", "Sexual Slang / Profanity",   _IMGGEN_SEXUAL_SLANG,   "block"),
]


# ---------------------------------------------------------------------------
# Universal explicit-content block — runs on EVERY request at the input gate
# (text chat, vision, image-gen), so the same content is refused regardless of
# which modality the profiler picks (a typo'd "imagfe of…" that lands on the
# text model must not yield explicit text). High-signal slang / hardcore terms
# only; word forms deliberately spare conceptual discussion — "a prostitute"
# blocks, "prostitution [policy]" does not; "hardcore porn" blocks, "hardcore
# fan" does not; "market penetration" is untouched.
# ---------------------------------------------------------------------------
_INPUT_EXPLICIT_HARDCORE = _compile_all([
    r"\b(?:porn(?:o(?:graph(?:y|ic))?)?|hardcore\s+(?:sex|porn|scene|xxx)|"
    r"xxx\s+(?:porn|rated|video|scene|content)|hentai|rule\s*34|bukkake|"
    r"gangbang)\b",
    r"\b(?:oral\s+sex|cum\s*shot|money\s*shot|ejaculat\w*)\b",
])
# CSAM (never legitimate) + explicit slang/acts + hardcore terms.
_INPUT_EXPLICIT_GROUP = _IMGGEN_CSAM + _IMGGEN_SEXUAL_SLANG + _INPUT_EXPLICIT_HARDCORE
_INPUT_GROUPS.append(
    ("I.8", "Explicit Sexual / CSAM", _INPUT_EXPLICIT_GROUP, "block")
)


# ---------------------------------------------------------------------------
# Categorized output pattern groups (rendered as O.1, O.2, ...).
# ---------------------------------------------------------------------------

_OUTPUT_JAILBREAK_ECHO = _compile_all([
    r"\bignore\s+all\s+previous\s+instructions\b",
    r"\bdan\s+mode\b",
])

# Key=value secret leaks. The value is captured (group 1) so a validator can
# tell a real high-entropy secret apart from an instructional placeholder.
# The old `\S+` value matcher blocked *any* token after `password:` — which
# false-positived on legitimate DevOps / Docker / CLI answers like
# "password: <your-password>", "--password=secret", or even prose such as
# "set the password: in the config file". See _looks_like_secret_value.
_OUTPUT_SECRET_KV = _compile_all([
    r"\b(?:password|passwd|pwd|secret[_\-]?key|api[_\-]?key|access[_\-]?token"
    r"|auth[_\-]?token|client[_\-]?secret)\s*[:=]\s*([^\s,;]+)",
])

# Private-key / SSH key material is an unambiguous leak regardless of value,
# so it is matched structurally (full PEM header / real ssh-rsa key body)
# rather than via the value validator.
_OUTPUT_PRIVATE_KEY = _compile_all([
    r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PGP|ENCRYPTED)?\s*PRIVATE\s+KEY-----",
    r"\bssh-rsa\s+AAAA[0-9A-Za-z+/]{20,}",
])

# Tokens that look like a key value but are placeholders / prose, never a leak.
_SECRET_PLACEHOLDER_TOKENS = {
    "password", "passwd", "pwd", "secret", "yourpassword", "mypassword",
    "changeme", "example", "placeholder", "redacted", "none", "null",
    "string", "value", "here", "your", "the", "in", "a", "token",
    "credentials", "credential",
}

_OUTPUT_VIOLENT_ECHO = _compile_all([
    r"\bstep\s*\d+\s*[:\-]\s*(?:stab|shoot|poison|strangle|behead|strik|attack|assault)",
    r"\b(?:stab|shoot|strangle|behead|poison|garrote)\s+(?:the|your|them|him|her)\b",
    r"\bto\s+(?:kill|murder|assassinate)\s+(?:someone|a\s+person|them|him|her)\b",
    r"\b(?:method|process|procedure|way)\s+(?:of|to|for)\s+(?:killing|murdering|assassinating)\b",
    r"\b(?:use|apply|inject)\s+\w+\s+to\s+(?:kill|poison|harm|hurt)\b",
])

# Output PII / financial redaction.
# (pattern, replacement, type_tag, validator_or_None)
_OUTPUT_PII_REDACTION_PATTERNS: list[tuple] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]", "ssn", None),
    (
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|6011|65\d{2}|3[47]\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        ),
        "[REDACTED-CC]",
        "credit_card",
        _luhn_valid,
    ),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[REDACTED-IBAN]", "iban", None),
    (
        re.compile(
            r"\b\d{3}-\d{3}-\d{4}\b"
            r"|\(\d{3}\)\s*\d{3}[\s\-]?\d{4}"
            r"|\+?1[\s\-]\d{3}[\s\-]\d{3}[\s\-]\d{4}"
        ),
        "[REDACTED-PHONE]",
        "phone",
        None,
    ),
]

# (id, label, kind)  — kind: "structural" | "block" | "redact"
_OUTPUT_GROUPS_META: list[tuple[str, str, str]] = [
    ("O.1", "Jailbreak Echo",                "block"),
    ("O.2", "Credential / Secret Leak",      "block"),
    ("O.3", "Violent Instruction Echo",      "block"),
    ("O.4", "PII / Financial Redaction",     "redact"),
    ("O.5", "Output Sanity",                 "structural"),
]
# O.2 (credential leak) is handled by the dedicated `_run_credential_leak`
# validator below, not by the generic regex-search path, so it is not listed
# here.
_OUTPUT_BLOCK_PATTERN_MAP: dict[str, list[re.Pattern]] = {
    "O.1": _OUTPUT_JAILBREAK_ECHO,
    "O.3": _OUTPUT_VIOLENT_ECHO,
}


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _step(step_id: str, label: str, status: str, detail: str, latency_ms: float) -> dict[str, Any]:
    return {
        "id": step_id,
        "label": label,
        "status": status,
        "detail": detail,
        "latency_ms": round(latency_ms * 1000, 3),
    }


def _run_structural_input(text: str) -> tuple[str, str]:
    """Return (status, detail) for the structural input check."""
    n = len(text)
    if "\x00" in text or re.search(r"[\x01-\x08\x0e-\x1f]", text):
        return "warn", "Control characters present"
    if n > 4000:
        return "warn", f"Long input ({n} chars)"
    return "allow", f"No structural anomalies ({n} chars)"


def _run_structural_output(text: str) -> tuple[str, str]:
    n = len(text)
    if n == 0:
        return "warn", "Empty response"
    return "allow", f"Sanity OK ({n} chars)"


def _run_block_group(patterns: list[re.Pattern], lowered: str) -> tuple[str, str]:
    for pat in patterns:
        if pat.search(lowered):
            return "block", f"matched: /{pat.pattern[:64]}…/"
    return "allow", f"no match ({len(patterns)} patterns)"


def _run_warn_group(patterns: list[re.Pattern], lowered: str) -> tuple[str, str]:
    hits: list[str] = []
    for pat in patterns:
        m = pat.search(lowered)
        if m:
            hits.append(m.group(0)[:30])
    if hits:
        return "warn", f"sensitive: {', '.join(hits[:3])}"
    return "allow", "intent=safe"


def _call_safety_model(snippet: str, url: str, model: str, protocol: str) -> tuple[str, str]:
    """One safety-model round-trip. Returns (status, detail):

        ("block", "flagged: <reason>")   model judged the request UNSAFE
        ("allow", "safe")                model judged it SAFE
        ("error", "unavailable (...)")   endpoint unreachable / malformed

    Self-contained (stdlib only) so this module never imports the request
    pipeline. `protocol` selects the Llama Guard vs instruct wire format.
    """
    if protocol == "guard":
        # Llama Guard: pass only the turn to moderate. vLLM applies the model's
        # built-in chat template, which supplies the S1–S14 policy prompt; a
        # custom system message would corrupt that template.
        messages = [{"role": "user", "content": snippet}]
    else:
        messages = [
            {"role": "system", "content": _SAFETY_SYSTEM_PROMPT},
            {"role": "user", "content": snippet},
        ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 24,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GUARDRAILS_LLM_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        verdict = (body["choices"][0]["message"]["content"] or "").strip()
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        # FAIL-OPEN: never block real users because a safety model is down.
        logger.warning("I.9 safety model unavailable (fail-open): %s", exc)
        return "error", f"unavailable ({type(exc).__name__})"

    # Conservative parse (both protocols): block only on an explicit UNSAFE
    # token. `\bsafe\b` does not match inside "unsafe" (no word boundary), so a
    # bare "safe" reply never trips the block, and ambiguous chatter allows.
    if not re.search(r"\bunsafe\b", verdict, re.IGNORECASE):
        return "allow", "safe"

    if protocol == "guard":
        # Llama Guard emits e.g. "unsafe\nS1,S9" — map the hazard codes to names.
        codes = re.findall(r"\bS(?:1[0-4]|[1-9])\b", verdict, re.IGNORECASE)
        names = [_LLAMA_GUARD_CATEGORIES.get(c.upper(), c.upper()) for c in codes]
        reason = ", ".join(dict.fromkeys(names)) if names else "policy violation"
        return "block", f"flagged: {reason[:60]}"

    m = re.search(r"unsafe\s*[:\-]?\s*(.*)", verdict, re.IGNORECASE | re.DOTALL)
    reason = (m.group(1).strip().splitlines()[0] if m and m.group(1).strip() else "") or "harmful request"
    return "block", f"flagged: {reason[:60]}"


def _safety_layers() -> list[tuple[str, str, str, str]]:
    """Active I.9 layers as (name, url, model, protocol), in evaluation order.

    Primary is the configured classifier (Llama Guard by default). When
    defense-in-depth is on, a second independent instruct-prompted classifier is
    appended — unless it would just duplicate the primary (same url+model).
    """
    layers = [(
        "Llama Guard" if _classifier_protocol() == "guard" else "LLM classifier",
        GUARDRAILS_LLM_URL, GUARDRAILS_LLM_MODEL, _classifier_protocol(),
    )]
    if GUARDRAILS_DEFENSE_IN_DEPTH:
        secondary = (GUARDRAILS_INSTRUCT_URL, GUARDRAILS_INSTRUCT_MODEL, "instruct")
        if secondary != (GUARDRAILS_LLM_URL, GUARDRAILS_LLM_MODEL, _classifier_protocol()):
            layers.append(("Instruct classifier", *secondary))
    return layers


def _run_llm_safety_classifier(text: str) -> tuple[str, str]:
    """I.9 input check: block if ANY active safety layer flags the request.

    Layers evaluate in order and SHORT-CIRCUIT on the first block, so a request
    the primary already catches costs a single call. Returns ("block", reason)
    if any layer blocks; ("error", ...) only when every layer failed (FAIL-OPEN,
    treated as non-blocking upstream); otherwise ("allow", per-layer summary).
    """
    if not GUARDRAILS_LLM_CLASSIFIER_ENABLED:
        return "skipped", "classifier disabled"
    snippet = text.strip()
    if not snippet:
        return "allow", "empty input"
    # Cap what we send — bounds latency and each layer's carbon cost.
    snippet = snippet[:2000]

    statuses: list[str] = []
    summary: list[str] = []
    for name, url, model, protocol in _safety_layers():
        status, detail = _call_safety_model(snippet, url, model, protocol)
        statuses.append(status)
        if status == "block":
            return "block", f"{name} {detail}"
        summary.append(f"{name}:{status}")

    if statuses and all(s == "error" for s in statuses):
        return "error", "all safety classifiers unavailable"
    return "allow", "; ".join(summary)


def _looks_like_secret_value(raw: str) -> bool:
    """True only when *raw* looks like a real leaked secret, not a placeholder.

    Distinguishes an actual credential (``Xk9mP2vL8qZ``, ``sk-1a2b3c...``) from
    the instructional / templated text that dominates legitimate DevOps answers
    (``<your-password>``, ``${DOCKER_PASSWORD}``, ``changeme``, ``****``, prose
    words). A real secret is reasonably long *and* high-entropy: it mixes
    character classes, or is a long mixed token. Placeholders, shell/template
    variables, masked values, and dictionary words are never treated as leaks.
    """
    if not raw:
        return False
    v = raw.strip().strip("\"'`.,);:")
    # Empty after stripping, or a shell/template/markup placeholder.
    if not v or v[0] in "<[{$%" or v.startswith(("${", "{{")):
        return False
    # Masked values: ****, xxxx, ----, ....
    if set(v) <= set("*•xX-_."):
        return False
    low = v.lower()
    if low in _SECRET_PLACEHOLDER_TOKENS:
        return False
    if any(marker in low for marker in (
        "your_", "your-", "<your", "example", "changeme",
        "placeholder", "redacted", "_here", "-here", "xxxx",
    )):
        return False
    if len(v) < 8:
        return False
    has_alpha = any(c.isalpha() for c in v)
    has_digit = any(c.isdigit() for c in v)
    has_symbol = any((not c.isalnum()) for c in v)
    has_upper = any(c.isupper() for c in v)
    has_lower = any(c.islower() for c in v)
    if (has_alpha and has_digit) or (has_alpha and has_symbol):
        return True
    # Long mixed-case / digit / symbol blobs (base64- / hex-style tokens).
    if len(v) >= 20 and (has_digit or has_symbol or (has_upper and has_lower)):
        return True
    return False


def _run_credential_leak(text: str) -> tuple[str, str]:
    """O.2 output check: block only genuine secret leaks, not how-to text.

    Searches the original-case *text* (entropy detection needs case preserved).
    Private-key / ssh-key material always blocks; ``key: value`` matches block
    only when the value passes :func:`_looks_like_secret_value`.
    """
    for pat in _OUTPUT_PRIVATE_KEY:
        if pat.search(text):
            return "block", "private-key / ssh key material"
    for pat in _OUTPUT_SECRET_KV:
        m = pat.search(text)
        if m and _looks_like_secret_value(m.group(1)):
            key = m.group(0)[: m.start(1) - m.start(0)].strip()
            return "block", f"leaked secret after '{key[:32]}'"
    n = len(_OUTPUT_PRIVATE_KEY) + len(_OUTPUT_SECRET_KV)
    return "allow", f"no credential leak ({n} patterns)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_guardrails(
    text: str,
    phase: str = "input",
    context: str | None = None,
) -> dict[str, Any]:
    """Run NemoGuardrails programmable rails on *text*.

    Returns a trace dict with:
        blocked          bool
        reason           str (set when blocked)
        safe_replacement str (replacement for blocked content)
        warnings         list  (non-blocking sensitive-topic flags)
        checks           list  (legacy names of phases run)
        steps            list  (categorized pipeline steps for UI rendering)
        redacted_text    str   (output-phase: response with PII masked)
        redactions       list  ([{type, count}, ...])
        latency_ms       float (total rail wall-clock)
        guardrails_version str
        phase            'input' | 'output'
    """
    t0 = time.monotonic()
    trace: dict[str, Any] = {
        "blocked": False,
        "reason": "",
        "safe_replacement": "",
        "warnings": [],
        "checks": [],
        "steps": [],
        "latency_ms": 0.0,
        "guardrails_version": "nemoguardrails-0.21.0",
        "phase": phase,
        "redacted_text": "",
        "redactions": [],
    }

    if not GUARDRAILS_ENABLED or not text:
        trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 2)
        return trace

    lowered = text.lower()

    if phase == "input":
        trace["checks"].append("input_pipeline")
        first_block_logged = False
        for step_id, label, patterns, mode in _INPUT_GROUPS:
            s0 = time.monotonic()
            if mode == "structural":
                status, detail = _run_structural_input(text)
            elif mode == "block":
                status, detail = _run_block_group(patterns or [], lowered)
            elif mode == "warn":
                status, detail = _run_warn_group(patterns or [], lowered)
            else:
                status, detail = "allow", ""

            elapsed = time.monotonic() - s0
            trace["steps"].append(_step(step_id, label, status, detail, elapsed))

            if status == "block" and not first_block_logged:
                trace["blocked"] = True
                trace["reason"] = f"{step_id} {label}: {detail}"
                trace["safe_replacement"] = (
                    "I can't help with that request. If you're in crisis or need "
                    "support, please contact local emergency services or a trusted "
                    "professional."
                )
                first_block_logged = True
                logger.warning("NemoGuardrails INPUT BLOCKED (%s %s)", step_id, label)
            elif status == "warn":
                trace["warnings"].append(f"{step_id}: {detail}")

        # I.9 — LLM safety net. Runs only when the deterministic rails let the
        # prompt through (common path pays no inference), and fails open so a
        # classifier outage can never take down legitimate traffic.
        if GUARDRAILS_LLM_CLASSIFIER_ENABLED and not trace["blocked"]:
            s0 = time.monotonic()
            status, detail = _run_llm_safety_classifier(text)
            elapsed = time.monotonic() - s0
            trace["steps"].append(_step("I.9", "LLM Safety Classifier", status, detail, elapsed))
            if status == "block":
                trace["blocked"] = True
                trace["reason"] = f"I.9 LLM Safety Classifier: {detail}"
                trace["safe_replacement"] = (
                    "I can't help with that request. If you're in crisis or need "
                    "support, please contact local emergency services or a trusted "
                    "professional."
                )
                logger.warning("NemoGuardrails INPUT BLOCKED (I.9 LLM Safety Classifier)")

    elif phase == "image-gen":
        # Description-oriented rail for text-to-image prompts. The generic input
        # pipeline is expected to have run first; this adds visual-content policy
        # (CSAM, explicit imagery, gore, deepfakes) the verb-keyed chat rails miss.
        trace["checks"].append("image_gen_pipeline")
        first_block_logged = False
        for step_id, label, patterns, mode in _IMGGEN_GROUPS:
            s0 = time.monotonic()
            status, detail = _run_block_group(patterns or [], lowered)
            elapsed = time.monotonic() - s0
            trace["steps"].append(_step(step_id, label, status, detail, elapsed))
            if status == "block" and not first_block_logged:
                trace["blocked"] = True
                trace["reason"] = f"{step_id} {label}: {detail}"
                trace["safe_replacement"] = (
                    "I can't generate that image. This request violates the image "
                    "content policy."
                )
                first_block_logged = True
                logger.warning("NemoGuardrails IMAGE-GEN BLOCKED (%s %s)", step_id, label)

    elif phase == "output":
        trace["checks"].append("output_pipeline")
        first_block_logged = False
        for step_id, label, kind in _OUTPUT_GROUPS_META:
            s0 = time.monotonic()
            if kind == "block":
                if first_block_logged:
                    status, detail = "skipped", "skipped — earlier block"
                else:
                    if step_id == "O.2":
                        # Credential leak: entropy-aware so instructional CLI
                        # text ("password: <your-password>") is not blocked.
                        status, detail = _run_credential_leak(text)
                    else:
                        pats = _OUTPUT_BLOCK_PATTERN_MAP.get(step_id, [])
                        status, detail = _run_block_group(pats, lowered)
                    if status == "block":
                        first_block_logged = True
                        trace["blocked"] = True
                        trace["reason"] = f"{step_id} {label}: {detail}"
                        trace["safe_replacement"] = (
                            "I'm sorry, I'm unable to provide that information."
                        )
                        logger.warning(
                            "NemoGuardrails OUTPUT BLOCKED (%s %s)", step_id, label
                        )
            elif kind == "redact":
                if first_block_logged:
                    status, detail = "skipped", "skipped — earlier block"
                else:
                    redacted = text
                    found: list[dict[str, Any]] = []
                    for pat, replacement, tag, validator in _OUTPUT_PII_REDACTION_PATTERNS:
                        if validator is None:
                            new_redacted, count = pat.subn(replacement, redacted)
                        else:
                            count_box = [0]

                            def _sub(m, _v=validator, _r=replacement, _box=count_box):
                                if _v(m.group(0)):
                                    _box[0] += 1
                                    return _r
                                return m.group(0)

                            new_redacted = pat.sub(_sub, redacted)
                            count = count_box[0]
                        if count > 0:
                            redacted = new_redacted
                            found.append({"type": tag, "count": count})
                    if found:
                        status = "redact"
                        detail = ", ".join(f"{r['type']} × {r['count']}" for r in found)
                        trace["redacted_text"] = redacted
                        trace["redactions"] = found
                        logger.info(
                            "NemoGuardrails OUTPUT REDACTED types=%s",
                            [r["type"] for r in found],
                        )
                    else:
                        status, detail = "allow", "No PII detected"
            elif kind == "structural":
                if first_block_logged:
                    status, detail = "skipped", "skipped — earlier block"
                else:
                    status, detail = _run_structural_output(text)
            else:
                status, detail = "allow", ""

            elapsed = time.monotonic() - s0
            trace["steps"].append(_step(step_id, label, status, detail, elapsed))

    trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 2)
    return trace

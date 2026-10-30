"""Small language model (SLM) wrapper.

Uses Llama 3.1 8B on Groq when GROQ_API_KEY is set. Without a key the agent
falls back to deterministic rules so the graph still runs offline (the trace
shows which mode produced each step).
"""
import json
import re
import time

from . import config
from .trace import tracer

_client = None


def available() -> bool:
    return bool(config.GROQ_API_KEY)


def mode() -> str:
    return f"Groq · {config.GROQ_MODEL}" if available() else "offline rules (no GROQ_API_KEY)"


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def chat_json(node: str, system: str, user: str) -> dict | None:
    """One narrow SLM call that must return a JSON object. Returns None on failure."""
    if not available():
        return None
    t0 = time.time()
    try:
        resp = _get_client().chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content
        out = json.loads(text)
        tracer.log(node, "llm", config.GROQ_MODEL, {"system": system[:200], "user": user[:500]}, out,
                   ms=int((time.time() - t0) * 1000), tokens=getattr(resp.usage, "total_tokens", None))
        return out
    except Exception as e:  # network / JSON errors -> rules fallback
        tracer.log(node, "llm_error", config.GROQ_MODEL, {"user": user[:300]}, {"error": str(e)})
        return None


# ---------- rule fallbacks used when the SLM is unavailable ----------

FIELD_PATTERNS = {
    "gi_no": r"GI\s*(?:Tag)?\s*(?:No\.?|Number|#)\s*[:\-]?\s*(\d+)",
    "woven_at": r"Woven\s*(?:at|in)\s*[:\-]?\s*([^|\n]+)",
    "weaver_id": r"Weaver\s*(?:ID)?\s*[:\-]?\s*([A-Z]{3}-W-\d{3})",
    "batch_id": r"Batch\s*(?:ID|No\.?)?\s*[:\-]?\s*([A-Z]{3}-B-\d{3,5})",
}


def rule_extract(label: str) -> dict:
    fields = {}
    for k, pat in FIELD_PATTERNS.items():
        m = re.search(pat, label, re.I)
        fields[k] = m.group(1).strip() if m else None
    first = re.split(r"[|\n]", label)[0]
    first = re.sub(r"\b(pure|genuine|authentic|original|handloom|saree|sari|sarees|dupatta)\b", "", first, flags=re.I)
    fields["product"] = re.sub(r"\s+", " ", first).strip(" -:,") or None
    low = label.lower()
    fields["materials"] = [m for m in ["mulberry silk", "silk", "zari", "cotton", "polyester", "art silk",
                                       "viscose", "nylon", "powerloom", "handloom"] if m in low]
    if fields["woven_at"]:
        parts = [p.strip() for p in fields["woven_at"].split(",")]
        fields["district"] = parts[0]
        fields["state"] = parts[1] if len(parts) > 1 else None
    else:
        fields["district"] = fields["state"] = None
    return fields

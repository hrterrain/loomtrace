"""The seven typed tools. Every tool returns a ToolResult with a source_id so the
verdict can cite it, and every call is logged by the tracer."""
import csv
import difflib
import json
import sqlite3
import time
from datetime import datetime
from functools import lru_cache

from pydantic import BaseModel

from . import config, llm, rag
from .trace import tracer


class ToolResult(BaseModel):
    tool: str
    found: bool
    source_id: str | None = None
    data: dict | list | None = None
    note: str = ""


def _logged(fn):
    def wrapper(*args, node="retriever", **kwargs):
        t0 = time.time()
        res: ToolResult = fn(*args, **kwargs)
        tracer.log(node, "tool", fn.__name__, {"args": args, **kwargs}, res.model_dump(),
                   ms=int((time.time() - t0) * 1000))
        return res
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@lru_cache(maxsize=None)
def _csv(name: str) -> list[dict]:
    with open(config.DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def db() -> sqlite3.Connection:
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS review_queue(
        id INTEGER PRIMARY KEY, case_id TEXT, label_text TEXT, agent_verdict TEXT, reason TEXT,
        status TEXT DEFAULT 'pending', reviewer_verdict TEXT, reviewer_note TEXT,
        created_at TEXT, reviewed_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS verified_facts(
        id INTEGER PRIMARY KEY, kind TEXT, key TEXT, value TEXT, review_id INTEGER, created_at TEXT)""")
    return con


# 1 ---------------------------------------------------------------
def extract_label(label_text: str) -> ToolResult:
    """Label text (OCR output) -> structured fields, via the SLM with a rules fallback."""
    out = llm.chat_json(
        "extract",
        "You read handloom product labels. Return JSON with keys: product, gi_no, woven_at, district, state, "
        "weaver_id, batch_id, materials (list of strings). Use null when a field is absent. Never invent values.",
        label_text,
    )
    rules = llm.rule_extract(label_text)
    if out:
        # IDs and numbers are copied verbatim from regex; SLM handles the free-text fields.
        for k in ("gi_no", "weaver_id", "batch_id"):
            out[k] = rules[k]
        out["materials"] = sorted(set(out.get("materials") or []) | set(rules["materials"]))
        fields, src = out, "SLM+regex"
    else:
        fields, src = rules, "regex"
    return ToolResult(tool="extract_label", found=True, source_id=f"LABEL-OCR({src})", data=fields)


extract_label = _logged(extract_label)


# 2 ---------------------------------------------------------------
@_logged
def gi_registry_lookup(product: str | None = None, gi_no: str | None = None, fuzzy: bool = False) -> ToolResult:
    """Product name (or GI number) -> GI no., state, goods class, GI districts."""
    rows = _csv("gi_registry.csv")
    if gi_no:
        for r in rows:
            if str(r["gi_no"]) == str(gi_no):
                return ToolResult(tool="gi_registry_lookup", found=True, source_id=r["source_id"], data=r,
                                  note=f"matched by GI number {gi_no}")
    if product:
        p = product.lower().strip()
        for r in rows:
            names = [r["product"].lower()] + [a.strip() for a in r["aliases"].lower().split("|")]
            if p in names or any(n in p for n in names):
                return ToolResult(tool="gi_registry_lookup", found=True, source_id=r["source_id"], data=r)
        if fuzzy:
            cand = {n: r for r in rows for n in [r["product"].lower()] + r["aliases"].lower().split("|")}
            m = difflib.get_close_matches(p, cand.keys(), n=1, cutoff=0.5)
            if m:
                r = cand[m[0]]
                return ToolResult(tool="gi_registry_lookup", found=True, source_id=r["source_id"], data=r,
                                  note=f"fuzzy match '{product}' -> '{m[0]}'")
    return ToolResult(tool="gi_registry_lookup", found=False, note=f"no GI registration for '{product}'")


# 3 ---------------------------------------------------------------
@_logged
def gi_doc_search(query: str, product: str | None = None, k: int = 5) -> ToolResult:
    """Query + product -> top-k GI application passages with page numbers."""
    hits = rag.index().search(query, product=product, k=k)
    return ToolResult(tool="gi_doc_search", found=bool(hits),
                      source_id=",".join(h["id"] for h in hits) or None,
                      data=[{k_: h[k_] for k_ in ("id", "product", "page", "section", "text", "source", "score")}
                            for h in hits])


# 4 ---------------------------------------------------------------
@_logged
def census_lookup(state: str, district: str | None = None) -> ToolResult:
    """State / district -> handloom weaver and loom counts (Handloom Census 2019-20, cached CSV)."""
    rows = _csv("census.csv")
    s = (state or "").lower()
    d = (district or "").lower()
    match = [r for r in rows if r["state"].lower() == s and r["district"].lower() in (d, "all")]
    if not match:
        return ToolResult(tool="census_lookup", found=False, note=f"no census rows for {state}/{district}")
    best = next((r for r in match if r["district"].lower() == d), match[0])
    return ToolResult(tool="census_lookup", found=True, source_id=best["source_id"], data=best)


# 5 ---------------------------------------------------------------
@_logged
def weaver_lookup(weaver_id: str) -> ToolResult:
    """Weaver ID -> name, cluster, district, product, status (simulated registry + reviewer-verified facts)."""
    for r in _csv("weavers.csv"):
        if r["weaver_id"].upper() == (weaver_id or "").upper():
            return ToolResult(tool="weaver_lookup", found=True, source_id=f"SIM-WEAVER-{r['weaver_id']}", data=r)
    fact = _verified("weaver", weaver_id)
    if fact:
        return ToolResult(tool="weaver_lookup", found=True, source_id=f"HUMAN-REVIEW-{fact['review_id']}",
                          data=json.loads(fact["value"]), note="verified earlier by a human reviewer (feedback loop)")
    return ToolResult(tool="weaver_lookup", found=False, note=f"weaver {weaver_id} not in registry")


# 6 ---------------------------------------------------------------
@_logged
def provenance_lookup(batch_id: str) -> ToolResult:
    """Batch ID -> loom type, yarn, dates, weaver, district (simulated records + reviewer-verified facts)."""
    for r in _csv("provenance.csv"):
        if r["batch_id"].upper() == (batch_id or "").upper():
            return ToolResult(tool="provenance_lookup", found=True, source_id=f"SIM-BATCH-{r['batch_id']}", data=r)
    fact = _verified("batch", batch_id)
    if fact:
        return ToolResult(tool="provenance_lookup", found=True, source_id=f"HUMAN-REVIEW-{fact['review_id']}",
                          data=json.loads(fact["value"]), note="verified earlier by a human reviewer (feedback loop)")
    return ToolResult(tool="provenance_lookup", found=False, note=f"batch {batch_id} not in records")


# 7 ---------------------------------------------------------------
@_logged
def escalate_to_human(case_id: str, label_text: str, agent_verdict: str, reason: str) -> ToolResult:
    """Case + reason -> review-queue entry."""
    con = db()
    cur = con.execute(
        "INSERT INTO review_queue(case_id,label_text,agent_verdict,reason,created_at) VALUES(?,?,?,?,?)",
        (case_id, label_text, agent_verdict, reason, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    return ToolResult(tool="escalate_to_human", found=True, source_id=f"REVIEW-{cur.lastrowid}",
                      data={"review_id": cur.lastrowid, "reason": reason})


# ---------- human feedback memory ----------

def _verified(kind: str, key: str | None):
    if not key:
        return None
    row = db().execute("SELECT * FROM verified_facts WHERE kind=? AND key=? ORDER BY id DESC LIMIT 1",
                       (kind, key.upper())).fetchone()
    return dict(row) if row else None


def record_review(review_id: int, reviewer_verdict: str, note: str, fields: dict | None = None):
    """Store a reviewer decision. If the reviewer confirms the claim, the weaver/batch they
    checked becomes a verified fact the agent will use on the next run (feedback loop)."""
    con = db()
    now = datetime.now().isoformat(timespec="seconds")
    con.execute("UPDATE review_queue SET status='reviewed', reviewer_verdict=?, reviewer_note=?, reviewed_at=? "
                "WHERE id=?", (reviewer_verdict, note, now, review_id))
    if reviewer_verdict == "Supported" and fields:
        if fields.get("weaver_id"):
            con.execute("INSERT INTO verified_facts(kind,key,value,review_id,created_at) VALUES(?,?,?,?,?)",
                        ("weaver", fields["weaver_id"].upper(), json.dumps({
                            "weaver_id": fields["weaver_id"], "name": "(verified by reviewer)",
                            "cluster": fields.get("district"), "district": fields.get("district"),
                            "state": fields.get("state"), "product": fields.get("registry_product"),
                            "status": "active"}), review_id, now))
        if fields.get("batch_id"):
            con.execute("INSERT INTO verified_facts(kind,key,value,review_id,created_at) VALUES(?,?,?,?,?)",
                        ("batch", fields["batch_id"].upper(), json.dumps({
                            "batch_id": fields["batch_id"], "weaver_id": fields.get("weaver_id"),
                            "product": fields.get("registry_product"), "loom_type": "handloom",
                            "district": fields.get("district")}), review_id, now))
    con.commit()


TOOLS = [extract_label, gi_registry_lookup, gi_doc_search, census_lookup, weaver_lookup,
         provenance_lookup, escalate_to_human]

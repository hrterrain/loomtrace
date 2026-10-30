r"""LoomTrace agent graph (LangGraph).

extract -> planner -> retriever -> verifier -> grader --(gaps, <3 loops)--> planner
                                                     \--(enough)--> guardrail -> END
CaseState is checkpointed to SQLite after every node.
"""
import sqlite3
import uuid
from typing import TypedDict

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph
    ENGINE = "LangGraph"
except ImportError:  # pip install -r requirements.txt to use the real LangGraph
    from .minigraph import END, StateGraph
    from .minigraph import SqliteCheckpoint
    ENGINE = "built-in mini-graph (LangGraph not installed)"

from . import config, llm, tools
from .trace import tracer


class CaseState(TypedDict, total=False):
    case_id: str
    label_text: str
    fields: dict
    plan: list[dict]
    tried: list[str]
    evidence: dict          # tool key -> ToolResult dict (latest)
    checks: dict            # check -> {"status": pass|fail|unknown, "detail": str, "source": str}
    grade: dict
    iteration: int
    gaps: list[str]
    verdict: str
    confidence: float
    citations: list[str]
    explanation: str
    review_id: int | None


def _in_gi_area(place: str | None, reg: dict) -> bool | None:
    if not place:
        return None
    p = place.lower()
    areas = [a.strip().lower() for a in reg.get("gi_districts", "").split("|") if a.strip()]
    return any(a in p or p in a for a in areas)


# ------------------------------------------------------------------ nodes

def extract(state: CaseState) -> CaseState:
    res = tools.extract_label(state["label_text"], node="extract")
    return {"fields": res.data, "iteration": 0, "tried": [], "evidence": {}, "gaps": []}


def planner(state: CaseState) -> CaseState:
    f, it, gaps, ev = state["fields"], state.get("iteration", 0), state.get("gaps", []), state.get("evidence", {})
    tried = list(state.get("tried", []))
    plan: list[dict] = []

    if it == 0:
        plan.append({"tool": "gi_registry_lookup", "args": {"product": f.get("product")},
                     "why": "is the claimed product GI-registered?"})
        if f.get("weaver_id"):
            plan.append({"tool": "weaver_lookup", "args": {"weaver_id": f["weaver_id"]}, "why": "does the weaver exist?"})
        if f.get("batch_id"):
            plan.append({"tool": "provenance_lookup", "args": {"batch_id": f["batch_id"]},
                         "why": "is there a production record?"})
        q = llm.chat_json(
            "planner",
            "You plan evidence retrieval for verifying a handloom GI claim. Return JSON {\"doc_query\": str} - "
            "a short keyword query to find the GI application passages about geographical area, raw materials "
            "and method of production relevant to this label.",
            f"Label fields: {f}",
        )
        doc_q = (q or {}).get("doc_query") or f"geographical area raw materials method of production {f.get('product') or ''}"
        plan.append({"tool": "gi_doc_search", "args": {"query": doc_q}, "why": "official GI text to cite"})
    else:
        # Re-plan: one sharper strategy per gap reported by the grader.
        if "registry_not_found" in gaps:
            if f.get("gi_no") and "registry_by_gi_no" not in tried:
                plan.append({"tool": "gi_registry_lookup", "args": {"gi_no": f["gi_no"]},
                             "why": "re-plan: look up by GI number printed on label"})
                tried.append("registry_by_gi_no")
            elif "registry_fuzzy" not in tried:
                plan.append({"tool": "gi_registry_lookup", "args": {"product": f.get("product"), "fuzzy": True},
                             "why": "re-plan: fuzzy match product name against aliases"})
                tried.append("registry_fuzzy")
        if "weaver_from_batch" in gaps:
            wid = ev["provenance_lookup"]["data"]["weaver_id"]
            plan.append({"tool": "weaver_lookup", "args": {"weaver_id": wid},
                         "why": "re-plan: label has no weaver ID; use the weaver on the batch record"})
            tried.append("weaver_from_batch")
        if "docs_weak" in gaps:
            reg = ev.get("gi_registry_lookup", {}).get("data") or {}
            plan.append({"tool": "gi_doc_search",
                         "args": {"query": f"geographical area districts region {reg.get('product', '')} "
                                           f"{reg.get('state', '')} {' '.join(f.get('materials') or [])} raw materials"},
                         "why": "re-plan: sharper query with registry product + region + materials"})
            tried.append("docs_expanded")
    if plan and not any(p["tool"] == "census_lookup" for p in plan) and "census" not in tried:
        st = f.get("state") or (ev.get("gi_registry_lookup", {}).get("data") or {}).get("state")
        if st:
            plan.append({"tool": "census_lookup", "args": {"state": st, "district": f.get("district")},
                         "why": "is there a handloom weaver base in this region?"})
            tried.append("census")
    tracer.log("planner", "plan", f"iteration {it}", {"gaps": gaps}, plan)
    return {"plan": plan, "tried": tried}


def retriever(state: CaseState) -> CaseState:
    ev = dict(state.get("evidence", {}))
    reg = (ev.get("gi_registry_lookup") or {}).get("data") or {}
    for step in state["plan"]:
        fn = getattr(tools, step["tool"])
        args = dict(step["args"])
        if step["tool"] == "gi_doc_search":
            # filter by registered product if we know it (metadata filter)
            reg_now = (ev.get("gi_registry_lookup") or {}).get("data") or reg
            args.setdefault("product", reg_now.get("product"))
        res = fn(**args, node="retriever")
        if step["tool"] == "census_lookup" and not res.found and ev.get("gi_registry_lookup", {}).get("found"):
            pass
        if step["tool"] == "gi_registry_lookup" and not res.found and (ev.get(step["tool"]) or {}).get("found"):
            continue  # don't overwrite a good result
        ev[step["tool"]] = res.model_dump()
    return {"evidence": ev}


def verifier(state: CaseState) -> CaseState:
    f, ev = state["fields"], state["evidence"]
    checks = {}
    reg_r = ev.get("gi_registry_lookup", {})
    reg = reg_r.get("data") or {}

    def put(name, status, detail, source=None):
        checks[name] = {"status": status, "detail": detail, "source": source}

    if reg_r.get("found"):
        put("registered", "pass", f"{reg['product']} is GI No. {reg['gi_no']} ({reg['state']})", reg_r["source_id"])
        if f.get("gi_no") and str(f["gi_no"]) != str(reg["gi_no"]):
            put("gi_number", "fail", f"label says GI No. {f['gi_no']} but registry says {reg['gi_no']}", reg_r["source_id"])
        elif f.get("gi_no"):
            put("gi_number", "pass", f"GI No. {f['gi_no']} matches registry", reg_r["source_id"])
    else:
        put("registered", "unknown", f"no GI registration found for '{f.get('product')}'")

    docs = ev.get("gi_doc_search", {})
    area_doc = next((d for d in (docs.get("data") or []) if "geograph" in d["section"].lower()), None)
    region_src = ", ".join(s for s in [reg_r.get("source_id"), area_doc and area_doc["id"]] if s) or None
    in_area = _in_gi_area(f.get("woven_at") or f.get("district"), reg) if reg else None
    if in_area is None:
        put("region", "unknown", "label does not say where it was woven" if not f.get("woven_at")
            else "cannot check region without registry entry")
    elif in_area:
        put("region", "pass", f"'{f.get('woven_at')}' lies in the GI area", region_src)
    else:
        put("region", "fail", f"'{f.get('woven_at')}' is outside the GI area ({reg['gi_districts'].replace('|', ', ')})",
            region_src)

    bad = [m for m in (f.get("materials") or []) if m in ("polyester", "art silk", "viscose", "nylon", "powerloom")]
    mat_doc = next((d for d in (docs.get("data") or []) if "material" in d["section"].lower()), None)
    if bad:
        put("materials", "fail", f"label mentions {', '.join(bad)}, not allowed by the GI specification",
            mat_doc["id"] if mat_doc else None)
    elif f.get("materials"):
        put("materials", "pass", f"materials {f['materials']} consistent with GI spec", mat_doc["id"] if mat_doc else None)

    w = ev.get("weaver_lookup")
    if w and w.get("found"):
        wd = w["data"]
        problems = []
        if reg and wd.get("product") and wd["product"].lower() != reg["product"].lower():
            problems.append(f"weaver is registered for {wd['product']}")
        if wd.get("status", "active") != "active":
            problems.append("weaver is inactive")
        if reg and _in_gi_area(wd.get("district"), reg) is False:
            problems.append(f"weaver's district {wd['district']} is outside GI area")
        put("weaver", "fail" if problems else "pass",
            "; ".join(problems) or f"{wd['weaver_id']} ({wd.get('name')}) active in {wd.get('district')}", w["source_id"])
    elif w:
        put("weaver", "unknown", w.get("note", "weaver not found"))
    else:
        put("weaver", "unknown", "no weaver ID on label")

    b = ev.get("provenance_lookup")
    if b and b.get("found"):
        bd = b["data"]
        problems = []
        wid = f.get("weaver_id") or (w or {}).get("data", {}).get("weaver_id") if w and w.get("found") else f.get("weaver_id")
        if wid and bd.get("weaver_id") and bd["weaver_id"].upper() != wid.upper():
            problems.append(f"batch was woven by {bd['weaver_id']}, not {wid}")
        if "handloom" not in bd.get("loom_type", "").lower() or "power" in bd.get("loom_type", "").lower():
            problems.append(f"batch made on {bd.get('loom_type')}")
        if reg and bd.get("product") and bd["product"].lower() != reg["product"].lower():
            problems.append(f"batch is {bd['product']}")
        if w and w.get("found") and bd.get("district") and w["data"].get("district") \
                and bd["district"].lower() != w["data"]["district"].lower():
            problems.append(f"batch district {bd['district']} != weaver district {w['data']['district']}")
        put("batch", "fail" if problems else "pass",
            "; ".join(problems) or f"{bd['batch_id']}: {bd.get('loom_type')}, {bd.get('yarn')}, {bd.get('district')}",
            b["source_id"])
    elif b:
        put("batch", "unknown", b.get("note", "batch not found"))
    else:
        put("batch", "unknown", "no batch ID on label")

    c = ev.get("census_lookup")
    if c and c.get("found"):
        put("census", "pass" if int(c["data"]["weavers"]) > 0 else "fail",
            f"{c['data']['weavers']} handloom weavers in {c['data']['district']}, {c['data']['state']}", c["source_id"])

    fails = [k for k, v in checks.items() if v["status"] == "fail"]
    core = ["registered", "region", "weaver", "batch"]
    if fails:
        verdict = "Conflicting"
    elif all(checks.get(k, {}).get("status") == "pass" for k in core):
        verdict = "Supported"
    else:
        verdict = "Insufficient"
    citations = sorted({s.strip() for v in checks.values() if v["status"] != "unknown" and v["source"]
                        for s in v["source"].split(",")})

    expl = llm.chat_json(
        "verifier",
        "You are a careful verifier. Given rule-check results for a handloom GI claim, write JSON "
        "{\"explanation\": str} - 2-3 sentences, cite source IDs in [brackets], do not add facts not in the checks.",
        f"Proposed verdict: {verdict}\nChecks: {checks}",
    )
    explanation = (expl or {}).get("explanation") or _template_explanation(verdict, checks)
    tracer.log("verifier", "checks", "rule checks", {"fields": f}, {"checks": checks, "verdict": verdict})
    return {"checks": checks, "verdict": verdict, "citations": citations, "explanation": explanation}


def _template_explanation(verdict, checks):
    parts = [f"{k}: {v['detail']}" + (f" [{v['source']}]" if v["source"] else "") for k, v in checks.items()
             if v["status"] in ("fail", "pass" if verdict == "Supported" else "fail", "unknown")]
    return f"{verdict}. " + "; ".join(parts[:5])


def grader(state: CaseState) -> CaseState:
    checks, ev, f = state["checks"], state["evidence"], state["fields"]
    core = ["registered", "region", "weaver", "batch"]
    known = sum(checks.get(k, {}).get("status") in ("pass", "fail") for k in core)
    completeness = known / len(core)
    cited = [v for v in checks.values() if v["status"] != "unknown"]
    consistency = sum(bool(v["source"]) for v in cited) / max(len(cited), 1)
    top = max((d["score"] for d in (ev.get("gi_doc_search", {}).get("data") or [])), default=0)
    has_area = any("geograph" in d["section"].lower() for d in (ev.get("gi_doc_search", {}).get("data") or []))

    tried = state.get("tried", [])
    gaps = []
    if not ev.get("gi_registry_lookup", {}).get("found") and not {"registry_by_gi_no", "registry_fuzzy"} <= set(tried):
        if not ("registry_fuzzy" in tried and not f.get("gi_no")):
            gaps.append("registry_not_found")
    if (not has_area or top < 2.0) and "docs_expanded" not in tried:
        gaps.append("docs_weak")
    b = ev.get("provenance_lookup", {})
    if not f.get("weaver_id") and b.get("found") and "weaver_from_batch" not in tried:
        gaps.append("weaver_from_batch")

    it = state.get("iteration", 0) + 1
    grade = {"completeness": round(completeness, 2), "consistency": round(consistency, 2),
             "doc_top_score": top, "iteration": it,
             "decision": "re-plan" if gaps and it < config.MAX_ITERATIONS else "finish"}
    if state["verdict"] == "Conflicting":
        conf = 0.9 if consistency == 1 else 0.6
    elif state["verdict"] == "Supported":
        conf = round(0.5 * completeness + 0.5 * consistency, 2)
    else:
        conf = round(completeness, 2)
    tracer.log("grader", "grade", f"iteration {it}", {"gaps": gaps}, grade)
    return {"grade": grade, "gaps": gaps, "iteration": it, "confidence": conf}


def route_after_grade(state: CaseState) -> str:
    return "planner" if state["grade"]["decision"] == "re-plan" else "guardrail"


def guardrail(state: CaseState) -> CaseState:
    verdict, conf, cites = state["verdict"], state["confidence"], state["citations"]
    reason = None
    if verdict != "Insufficient" and not cites:
        reason = "verdict has no citation"
    elif verdict != "Insufficient" and conf < config.CONFIDENCE_THRESHOLD:
        reason = f"confidence {conf} below {config.CONFIDENCE_THRESHOLD}"
    elif verdict == "Insufficient":
        unknown = [k for k, v in state["checks"].items() if v["status"] == "unknown"]
        reason = "missing evidence: " + ", ".join(unknown)
    out: CaseState = {}
    if reason:
        if verdict != "Insufficient":
            out["explanation"] = f"[Guardrail] Downgraded from {verdict}: {reason}. " + state["explanation"]
            out["verdict"] = "Insufficient"
        res = tools.escalate_to_human(state["case_id"], state["label_text"], out.get("verdict", verdict), reason,
                                      node="guardrail")
        out["review_id"] = res.data["review_id"]
    tracer.log("guardrail", "gate", "guardrail", {"verdict": verdict, "confidence": conf},
               {"final": out.get("verdict", verdict), "escalated": bool(reason), "reason": reason})
    return out


# ------------------------------------------------------------------ graph

def build_graph(checkpoint: bool = True):
    g = StateGraph(CaseState)
    for name, fn in [("extract", extract), ("planner", planner), ("retriever", retriever),
                     ("verifier", verifier), ("grader", grader), ("guardrail", guardrail)]:
        g.add_node(name, fn)
    g.set_entry_point("extract")
    g.add_edge("extract", "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "verifier")
    g.add_edge("verifier", "grader")
    g.add_conditional_edges("grader", route_after_grade, {"planner": "planner", "guardrail": "guardrail"})
    g.add_edge("guardrail", END)
    if not checkpoint:
        saver = None
    elif ENGINE == "LangGraph":
        saver = SqliteSaver(sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False))
    else:
        saver = SqliteCheckpoint(config.CHECKPOINT_DB)
    return g.compile(checkpointer=saver)


_graph = None


def stream_case(label_text: str, case_id: str | None = None):
    """Yields (node, update) as each node finishes; final state via get_state."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    case_id = case_id or "case-" + uuid.uuid4().hex[:8]
    tracer.start(case_id)
    cfg = {"configurable": {"thread_id": case_id}, "run_name": f"LoomTrace {case_id}",
           "tags": ["loomtrace"], "metadata": {"case_id": case_id}}
    for chunk in _graph.stream({"case_id": case_id, "label_text": label_text}, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update or {}
    yield "__final__", _graph.get_state(cfg).values


def run_case(label_text: str, case_id: str | None = None) -> dict:
    final = {}
    for node, upd in stream_case(label_text, case_id):
        if node == "__final__":
            final = upd
    return final

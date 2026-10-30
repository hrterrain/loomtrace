"""LoomTrace Streamlit UI: label in -> live agent trace -> cited verdict; human review queue."""
import json

import pandas as pd
import streamlit as st

from loomtrace import config, llm, tools
from loomtrace.graph import stream_case
from loomtrace.trace import load_trace

st.set_page_config(page_title="LoomTrace", page_icon="🧵", layout="wide")
COLOR = {"Supported": "green", "Conflicting": "red", "Insufficient": "orange"}
NODE_LABEL = {"extract": "1 · Extract (OCR + SLM)", "planner": "2 · Planner", "retriever": "3 · Retriever agent",
              "verifier": "4 · Verifier agent", "grader": "5 · Grader (reflection)", "guardrail": "6 · Guardrail + HITL"}

golden = json.loads((config.DATA / "golden_cases.json").read_text())

st.sidebar.title("🧵 LoomTrace")
st.sidebar.caption("Checks whether a handloom product's GI-tag claim is backed by official evidence.")
page = st.sidebar.radio("Page", ["Verify a claim", "Human review queue", "Traces", "Knowledge base"])
st.sidebar.markdown(f"**SLM:** {llm.mode()}")
st.sidebar.markdown("**Scope:** Kancheepuram Silk · Pochampalli Ikat · Banaras Brocades")

# ------------------------------------------------------------------ verify
if page == "Verify a claim":
    st.header("Verify a GI claim")
    c1, c2 = st.columns([2, 1])
    with c2:
        pick = st.selectbox("Load a golden test case", ["—"] + [f"{g['case_id']} · {g['expected']} · {g['product']}"
                                                                 for g in golden])
        img = st.file_uploader("…or upload a label photo (OCR)", type=["png", "jpg", "jpeg"])
    default = ""
    if pick != "—":
        default = next(g for g in golden if g["case_id"] == pick.split(" · ")[0])["label_text"]
    if img is not None:
        try:
            import easyocr  # optional, heavy
            reader = easyocr.Reader(["en"], gpu=False)
            default = " | ".join(reader.readtext(img.getvalue(), detail=0))
            st.success("OCR done")
        except ImportError:
            st.info("EasyOCR not installed (`pip install easyocr`) — paste the label text instead.")
    with c1:
        label = st.text_area("Label text (as read from the product tag)", value=default, height=120)
        go = st.button("Run agent", type="primary") and bool(label.strip())

    if go:
        left, right = st.columns([3, 2])
        final = {}
        with left:
            st.subheader("Live agent trace")
            for node, upd in stream_case(label):
                if node == "__final__":
                    final = upd
                    continue
                it = upd.get("iteration")
                with st.expander(NODE_LABEL.get(node, node), expanded=node in ("planner", "verifier", "grader")):
                    if node == "extract":
                        st.json(upd["fields"])
                    elif node == "planner":
                        for p in upd["plan"]:
                            st.markdown(f"🔧 `{p['tool']}({', '.join(f'{k}={v!r}' for k, v in p['args'].items())})` — {p['why']}")
                    elif node == "retriever":
                        for k, v in upd["evidence"].items():
                            st.markdown(f"{'🟢' if v['found'] else '⚪'} **{k}** → `{v['source_id'] or v['note']}`")
                    elif node == "verifier":
                        rows = [{"check": k, "status": v["status"], "detail": v["detail"], "source": v["source"]}
                                for k, v in upd["checks"].items()]
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                        st.caption(f"Proposed verdict: **{upd['verdict']}**")
                    elif node == "grader":
                        g = upd["grade"]
                        st.markdown(f"completeness **{g['completeness']}** · consistency **{g['consistency']}** · "
                                    f"iteration **{g['iteration']}/{config.MAX_ITERATIONS}** → **{g['decision']}**")
                        if upd["gaps"]:
                            st.warning("Gaps found → re-plan: " + ", ".join(upd["gaps"]))
                    elif node == "guardrail":
                        st.write("Escalated to human review" if upd.get("review_id") else "Passed guardrail")
        with right:
            v = final["verdict"]
            st.subheader("Verdict")
            st.markdown(f"## :{COLOR[v]}[{v}]")
            st.metric("Confidence", final["confidence"])
            st.markdown("**Explanation**")
            st.write(final["explanation"])
            st.markdown("**Citations**")
            docs = {d["id"]: d for d in (final["evidence"].get("gi_doc_search", {}).get("data") or [])}
            for c in final["citations"]:
                if c in docs:
                    with st.expander(f"📄 {c} — {docs[c]['section']} (p.{docs[c]['page']})"):
                        st.write(docs[c]["text"])
                        st.caption(docs[c]["source"])
                else:
                    st.markdown(f"- `{c}`")
            if final.get("review_id"):
                st.warning(f"Sent to human review queue (#{final['review_id']})")
            st.caption(f"Case `{final['case_id']}` · loops: {final['iteration']} · "
                       f"tool calls: {sum(e['kind'] == 'tool' for e in load_trace(final['case_id']))}")
            st.session_state["last_case"] = final["case_id"]

# ------------------------------------------------------------------ review
elif page == "Human review queue":
    st.header("Human review queue")
    st.caption("Feedback loop: when a reviewer confirms a claim, the weaver/batch they checked is saved as a "
               "verified fact. The agent's tools read it on the next run, so the same claim no longer needs a human.")
    con = tools.db()
    pending = [dict(r) for r in con.execute("SELECT * FROM review_queue WHERE status='pending' ORDER BY id DESC")]
    if not pending:
        st.info("No pending cases.")
    for r in pending:
        with st.container(border=True):
            st.markdown(f"**#{r['id']}** · case `{r['case_id']}` · agent said **{r['agent_verdict']}**")
            st.code(r["label_text"])
            st.caption(f"Reason: {r['reason']}")
            c1, c2, c3 = st.columns([1, 2, 1])
            verdict = c1.selectbox("Your verdict", ["Supported", "Conflicting", "Insufficient"], key=f"v{r['id']}")
            note = c2.text_input("Note", key=f"n{r['id']}", placeholder="e.g. weaver verified by phone with cluster office")
            if c3.button("Submit", key=f"b{r['id']}"):
                fields = tools.llm.rule_extract(r["label_text"])
                reg = tools.gi_registry_lookup(fields.get("product"), fields.get("gi_no"), fuzzy=True, node="review")
                fields["registry_product"] = (reg.data or {}).get("product")
                tools.record_review(r["id"], verdict, note, fields)
                st.success("Saved. Re-run this label on the Verify page to see the feedback used.")
                st.rerun()
    done = pd.read_sql("SELECT id, case_id, agent_verdict, reviewer_verdict, reviewer_note, reviewed_at "
                       "FROM review_queue WHERE status='reviewed' ORDER BY id DESC", con)
    if len(done):
        st.subheader("Reviewed")
        st.dataframe(done, hide_index=True, use_container_width=True)
    facts = pd.read_sql("SELECT kind, key, review_id, created_at FROM verified_facts ORDER BY id DESC", con)
    if len(facts):
        st.subheader("Verified facts learned from reviewers")
        st.dataframe(facts, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ traces
elif page == "Traces":
    st.header("Agent traces")
    st.caption("Every node, tool call and SLM call, saved per case in traces/*.jsonl"
               + (" and sent to LangSmith." if config.os.getenv("LANGSMITH_API_KEY") else "."))
    files = sorted(config.TRACES.glob("*.jsonl"), key=lambda p: -p.stat().st_mtime)
    if not files:
        st.info("Run a case first.")
    else:
        names = [f.stem for f in files]
        last = st.session_state.get("last_case")
        sel = st.selectbox("Case", names, index=names.index(last) if last in names else 0)
        ev = load_trace(sel)
        df = pd.DataFrame([{"t (s)": e["t"], "node": e["node"], "kind": e["kind"], "name": e["name"],
                            "ms": e.get("ms"), "summary": json.dumps(e["output"], default=str)[:140]} for e in ev])
        st.dataframe(df, hide_index=True, use_container_width=True)
        i = st.number_input("Inspect event #", 0, max(len(ev) - 1, 0), 0)
        st.json(ev[int(i)])

# ------------------------------------------------------------------ KB
else:
    st.header("Knowledge base")
    t1, t2, t3, t4 = st.tabs(["GI registry", "GI documents (RAG)", "Handloom census", "Weavers & batches (simulated)"])
    t1.dataframe(pd.read_csv(config.DATA / "gi_registry.csv"), hide_index=True)
    t2.dataframe(pd.DataFrame(json.loads((config.DATA / "gi_docs.json").read_text())), hide_index=True)
    t3.dataframe(pd.read_csv(config.DATA / "census.csv"), hide_index=True)
    t4.dataframe(pd.read_csv(config.DATA / "weavers.csv"), hide_index=True)
    t4.dataframe(pd.read_csv(config.DATA / "provenance.csv"), hide_index=True)

"""Evaluate LoomTrace on the 45-case golden set and A/B against a baseline.

Baseline = one registry lookup + one doc search + one decision (no planning,
no weaver/batch tools, no re-plan loop, no guardrail).
"""
import json
import time
from collections import Counter

from loomtrace import llm, rag, tools
from loomtrace.graph import run_case
from loomtrace.trace import load_trace

LABELS = ["Supported", "Conflicting", "Insufficient"]


def baseline(label: str) -> str:
    f = llm.rule_extract(label)
    reg = tools.gi_registry_lookup(f.get("product"), node="baseline")
    if not reg.found:
        return "Insufficient"
    tools.gi_doc_search(f"{f.get('product')} geographical area", product=reg.data["product"], node="baseline")
    return "Supported"  # a single lookup sees a registered product and accepts the claim


def macro_f1(pairs):
    f1s = []
    for c in LABELS:
        tp = sum(p == c and g == c for g, p in pairs)
        fp = sum(p == c and g != c for g, p in pairs)
        fn = sum(p != c and g == c for g, p in pairs)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return sum(f1s) / len(f1s)


def main():
    cases = json.load(open("data/golden_cases.json"))
    agent, base, rows = [], [], []
    recall_hits, lat, tool_calls, cited = 0, [], [], 0
    for c in cases:
        t0 = time.time()
        out = run_case(c["label_text"], case_id=f"eval-{c['case_id']}")
        lat.append(time.time() - t0)
        tr = load_trace(f"eval-{c['case_id']}")
        tool_calls.append(sum(e["kind"] == "tool" for e in tr))
        agent.append((c["expected"], out["verdict"]))
        base.append((c["expected"], baseline(c["label_text"])))
        # Recall@5: the product's "Geographical area" passage is in the top 5
        hits = rag.index().search(f"{c['product']} geographical area region districts", product=c["product"], k=5)
        recall_hits += any("geograph" in h["section"].lower() for h in hits)
        if out["verdict"] != "Insufficient":
            cited += bool(out["citations"])
        ok = "✓" if out["verdict"] == c["expected"] else "✗"
        rows.append(f"{ok} {c['case_id']} expected={c['expected']:<12} agent={out['verdict']:<12} loops={out['iteration']}")

    print("\n".join(rows))
    n = len(cases)
    acc = lambda p: sum(g == x for g, x in p) / n
    abst = lambda p: (sum(x == "Insufficient" for g, x in p if g == "Insufficient")
                      / max(sum(g == "Insufficient" for g, _ in p), 1))
    decided = sum(v != "Insufficient" for _, v in agent)
    print("\n================ RESULTS (45 golden cases) ================")
    print(f"{'Metric':<28}{'Baseline':>10}{'LoomTrace':>12}{'Target':>10}")
    print(f"{'Verdict accuracy':<28}{acc(base):>10.0%}{acc(agent):>12.0%}{'>=80%':>10}")
    print(f"{'Macro-F1':<28}{macro_f1(base):>10.2f}{macro_f1(agent):>12.2f}")
    print(f"{'Correct abstention':<28}{abst(base):>10.0%}{abst(agent):>12.0%}{'>=90%':>10}")
    print(f"{'Retrieval Recall@5':<28}{'-':>10}{recall_hits / n:>12.2f}{'>=0.80':>10}")
    print(f"{'Decided verdicts w/ citation':<28}{'-':>10}{cited / max(decided, 1):>12.0%}{'>=90%':>10}")
    print(f"{'Avg latency (s)':<28}{'-':>10}{sum(lat) / n:>12.2f}{'<15':>10}")
    print(f"{'Avg tool calls / case':<28}{'2':>10}{sum(tool_calls) / n:>12.1f}")
    print("\nConfusion (expected -> agent):", dict(Counter(agent)))
    print(f"SLM mode: {llm.mode()}")
    con = tools.db()  # keep the demo review queue clean
    con.execute("DELETE FROM review_queue WHERE case_id LIKE 'eval-%'")
    con.commit()


if __name__ == "__main__":
    main()

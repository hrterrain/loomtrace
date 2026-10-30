"""CLI: python run_case.py "<label text>"   or   python run_case.py --golden G07"""
import json
import sys

from loomtrace import llm
from loomtrace.graph import stream_case

ICON = {"Supported": "✅", "Conflicting": "❌", "Insufficient": "⚠️"}

if __name__ == "__main__":
    args = sys.argv[1:]
    case_id = None
    if args and args[0] == "--golden":
        case = next(c for c in json.load(open("data/golden_cases.json")) if c["case_id"] == args[1])
        label, case_id = case["label_text"], f"{case['case_id']}-cli"
        print(f"Golden case {case['case_id']} (expected: {case['expected']})")
    else:
        label = " ".join(args) or "PURE KANCHEEPURAM SILK SAREE | GI Tag No. 15 | Woven at: Kancheepuram, Tamil Nadu | Weaver ID: KAN-W-001 | Batch: KAN-B-1001"
    print(f"SLM mode: {llm.mode()}\nLabel: {label}\n")
    final = {}
    for node, upd in stream_case(label, case_id):
        if node == "__final__":
            final = upd
            continue
        print(f"── {node}")
        if node == "planner":
            for p in upd.get("plan", []):
                print(f"   plan: {p['tool']}({p['args']})  # {p['why']}")
        if node == "grader":
            print(f"   grade: {upd['grade']}  gaps: {upd['gaps']}")
        if node == "verifier":
            for k, v in upd["checks"].items():
                print(f"   {v['status']:7} {k:10} {v['detail']}  [{v['source'] or '-'}]")
    print(f"\n{ICON[final['verdict']]} VERDICT: {final['verdict']}  (confidence {final['confidence']})")
    print("Citations:", ", ".join(final["citations"]) or "none")
    print("Explanation:", final["explanation"])
    if final.get("review_id"):
        print(f"→ Escalated to human review queue (#{final['review_id']})")
    print(f"Trace: traces/{final['case_id']}.jsonl")

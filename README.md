# LoomTrace

An agent that checks whether a handloom product's GI-tag claim is backed by official government evidence.
It returns one of three advisory verdicts, each with citations: **Supported**, **Conflicting**, or **Insufficient** (the case goes to human review).

Agentic AI mini-project (26CD7PEAAI), BMSCE, Dept. of CSE (Data Science)
Team: Sudeeksha (1BM23CD027, lead) · Khushi Vernekar (1BM23CD026) · Trishul J (1BM23CD066) · Geetha R (1BM24CD400)

Scope: Kancheepuram Silk (GI 15), Pochampalli Ikat (GI 4), Banaras Brocades & Sarees (GI 99).

## Run instructions

```bash
cd loomtrace
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add GROQ_API_KEY / LANGSMITH_API_KEY

streamlit run app.py          # UI: verify a claim, live trace, review queue, traces
python run_case.py --golden G07          # CLI: one golden case with the full trace
python run_case.py "PURE KANCHEEPURAM SILK SAREE | GI Tag No. 15 | Woven at: Kancheepuram, Tamil Nadu | Weaver ID: KAN-W-001 | Batch: KAN-B-1001"
python eval.py                # 45-case evaluation + A/B against the baseline
```

With no API key, the agent runs in **offline rules mode**: the SLM steps fall back to deterministic rules.
If `langgraph` is not installed, a built-in runner with the same graph API is used (`loomtrace/minigraph.py`).

## Architecture

```
label ─► Extract ─► Planner ─► Retriever agent ─► Verifier agent ─► Grader ──enough──► Guardrail+HITL ─► cited verdict
          (OCR+SLM)   ▲        (calls 7 tools)    (rules + SLM)     (reflection)             │
                      └──────── gaps → re-plan (max 3 loops) ◄──────────┘          Insufficient → review queue
                CaseState is checkpointed to SQLite after every node · every step traced
```

| File | What it is |
|---|---|
| `loomtrace/graph.py` | LangGraph state graph: the 6 nodes, the re-plan loop, and the guardrail |
| `loomtrace/tools.py` | 7 typed tools (Pydantic `ToolResult`, each returns a `source_id`), review queue, feedback memory |
| `loomtrace/rag.py` | BM25 retrieval over GI application passages, filtered by product |
| `loomtrace/llm.py` | Llama 3.1 8B on Groq (JSON mode), with a rules fallback |
| `loomtrace/trace.py` | Tracer: every node, tool and SLM call goes to `traces/<case>.jsonl` (and to LangSmith if a key is set) |
| `app.py` | Streamlit UI · `run_case.py` CLI · `eval.py` evaluation |
| `data/` | GI registry, GI passages, Handloom Census 2019-20, simulated weaver and batch records, 45 golden cases |

## Agentic patterns

- **Tool use.** 7 tools: `extract_label`, `gi_registry_lookup`, `gi_doc_search`, `census_lookup`, `weaver_lookup`, `provenance_lookup`, `escalate_to_human`.
- **Agentic RAG.** GI passages are retrieved with a product filter. When results are weak, the agent re-queries with a sharper query, and the verdict cites passage IDs (e.g. `KAN-p2-2`).
- **Planning and re-planning.** The planner picks checks for each claim. The grader reports gaps (`registry_not_found`, `docs_weak`, `weaver_from_batch`), and the planner retries with a new strategy, at most 3 times.
- **Multi-agent.** The Retriever gathers evidence and the Verifier judges it.
- **Reflection.** The grader scores completeness and consistency before any verdict is issued.
- **Guardrail and human-in-the-loop.** A verdict with no citation or low confidence, and any Insufficient verdict, goes to the review queue.

## Feedback loops (demo)

1. **Re-plan loop (automatic).**
   `python run_case.py "KANJEEVARAM BRIDAL SILK SAREE | GI Tag No. 15 | Woven at: Kancheepuram, Tamil Nadu | Weaver ID: KAN-W-002 | Batch: KAN-B-1002"`
   Iteration 1 misses the registry because the spelling is unknown. The grader flags `registry_not_found`, the planner re-queries by GI No. 15, and iteration 2 returns **Supported**.
2. **Human feedback loop.** Run golden case `G12` (weaver KAN-W-099 is unknown), which gives **Insufficient** and adds the case to the review queue.
   On the *Human review queue* page, confirm the case as Supported. The weaver and batch are saved as verified facts.
   Re-run G12: `weaver_lookup` now returns the evidence with source `HUMAN-REVIEW-1`, and the verdict is **Supported**.

## Evaluation (`python eval.py`, offline rules mode)

| Metric | Baseline | LoomTrace | Target |
|---|---|---|---|
| Verdict accuracy | 38% | 100% | ≥ 80% |
| Macro-F1 | 0.30 | 1.00 | |
| Correct abstention | 47% | 100% | ≥ 90% |
| Retrieval Recall@5 | – | 1.00 | ≥ 0.80 |
| Decided verdicts with citation | – | 100% | ≥ 90% |

The baseline is one registry lookup plus one doc search, with no planning, loop or guardrail. These results are on our own 45-case golden set. That set uses simulated weaver and batch records, and the rule checks were written alongside it, so the scores are optimistic. The SLM-mode run and a set of harder cases are planned for Review 4.

## Data honesty

GI numbers, states and state-level weaver counts are from IP India and the 4th All India Handloom Census 2019-20.
GI passages are paraphrased from the GI applications. District-level census figures, weaver records and batch records are **simulated** and labelled as such: individual weaver data is not public.

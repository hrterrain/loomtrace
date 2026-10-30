"""Observability: every node, tool call and SLM call is recorded.

Events are written to traces/<case_id>.jsonl locally. If LANGSMITH_API_KEY is
set, LangGraph also sends the full run tree to LangSmith automatically
(LANGSMITH_TRACING=true).
"""
import json
import time
from pathlib import Path

from . import config


class Tracer:
    def __init__(self):
        self.case_id = None
        self.events: list[dict] = []

    def start(self, case_id: str):
        self.case_id = case_id
        self.events = []
        self.t0 = time.time()

    def log(self, node: str, kind: str, name: str, inp, out, **extra):
        ev = {"t": round(time.time() - getattr(self, "t0", time.time()), 3), "node": node,
              "kind": kind, "name": name, "input": inp, "output": out, **extra}
        self.events.append(ev)
        if self.case_id:
            with open(config.TRACES / f"{self.case_id}.jsonl", "a") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev


tracer = Tracer()


def load_trace(case_id: str) -> list[dict]:
    p = Path(config.TRACES) / f"{case_id}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

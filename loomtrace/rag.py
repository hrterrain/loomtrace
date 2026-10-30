"""Agentic RAG over GI application passages (IP India).

BM25 keyword retrieval with a product metadata filter. Pure Python so it runs
on any laptop; Chroma + bge-small embeddings is the planned upgrade.
"""
import json
import math
import re
from collections import Counter
from functools import lru_cache

from . import config

STOP = set("the a an of and or to in is are was for on with by as at from this that be it its which".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs: list[dict], k1=1.5, b=0.75):
        self.docs = docs
        self.toks = [tokenize(d["section"] + " " + d["text"]) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]
        self.k1, self.b = k1, b

    def score(self, q: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.toks[i])
        s = 0.0
        for w in q:
            if w in tf:
                s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, product: str | None = None, k: int = 5) -> list[dict]:
        q = tokenize(query)
        hits = []
        for i, d in enumerate(self.docs):
            if product and d["product"].lower() != product.lower():
                continue
            s = self.score(q, i)
            if s > 0:
                hits.append({**d, "score": round(s, 3)})
        return sorted(hits, key=lambda h: -h["score"])[:k]


@lru_cache(maxsize=1)
def index() -> BM25:
    docs = json.loads((config.DATA / "gi_docs.json").read_text())
    return BM25(docs)

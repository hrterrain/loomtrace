"""Fallback used only when `langgraph` is not installed: the same StateGraph API
(add_node / add_edge / add_conditional_edges / compile / stream / get_state) and
a SQLite checkpoint after every node, so the demo runs on any laptop."""
import json
import sqlite3
from types import SimpleNamespace

END = "__end__"


class StateGraph:
    def __init__(self, _schema):
        self.nodes, self.edges, self.cond, self.entry = {}, {}, {}, None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, a, router, mapping):
        self.cond[a] = (router, mapping)

    def compile(self, checkpointer=None):
        return _Compiled(self, checkpointer)


class SqliteCheckpoint:
    def __init__(self, path):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS checkpoints(thread_id TEXT, step INTEGER, node TEXT, state TEXT)")

    def put(self, thread_id, step, node, state):
        self.con.execute("INSERT INTO checkpoints VALUES(?,?,?,?)", (thread_id, step, node, json.dumps(state, default=str)))
        self.con.commit()


class _Compiled:
    def __init__(self, g, saver):
        self.g, self.saver, self.states = g, saver, {}

    def stream(self, inp, cfg, stream_mode="updates"):
        tid = cfg["configurable"]["thread_id"]
        state, node, step = dict(inp), self.g.entry, 0
        while node != END:
            upd = self.g.nodes[node](state) or {}
            state.update(upd)
            self.states[tid] = state
            if self.saver:
                self.saver.put(tid, step, node, state)
            yield {node: upd}
            if node in self.g.cond:
                router, mapping = self.g.cond[node]
                node = mapping[router(state)]
            else:
                node = self.g.edges.get(node, END)
            step += 1

    def get_state(self, cfg):
        return SimpleNamespace(values=self.states[cfg["configurable"]["thread_id"]])

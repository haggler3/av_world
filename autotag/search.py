"""Find scenes: boolean tag queries with time overlap, and "more scenes like this one".

Query language:  name, &, |, !, parentheses.   "cut_in & hidden_ahead & !sim_gap"
  scene mode    a scene matches if the boolean formula holds over the set of tags it has
  overlap mode  for a conjunction of positive interval tags, the tags must also overlap in time (on any agents): the cut-in happens
                while someone is hidden ahead, not 8 s apart
Similarity: cosine similarity of standardised scene vectors (scene features + tag counts); the nearest k scenes to a query scene.
"""
import re

import numpy as np


def tokenize(q):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[&|!()]", q)


def parse(q):
    """-> a function(set_of_tag_names) -> bool, and the positive names (for overlap)."""
    toks = tokenize(q)
    pos = []

    def expr(i):
        a, i = term(i)
        while i < len(toks) and toks[i] == "|":
            b, i = term(i + 1)
            a = (lambda a, b: lambda s: a(s) or b(s))(a, b)
        return a, i

    def term(i):
        a, i = factor(i)
        while i < len(toks) and toks[i] == "&":
            b, i = factor(i + 1)
            a = (lambda a, b: lambda s: a(s) and b(s))(a, b)
        return a, i

    def factor(i):
        if i >= len(toks):
            raise ValueError(f"query {q!r} ends early")
        t = toks[i]
        if t == "!":
            a, i = factor(i + 1)
            return (lambda a: lambda s: not a(s))(a), i
        if t == "(":
            a, i = expr(i + 1)
            if i >= len(toks) or toks[i] != ")":
                raise ValueError(f"missing ')' in {q!r}")
            return a, i + 1
        if not re.match(r"[A-Za-z_]", t):
            raise ValueError(f"unexpected {t!r} in {q!r}")
        pos.append(t)
        return (lambda n: lambda s: n in s)(t), i + 1

    f, i = expr(0)
    if i != len(toks):
        raise ValueError(f"unexpected {toks[i]!r} in {q!r}")
    return f, pos


def query(df, q, overlap=False):
    """df: a tag table (TagStore.frame(), status 'auto'). -> list of (scene_id, [(t_start, t_end)] overlap windows or None)."""
    f, names = parse(q)
    df = df[df.status == "auto"]
    out = []
    for sid, g in df.groupby("scene_id"):
        have = set(g.name)
        if not f(have):
            continue
        if overlap and "|" not in q and "!" not in q and len(names) > 1:
            win = [(a, b) for a, b in zip(g[g.name == names[0]].t_start, g[g.name == names[0]].t_end)]
            for n in names[1:]:
                other = list(zip(g[g.name == n].t_start, g[g.name == n].t_end))
                win = [(max(a, c), min(b, d)) for a, b in win for c, d in other if max(a, c) < min(b, d)]
            if not win:
                continue
            out.append((sid, win))
        else:
            out.append((sid, None))
    return out


class Index:
    def __init__(self, ids, vectors):
        self.ids = list(ids)
        v = np.asarray(vectors, float)
        self.mu, self.sd = v.mean(0), v.std(0) + 1e-6
        z = (v - self.mu) / self.sd
        self.z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    def similar(self, scene_id, k=5):
        i = self.ids.index(scene_id)
        sim = self.z @ self.z[i]
        o = np.argsort(-sim)[:k]
        return [(self.ids[j], float(sim[j])) for j in o]


def embedding(features, tag_names, scene_tags, vocab):
    """Scene vector: features + per-tag counts (log1p)."""
    counts = np.array([[np.log1p(sum(1 for t in scene_tags if t == n)) for n in vocab]])
    return np.concatenate([np.asarray(features, float)[None], counts], 1)[0]

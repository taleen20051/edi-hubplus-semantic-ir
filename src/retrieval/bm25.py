from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


@dataclass
class BM25Index:
    doc_ids: List[str]
    doc_len: List[int]
    avgdl: float
    idf: Dict[str, float]
    tf: List[Counter]

    @classmethod
    def build(cls, docs: Sequence[Tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        # We store k1/b only in scoring; build computes TF + IDF.
        doc_ids: List[str] = []
        tf: List[Counter] = []
        doc_len: List[int] = []

        df: Counter = Counter()
        for doc_id, text in docs:
            doc_ids.append(str(doc_id))
            tokens = tokenize(text)
            c = Counter(tokens)
            tf.append(c)
            doc_len.append(len(tokens))
            for term in c.keys():
                df[term] += 1

        n_docs = len(docs)
        avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0

        # BM25+ style IDF (robust; avoids negative IDF)
        idf: Dict[str, float] = {}
        for term, dfi in df.items():
            idf[term] = math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5))

        return cls(doc_ids=doc_ids, doc_len=doc_len, avgdl=avgdl, idf=idf, tf=tf)

    def score(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return [(doc_id, 0.0) for doc_id in self.doc_ids]

        scores: List[float] = [0.0 for _ in self.doc_ids]
        for i, doc_id in enumerate(self.doc_ids):
            dl = self.doc_len[i]
            denom_norm = (1 - b) + b * (dl / self.avgdl) if self.avgdl > 0 else 1.0
            doc_tf = self.tf[i]

            s = 0.0
            for term in q_terms:
                if term not in doc_tf:
                    continue
                f = doc_tf[term]
                term_idf = self.idf.get(term, 0.0)
                s += term_idf * (f * (k1 + 1)) / (f + k1 * denom_norm)
            scores[i] = s

        return list(zip(self.doc_ids, scores))
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


# Token pattern for simple lexical retrieval.
# Keeps words, numbers, and contractions such as "user's".
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    """
    Convert text into lowercase word tokens for BM25 retrieval.

    This is intentionally simple because BM25 is used as a lexical baseline.
    """
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


@dataclass
# Lightweight BM25 index used for the lexical retrieval baseline.
class BM25Index:

    doc_ids: List[str]
    doc_len: List[int]
    avgdl: float
    idf: Dict[str, float]
    tf: List[Counter]

    @classmethod
    # Build a BM25 index from a sequence of (document_id, text) pairs.
    def build(cls, docs: Sequence[Tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        doc_ids: List[str] = []
        tf: List[Counter] = []
        doc_len: List[int] = []

        # Document frequency: number of documents containing each term.
        df: Counter = Counter()

        for doc_id, text in docs:
            doc_ids.append(str(doc_id))

            # Tokenise document and count term frequencies.
            tokens = tokenize(text)
            counts = Counter(tokens)

            tf.append(counts)
            doc_len.append(len(tokens))

            # Count each term once per document for IDF.
            for term in counts.keys():
                df[term] += 1

        n_docs = len(docs)
        avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0

        # BM25-style IDF with smoothing.
        # The +1 inside log avoids negative IDF values for very common terms.
        idf: Dict[str, float] = {}
        for term, dfi in df.items():
            idf[term] = math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5))

        return cls(doc_ids=doc_ids, doc_len=doc_len, avgdl=avgdl, idf=idf, tf=tf)


# Score every indexed document against the query using BM25.
    def score(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)

        # Empty query receives zero score for all documents.
        if not q_terms:
            return [(doc_id, 0.0) for doc_id in self.doc_ids]

        scores: List[float] = [0.0 for _ in self.doc_ids]

        for i, doc_id in enumerate(self.doc_ids):
            doc_length = self.doc_len[i]
            doc_tf = self.tf[i]

            # BM25 document-length normalisation.
            denom_norm = (1 - b) + b * (doc_length / self.avgdl) if self.avgdl > 0 else 1.0

            score = 0.0
            for term in q_terms:
                if term not in doc_tf:
                    continue

                term_frequency = doc_tf[term]
                term_idf = self.idf.get(term, 0.0)

                score += term_idf * (term_frequency * (k1 + 1)) / (
                    term_frequency + k1 * denom_norm
                )

            scores[i] = score

        return list(zip(self.doc_ids, scores))
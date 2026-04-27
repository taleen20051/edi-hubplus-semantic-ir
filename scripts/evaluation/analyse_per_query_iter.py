import json
import argparse

def load_qrels(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {q: set(map(str, rels)) for q, rels in raw.items()}

def load_run(path):
    """Load a run JSONL file.

    Supports multiple run schemas:
      - {"qid": "q1", "ranking": [{"rid": "44", "score": ...}, ...]}
      - {"qid": "q1", "results": [{"docid": "44", "score": ...}, ...]}

    Defensive behavior:
      - Treat missing or null rankings as an empty list.
      - Accept both `rid` and `docid` keys for document IDs.
      - Skip blank lines.
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            if not qid:
                continue

            ranking = obj.get("ranking")
            if ranking is None:
                ranking = obj.get("results")
            if ranking is None:
                ranking = []

            rids = []
            for x in ranking:
                if not isinstance(x, dict):
                    continue
                rid = x.get("rid")
                if rid is None:
                    rid = x.get("docid")
                if rid is None:
                    continue
                rids.append(str(rid))

            out[qid] = rids
    return out

def precision_recall_f1_at_k(ranked, relevant, k=10):
    topk = ranked[:k]
    hits = sum(1 for rid in topk if rid in relevant)
    p = hits / k if k > 0 else 0
    r = hits / len(relevant) if relevant else 0
    f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0
    return hits, p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    print(f"{'QID':<6} {'RelDocs':<8} {'Hits@10':<8} {'P@10':<8} {'R@10':<8} {'F1@10'}")
    print("-"*50)

    for qid, relevant in qrels.items():
        if not relevant:
            continue
        ranked = run.get(qid, [])
        hits, p, r, f1 = precision_recall_f1_at_k(ranked, relevant, args.k)
        print(f"{qid:<6} {len(relevant):<8} {hits:<8} {p:<8.3f} {r:<8.3f} {f1:.3f}")

if __name__ == "__main__":
    main()
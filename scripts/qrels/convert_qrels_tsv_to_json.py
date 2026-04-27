from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_tsv_qrels(path: Path) -> Dict[str, List[str]]:
    """
    TREC qrels format:
        qid 0 docid rel

    We convert this into:
        {
            "q1": ["83", "92", ...],
            "q2": [...]
        }

    Only rel > 0 are considered relevant.
    """

    qrels: Dict[str, List[str]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid qrels line at {path}:{line_no}. "
                    f"Expected 4 columns, got {len(parts)}"
                )

            qid, _unused, docid, rel = parts

            if int(rel) > 0:
                qrels.setdefault(qid, []).append(str(docid))

            else:
                qrels.setdefault(qid, [])

    return qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels_tsv", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    args = parser.parse_args()

    qrels_dict = load_tsv_qrels(args.qrels_tsv)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(qrels_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Converted qrels TSV → JSON:", args.out_json)


if __name__ == "__main__":
    main()
import json
import sys

# Usage:
#   python make_qrels.py input.jsonl qrels.tsv
#
# input.jsonl = the file containing the lines like:
# {"qid":"q1","rid":"1","rel":0}

in_path = sys.argv[1]
out_path = sys.argv[2]

with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["qid"]
        docid = obj["rid"]   # keep as string (matches your run docids)
        rel = int(obj["rel"])
        f_out.write(f"{qid} 0 {docid} {rel}\n")

print(f"Wrote qrels to: {out_path}")

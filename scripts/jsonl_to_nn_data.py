#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Claire Ivanenka <claire@gnu-ai.org>
#
# jsonl_to_nn_data.py — convert JSON(L) text data into the vector
# JSONL consumed by train_nn.py ({"input": [...], "output": [...]}),
# so a text dataset can be trained into a .nn model loadable by
# neuron-translator:
#
#     python3 scripts/jsonl_to_nn_data.py --in conversations.jsonl \
#         --out data.jsonl --dim 16 --out-dim 4
#     python3 scripts/train_nn.py --data data.jsonl --topology 16,32,4 \
#         --epochs 200 --out model.nn
#     echo 'load /root/model.nn' > /llm1
#
# Accepted input records (auto-detected per line):
#   {"conversations": [{"from": "human", "value": ...},
#                      {"from": "gpt",   "value": ...}, ...]}
#       -> one {"input", "output"} pair per (human, next gpt) turn
#   {"question": ..., "answer": ...}          -> one pair
#   {"text": ...}                              -> autoencoder pair (in == out)
#   {"input": [...], "output": [...]}          -> passthrough (already vectors)
#
# Vectorization: hashing trick (md5, stable across runs), lowercase
# word tokens, binary presence, fixed dimension --dim.  Pure standard
# library.

import argparse
import hashlib
import json
import re
import sys

TOKEN_RE = re.compile(r"[a-z0-9à-ÿ_]+")


def vectorize(text, dim):
    v = [0.0] * dim
    toks = TOKEN_RE.findall(text.lower())
    for t in toks:
        v[int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % dim] = 1.0
    return v


def records_from_line(line, dim, out_dim, src):
    """Yield {"input":…, "output":…} dicts for one input line."""
    rec = json.loads(line)
    if isinstance(rec.get("input"), list) and isinstance(rec.get("output"), list):
        yield {"input": [float(v) for v in rec["input"]],
               "output": [float(v) for v in rec["output"]]}
        return
    if "conversations" in rec:
        conv = rec["conversations"]
        human = None
        for turn in conv:
            who = turn.get("from", "")
            val = str(turn.get("value", ""))
            if who == "human":
                human = val
            elif who in ("gpt", "assistant") and human is not None:
                yield {"input": vectorize(human, dim),
                       "output": vectorize(val, out_dim)}
                human = None
        return
    if "question" in rec and "answer" in rec:
        yield {"input": vectorize(str(rec["question"]), dim),
               "output": vectorize(str(rec["answer"]), out_dim)}
        return
    if "text" in rec:
        v = vectorize(str(rec["text"]), dim)
        yield {"input": v, "output": [float(x) for x in v[:out_dim]] or v}
        return
    raise ValueError(f"unrecognized record format: {sorted(rec)[:5]}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input JSONL (conversations/question-answer/text/vectors)")
    ap.add_argument("--out", required=True, help="output JSONL for train_nn.py")
    ap.add_argument("--dim", type=int, default=16, help="input vector size (topology first layer)")
    ap.add_argument("--out-dim", type=int, default=4, help="output vector size (topology last layer)")
    ap.add_argument("--max-records", type=int, default=0, help="stop after N output pairs (0 = all)")
    args = ap.parse_args()

    if args.dim < 1 or args.out_dim < 1:
        sys.exit("--dim and --out-dim must be >= 1")

    n_in = n_out = n_err = 0
    with open(args.inp) as fi, open(args.out, "w") as fo:
        for lineno, line in enumerate(fi, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                for rec in records_from_line(line, args.dim, args.out_dim, args.inp):
                    fo.write(json.dumps(rec) + "\n")
                    n_out += 1
                    if args.max_records and n_out >= args.max_records:
                        break
            except Exception as e:
                n_err += 1
                print(f"{args.inp}:{lineno}: skipped: {e}", file=sys.stderr)
            if args.max_records and n_out >= args.max_records:
                break

    print(f"{n_in} input lines -> {n_out} vector pairs in {args.out} "
          f"({n_err} skipped)", file=sys.stderr)
    print(f"suggested topology: {args.dim},<hidden>,{args.out_dim} — e.g. "
          f"python3 scripts/train_nn.py --data {args.out} "
          f"--topology {args.dim},32,{args.out_dim} --out model.nn",
          file=sys.stderr)
    if n_out == 0:
        sys.exit("no output pairs produced")


if __name__ == "__main__":
    main()

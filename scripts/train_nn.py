#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Claire Ivanenka <claire@gnu-ai.org>
#
# External trainer for neuron-translator: trains a small dense sigmoid
# feedforward network on vector data and writes the .nn v2 binary
# format documented in README.md ("The .nn Model Format (v2)"), so the
# result loads with:
#
#     echo 'load /path/to/model.nn' > /llm1
#
# Input data: JSONL, one record per line:
#     {"input": [0.5, 0.3], "output": [1.0]}
#
# Usage:
#     python3 train_nn.py --data train.jsonl --topology 2,8,1 \
#         --epochs 500 --lr 0.5 --out model.nn
#
# Without --topology, the input/output sizes come from the first
# record and a single hidden layer of max(2, in+out) neurons is used.
#
# Pure standard library (no numpy): the networks targeted here are
# small.  The .nn format is a plain dense MLP layout - a transformer
# LLM does not fit it; use this for small vector models only.

import argparse
import json
import math
import random
import struct
import sys

MAGIC = 0x4E455552          # bytes 52 55 45 4e on little-endian
VERSION = 2
MAX_LAYERS = 8
MAX_NEURONS_PER_LAYER = 8192
DEFAULT_RESET_POTENTIAL = -80.0   # mV, as shown by cat /llm1


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def forward(sizes, weights, biases, x):
    """weights[k] / biases[k] are the layer-k (1-based) blocks."""
    a = list(x)
    activations = [a]
    for k in range(1, len(sizes)):
        wk, bk = weights[k], biases[k]
        nxt = []
        for r in range(sizes[k]):
            s = bk[r]
            row = wk[r]
            for c in range(sizes[k - 1]):
                s += row[c] * a[c]
            nxt.append(sigmoid(s))
        a = nxt
        activations.append(a)
    return activations


def train(sizes, data, epochs, lr, seed):
    random.seed(seed)
    weights = [None] + [
        [[random.uniform(-0.5, 0.5) for _ in range(sizes[k - 1])]
         for _ in range(sizes[k])]
        for k in range(1, len(sizes))
    ]
    biases = [None] + [
        [random.uniform(-0.5, 0.5) for _ in range(sizes[k])]
        for k in range(1, len(sizes))
    ]
    n = len(data)
    for epoch in range(epochs):
        random.shuffle(data)
        total = 0.0
        for x, y in data:
            acts = forward(sizes, weights, biases, x)
            out = acts[-1]
            # output delta (MSE + sigmoid)
            delta = [(out[i] - y[i]) * out[i] * (1.0 - out[i])
                     for i in range(sizes[-1])]
            deltas = [None] * len(sizes)
            deltas[-1] = delta
            for k in range(len(sizes) - 2, 0, -1):
                nxt = deltas[k + 1]
                wk = weights[k + 1]
                d = []
                for c in range(sizes[k]):
                    s = 0.0
                    for r in range(sizes[k + 1]):
                        s += wk[r][c] * nxt[r]
                    a = acts[k][c]
                    d.append(s * a * (1.0 - a))
                deltas[k] = d
            for k in range(1, len(sizes)):
                for r in range(sizes[k]):
                    for c in range(sizes[k - 1]):
                        weights[k][r][c] -= lr * deltas[k][r] * acts[k - 1][c]
                    biases[k][r] -= lr * deltas[k][r]
            total += sum((out[i] - y[i]) ** 2 for i in range(sizes[-1]))
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{epochs}  mse={total / n:.6f}",
                  file=sys.stderr)
    return weights, biases


def write_nn(path, sizes, weights, biases, reset_potential):
    lc = len(sizes)
    if not (2 <= lc <= MAX_LAYERS):
        sys.exit(f"layer_count {lc} out of range 2..{MAX_LAYERS}")
    for s in sizes:
        if s == 0 or s > MAX_NEURONS_PER_LAYER:
            sys.exit(f"layer size {s} out of range 1..{MAX_NEURONS_PER_LAYER}")
    total_neurons = sum(sizes)
    total_weights = sum(sizes[k] * sizes[k - 1] for k in range(1, lc))
    total_biases = sum(sizes[k] for k in range(1, lc))
    # canonical offsets, as recomputed by network_load()
    layer_offsets = [sum(sizes[:i]) for i in range(lc)]
    weight_offsets = [sum(sizes[k] * sizes[k - 1] for k in range(1, j + 1))
                      for j in range(lc - 1)]
    bias_offsets = [sum(sizes[k] for k in range(1, i + 1)) for i in range(lc)]

    header = struct.pack(
        "<I H B B 8H H H 2x f 4x Q Q Q",
        MAGIC, VERSION, lc, 0,
        *(sizes + [0] * (8 - lc)),
        sizes[0], sizes[-1],
        reset_potential,
        total_neurons, total_weights, total_biases,
    )
    offsets = b"".join(struct.pack("<Q", o) for o in layer_offsets)
    offsets += b"".join(struct.pack("<Q", o) for o in weight_offsets)
    offsets += b"".join(struct.pack("<Q", o) for o in bias_offsets)
    voltages = struct.pack(f"<{total_neurons}f",
                           *([reset_potential] * total_neurons))
    wdata = b""
    for k in range(1, lc):
        for r in range(sizes[k]):
            wdata += struct.pack(f"<{sizes[k - 1]}f", *weights[k][r])
    bdata = b""
    for k in range(1, lc):
        bdata += struct.pack(f"<{sizes[k]}f", *biases[k])
    with open(path, "wb") as f:
        f.write(header + offsets + voltages + wdata + bdata)
    return total_neurons, total_weights, total_biases


def read_nn(path):
    """Parse a .nn v2 file back (self-check of what we just wrote)."""
    with open(path, "rb") as f:
        blob = f.read()
    magic, version, lc = struct.unpack_from("<I H B", blob, 0)
    if magic != MAGIC or version != VERSION:
        sys.exit("self-check: bad magic/version")
    sizes = list(struct.unpack_from("<8H", blob, 8))[:lc]
    in_size, out_size = struct.unpack_from("<H H", blob, 24)
    reset_potential, = struct.unpack_from("<f", blob, 30)
    tn, tw, tb = struct.unpack_from("<Q Q Q", blob, 38)
    off = 62
    layer_offsets = struct.unpack_from(f"<{lc}Q", blob, off); off += 8 * lc
    weight_offsets = struct.unpack_from(f"<{lc - 1}Q", blob, off); off += 8 * (lc - 1)
    bias_offsets = struct.unpack_from(f"<{lc}Q", blob, off); off += 8 * lc
    voltages = struct.unpack_from(f"<{tn}f", blob, off); off += 4 * tn
    weights_flat = struct.unpack_from(f"<{tw}f", blob, off); off += 4 * tw
    biases_flat = struct.unpack_from(f"<{tb}f", blob, off); off += 4 * tb
    if off != len(blob):
        sys.exit("self-check: trailing bytes")
    weights = [None]
    for k in range(1, lc):
        rows = sizes[k]
        cols = sizes[k - 1]
        start = weight_offsets[k - 1]
        weights.append([list(weights_flat[start + r * cols:start + (r + 1) * cols])
                        for r in range(rows)])
    biases = [None]
    for k in range(1, lc):
        start = bias_offsets[k - 1]
        biases.append(list(biases_flat[start:start + sizes[k]]))
    return sizes, weights, biases, in_size, out_size, reset_potential


def load_data(path):
    data = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                data.append(([float(v) for v in rec["input"]],
                             [float(v) for v in rec["output"]]))
            except Exception as e:
                sys.exit(f"{path}:{lineno}: bad record: {e}")
    if not data:
        sys.exit(f"{path}: no records")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="JSONL of {input, output} vectors")
    ap.add_argument("--topology", help="e.g. 2,8,1 (default: in,H,out from the data)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output .nn file")
    ap.add_argument("--reset-potential", type=float, default=DEFAULT_RESET_POTENTIAL)
    args = ap.parse_args()

    data = load_data(args.data)
    in_size = len(data[0][0])
    out_size = len(data[0][1])
    for x, y in data:
        if len(x) != in_size or len(y) != out_size:
            sys.exit("inconsistent vector sizes in the data")
    if args.topology:
        sizes = [int(s) for s in args.topology.split(",")]
    else:
        sizes = [in_size, max(2, in_size + out_size), out_size]
    if sizes[0] != in_size or sizes[-1] != out_size:
        sys.exit(f"topology {sizes} does not match data sizes {in_size}->{out_size}")

    weights, biases = train(sizes, list(data), args.epochs, args.lr, args.seed)
    tn, tw, tb = write_nn(args.out, sizes, weights, biases, args.reset_potential)
    print(f"wrote {args.out}: topology {sizes}, {tn} neurons, "
          f"{tw} weights, {tb} biases", file=sys.stderr)

    # self-check: parse the file back and run one sample through it
    s2, w2, b2, i2, o2, rp = read_nn(args.out)
    assert s2 == sizes and (i2, o2) == (in_size, out_size)
    x, y = data[0]
    got = forward(s2, w2, b2, x)[-1]
    print(f"self-check: input={x} expected={y} got={[round(v, 4) for v in got]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

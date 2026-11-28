/*
 * llm.c - Generative Language Model (mini transformer) for the LLM Translator
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire Ivanenka <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file llm.c
 *  @brief Character-level autoregressive transformer (forward, backprop,
 *         sampling, persistence)
 *
 *  Architecture: byte tokenizer (ids 0..255 + BOS/EOS/PAD), token and
 *  position embeddings, LLM_BLOCKS causal multi-head self-attention
 *  blocks with RMSNorm and a ReLU feed-forward, a final RMSNorm and a
 *  linear head over the vocabulary.  Training is a full manual
 *  backpropagation of the next-token cross-entropy with SGD.
 *
 *  Memory: one contiguous block per model, [params][grads][scratch],
 *  exactly like CompactNeuralNetwork in neuron.c.  Every per-block
 *  activation is indexed by block (the backward pass of block 0 must
 *  not read the activations of block 1).  Pure C23/POSIX+libm.
 */

#include "llm.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*****************************************************************************
 *                                                                           *
 *                    LAYOUT MACROS (offsets in floats)                      *
 *                                                                           *
 *****************************************************************************/

/* Parameters, per transformer block (block base + these offsets). */
#define B_LN1G  0u                              /* [D] */
#define B_WQ    (B_LN1G + LLM_D_MODEL)          /* [D][D] */
#define B_WK    (B_WQ + LLM_D_MODEL * LLM_D_MODEL)
#define B_WV    (B_WK + LLM_D_MODEL * LLM_D_MODEL)
#define B_WO    (B_WV + LLM_D_MODEL * LLM_D_MODEL)
#define B_LN2G  (B_WO + LLM_D_MODEL * LLM_D_MODEL)
#define B_W1    (B_LN2G + LLM_D_MODEL)          /* [D][FF] */
#define B_B1    (B_W1 + LLM_D_MODEL * LLM_D_FF) /* [FF] */
#define B_W2    (B_B1 + LLM_D_FF)               /* [FF][D] */
#define B_B2    (B_W2 + LLM_D_FF * LLM_D_MODEL) /* [D] */

/* Scratch: [BLOCKS] marks a per-block array. */
#define SEQ  LLM_MAX_SEQ
#define D    LLM_D_MODEL
#define BLKSEQ(b, t) ((size_t)(b) * SEQ + (size_t)(t))

#define OFF_X       0u                                        /* [SEQ][D] residual stream */
#define OFF_XPRE    (OFF_X + SEQ * D)                         /* [BLOCKS][SEQ][D] block input */
#define OFF_XMID    (OFF_XPRE + LLM_BLOCKS * SEQ * D)         /* [BLOCKS][SEQ][D] after attention */
#define OFF_XFIN    (OFF_XMID + LLM_BLOCKS * SEQ * D)         /* [SEQ][D] before final norm */
#define OFF_H1      (OFF_XFIN + SEQ * D)                      /* [BLOCKS][SEQ][D] after ln1 */
#define OFF_H2      (OFF_H1 + LLM_BLOCKS * SEQ * D)           /* [BLOCKS][SEQ][D] after ln2 */
#define OFF_Q       (OFF_H2 + LLM_BLOCKS * SEQ * D)           /* [BLOCKS][SEQ][D] */
#define OFF_K       (OFF_Q + LLM_BLOCKS * SEQ * D)
#define OFF_V       (OFF_K + LLM_BLOCKS * SEQ * D)
#define OFF_Z       (OFF_V + LLM_BLOCKS * SEQ * D)
#define OFF_AOUT    (OFF_Z + LLM_BLOCKS * SEQ * D)
#define OFF_FF      (OFF_AOUT + LLM_BLOCKS * SEQ * D)         /* [BLOCKS][SEQ][FF] post-ReLU */
#define OFF_ATTN    (OFF_FF + LLM_BLOCKS * SEQ * LLM_D_FF)    /* [BLOCKS][HEADS][SEQ][SEQ] */
#define OFF_XF      (OFF_ATTN + LLM_BLOCKS * LLM_HEADS * SEQ * SEQ) /* [SEQ][D] after final norm */
#define OFF_LOGITS  (OFF_XF + SEQ * D)                        /* [SEQ][VOCAB] */
#define OFF_PROBS   (OFF_LOGITS + SEQ * LLM_VOCAB)            /* [SEQ][VOCAB] */
#define OFF_GX      (OFF_PROBS + SEQ * LLM_VOCAB)             /* [SEQ][D] grad on current block output */
#define OFF_GH1     (OFF_GX + SEQ * D)                        /* [BLOCKS][SEQ][D] */
#define OFF_GH2     (OFF_GH1 + LLM_BLOCKS * SEQ * D)          /* [BLOCKS][SEQ][D] */
#define OFF_GQ      (OFF_GH2 + LLM_BLOCKS * SEQ * D)          /* [BLOCKS][SEQ][D] */
#define OFF_GK      (OFF_GQ + LLM_BLOCKS * SEQ * D)
#define OFF_GV      (OFF_GK + LLM_BLOCKS * SEQ * D)
#define OFF_GZ      (OFF_GV + LLM_BLOCKS * SEQ * D)
#define OFF_GAOUT   (OFF_GZ + LLM_BLOCKS * SEQ * D)
#define OFF_GFF     (OFF_GAOUT + LLM_BLOCKS * SEQ * D)       /* [BLOCKS][SEQ][FF] */
#define OFF_GATTN   (OFF_GFF + LLM_BLOCKS * SEQ * LLM_D_FF)   /* [BLOCKS][HEADS][SEQ][SEQ] */
#define OFF_GXF     (OFF_GATTN + LLM_BLOCKS * LLM_HEADS * SEQ * SEQ) /* [SEQ][D] */
#define OFF_DLOGITS (OFF_GXF + SEQ * D)                       /* [SEQ][VOCAB] */
#define SCRATCH_FLOATS (OFF_DLOGITS + SEQ * LLM_VOCAB)

/* Scratch accessors.  b = block index, t = position, d/f/h/v = component. */
#define X(m, t, d)       ((m)->scratch[OFF_X + (size_t)(t) * D + (d)])
#define HPRE(m, b, t, d) ((m)->scratch[OFF_XPRE + BLKSEQ (b, t) * D + (d)])
#define HMID(m, b, t, d) ((m)->scratch[OFF_XMID + BLKSEQ (b, t) * D + (d)])
#define XFIN(m, t, d)    ((m)->scratch[OFF_XFIN + (size_t)(t) * D + (d)])
#define H1(m, b, t, d)   ((m)->scratch[OFF_H1 + BLKSEQ (b, t) * D + (d)])
#define H2(m, b, t, d)   ((m)->scratch[OFF_H2 + BLKSEQ (b, t) * D + (d)])
#define Q(m, b, t, d)    ((m)->scratch[OFF_Q + BLKSEQ (b, t) * D + (d)])
#define K(m, b, t, d)    ((m)->scratch[OFF_K + BLKSEQ (b, t) * D + (d)])
#define V(m, b, t, d)    ((m)->scratch[OFF_V + BLKSEQ (b, t) * D + (d)])
#define Z(m, b, t, d)    ((m)->scratch[OFF_Z + BLKSEQ (b, t) * D + (d)])
#define AOUT(m, b, t, d) ((m)->scratch[OFF_AOUT + BLKSEQ (b, t) * D + (d)])
#define FF(m, b, t, f)   ((m)->scratch[OFF_FF + BLKSEQ (b, t) * LLM_D_FF + (f)])
#define ATT(m, b, h, t, s) ((m)->scratch[OFF_ATTN + (((size_t)(b) * LLM_HEADS + (h)) * SEQ + (size_t)(t)) * SEQ + (s)])
#define XF(m, t, d)      ((m)->scratch[OFF_XF + (size_t)(t) * D + (d)])
#define LOGIT(m, t, v)   ((m)->scratch[OFF_LOGITS + (size_t)(t) * LLM_VOCAB + (v)])
#define PROB(m, t, v)    ((m)->scratch[OFF_PROBS + (size_t)(t) * LLM_VOCAB + (v)])
#define GX(m, t, d)      ((m)->scratch[OFF_GX + (size_t)(t) * D + (d)])
#define GH1(m, b, t, d)  ((m)->scratch[OFF_GH1 + BLKSEQ (b, t) * D + (d)])
#define GH2(m, b, t, d)  ((m)->scratch[OFF_GH2 + BLKSEQ (b, t) * D + (d)])
#define GQ(m, b, t, d)   ((m)->scratch[OFF_GQ + BLKSEQ (b, t) * D + (d)])
#define GK(m, b, t, d)   ((m)->scratch[OFF_GK + BLKSEQ (b, t) * D + (d)])
#define GV(m, b, t, d)   ((m)->scratch[OFF_GV + BLKSEQ (b, t) * D + (d)])
#define GZ(m, b, t, d)   ((m)->scratch[OFF_GZ + BLKSEQ (b, t) * D + (d)])
#define GAOUT(m, b, t, d) ((m)->scratch[OFF_GAOUT + BLKSEQ (b, t) * D + (d)])
#define GFF(m, b, t, f)  ((m)->scratch[OFF_GFF + BLKSEQ (b, t) * LLM_D_FF + (f)])
#define GATT(m, b, h, t, s) ((m)->scratch[OFF_GATTN + (((size_t)(b) * LLM_HEADS + (h)) * SEQ + (size_t)(t)) * SEQ + (s)])
#define GXF(m, t, d)     ((m)->scratch[OFF_GXF + (size_t)(t) * D + (d)])
#define DLOGIT(m, t, v)  ((m)->scratch[OFF_DLOGITS + (size_t)(t) * LLM_VOCAB + (v)])


/*****************************************************************************
 *                                                                           *
 *                          SMALL MATH HELPERS                               *
 *                                                                           *
 *****************************************************************************/

/* y = x @ W, W is [din][dout] row-major. */
static void matvec(const float *x, const float *W,
                   size_t din, size_t dout, float *y)
{
    for (size_t o = 0; o < dout; o++) {
        float s = 0.0f;
        for (size_t i = 0; i < din; i++)
            s += x[i] * W[i * dout + o];
        y[o] = s;
    }
}

/* y = W @ x, W is [dout][din] row-major. */
static void matvec_T(const float *x, const float *W,
                     size_t din, size_t dout, float *y)
{
    for (size_t o = 0; o < dout; o++) {
        float s = 0.0f;
        for (size_t i = 0; i < din; i++)
            s += x[i] * W[o * din + i];
        y[o] = s;
    }
}

static void softmax(float *v, size_t n)
{
    float max = v[0];
    for (size_t i = 1; i < n; i++)
        if (v[i] > max) max = v[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        v[i] = expf(v[i] - max);
        sum += v[i];
    }
    for (size_t i = 0; i < n; i++)
        v[i] /= sum;
}

/* y = RMSNorm(x) * g */
static void rmsnorm(const float *x, const float *g, float *y, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; i++)
        s += x[i] * x[i];
    float rms = sqrtf(s / (float)n + 1e-5f);
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] * g[i] / rms;
}

/* Backward of RMSNorm: dx (written) and dg (accumulated) from x, g, dy. */
static void rmsnorm_back(const float *x, const float *g, const float *dy,
                         size_t n, float *dx, float *dg)
{
    float s = 0.0f, dot = 0.0f;
    for (size_t i = 0; i < n; i++)
        s += x[i] * x[i];
    float rms = sqrtf(s / (float)n + 1e-5f);
    for (size_t i = 0; i < n; i++)
        dot += x[i] * g[i] * dy[i];
    for (size_t i = 0; i < n; i++) {
        dx[i] = g[i] * dy[i] / rms
              - x[i] * dot / ((float)n * rms * rms * rms);
        if (dg)
            dg[i] += dy[i] * x[i] / rms;
    }
}

/* Deterministic PRNG, same spirit as the seeding of neuron.c. */
static float prng_float(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x ? x : 1u;
    return (float)(x & 0x007FFFFFu) / (float)0x007FFFFFu;
}


/*****************************************************************************
 *                                                                           *
 *                    INIT / FREE / LAYOUT / PERSISTENCE                     *
 *                                                                           *
 *****************************************************************************/

LlmLayout llm_layout(void)
{
    LlmLayout l;
    l.tok_emb = 0;
    l.pos_emb = l.tok_emb + LLM_VOCAB * D;
    l.block   = l.pos_emb + SEQ * D;
    l.lnf_g   = l.block + LLM_BLOCKS * LLM_BLOCK_FLOATS;
    l.head_w  = l.lnf_g + D;
    l.head_b  = l.head_w + LLM_VOCAB * D;
    return l;
}

int llm_init(LlmModel *m)
{
    if (!m) { errno = EINVAL; return -1; }
    LlmModel fresh = {0};
    fresh.mem_size = (2u * LLM_PARAM_FLOATS + SCRATCH_FLOATS) * sizeof(float);
    fresh.mem = calloc(1, fresh.mem_size);
    if (!fresh.mem) return -1;
    fresh.params  = fresh.mem;
    fresh.grads   = fresh.params + LLM_PARAM_FLOATS;
    fresh.scratch = fresh.grads + LLM_PARAM_FLOATS;

    LlmLayout l = llm_layout();
    /* Weights: deterministic pseudo-random in [-0.2, 0.2], exactly like
     * network_init() (seed = i * Knuth's multiplicative constant). */
    for (size_t i = 0; i < LLM_PARAM_FLOATS; i++) {
        uint32_t seed = (uint32_t)i * 2654435761U;
        float random = (float)(seed & 0x007FFFFFu) / (float)0x007FFFFFu;
        fresh.params[i] = random * 0.4f - 0.2f;
    }
    /* Norm gains are 1, biases are 0: overwrite the random fill. */
    for (size_t b = 0; b < LLM_BLOCKS; b++) {
        float *bp = fresh.params + l.block + b * LLM_BLOCK_FLOATS;
        for (size_t i = 0; i < D; i++) {
            bp[B_LN1G + i] = 1.0f;
            bp[B_LN2G + i] = 1.0f;
        }
        for (size_t i = 0; i < LLM_D_FF; i++)
            bp[B_B1 + i] = 0.0f;
        for (size_t i = 0; i < D; i++)
            bp[B_B2 + i] = 0.0f;
    }
    for (size_t i = 0; i < D; i++)
        fresh.params[l.lnf_g + i] = 1.0f;
    for (size_t i = 0; i < LLM_VOCAB; i++)
        fresh.params[l.head_b + i] = 0.0f;

    fresh.initialized = true;
    llm_free(m);
    *m = fresh;
    return 0;
}

void llm_free(LlmModel *m)
{
    if (!m) return;
    free(m->mem);
    m->mem = NULL;
    m->params = m->grads = m->scratch = NULL;
    m->mem_size = 0;
    m->initialized = false;
}

int llm_save(const LlmModel *m, const char *path)
{
    if (!m || !m->initialized || !path) { errno = EINVAL; return -1; }
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t header[3] = { LLM_FILE_MAGIC, LLM_FILE_VERSION, LLM_PARAM_FLOATS };
    int ok = fwrite(header, sizeof header, 1, f) == 1
          && fwrite(m->params, sizeof(float), LLM_PARAM_FLOATS, f)
             == LLM_PARAM_FLOATS;
    if (fclose(f) != 0)
        ok = 0;
    return ok ? 0 : -1;
}

int llm_load(LlmModel *m, const char *path)
{
    if (!m || !path) { errno = EINVAL; return -1; }
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t header[3];
    if (fread(header, sizeof header, 1, f) != 1
        || header[0] != LLM_FILE_MAGIC
        || header[1] != LLM_FILE_VERSION
        || header[2] != LLM_PARAM_FLOATS) {
        fclose(f);
        errno = EINVAL;
        return -1;
    }
    if (!m->initialized && llm_init(m) != 0) {
        fclose(f);
        return -1;
    }
    int ok = fread(m->params, sizeof(float), LLM_PARAM_FLOATS, f)
             == LLM_PARAM_FLOATS;
    fclose(f);
    if (!ok) { errno = EIO; return -1; }
    return 0;
}


/*****************************************************************************
 *                                                                           *
 *                       TOKENIZER (byte-level)                              *
 *                                                                           *
 *****************************************************************************/

size_t llm_tokenize(const char *text, uint16_t *ids, size_t max_ids)
{
    if (!text || !ids || max_ids == 0) return 0;
    size_t n = strlen(text);
    if (n > max_ids)
        n = max_ids;
    for (size_t i = 0; i < n; i++)
        ids[i] = (uint16_t)(unsigned char)text[i];
    return n;
}

size_t llm_detokenize(const uint16_t *ids, size_t n, char *out, size_t outsz)
{
    if (!out || outsz == 0) return 0;
    size_t w = 0;
    if (ids)
        for (size_t i = 0; i < n && w + 1 < outsz; i++)
            if (ids[i] < 256u)   /* skip BOS/EOS/PAD */
                out[w++] = (char)ids[i];
    out[w] = '\0';
    return w;
}


/*****************************************************************************
 *                                                                           *
 *                          FORWARD PASS                                     *
 *                                                                           *
 *****************************************************************************/

/* Forward on ids[0..n-1] (n <= LLM_MAX_SEQ): fills logits[] and probs[]. */
static void llm_forward(LlmModel *m, const uint16_t *ids, size_t n)
{
    const LlmLayout l = llm_layout();
    const float *P = m->params;

    for (size_t t = 0; t < n; t++)
        for (size_t d = 0; d < D; d++)
            X(m, t, d) = P[l.tok_emb + (size_t)ids[t] * D + d]
                       + P[l.pos_emb + t * D + d];

    for (size_t b = 0; b < LLM_BLOCKS; b++) {
        const float *bp = P + l.block + b * LLM_BLOCK_FLOATS;
        for (size_t t = 0; t < n; t++) {
            for (size_t d = 0; d < D; d++)
                HPRE(m, b, t, d) = X(m, t, d);
            rmsnorm(&X(m, t, 0), bp + B_LN1G, &H1(m, b, t, 0), D);
        }
        for (size_t t = 0; t < n; t++) {
            matvec(&H1(m, b, t, 0), bp + B_WQ, D, D, &Q(m, b, t, 0));
            matvec(&H1(m, b, t, 0), bp + B_WK, D, D, &K(m, b, t, 0));
            matvec(&H1(m, b, t, 0), bp + B_WV, D, D, &V(m, b, t, 0));
        }
        const float scale = 1.0f / sqrtf((float)LLM_D_HEAD);
        for (size_t h = 0; h < LLM_HEADS; h++) {
            for (size_t t = 0; t < n; t++) {
                for (size_t s = 0; s < SEQ; s++)
                    ATT(m, b, h, t, s) = -1e30f;
                for (size_t s = 0; s <= t; s++) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < LLM_D_HEAD; d++)
                        dot += Q(m, b, t, h * LLM_D_HEAD + d)
                             * K(m, b, s, h * LLM_D_HEAD + d);
                    ATT(m, b, h, t, s) = dot * scale;
                }
                softmax(&ATT(m, b, h, t, 0), SEQ);
            }
            for (size_t t = 0; t < n; t++)
                for (size_t d = 0; d < LLM_D_HEAD; d++) {
                    float s = 0.0f;
                    for (size_t u = 0; u <= t; u++)
                        s += ATT(m, b, h, t, u) * V(m, b, u, h * LLM_D_HEAD + d);
                    Z(m, b, t, h * LLM_D_HEAD + d) = s;
                }
        }
        for (size_t t = 0; t < n; t++) {
            matvec(&Z(m, b, t, 0), bp + B_WO, D, D, &AOUT(m, b, t, 0));
            for (size_t d = 0; d < D; d++) {
                X(m, t, d) += AOUT(m, b, t, d);
                HMID(m, b, t, d) = X(m, t, d);
            }
            rmsnorm(&X(m, t, 0), bp + B_LN2G, &H2(m, b, t, 0), D);
            matvec(&H2(m, b, t, 0), bp + B_W1, D, LLM_D_FF, &FF(m, b, t, 0));
            for (size_t f = 0; f < LLM_D_FF; f++) {
                float v = FF(m, b, t, f) + bp[B_B1 + f];
                FF(m, b, t, f) = v > 0.0f ? v : 0.0f;   /* ReLU */
            }
            float ffo[D];
            matvec(&FF(m, b, t, 0), bp + B_W2, LLM_D_FF, D, ffo);
            for (size_t d = 0; d < D; d++)
                X(m, t, d) += ffo[d] + bp[B_B2 + d];
        }
    }

    for (size_t t = 0; t < n; t++) {
        for (size_t d = 0; d < D; d++)
            XFIN(m, t, d) = X(m, t, d);
        rmsnorm(&X(m, t, 0), P + l.lnf_g, &XF(m, t, 0), D);
        for (size_t v = 0; v < LLM_VOCAB; v++) {
            float s = P[l.head_b + v];
            for (size_t d = 0; d < D; d++)
                s += P[l.head_w + v * D + d] * XF(m, t, d);
            LOGIT(m, t, v) = s;
        }
        for (size_t v = 0; v < LLM_VOCAB; v++)
            PROB(m, t, v) = LOGIT(m, t, v);
        softmax(&PROB(m, t, 0), LLM_VOCAB);
    }
}

/* Mean next-token cross-entropy of the window, from probs[]. */
static double window_loss(LlmModel *m, const uint16_t *ids, size_t n)
{
    if (n < 2)
        return 0.0;
    double loss = 0.0;
    for (size_t t = 0; t + 1 < n; t++) {
        float p = PROB(m, t, ids[t + 1]);
        loss -= logf(p > 1e-30f ? p : 1e-30f);
    }
    return loss / (double)(n - 1);
}

double llm_loss(LlmModel *m, const uint16_t *ids, size_t n)
{
    if (!m || !m->initialized || !ids || n < 2) { errno = EINVAL; return -1.0; }
    if (n > LLM_MAX_SEQ) {
        ids += n - LLM_MAX_SEQ;
        n = LLM_MAX_SEQ;
    }
    llm_forward(m, ids, n);
    return window_loss(m, ids, n);
}


/*****************************************************************************
 *                                                                           *
 *                    BACKWARD PASS (manual backprop)                        *
 *                                                                           *
 *****************************************************************************/

/* Backward on the window ids[0..n-1] whose forward filled the scratch.
 * Fills grads[] (same layout as params[]). */
static void llm_backward(LlmModel *m, const uint16_t *ids, size_t n)
{
    const LlmLayout l = llm_layout();
    float *P = m->params;
    float *G = m->grads;

    memset(G, 0, LLM_PARAM_FLOATS * sizeof(float));
    memset(m->scratch + OFF_GX, 0, (SCRATCH_FLOATS - OFF_GX) * sizeof(float));

    /* dlogits = (probs - onehot(target)) / (n-1) for t < n-1. */
    for (size_t t = 0; t + 1 < n; t++)
        for (size_t v = 0; v < LLM_VOCAB; v++)
            DLOGIT(m, t, v) = (PROB(m, t, v)
                - (v == ids[t + 1] ? 1.0f : 0.0f)) / (float)(n - 1);

    /* Head: logits[t] = head_w @ xf[t] + head_b. */
    for (size_t t = 0; t < n; t++)
        for (size_t v = 0; v < LLM_VOCAB; v++) {
            float dl = DLOGIT(m, t, v);
            if (dl == 0.0f)
                continue;
            G[l.head_b + v] += dl;
            for (size_t d = 0; d < D; d++) {
                G[l.head_w + v * D + d] += dl * XF(m, t, d);
                GXF(m, t, d) += dl * P[l.head_w + v * D + d];
            }
        }

    /* Final RMSNorm: xf = RMSNorm(xfin) * lnf_g. */
    for (size_t t = 0; t < n; t++)
        rmsnorm_back(&XFIN(m, t, 0), P + l.lnf_g, &GXF(m, t, 0),
                     D, &GX(m, t, 0), G + l.lnf_g);

    /* Blocks, in reverse.  GX carries dL/dX_out of the current block. */
    for (size_t rb = LLM_BLOCKS; rb-- > 0; ) {
        const float *bp = P + l.block + rb * LLM_BLOCK_FLOATS;
        float *gb = G + l.block + rb * LLM_BLOCK_FLOATS;
        float tmp[D];

        /* Feed-forward: X_out = X_mid + (ReLU(H2 @ W1 + b1)) @ W2 + b2.
         * GX is also dL/dX_mid (residual). */
        for (size_t t = 0; t < n; t++)
            for (size_t d = 0; d < D; d++) {
                gb[B_B2 + d] += GX(m, t, d);
                for (size_t f = 0; f < LLM_D_FF; f++) {
                    gb[B_W2 + f * D + d] += FF(m, rb, t, f) * GX(m, t, d);
                    GFF(m, rb, t, f) += GX(m, t, d) * bp[B_W2 + f * D + d];
                }
            }
        for (size_t t = 0; t < n; t++)
            for (size_t f = 0; f < LLM_D_FF; f++) {
                if (FF(m, rb, t, f) <= 0.0f)   /* ReLU mask */
                    GFF(m, rb, t, f) = 0.0f;
                gb[B_B1 + f] += GFF(m, rb, t, f);
                for (size_t d = 0; d < D; d++) {
                    gb[B_W1 + d * LLM_D_FF + f] += H2(m, rb, t, d) * GFF(m, rb, t, f);
                    GH2(m, rb, t, d) += GFF(m, rb, t, f) * bp[B_W1 + d * LLM_D_FF + f];
                }
            }
        /* ln2: H2 = RMSNorm(X_mid) * ln2g.
         * dL/dX_mid = GX (residual) + d(ln2), collected in GAOUT. */
        for (size_t t = 0; t < n; t++) {
            rmsnorm_back(&HMID(m, rb, t, 0), bp + B_LN2G, &GH2(m, rb, t, 0),
                         D, tmp, gb + B_LN2G);
            for (size_t d = 0; d < D; d++)
                GAOUT(m, rb, t, d) = GX(m, t, d) + tmp[d];
        }
        /* Attention output: AOUT = Z @ Wo ; X_mid = X_pre + AOUT. */
        for (size_t t = 0; t < n; t++)
            for (size_t i = 0; i < D; i++)
                for (size_t o = 0; o < D; o++) {
                    gb[B_WO + i * D + o] += Z(m, rb, t, i) * GAOUT(m, rb, t, o);
                    GZ(m, rb, t, i) += GAOUT(m, rb, t, o) * bp[B_WO + i * D + o];
                }
        const float scale = 1.0f / sqrtf((float)LLM_D_HEAD);
        for (size_t h = 0; h < LLM_HEADS; h++) {
            for (size_t t = 0; t < n; t++)
                for (size_t s = 0; s <= t; s++) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < LLM_D_HEAD; d++)
                        dot += GZ(m, rb, t, h * LLM_D_HEAD + d)
                             * V(m, rb, s, h * LLM_D_HEAD + d);
                    GATT(m, rb, h, t, s) = dot;
                    for (size_t d = 0; d < LLM_D_HEAD; d++)
                        GV(m, rb, s, h * LLM_D_HEAD + d) +=
                            ATT(m, rb, h, t, s) * GZ(m, rb, t, h * LLM_D_HEAD + d);
                }
            /* Softmax backward, per row t. */
            for (size_t t = 0; t < n; t++) {
                float acc = 0.0f;
                for (size_t s = 0; s <= t; s++)
                    acc += ATT(m, rb, h, t, s) * GATT(m, rb, h, t, s);
                for (size_t s = 0; s <= t; s++) {
                    float gs = ATT(m, rb, h, t, s) * (GATT(m, rb, h, t, s) - acc);
                    for (size_t d = 0; d < LLM_D_HEAD; d++) {
                        GQ(m, rb, t, h * LLM_D_HEAD + d) +=
                            gs * K(m, rb, s, h * LLM_D_HEAD + d) * scale;
                        GK(m, rb, s, h * LLM_D_HEAD + d) +=
                            gs * Q(m, rb, t, h * LLM_D_HEAD + d) * scale;
                    }
                }
            }
        }
        /* Q/K/V projections from H1. */
        for (size_t t = 0; t < n; t++)
            for (size_t i = 0; i < D; i++)
                for (size_t o = 0; o < D; o++) {
                    gb[B_WQ + i * D + o] += H1(m, rb, t, i) * GQ(m, rb, t, o);
                    gb[B_WK + i * D + o] += H1(m, rb, t, i) * GK(m, rb, t, o);
                    gb[B_WV + i * D + o] += H1(m, rb, t, i) * GV(m, rb, t, o);
                }
        for (size_t t = 0; t < n; t++) {
            float gq[D], gk[D], gv[D];
            matvec_T(&GQ(m, rb, t, 0), bp + B_WQ, D, D, gq);
            matvec_T(&GK(m, rb, t, 0), bp + B_WK, D, D, gk);
            matvec_T(&GV(m, rb, t, 0), bp + B_WV, D, D, gv);
            for (size_t d = 0; d < D; d++)
                GH1(m, rb, t, d) = gq[d] + gk[d] + gv[d];
        }
        /* ln1: H1 = RMSNorm(X_pre) * ln1g.
         * dL/dX_pre = GAOUT (attention residual) + d(ln1); it becomes the
         * GX of the previous block, or of the embeddings after the last. */
        for (size_t t = 0; t < n; t++) {
            rmsnorm_back(&HPRE(m, rb, t, 0), bp + B_LN1G, &GH1(m, rb, t, 0),
                         D, tmp, gb + B_LN1G);
            for (size_t d = 0; d < D; d++)
                GX(m, t, d) = GAOUT(m, rb, t, d) + tmp[d];
        }
    }

    /* Embeddings. */
    for (size_t t = 0; t < n; t++)
        for (size_t d = 0; d < D; d++) {
            G[l.tok_emb + (size_t)ids[t] * D + d] += GX(m, t, d);
            G[l.pos_emb + t * D + d] += GX(m, t, d);
        }
}


/*****************************************************************************
 *                                                                           *
 *                    TRAINING (SGD) / GENERATION                            *
 *                                                                           *
 *****************************************************************************/

double llm_train(LlmModel *m, const uint16_t *ids, size_t n,
                 size_t steps, double lr)
{
    if (!m || !m->initialized || !ids || n < 2 || steps == 0) {
        errno = EINVAL;
        return -1.0;
    }
    double last = -1.0;
    for (size_t s = 0; s < steps; s++) {
        size_t max_off = n > LLM_MAX_SEQ ? n - LLM_MAX_SEQ : 0;
        size_t off = max_off ? (s * 13u) % max_off : 0;
        size_t len = n - off;
        if (len > LLM_MAX_SEQ)
            len = LLM_MAX_SEQ;
        llm_forward(m, ids + off, len);
        last = window_loss(m, ids + off, len);
        llm_backward(m, ids + off, len);
        for (size_t i = 0; i < LLM_PARAM_FLOATS; i++)
            m->params[i] -= (float)lr * m->grads[i];
    }
    return last;
}

size_t llm_generate(LlmModel *m, const char *prompt, char *out,
                    size_t outsz, size_t max_new_tokens,
                    double temperature, uint32_t seed)
{
    if (!m || !m->initialized || !out || outsz == 0) {
        errno = EINVAL;
        return 0;
    }
    uint16_t ctx[LLM_MAX_SEQ];
    size_t n = prompt ? llm_tokenize(prompt, ctx, LLM_MAX_SEQ) : 0;
    if (n == 0)
        ctx[n++] = LLM_TOK_BOS;
    uint32_t rng = seed ? seed : 1u;
    size_t written = 0;

    for (size_t step = 0; step < max_new_tokens; step++) {
        llm_forward(m, ctx, n);
        /* Sample from the last position; BOS/PAD are never generated. */
        float *row = &PROB(m, n - 1, 0);
        uint16_t next;
        if (temperature <= 0.0) {
            float best = -1.0f;
            next = LLM_TOK_EOS;
            for (size_t v = 0; v < 256u; v++)   /* bytes only */
                if (row[v] > best) {
                    best = row[v];
                    next = (uint16_t)v;
                }
        } else {
            float scaled[LLM_VOCAB];
            for (size_t v = 0; v < LLM_VOCAB; v++)
                scaled[v] = LOGIT(m, n - 1, v) / (float)temperature;
            softmax(scaled, LLM_VOCAB);
            float r = prng_float(&rng), acc = 0.0f;
            next = LLM_TOK_EOS;
            for (size_t v = 0; v < 256u; v++) {
                acc += scaled[v];
                if (r < acc) {
                    next = (uint16_t)v;
                    break;
                }
            }
        }
        if (next == LLM_TOK_EOS)
            break;
        if (n == LLM_MAX_SEQ) {   /* sliding window */
            memmove(ctx, ctx + 1, (LLM_MAX_SEQ - 1) * sizeof *ctx);
            n--;
        }
        ctx[n++] = next;
        if (written + 1 < outsz)
            out[written++] = (char)(next & 0xFFu);
    }
    out[written] = '\0';
    return written;
}

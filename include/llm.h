/*
 * llm.h - Generative Language Model (mini transformer) for the LLM Translator
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

/** @file llm.h
 *  @brief Character-level generative language model (transformer)
 *
 *  A small but real autoregressive language model: byte-level
 *  tokenizer, token + position embeddings, a causal multi-head
 *  self-attention transformer, a vocabulary head, and top-k-free
 *  sampling (argmax or temperature).  Training is a full manual
 *  backpropagation of the cross-entropy loss with SGD, so the model
 *  really learns the corpus instead of echoing it.
 *
 *  Pure C23/POSIX + libm: no Hurd dependency, the suite runs on any
 *  system, exactly like neuron.c.
 */

#ifndef LLM_H
#define LLM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/*****************************************************************************
 *                                                                           *
 *                         CONSTANT DEFINITIONS                             *
 *                                                                           *
 *****************************************************************************/

/* --- Vocabulary: one id per byte, plus three control tokens ------ */
#define LLM_TOK_BOS   256u   /**< beginning of sequence */
#define LLM_TOK_EOS   257u   /**< end of sequence (stop token) */
#define LLM_TOK_PAD   258u   /**< padding (never generated) */
#define LLM_VOCAB     259u

/* --- Architecture (kept small: this runs on a Hurd translator) ---- */
#define LLM_D_MODEL   32u    /**< residual stream width */
#define LLM_HEADS     4u     /**< attention heads (D_MODEL % HEADS == 0) */
#define LLM_D_HEAD    (LLM_D_MODEL / LLM_HEADS)
#define LLM_D_FF      64u    /**< feed-forward hidden width */
#define LLM_BLOCKS    2u     /**< transformer blocks */
#define LLM_MAX_SEQ   64u    /**< maximum context length, in tokens */

/* --- Model file format -------------------------------------------- */
#define LLM_FILE_MAGIC    0x4C4C4D31u   /* 'L' 'L' 'M' '1' */
#define LLM_FILE_VERSION  1u

/** Per-block parameter count, in floats: ln1 gamma, Wq, Wk, Wv, Wo,
 *  ln2 gamma, W1, b1, W2, b2. */
#define LLM_BLOCK_FLOATS (LLM_D_MODEL + 4u * LLM_D_MODEL * LLM_D_MODEL \
    + LLM_D_MODEL + LLM_D_MODEL * LLM_D_FF + LLM_D_FF \
    + LLM_D_FF * LLM_D_MODEL + LLM_D_MODEL)

/** Total parameter count, in floats. */
#define LLM_PARAM_FLOATS (LLM_VOCAB * LLM_D_MODEL \
    + LLM_MAX_SEQ * LLM_D_MODEL + LLM_BLOCKS * LLM_BLOCK_FLOATS \
    + LLM_D_MODEL + LLM_VOCAB * LLM_D_MODEL + LLM_VOCAB)


/*****************************************************************************
 *                                                                           *
 *                          DATA STRUCTURES                                 *
 *                                                                           *
 *****************************************************************************/

/** Generative language model.  All parameters and all forward/backward
 *  scratch live in one contiguous memory block (same design as
 *  CompactNeuralNetwork): [params][grads][scratch]. */
typedef struct LlmModel {
    float *params;        /**< LLM_PARAM_FLOATS parameters */
    float *grads;         /**< same layout, gradients */
    float *scratch;       /**< forward/backward activations */
    void  *mem;           /**< the single allocated block */
    size_t mem_size;
    bool   initialized;
} LlmModel;

/* Parameter offsets inside params[], computed by llm_layout(). */
typedef struct LlmLayout {
    size_t tok_emb;       /* [VOCAB][D] */
    size_t pos_emb;       /* [MAX_SEQ][D] */
    size_t block;         /* [BLOCKS][LLM_BLOCK_FLOATS] */
    size_t lnf_g;         /* [D] final norm gain */
    size_t head_w;        /* [VOCAB][D] */
    size_t head_b;        /* [VOCAB] */
} LlmLayout;


/*****************************************************************************
 *                                                                           *
 *                      FUNCTION DECLARATIONS                              *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Initialize a model with deterministic pseudo-random weights
 *
 * Zero-initialize the object before first use.  Reinitialization is
 * transactional: on failure the previous model is preserved.
 * Weights are seeded exactly like network_init() (seed = i * Knuth's
 * constant, range [-0.2, 0.2]); norm gains are 1, biases 0.
 *
 * @param m Pointer to model structure
 * @return  0 on success, -1 on error (errno set)
 */
int llm_init(LlmModel *m);

/** @brief Free all model resources */
void llm_free(LlmModel *m);

/** @brief Parameter layout (offsets into params[]) */
LlmLayout llm_layout(void);

/**
 * @brief Tokenize text: one id per byte (UTF-8 passes through as bytes)
 * @param text    NUL-terminated text
 * @param ids     output array of at least max_ids bytes
 * @param max_ids capacity of ids
 * @return number of ids written (truncated to max_ids)
 */
size_t llm_tokenize(const char *text, uint16_t *ids, size_t max_ids);

/**
 * @brief Detokenize: ids back to bytes; BOS/EOS/PAD are skipped
 * @return number of bytes written (out is always NUL-terminated)
 */
size_t llm_detokenize(const uint16_t *ids, size_t n, char *out, size_t outsz);

/**
 * @brief Mean cross-entropy loss of the next-token prediction over the
 *        last min(n, LLM_MAX_SEQ) tokens
 * @return loss in nats, or -1 on error
 */
double llm_loss(LlmModel *m, const uint16_t *ids, size_t n);

/**
 * @brief Autoregressive generation from a prompt
 *
 * Runs the model token by token: forward pass on the current context,
 * sample the next token from the last position (argmax when
 * temperature <= 0, else softmax(logits / temperature) drawn with a
 * deterministic xorshift PRNG), append it, repeat until EOS,
 * max_new_tokens, or the output buffer is full.  When the context
 * reaches LLM_MAX_SEQ the oldest tokens are dropped (sliding window).
 *
 * @param m              initialized model
 * @param prompt         NUL-terminated prompt (may be empty: BOS only)
 * @param out            output buffer, always NUL-terminated
 * @param outsz          capacity of out
 * @param max_new_tokens maximum number of tokens to generate
 * @param temperature    0 = argmax (deterministic), > 0 = sampling
 * @param seed           PRNG seed for sampling (ignored when temp <= 0)
 * @return number of bytes written to out (excluding the NUL)
 */
size_t llm_generate(LlmModel *m, const char *prompt, char *out,
                    size_t outsz, size_t max_new_tokens,
                    double temperature, uint32_t seed);

/**
 * @brief Train the model on a tokenized corpus with full backprop + SGD
 *
 * Each step draws a deterministic window of the corpus
 * (offset = step * 13 modulo the corpus length), runs forward +
 * backward on it and applies one SGD update (params -= lr * grads).
 *
 * @param m     initialized model
 * @param ids   tokenized corpus
 * @param n     corpus length in tokens (must be >= 2)
 * @param steps number of SGD steps
 * @param lr    learning rate
 * @return loss of the last window before its update, or -1 on error
 */
double llm_train(LlmModel *m, const uint16_t *ids, size_t n,
                 size_t steps, double lr);

/**
 * @brief Save model parameters to a file (.llm format, magic + version)
 * @return 0 on success, -1 on error (errno set)
 */
int llm_save(const LlmModel *m, const char *path);

/**
 * @brief Load model parameters from a .llm file; unknown magic/version
 *        is rejected with errno EINVAL and the model is left untouched
 * @return 0 on success, -1 on error (errno set)
 */
int llm_load(LlmModel *m, const char *path);

#endif /* LLM_H */

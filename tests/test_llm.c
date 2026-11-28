/* SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 Claire Ivanenka <claire@gnu-ai.org> */

/*
 * test_llm.c — conformance tests of the generative LLM core (llm.c).
 *
 * The .llm format and the autoregressive contract are pinned down:
 *
 *   - byte tokenizer round-trips UTF-8 as bytes, skips control tokens;
 *   - init is deterministic (same seeding rule as neuron.c);
 *   - the manual backprop matches a numerical gradient check on a
 *     spread of parameters (embeddings, attention, FFN, head, norms);
 *   - SGD training really lowers the next-token loss and the model
 *     then completes a learned periodic pattern (argmax generation);
 *   - generation is deterministic for a fixed seed/temperature;
 *   - save / load round-trips the weights, and a forged version is
 *     rejected with errno EINVAL.
 *
 * Pure C23/POSIX + libm: no Hurd, no trivfs — runs on any system.
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm.h"

static int failures = 0;

static void check (const char *name, int ok)
{
    printf ("%s: %s\n", name, ok ? "OK" : "FAIL");
    if (!ok)
        failures++;
}

/* Numerical gradient of llm_loss at params[i], by central difference. */
static double num_grad (LlmModel *m, const uint16_t *ids, size_t n,
                        size_t i, double eps)
{
    double save = m->params[i];
    m->params[i] = (float) (save + eps);
    double l1 = llm_loss (m, ids, n);
    m->params[i] = (float) (save - eps);
    double l2 = llm_loss (m, ids, n);
    m->params[i] = (float) save;
    return (l1 - l2) / (2.0 * eps);
}

int main (void)
{
    /* --- Tokenizer round-trip (UTF-8 passes as bytes) --------- */
    {
        const char *text = "r\xc3\xa9sum\xc3\xa9 abc";  /* "résumé abc" */
        uint16_t ids[64];
        size_t n = llm_tokenize (text, ids, 64);
        char back[64];
        size_t w = llm_detokenize (ids, n, back, sizeof back);
        check ("tokenize length", n == strlen (text));
        check ("tokenize/detokenize round-trip",
               w == n && memcmp (back, text, n) == 0);
        uint16_t ctrl[3] = { LLM_TOK_BOS, 'a', LLM_TOK_EOS };
        char out[8];
        llm_detokenize (ctrl, 3, out, sizeof out);
        check ("control tokens skipped", strcmp (out, "a") == 0);
        check ("tokenize truncates to max_ids",
               llm_tokenize (text, ids, 4) == 4);
    }

    /* --- Deterministic init ------------------------------------ */
    LlmModel a = {0}, b = {0};
    if (llm_init (&a) != 0 || llm_init (&b) != 0)
        {
            perror ("llm_init");
            return 2;
        }
    check ("deterministic init",
           memcmp (a.params, b.params,
                   LLM_PARAM_FLOATS * sizeof (float)) == 0);

    /* --- Forward is deterministic, loss is finite -------------- */
    const char *corpus = "abcdefghabcdefghabcdefghabcdefghabcdefgh"
                         "abcdefghabcdefghabcdefghabcdefghabcdefgh";
    uint16_t ids[128];
    size_t n = llm_tokenize (corpus, ids, 128);
    double l1 = llm_loss (&a, ids, n);
    double l2 = llm_loss (&a, ids, n);
    check ("loss finite", l1 > 0.0 && l1 < 100.0);
    check ("forward deterministic", l1 == l2);

    /* --- Gradient check: manual backprop vs central difference -- */
    {
        LlmLayout l = llm_layout ();
        /* llm_backward is static: one train step with lr 0 fills grads
         * (backward runs) while leaving the parameters untouched. */
        LlmModel g = {0};
        if (llm_init (&g) != 0)
            return 2;
        memcpy (g.params, a.params, LLM_PARAM_FLOATS * sizeof (float));
        double last = llm_train (&g, ids, n, 1, 0.0);
        check ("zero-lr train returns window loss", last > 0.0);
        /* grads of g are the analytic gradients of the last window;
         * probe a spread across every parameter group */
        size_t idx[8];
        idx[0] = l.tok_emb + 3 * LLM_D_MODEL + 1;
        idx[1] = l.pos_emb + 4 * LLM_D_MODEL + 2;
        idx[2] = l.block + 0 * LLM_BLOCK_FLOATS + 10;             /* ln1g */
        idx[3] = l.block + 0 * LLM_BLOCK_FLOATS + 40 + 5 * LLM_D_MODEL + 6; /* Wq */
        idx[4] = l.block + 1 * LLM_BLOCK_FLOATS + 4000;            /* W2 area */
        idx[5] = l.lnf_g + 9;
        idx[6] = l.head_w + 100 * LLM_D_MODEL + 11;
        idx[7] = l.head_b + 42;
        int ok = 1;
        for (int k = 0; k < 8; k++)
          {
            double gn = num_grad (&g, ids, n, idx[k], 1e-3);
            double ga = g.grads[idx[k]];
            double denom = fabs (gn) + fabs (ga) + 1e-6;
            if (fabs (gn - ga) / denom > 0.10)
              {
                printf ("  grad mismatch at %zu: num=%.6f analytic=%.6f\n",
                        idx[k], gn, ga);
                ok = 0;
              }
          }
        check ("backprop matches numerical gradient", ok);
        llm_free (&g);
    }

    /* --- Training lowers the loss and teaches the pattern ------ */
    {
        double loss0 = llm_loss (&a, ids, n);
        llm_train (&a, ids, n, 400, 0.05);
        double loss1 = llm_loss (&a, ids, n);
        printf ("  loss before=%.4f after=%.4f\n", loss0, loss1);
        check ("training lowers the loss", loss1 < loss0 * 0.7);

        char out[64];
        size_t w = llm_generate (&a, "abc", out, sizeof out, 8, 0.0, 1);
        check ("argmax generation continues the pattern",
               w >= 1 && out[0] == 'd');
    }

    /* --- Sampling is deterministic for a fixed seed ----------- */
    {
        char o1[64], o2[64];
        llm_generate (&a, "ab", o1, sizeof o1, 12, 0.8, 42);
        llm_generate (&a, "ab", o2, sizeof o2, 12, 0.8, 42);
        check ("sampled generation deterministic per seed",
               strcmp (o1, o2) == 0);
    }

    /* --- save / load round-trip -------------------------------- */
    {
        const char *path = "test-llm.llm";
        check ("llm save", llm_save (&a, path) == 0);
        LlmModel c = {0};
        check ("llm load", llm_load (&c, path) == 0);
        check ("weights round-trip",
               memcmp (a.params, c.params,
                       LLM_PARAM_FLOATS * sizeof (float)) == 0);
        char o1[64], o2[64];
        llm_generate (&a, "ab", o1, sizeof o1, 12, 0.0, 1);
        llm_generate (&c, "ab", o2, sizeof o2, 12, 0.0, 1);
        check ("generation identical after load", strcmp (o1, o2) == 0);

        /* forged version must be rejected, errno EINVAL */
        FILE *fp = fopen (path, "r+b");
        if (!fp)
            return 2;
        uint32_t forged = 99u;
        fseek (fp, 4, SEEK_SET);
        fwrite (&forged, sizeof forged, 1, fp);
        fclose (fp);
        LlmModel d = {0};
        errno = 0;
        check ("forged version rejected",
               llm_load (&d, path) != 0 && errno == EINVAL);
        llm_free (&c);
        llm_free (&d);
        remove (path);
    }

    llm_free (&a);
    llm_free (&b);

    printf ("%s\n", failures ? "SUITE FAIL" : "SUITE OK");
    return failures;
}

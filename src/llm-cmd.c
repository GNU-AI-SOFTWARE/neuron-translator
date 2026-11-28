/*
 * llm-cmd.c - Node-level commands driving the generative LLM
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

#include "llm-cmd.h"
#include "llm.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One generative model per node, lazily initialized. */
static LlmModel g_llm;
static int g_llm_ready;
static char g_last[4096];

static int ensure_llm(void)
{
    if (g_llm_ready)
        return 0;
    if (llm_init(&g_llm) != 0)
        return errno ? errno : ENOMEM;
    g_llm_ready = 1;
    return 0;
}

const char *llm_last_output(void)
{
    return g_last;
}

/* Read a whole text file (bounded), NUL-terminated. */
static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || sz > 4 * 1024 * 1024) { fclose(f); errno = EMSGSIZE; return NULL; }
    rewind(f);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        errno = EIO;
        return NULL;
    }
    fclose(f);
    buf[sz] = '\0';
    *len = (size_t)sz;
    return buf;
}

static char *status_line(const char *format, ...)
{
    char *buf = malloc(512);
    if (!buf)
        return NULL;
    va_list args;
    va_start(args, format);
    vsnprintf(buf, 512, format, args);
    va_end(args);
    return buf;
}

int llm_node_command(const char *cmd, char **out, size_t *outlen)
{
    if (!cmd || !out || !outlen)
        return EINVAL;
    *out = NULL;
    *outlen = 0;

    if (!strncmp(cmd, "gen ", 4)) {
        const char *prompt = cmd + 4;
        int e = ensure_llm();
        if (e)
            return e;
        size_t w = llm_generate(&g_llm, prompt, g_last, sizeof g_last,
                                64, 0.8, 1);
        *out = status_line("gen: %zu token(s) generated", w);
        *outlen = *out ? strlen(*out) : 0;
        return *out ? 0 : ENOMEM;
    }
    if (!strncmp(cmd, "train ", 6)) {
        const char *path = cmd + 6;
        size_t len = 0;
        char *text = read_file(path, &len);
        if (!text)
            return errno ? errno : EIO;
        int e = ensure_llm();
        if (e) { free(text); return e; }
        uint16_t *ids = malloc((len + 1) * sizeof *ids);
        if (!ids) { free(text); return ENOMEM; }
        size_t n = llm_tokenize(text, ids, len);
        free(text);
        if (n < 2) { free(ids); return EINVAL; }
        double before = llm_loss(&g_llm, ids, n);
        llm_train(&g_llm, ids, n, 500, 0.05);
        double after = llm_loss(&g_llm, ids, n);
        free(ids);
        *out = status_line("train %s: %zu tokens, loss %.4f -> %.4f",
                           path, n, before, after);
        *outlen = *out ? strlen(*out) : 0;
        return *out ? 0 : ENOMEM;
    }
    if (!strncmp(cmd, "llm save ", 9) || !strncmp(cmd, "llm load ", 9)) {
        int saving = cmd[4] == 's';
        const char *path = cmd + 9;
        int e = ensure_llm();
        if (e)
            return e;
        int ok = saving ? llm_save(&g_llm, path) : llm_load(&g_llm, path);
        if (ok != 0)
            return errno ? errno : EINVAL;
        *out = status_line("llm %s %s: ok", saving ? "save" : "load", path);
        *outlen = *out ? strlen(*out) : 0;
        return *out ? 0 : ENOMEM;
    }
    return EINVAL;
}

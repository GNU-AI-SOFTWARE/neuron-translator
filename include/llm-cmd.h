/*
 * llm-cmd.h - Node-level commands driving the generative LLM
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

#ifndef LLM_CMD_H
#define LLM_CMD_H

#include <stddef.h>

/* Commands understood on the /llm node, routed by network_command():
 *   gen <prompt>        autoregressive generation from <prompt>
 *   train <path>        SGD training on the tokenized text file <path>
 *   llm save <path>     save the generative model (.llm format)
 *   llm load <path>     load a .llm model
 * llm_node_command() answers with a short status line in *out (malloc'd,
 * caller frees) and returns 0 on success, an errno-style code otherwise. */
int llm_node_command(const char *cmd, char **out, size_t *outlen);

/* Last generated text (for network_status), "" when none. */
const char *llm_last_output(void);

#endif /* LLM_CMD_H */

/*
 * main.c - Main Entry Point for Sigmoid Neuron Translator
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

/** @file main.c
 *  @brief Main entry point and initialization for the translator
 *
 *  This file contains the main() function that initializes the translator
 *  and starts the Hurd trivfs server loop. It follows GNU Hurd translator
 *  conventions and provides the entry point for the system.
 *
 *  IMPORTANT: On GNU/Hurd, the main() function is NEVER called for filesystem
 *  translators. The entry point is through the trivfs interface (trivfs_demuxer).
 *  This main() function is only for non-Hurd systems (Debian/Linux) for testing
 *  and compilation verification.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>  /* For errno */

#include "neuron.h"
#include "llm.h"
#include "trivfs-hooks.h"  /* Includes all necessary Hurd/Mach declarations */
#include "debug.h"

/* Default ON_HURD to 0 if not defined by Makefile */
#ifndef ON_HURD
#define ON_HURD 0
#endif

/*****************************************************************************
 *                                                                           *
 *                    THE GNU BASE COMMANDS                                  *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Print the standard GNU --version answer
 *
 * The version number comes from configure (config.h), the single
 * source of truth of the release; the layout follows the GNU
 * coding standards so scripts can grep it.
 */
static void print_version(void)
{
    printf("sigmoid-neuron-translator (GNU AI) %s\n", VERSION);
    printf("License GPLv3+: GNU GPL version 3 or later"
           " <https://gnu.org/licenses/gpl.html>.\n");
    printf("This is free software: you are free to change"
           " and redistribute it.\n");
    printf("There is NO WARRANTY, to the extent permitted by law.\n");
}

/**
 * @brief Print the standard GNU --help answer
 *
 * The usage documents both faces of the translator: the mounted
 * POSIX interface (the Hurd way to drive the network) and the
 * verification build of non-Hurd systems.
 */
static void print_help(void)
{
    printf("Usage: sigmoid-neuron-translator [OPTION]...\n");
    printf("Feedforward sigmoid network as a GNU/Hurd translator.\n");
    printf("\n");
    printf("  -h, --help     display this help and exit\n");
    printf("  -V, --version  output version information and exit\n");
    printf("\n");
    printf("Generative LLM mode (portable verification face):\n");
    printf("  --gen PROMPT     autoregressive generation from PROMPT\n");
    printf("  --train FILE     SGD training on the text file FILE first\n");
    printf("  --steps N        training steps (default 500)\n");
    printf("  --tokens N       tokens to generate (default 48)\n");
    printf("  --temp X         sampling temperature, 0 = argmax (default 0.8)\n");
    printf("  --seed N         PRNG seed for sampling (default 1)\n");
    printf("\n");
    printf("On GNU/Hurd, mount and drive it through the filesystem:\n");
    printf("  sudo settrans -c /llm /hurd/sigmoid-neuron-translator\n");
    printf("  cat /llm\n");
    printf("\n");
    printf("Report bugs at <https://github.com/gnu-ai/neuron-translator/issues>.\n");
}

/**
 * @brief Answer the GNU base commands --version and --help
 *
 * Every binary of the GNU AI stack answers the two base commands
 * of the GNU toolbox before anything else, so a translator stays
 * inspectable like any other GNU tool.  On GNU/Hurd, main-hurd.c
 * gets the same behavior from argp.
 *
 * @param argc  argument count
 * @param argv  argument vector
 * @return true when one of the commands was answered (exit 0),
 *         false when the normal startup should proceed
 */
static bool handle_gnu_options(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-V") == 0) {
            print_version();
            return true;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help();
            return true;
        }
    }
    return false;
}

/**
 * @brief Generative LLM mode: --gen PROMPT [--train FILE] [options]
 *
 * On any host this drives the mini transformer of llm.c directly:
 * optional SGD training on a text file, then autoregressive sampling
 * from the prompt, printed on stdout.  This is the portable face of
 * the same model the /llm node serves on GNU/Hurd (gen/train/llm
 * commands, see llm-cmd.c).
 *
 * @return true when --gen was present and handled (exit 0)
 */
static bool handle_llm_options(int argc, char *argv[])
{
    const char *gen = NULL, *train = NULL;
    long steps = 500, tokens = 48;
    double temp = 0.8;
    unsigned long seed = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc)
            gen = argv[++i];
        else if (strcmp(argv[i], "--train") == 0 && i + 1 < argc)
            train = argv[++i];
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            steps = strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            tokens = strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
            temp = strtod(argv[++i], NULL);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = strtoul(argv[++i], NULL, 10);
    }
    if (!gen)
        return false;

    LlmModel model;
    memset(&model, 0, sizeof model);
    if (llm_init(&model) != 0) {
        fprintf(stderr, "sigmoid-neuron-translator: llm_init: %s\n",
                strerror(errno));
        return true;
    }
    if (train) {
        FILE *f = fopen(train, "rb");
        if (!f) {
            fprintf(stderr, "sigmoid-neuron-translator: %s: %s\n",
                    train, strerror(errno));
            llm_free(&model);
            return true;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        if (sz < 2 || sz > 4 * 1024 * 1024) {
            fprintf(stderr, "sigmoid-neuron-translator: %s: bad corpus size\n",
                    train);
            fclose(f);
            llm_free(&model);
            return true;
        }
        char *text = malloc((size_t)sz + 1);
        uint16_t *ids = malloc((size_t)sz * sizeof *ids);
        if (!text || !ids
            || fread(text, 1, (size_t)sz, f) != (size_t)sz) {
            fprintf(stderr, "sigmoid-neuron-translator: %s: read error\n", train);
            free(text);
            free(ids);
            fclose(f);
            llm_free(&model);
            return true;
        }
        fclose(f);
        text[sz] = '\0';
        size_t n = llm_tokenize(text, ids, (size_t)sz);
        double before = llm_loss(&model, ids, n);
        llm_train(&model, ids, n, (size_t)steps, 0.05);
        double after = llm_loss(&model, ids, n);
        fprintf(stderr, "train %s: %zu tokens, loss %.4f -> %.4f\n",
                train, n, before, after);
        free(text);
        free(ids);
    }
    char out[4096];
    llm_generate(&model, gen, out, sizeof out, (size_t)tokens, temp,
                 (uint32_t)seed);
    printf("%s\n", out);
    llm_free(&model);
    return true;
}

/*****************************************************************************
 *                                                                           *
 *                      MAIN ENTRY POINT                                    *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Main entry point for the sigmoid neuron translator
 * 
 * On non-Hurd systems (like Debian/Linux), this initializes and tests the translator.
 * On GNU/Hurd, this function should NEVER be called for filesystem translators.
 * The actual entry point on Hurd is trivfs_demuxer() called by the Hurd filesystem system.
 * 
 * @return Exit code (0 on success, non-zero on error)
 */
int main(int argc, char *argv[])
{
    /* The GNU base commands --version and --help are answered
     * before anything else; the translator stays inspectable
     * like any other GNU tool. */
    if (handle_gnu_options(argc, argv))
        return EXIT_SUCCESS;

    /* Generative LLM mode answers --gen before the translator startup. */
    if (handle_llm_options(argc, argv))
        return EXIT_SUCCESS;

    /* On GNU/Hurd, when loaded as a translator by settrans, main() is called
     * and should start the trivfs server, which will handle the message loop
     * and dispatch to our fs_* functions.
     */
#if ON_HURD == 1
    /* On Hurd, when loaded as a translator by settrans, main() is NOT actually called.
     * The entry point is handled by libtrivfs via our fs_* functions.
     * If main() IS called (e.g., when running directly), return 0.
     */
    log_debug_message("[DEBUG] main(): Hurd translator - main() should not be called by settrans");
    return 0;
#else
    /* On non-Hurd systems (like Debian/Linux), display helpful message */
#ifdef DEBUG
    log_debug_message("[DEBUG] main(): Starting sigmoid-neuron-translator (non-Hurd system)");
#endif
    
    /* Initialize global network state to zero */
    extern CompactNeuralNetwork global_network;
    memset(&global_network, 0, sizeof(global_network));
    
    /* Set up trivfs control port */
    extern mach_port_t trivfs_control;
    trivfs_control = MACH_PORT_NULL;
    
    /* Initialize network with default topology */
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        /* Failed to initialize - this is fatal for a translator */
        fprintf(stderr, "sigmoid-neuron-translator: failed to initialize network: %s\n",
                strerror(errno));
        network_free(&global_network);
        return EXIT_FAILURE;
    }
    
    /* Start the trivfs server loop - this should not return on Hurd */
    /* On non-Hurd systems, trivfs_server_loop returns -1 (stub implementation) */
    int result = trivfs_server_loop();
    
    /* If trivfs_server_loop returns (which it shouldn't on Hurd),
       it means we're not on a Hurd system */
    if (result != 0) {
        /* Not running on GNU/Hurd - display helpful message */
        printf("\n" 
               "================================================================\n" 
               "  LLM SIGMOID NEURON TRANSLATOR - GNU/Hurd\n" 
               "================================================================\n\n" 
               "  This is a GNU/Hurd filesystem translator.\n" 
               "  It CANNOT run on Debian/Linux - it requires a GNU/Hurd system.\n\n" 
               "  WHAT YOU CAN DO:\n\n" 
               "  On Debian/Linux:\n" 
               "    - Compile only:  make\n" 
               "    - Test C23/POSIX: make\n\n" 
               "  On GNU/Hurd:\n" 
               "    1. Build:        make executable\n" 
               "    2. Install:      sudo make install\n" 
               "    3. Set:          sudo settrans -c /llm /hurd/sigmoid-neuron-translator\n" 
               "    4. Use:          cat /llm\n\n" 
               "  For more info: https://github.com/gnu-ai/neuron-translator\n" 
               "================================================================\n");
        network_free(&global_network);
        return EXIT_FAILURE;
    }
    
    return result;
#endif
}

/*
 * main.c - Main Entry Point for Sigmoid Neuron Translator
 *
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
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
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neuron.h"
#include "trivfs-hooks.h"


/*****************************************************************************
 *                                                                           *
 *                      MAIN ENTRY POINT                                    *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Main entry point for the sigmoid neuron translator
 * 
 * Initializes the global network and starts the trivfs server loop.
 * This function is called by the Hurd system when the translator is loaded.
 * 
 * @return Exit code (should not return for a translator)
 */
int main(void)
{
    /* Initialize global network state to zero */
    memset(&global_network, 0, sizeof(global_network));
    
    /* Set up trivfs control port */
    trivfs_control = MACH_PORT_NULL;
    
    /* Initialize network with default topology */
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        /* Failed to initialize - this is fatal for a translator */
        fprintf(stderr, "sigmoid-neuron-translator: failed to initialize network: %s\n",
                strerror(errno));
        return EXIT_FAILURE;
    }
    
    /* Set translator help text */
    fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
              "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator";
    
    /* Register our filesystem hooks with explicit type casting */
    /* This ensures type compatibility with Hurd's function pointer declarations */
    fs_open = (error_t (*)(struct iouser *, int, mode_t, struct node *, struct iobuf **))fs_open_hook;
    fs_read = (error_t (*)(struct iouser *, struct iobuf *, off_t, size_t *, size_t))fs_read_hook;
    fs_write = (error_t (*)(struct iouser *, struct iobuf *, off_t, size_t, size_t))fs_write_hook;
    
    /* Start the trivfs server loop - this should not return */
    return trivfs_server_loop();
}

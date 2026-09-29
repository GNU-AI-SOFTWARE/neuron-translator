/*
 * trivfs-hooks.c - Hurd Translator Hooks Implementation
 *
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file trivfs-hooks.c
 *  @brief Implementation of Hurd trivfs translator hooks
 *
 *  This file implements the filesystem hooks for the sigmoid neuron translator.
 *  It follows GNU Hurd translator conventions and provides the interface
 *  between the neural network implementation and the Hurd filesystem.
 *
 *  Style: Claude Delannoy - C23 standard, POSIX compliant, educational.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include our headers - trivfs-hooks.h handles Hurd/system includes */
#include "neuron.h"
#include "debug.h"
#include "trivfs-hooks.h"


/*****************************************************************************
 *                                                                           *
 *              FALLBACK DEFINITIONS FOR NON-HURD SYSTEMS                   *
 *                                                                           *
 *  These are only needed when compiling on non-Hurd systems (Linux, etc.) *
 *  On Hurd, these types are provided by system headers.                 *
 *                                                                           *
 *****************************************************************************/

#ifndef ON_HURD

/* Mach types - fallback for non-Hurd systems */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)

/* Forward declarations for Hurd types */
struct iouser;
struct node;

/* I/O buffer structure - used for read/write operations in translators */
struct iobuf {
    char *buf;              /* Pointer to the actual buffer data */
    size_t buf_size;        /* Size of the buffer */
    off_t offset;           /* Current offset within the buffer */
};

/* Server loop function - stub for non-Hurd */
int trivfs_server_loop(void) {
    log_debug_message("[DEBUG] trivfs_server_loop: STUB called - not on Hurd!");
    return -1;
}

/* Server message handler - stub for non-Hurd */
error_t trivfs_server(mach_msg_header_t inmsg, mach_msg_header_t outmsg) {
    (void)inmsg;
    (void)outmsg;
    log_debug_message("[DEBUG] trivfs_server: STUB called - not on Hurd!");
    return -1;
}

#endif /* ON_HURD */


/*****************************************************************************
 *                                                                           *
 *                         GLOBAL NETWORK INSTANCE                          *
 *                                                                           *
 *****************************************************************************/

/** Global network instance shared across all translator operations */
CompactNeuralNetwork global_network = {0};


/*****************************************************************************
 *                                                                           *
 *                    TRANSLATOR GLOBAL VARIABLES                            *
 *                                                                           *
 *  These variables are declared as extern in trivfs-hooks.h and must be   *
 *  defined here for the linker to find them.                               *
 *                                                                           *
 *****************************************************************************/

/* Translator control port */
mach_port_t trivfs_control = MACH_PORT_NULL;

/* Help text for the translator */
char *fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
                "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator";


/*****************************************************************************
 *                                                                           *
 *                    TRIVFS HOOK IMPLEMENTATIONS                         *
 *                                                                           *
 *  These override the default trivfs implementations via weak symbols.    *
 *                                                                           *
 *****************************************************************************/

/**
 * fs_open - Called by trivfs library when node is opened
 * This overrides the default trivfs fs_open with our implementation
 */
error_t fs_open(struct iouser *cred, int flags, mode_t mode,
               struct node *node, struct iobuf **iobuf)
{
    (void)cred;    /* Suppress unused parameter warning */
    (void)flags;   /* Suppress unused parameter warning */
    (void)mode;    /* Suppress unused parameter warning */
    (void)node;    /* Suppress unused parameter warning */
    return fs_open_hook(cred, flags, mode, node, iobuf);
}

/**
 * fs_read - Called by trivfs library when node is read
 * This overrides the default trivfs fs_read with our implementation
 */
error_t fs_read(struct iouser *cred, struct iobuf *iobuf,
               off_t offset, size_t *len, size_t count)
{
    (void)cred;    /* Suppress unused parameter warning */
    (void)offset;  /* Suppress unused parameter warning */
    (void)count;   /* Suppress unused parameter warning */
    return fs_read_hook(cred, iobuf, offset, len, count);
}

/**
 * fs_write - Called by trivfs library when node is written
 * This overrides the default trivfs fs_write with our implementation
 */
error_t fs_write(struct iouser *cred, struct iobuf *iobuf,
                off_t offset, size_t len, size_t count)
{
    (void)cred;    /* Suppress unused parameter warning */
    (void)offset;  /* Suppress unused parameter warning */
    (void)count;   /* Suppress unused parameter warning */
    return fs_write_hook(cred, iobuf, offset, len, count);
}


/*****************************************************************************
 *                                                                           *
 *              INITIALIZATION FOR GNU/HURD TRANSLATOR                      *
 *                                                                           *
 *  On GNU/Hurd, main() is never called for translators. Instead, we use     *
 *  a constructor attribute to run initialization before the first request. *
 *                                                                           *
 *****************************************************************************/

/**
 * translator_init - Constructor function for Hurd translator
 * This runs before main() on non-Hurd systems, and before any IPC
 * message is processed on Hurd. It ensures global variables are initialized.
 */
static void __attribute__((constructor)) translator_init(void)
{
    /* Initialize global network to zero */
    if (global_network.memory_block == NULL) {
        memset(&global_network, 0, sizeof(global_network));
    }
    
    /* Set trivfs control port to null */
    trivfs_control = MACH_PORT_NULL;
    
    /* Debug: Indicate initialization */
    log_debug_message("[DEBUG] Sigmoid Neuron Translator: Constructor ran");
}


/*****************************************************************************
 *                                                                           *
 *                      TRIVFS HOOK IMPLEMENTATIONS                         *
 *                                                                           *
 *  These are our internal hook implementations that do the actual work.   *
 *                                                                           *
 *****************************************************************************/

/**
 * fs_open_hook - Called when translator node is opened
 * 
 * Initializes the network. This is called by the Hurd system when the
 * translator node is opened.
 */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf)
{
    (void)cred;    /* Unused parameter */
    (void)flags;    /* Unused parameter */
    (void)mode;     /* Unused parameter */
    (void)node;     /* Unused parameter */
    
    log_debug_message("[DEBUG] fs_open_hook called");
    
    *iobuf = NULL;  /* No I/O buffer needed for this translator */
    
    /* Initialize network if not already done */
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return EIO;  /* Initialization failed */
        }
    }
    
    return 0;  /* Success */
}


/**
 * fs_read_hook - Called when translator node is read from
 * 
 * Returns network information and current output.
 */
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count)
{
    (void)cred;    /* Unused parameter */
    (void)offset;   /* Unused parameter */
    (void)count;    /* Unused parameter */
    
    {
        char debug_msg[256];
        snprintf(debug_msg, sizeof(debug_msg), "[DEBUG] fs_read_hook called, initialized=%d", global_network.initialized);
        log_debug_message(debug_msg);
    }
    
    /* Network should already be initialized by fs_open_hook */
    if (!global_network.initialized) {
        return EIO;  /* Network not initialized - should not happen */
    }
    
    /* Build output buffer with network information */
    char buffer[4096];
    size_t written = 0;
    int snprintf_result;
    
    /* Title and separator */
    snprintf_result = snprintf(buffer, sizeof(buffer),
                      "LLM Sigmoid Neuron Translator - GNU Hurd\n"
                      "===========================================\n\n");
    if (snprintf_result < 0) return EIO;
    written = (size_t)snprintf_result;
    
    /* Network topology */
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "Network: %d layers", global_network.topology.layer_count);
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    for (uint8_t i = 0; i < global_network.topology.layer_count; i++) {
        snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                          ", %d", global_network.topology.layer_sizes[i]);
        if (snprintf_result < 0) return EIO;
        written += (size_t)snprintf_result;
    }
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written, "\n\n");
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
    /* Memory usage */
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "Memory: %.2f KB, Neurons: %zu, Weights: %zu\n\n",
                       (double)global_network.memory_block_size / 1024.0,
                       global_network.total_neurons,
                       global_network.total_weights);
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
    /* Neuron parameters */
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "Parameters:\n"
                       "  Reset Potential: %.2f mV\n"
                       "  Threshold: %.2f mV\n"
                       "  Leak Rate: %.2f\n"
                       "  Refractory: %d steps\n\n",
                       global_network.topology.reset_potential,
                       global_network.topology.threshold,
                       global_network.topology.leak_rate,
                       global_network.topology.refractory_length);
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
    /* Current output */
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "Output:\n");
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                          "  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
        if (snprintf_result < 0) return EIO;
        written += (size_t)snprintf_result;
    }
    
    /* Statistics */
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "\nStatistics:\n"
                       "  Forward Passes: %zu\n"
                       "  Neuron Activations: %zu\n\n",
                       global_network.forward_pass_count,
                       global_network.neuron_activations);
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
    /* Usage instructions */
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "Usage:\n"
                       "  cat /llm                    - Show info\n"
                       "  echo '10,20,5' > /llm      - Set topology\n"
                       "  echo '0.5,0.3,0.8' > /llm  - Set input\n"
                       "  echo reset > /llm         - Reset network state\n"
                       "  echo 'save /tmp/net.bin' > /llm  - Save network\n"
                       "  echo 'load /tmp/net.bin' > /llm  - Load network\n");
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
    /* Ensure null-termination */
    if (written >= sizeof(buffer)) {
        written = sizeof(buffer) - 1;
    }
    buffer[written] = '\0';
    
    /* If no iobuf provided, just return length */
    if (iobuf == NULL || iobuf->buf == NULL) {
        *len = written;
        return 0;
    }
    
    /* Copy to provided buffer */
    size_t to_copy = (written < *len) ? written : *len;
    if (to_copy > 0 && iobuf->buf != NULL) {
        memcpy(iobuf->buf, buffer, to_copy);
    }
    *len = to_copy;
    
    return 0;  /* Success */
}


/**
 * fs_write_hook - Called when data is written to translator node
 * 
 * Handles configuration commands and input data.
 */
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count)
{
    (void)cred;    /* Unused parameter */
    (void)offset;   /* Unused parameter */
    (void)count;    /* Unused parameter */
    
    /* Validate input */
    if (iobuf == NULL || len == 0 || iobuf->buf == NULL) {
        return EINVAL;
    }
    
    /* Network should already be initialized */
    if (!global_network.initialized) {
        return EIO;  /* Network not initialized - should not happen */
    }
    
    /* Copy input to temporary buffer */
    char temp[1024];
    if (len >= sizeof(temp)) {
        len = sizeof(temp) - 1;
    }
    memcpy(temp, iobuf->buf, len);
    temp[len] = '\0';
    
    /* Remove trailing newlines */
    char *newline = strchr(temp, '\n');
    if (newline) *newline = '\0';
    newline = strchr(temp, '\r');
    if (newline) *newline = '\0';
    
    /* Handle special commands */
    /* Check for "reset" command */
    if (strlen(temp) >= 5 && strncmp(temp, "reset", 5) == 0) {
        if (temp[5] == '\0' || temp[5] == ' ' || temp[5] == '\t') {
            network_reset(&global_network);
            return 0;
        }
    }
    
    /* Check for "save <filename>" command */
    if (strlen(temp) >= 5 && strncmp(temp, "save ", 5) == 0) {
        char *filename = temp + 5;
        if (*filename != '\0') {
            if (network_save(&global_network, filename)) {
                return 0;
            }
        }
        return EINVAL;
    }
    
    /* Check for "load <filename>" command */
    if (strlen(temp) >= 5 && strncmp(temp, "load ", 5) == 0) {
        char *filename = temp + 5;
        if (*filename != '\0') {
            if (network_load(&global_network, filename)) {
                return 0;
            }
        }
        return EINVAL;
    }
    
    /* Try to parse as topology configuration */
    uint16_t layers[MAX_LAYERS];
    int layer_count = parse_config_string(temp, layers, MAX_LAYERS);
    
    if (layer_count > 0) {
        network_free(&global_network);
        if (network_init(&global_network, layer_count, layers) != 0) {
            return EINVAL;
        }
        return 0;
    }
    
    /* Try to parse as input data */
    if (parse_input_string(&global_network, temp)) {
        return 0;
    }
    
    /* Invalid command */
    return EINVAL;
}


/*****************************************************************************
 *                                                                           *
 *                      TRIVFS DEMUXER - ENTRY POINT                        *
 *                                                                           *
 *  On GNU/Hurd, this is the REAL entry point called by the Hurd filesystem.*
 *  On other systems, this is a stub.                                    *
 *                                                                           *
 *****************************************************************************/

/**
 * trivfs_demuxer - Message demultiplexer
 * 
 * This function is the entry point for all Mach IPC messages received by the
 * translator. It demultiplexes messages and dispatches them to the
 * appropriate trivfs server function.
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    /* This is the actual entry point for the translator on GNU/Hurd */
    log_debug_message("[DEBUG] trivfs_demuxer: Called (Hurd entry point)");
    
    (void)inmsg;  /* Suppress unused parameter warning */
    (void)outmsg; /* Suppress unused parameter warning */
    
#ifdef ON_HURD
    /* On Hurd, delegate to the trivfs server message handler from libtrivfs */
    return trivfs_server(*inmsg, *outmsg);
#else
    /* On non-Hurd systems, this should never be called */
    log_debug_message("[DEBUG] trivfs_demuxer: STUB called - not on Hurd!");
    return EOPNOTSUPP;
#endif
}

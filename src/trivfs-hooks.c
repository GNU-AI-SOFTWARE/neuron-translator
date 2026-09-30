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
 *  Style: Claude Delannoy - C23 standard, POSIX compliant, educational.
 *
 *  NOTE: This file provides implementations of the standard trivfs functions
 *  (fs_open, fs_read, fs_write) which libtrivfs will call. These functions
 *  delegate to our custom hook implementations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

/*****************************************************************************
 *  HURD DETECTION
 *****************************************************************************/

/* ON_HURD should be defined by the Makefile via -DON_HURD flag
 * If compiling manually on Hurd, use: -DON_HURD
 */
#ifndef ON_HURD
#define ON_HURD 0
#endif

/*****************************************************************************
 *  BASIC MACH TYPES
 *  On Hurd: Use system headers
 *  On non-Hurd: Provide our own definitions
 *****************************************************************************/

#if ON_HURD == 1
/* Try to include Hurd headers - if they don't exist, we'll use our own definitions */
#if __has_include(<mach.h>)
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <hurd.h>
#else
/* Hurd headers not available, use our own definitions */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)
struct mach_msg_header;
typedef struct mach_msg_header mach_msg_header_t;
typedef int error_t;
#endif
#else
/* Non-Hurd systems */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)
struct mach_msg_header;
typedef struct mach_msg_header mach_msg_header_t;
typedef int error_t;
#endif


/*****************************************************************************
 *  COMPLETE HURD FILESYSTEM TYPE DEFINITIONS
 *  We provide these ourselves because Hurd headers only forward-declare
 *  struct node and struct iobuf, and we need to access iobuf->buf.
 *  These match the internal libtrivfs type definitions.
 *****************************************************************************/

/* Complete definitions for all Hurd filesystem structures */
struct iouser {
    int uid;
    int gid;
    int *uids;
    int *gids;
    int nuids;
    int ngids;
};

struct node {
    void *data;
};

struct iobuf {
    char *buf;
    size_t size;
    off_t offset;
};


/*****************************************************************************
 *  INCLUDE PROJECT HEADERS
 *****************************************************************************/

#include "neuron.h"
#include "debug.h"


/*****************************************************************************
 *  GLOBAL VARIABLES
 *****************************************************************************/

CompactNeuralNetwork global_network = {0};
mach_port_t trivfs_control = MACH_PORT_NULL;
char *fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
                "Usage: settrans -a <node> /hurd/sigmoid-neuron-translator";


/*****************************************************************************
 *  INITIALIZATION
 *****************************************************************************/

static void __attribute__((constructor)) translator_init(void)
{
    if (global_network.memory_block == NULL) {
        memset(&global_network, 0, sizeof(global_network));
    }
    trivfs_control = MACH_PORT_NULL;
    log_debug_message("[DEBUG] Sigmoid Neuron Translator: Constructor ran");
}


/*****************************************************************************
 *  HOOK IMPLEMENTATIONS
 *  These are our custom implementations that do the actual work.
 *****************************************************************************/

/* Open hook implementation */
static error_t our_fs_open(struct iouser *cred, int flags, mode_t mode,
                           struct node *node, struct iobuf **iobuf)
{
    (void)cred; (void)flags; (void)mode; (void)node;
    
    log_debug_message("[DEBUG] our_fs_open called");
    *iobuf = NULL;
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return EIO;
        }
    }
    
    return 0;
}


/* Read hook implementation */
static error_t our_fs_read(struct iouser *cred, struct iobuf *iobuf,
                          off_t offset, size_t *len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "[DEBUG] our_fs_read called, initialized=%d",
            global_network.initialized);
    log_debug_message(debug_msg);
    
    if (!global_network.initialized) {
        return EIO;
    }
    
    char buffer[4096];
    size_t written = 0;
    int snprintf_result;
    
    /* Build output */
    snprintf_result = snprintf(buffer, sizeof(buffer),
              "LLM Sigmoid Neuron Translator - GNU Hurd\n"
              "===========================================\n\n");
    if (snprintf_result < 0) return EIO;
    written = (size_t)snprintf_result;
    
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
    
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "Memory: %.2f KB, Neurons: %zu, Weights: %zu\n\n",
                       (double)global_network.memory_block_size / 1024.0,
                       global_network.total_neurons,
                       global_network.total_weights);
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
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
    
    snprintf_result = snprintf(buffer + written, sizeof(buffer) - written,
                       "\nStatistics:\n"
                       "  Forward Passes: %zu\n"
                       "  Neuron Activations: %zu\n\n",
                       global_network.forward_pass_count,
                       global_network.neuron_activations);
    if (snprintf_result < 0) return EIO;
    written += (size_t)snprintf_result;
    
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
    
    if (written >= sizeof(buffer)) {
        written = sizeof(buffer) - 1;
    }
    buffer[written] = '\0';
    
    if (len == NULL) {
        return EINVAL;
    }
    
    size_t to_copy = (written < *len) ? written : *len;
    if (to_copy > 0 && iobuf != NULL && iobuf->buf != NULL) {
        memcpy(iobuf->buf, buffer, to_copy);
    }
    *len = to_copy;
    
    return 0;
}


/* Write hook implementation */
static error_t our_fs_write(struct iouser *cred, struct iobuf *iobuf,
                           off_t offset, size_t len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    if (len == 0) {
        return EINVAL;
    }
    
    if (!global_network.initialized) {
        return EIO;
    }
    
    if (iobuf == NULL || iobuf->buf == NULL) {
        return EINVAL;
    }
    
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
    
    /* Handle commands */
    if (strlen(temp) >= 5 && strncmp(temp, "reset", 5) == 0) {
        if (temp[5] == '\0' || temp[5] == ' ' || temp[5] == '\t') {
            network_reset(&global_network);
            return 0;
        }
    }
    
    if (strlen(temp) >= 5 && strncmp(temp, "save ", 5) == 0) {
        char *filename = temp + 5;
        if (*filename != '\0') {
            if (network_save(&global_network, filename)) {
                return 0;
            }
        }
        return EINVAL;
    }
    
    if (strlen(temp) >= 5 && strncmp(temp, "load ", 5) == 0) {
        char *filename = temp + 5;
        if (*filename != '\0') {
            if (network_load(&global_network, filename)) {
                return 0;
            }
        }
        return EINVAL;
    }
    
    /* Parse topology */
    uint16_t layers[MAX_LAYERS];
    int layer_count = parse_config_string(temp, layers, MAX_LAYERS);
    
    if (layer_count > 0) {
        network_free(&global_network);
        if (network_init(&global_network, layer_count, layers) != 0) {
            return EINVAL;
        }
        return 0;
    }
    
    /* Parse input */
    if (parse_input_string(&global_network, temp)) {
        return 0;
    }
    
    return EINVAL;
}


/*****************************************************************************
 *  STANDARD TRIVFS FUNCTIONS
 *  These are the functions that libtrivfs expects and will call.
 *  We define them to override the default libtrivfs implementations.
 *  They delegate to our custom implementations above.
 *****************************************************************************/

/* Standard trivfs open function */
error_t fs_open(struct iouser *cred, int flags, mode_t mode,
               struct node *node, struct iobuf **iobuf)
{
    return our_fs_open(cred, flags, mode, node, iobuf);
}

/* Standard trivfs read function */
error_t fs_read(struct iouser *cred, struct iobuf *iobuf,
               off_t offset, size_t *len, size_t count)
{
    return our_fs_read(cred, iobuf, offset, len, count);
}

/* Standard trivfs write function */
error_t fs_write(struct iouser *cred, struct iobuf *iobuf,
                off_t offset, size_t len, size_t count)
{
    return our_fs_write(cred, iobuf, offset, len, count);
}


/*****************************************************************************
 *  STUB IMPLEMENTATIONS FOR NON-HURD SYSTEMS
 *  Note: These stubs are provided even on Hurd in case the system
 *  doesn't have the actual libtrivfs libraries installed.
 *****************************************************************************/

int trivfs_server_loop(void) {
    log_debug_message("[DEBUG] trivfs_server_loop: STUB - not on Hurd!");
    return -1;
}

/* trivfs_server is provided by libtrivfs on Hurd systems.
 * On non-Hurd systems, we don't need it as we can't run as a translator.
 */
#if ON_HURD != 1
/* On non-Hurd systems, we don't have libtrivfs, so trivfs_server is not available
 * This is fine because we can't run as a translator on non-Hurd anyway.
 */
int trivfs_server(void) __attribute__((weak));
#endif


/*****************************************************************************
 *  TRIVFS DEMUXER - Optional entry point for custom message handling
 *  For standard libtrivfs usage, we don't need this - libtrivfs provides
 *  the server loop via trivfs_server() and calls our fs_* functions directly.
 *  
 *  However, some Hurd versions expect this symbol to exist. We provide a minimal
 *  implementation that delegates to libtrivfs.
 *****************************************************************************/

/* Minimal demuxer that works with libtrivfs */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    (void)inmsg; (void)outmsg;
    /* libtrivfs will call our fs_* functions automatically */
    return 0;
}

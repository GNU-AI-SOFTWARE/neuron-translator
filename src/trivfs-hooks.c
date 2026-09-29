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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

/*****************************************************************************
 *  HURD DETECTION AND TYPE DEFINITIONS
 *  
 *  On Hurd: Use system headers for Mach types (mach_port_t, error_t, etc.)
 *  On non-Hurd: Provide our own type definitions
 *  
 *  We provide complete definitions for struct iobuf, struct node, struct iouser
 *  in this file (where we need to access their members).
 *****************************************************************************/

#if defined(__GNU__) && !defined(__GNU_LIBRARY__)
#define ON_HURD 1
#endif

/* On Hurd: include system headers */
#if defined(ON_HURD)
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <hurd.h>
#include <hurd/trivfs.h>
#else
/* Non-Hurd systems: provide Mach type definitions */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)
struct mach_msg_header;
typedef struct mach_msg_header mach_msg_header_t;
typedef int error_t;
#endif

/*
 * Complete type definitions for Hurd filesystem structures.
 * These match the definitions used internally by libtrivfs.
 * We define them here regardless of platform to ensure we can access
 * members like iobuf->buf in our hook implementations.
 */
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
 *  FUNCTION FORWARD DECLARATIONS (only visible in this file)
 *  These are declared here to avoid type visibility issues with Hurd headers
 *****************************************************************************/

/* Hook implementations */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);


/*****************************************************************************
 *  STUB IMPLEMENTATIONS FOR NON-HURD SYSTEMS
 *****************************************************************************/

#ifndef ON_HURD

int trivfs_server_loop(void) {
    log_debug_message("[DEBUG] trivfs_server_loop: STUB - not on Hurd!");
    return -1;
}

error_t trivfs_server(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg) {
    (void)inmsg; (void)outmsg;
    log_debug_message("[DEBUG] trivfs_server: STUB - not on Hurd!");
    return -1;
}

/* On non-Hurd, provide the fs_* functions that would be in libtrivfs */
error_t fs_open(struct iouser *cred, int flags, mode_t mode,
               struct node *node, struct iobuf **iobuf)
{
    (void)cred; (void)flags; (void)mode; (void)node; (void)iobuf;
    return ENOSYS;
}

error_t fs_read(struct iouser *cred, struct iobuf *iobuf,
               off_t offset, size_t *len, size_t count)
{
    (void)cred; (void)iobuf; (void)offset; (void)len; (void)count;
    return ENOSYS;
}

error_t fs_write(struct iouser *cred, struct iobuf *iobuf,
                off_t offset, size_t len, size_t count)
{
    (void)cred; (void)iobuf; (void)offset; (void)len; (void)count;
    return ENOSYS;
}

#endif /* !ON_HURD */


/*****************************************************************************
 *  GLOBAL VARIABLES
 *****************************************************************************/

CompactNeuralNetwork global_network = {0};
mach_port_t trivfs_control = MACH_PORT_NULL;
char *fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
                "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator";


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
 *  These are our actual implementations that get called
 *****************************************************************************/

error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf)
{
    (void)cred; (void)flags; (void)mode; (void)node;
    
    log_debug_message("[DEBUG] fs_open_hook called");
    *iobuf = NULL;
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return EIO;
        }
    }
    
    return 0;
}


error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "[DEBUG] fs_read_hook called, initialized=%d",
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


error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
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
 *  TRIVFS DEMUXER - Entry point for Hurd translator
 *****************************************************************************/

int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    log_debug_message("[DEBUG] trivfs_demuxer: Called (Hurd entry point)");
    
    if (inmsg == NULL || outmsg == NULL) {
        return EINVAL;
    }
    
#ifdef ON_HURD
    /* On Hurd, delegate to libtrivfs */
    extern error_t trivfs_server(mach_msg_header_t, mach_msg_header_t);
    return trivfs_server(*inmsg, *outmsg);
#else
    log_debug_message("[DEBUG] trivfs_demuxer: STUB - not on Hurd!");
    return -1;
#endif
}

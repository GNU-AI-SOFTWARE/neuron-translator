/*
 * trivfs-hooks.h - Hurd Translator Hooks for LLM Neuron Translator
 *
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file trivfs-hooks.h
 *  @brief Hurd trivfs translator interface declarations
 *  Style: Claude Delannoy - C23 standard, POSIX compliant, educational.
 *
 *  NOTE: We provide our own definitions of struct iouser, struct node,
 *  struct iobuf to match libtrivfs internal types, as they are not
 *  fully exposed in public Hurd headers.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <sys/types.h>
#include <errno.h>

/*****************************************************************************
 *  HURD DETECTION
 *****************************************************************************/

#if defined(__GNU__) && !defined(__GNU_LIBRARY__)
#define ON_HURD 1
#endif

/*****************************************************************************
 *  TYPES
 *  On Hurd: Include Mach headers for basic types
 *  On non-Hurd: Provide Mach type definitions
 *  
 *  For struct iouser, struct node, struct iobuf: we always provide our own
 *  definitions to ensure we can access members like iobuf->buf.
 *  These match the internal libtrivfs definitions.
 *****************************************************************************/

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
 * Complete definitions for Hurd filesystem structures.
 * These match the internal libtrivfs definitions.
 * We define them here to ensure we can access members like iobuf->buf.
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
 *  FUNCTION DECLARATIONS
 *  Note: Hook implementations are in trivfs-hooks.c
 *****************************************************************************/

/* Translator entry point */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/*****************************************************************************
 *  GLOBAL VARIABLES
 *****************************************************************************/

extern mach_port_t trivfs_control;
extern char *fs_help;

/*****************************************************************************
 *  STUB FUNCTIONS FOR NON-HURD
 *****************************************************************************/

#ifndef ON_HURD
/* Stub functions for non-Hurd systems */
extern int trivfs_server_loop(void);
extern error_t trivfs_server(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);
#endif

/*****************************************************************************
 *  NEURAL NETWORK
 *****************************************************************************/

#include "neuron.h"

extern CompactNeuralNetwork global_network;

#endif /* TRIVFS_HOOKS_H */

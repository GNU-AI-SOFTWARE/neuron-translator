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
 *  NOTE: Hurd-specific type definitions (struct iobuf, struct node, struct iouser)
 *  are provided in trivfs-hooks.c to avoid conflicts with system headers.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <sys/types.h>

/*****************************************************************************
 *  HURD DETECTION
 *****************************************************************************/

#if defined(__GNU__) && !defined(__GNU_LIBRARY__)
#define ON_HURD 1
#endif

/*****************************************************************************
 *  TYPES - Must be defined before any function using them
 *  On Hurd: We include Mach headers for mach_port_t
 *  On non-Hurd: We typedef mach_port_t ourselves
 *****************************************************************************/

#if defined(ON_HURD)
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <hurd.h>
#else
/* Non-Hurd systems: provide Mach type definitions */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)
#endif

struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;

/*
 * Forward declarations for Hurd types.
 * Complete definitions are in trivfs-hooks.c where we need to access members.
 */
struct iouser;
struct node;
struct iobuf;

typedef int error_t;

/*****************************************************************************
 *  FUNCTION DECLARATIONS
 *  Note: Hook implementations are declared in trivfs-hooks.c only
 *  to avoid type visibility issues with Hurd system headers
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
extern error_t trivfs_server(mach_msg_header_t inmsg, mach_msg_header_t outmsg);
#endif

/*****************************************************************************
 *  NEURAL NETWORK
 *****************************************************************************/

#include "neuron.h"

extern CompactNeuralNetwork global_network;

#endif /* TRIVFS_HOOKS_H */

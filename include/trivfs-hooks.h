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
 *  NOTE: struct iouser, struct node, struct iobuf are forward-declared here.
 *  Complete definitions are in trivfs-hooks.c (from <hurd/iohelp.h> on Hurd,
 *  or our own definitions on non-Hurd).
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
 *  TYPES - Basic Mach/Hurd types
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

/* Forward declarations for Hurd types */
struct iouser;
struct node;
struct iobuf;

/*****************************************************************************
 *  FUNCTION DECLARATIONS
 *  Hook implementations are declared here, using forward-declared types
 *****************************************************************************/

/* Hook implementations */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);

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

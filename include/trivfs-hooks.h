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
 *
 *  On GNU/Hurd: uses system types from <hurd/trivfs.h>.
 *  On other systems: provides minimal opaque declarations.
 *  Style: Claude Delannoy - C23 standard, POSIX compliant, educational.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <errno.h>   /* For errno */
#include <sys/types.h>  /* For mode_t, off_t, size_t */

/*****************************************************************************
 *  HURD DETECTION
 *  On GNU/Hurd, __GNU__ is defined by gcc
 *****************************************************************************/

#if defined(__GNU__) && !defined(__GNU_LIBRARY__)
#define ON_HURD 1
#endif

/*****************************************************************************
 *  MACH/HURD TYPES
 *  On Hurd: use system headers. On other systems: opaque forward declarations.
 *****************************************************************************/

#if defined(ON_HURD)
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <hurd.h>
#include <hurd/trivfs.h>
#else
/* Fallback types for non-Hurd systems */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)
struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;
struct iouser;
struct node;
struct iobuf;
#endif

/*****************************************************************************
 *  TRANSLATOR VARIABLES
 *****************************************************************************/

extern mach_port_t trivfs_control;
extern char *fs_help;

/*****************************************************************************
 *  STUB FUNCTIONS FOR NON-HURD SYSTEMS
 *****************************************************************************/

#ifndef ON_HURD
/* On non-Hurd systems, provide stub declarations */
extern int trivfs_server_loop(void);
extern error_t trivfs_server(mach_msg_header_t inmsg, mach_msg_header_t outmsg);
#endif

/*****************************************************************************
 *  HOOK FUNCTION DECLARATIONS
 *****************************************************************************/

/* Main translator entry point */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/* Filesystem hooks - these override trivfs defaults via weak symbols */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);

/*****************************************************************************
 *  NEURAL NETWORK
 *****************************************************************************/

#include "neuron.h"

extern CompactNeuralNetwork global_network;

#endif /* TRIVFS_HOOKS_H */

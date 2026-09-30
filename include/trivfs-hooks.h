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
 *  NOTE: We provide our own definitions of Hurd types to avoid conflicts
 *  with system headers. The system's <hurd/iohelp.h> only provides partial
 *  definitions (struct iouser is complete, but struct node and struct iobuf
 *  are only forward-declared).
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <sys/types.h>
#include <errno.h>  /* For error_t */

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

/* mach_msg_header_t is either from system or our typedef above */

/*****************************************************************************
 *  HURD FILESYSTEM TYPES
 *  We always provide our own complete definitions to ensure we can access
 *  members like iobuf->buf. These match the internal libtrivfs types.
 *  Note: We avoid including <hurd/trivfs.h> and <hurd/iohelp.h> as they
 *  cause redefinition conflicts with struct iouser.
 *****************************************************************************/

/* Forward declarations for Hurd types */
struct iouser;
struct node;
struct iobuf;

/*****************************************************************************
 *  FUNCTION DECLARATIONS
 *****************************************************************************/

/* Standard trivfs functions - these are called by libtrivfs */
error_t fs_open(struct iouser *cred, int flags, mode_t mode,
                struct node *node, struct iobuf **iobuf);
error_t fs_read(struct iouser *cred, struct iobuf *iobuf,
                 off_t offset, size_t *len, size_t count);
error_t fs_write(struct iouser *cred, struct iobuf *iobuf,
                  off_t offset, size_t len, size_t count);

/* Hook implementations - our internal implementations */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);

/* Translator entry point */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/*****************************************************************************
 *  STUB FUNCTIONS FOR NON-HURD
 *****************************************************************************/

/* Stub for non-Hurd systems */
extern int trivfs_server_loop(void);

/*****************************************************************************
 *  GLOBAL VARIABLES
 *****************************************************************************/

extern mach_port_t trivfs_control;
extern char *fs_help;

/*****************************************************************************
 *  LIBTRIVFS FUNCTIONS
 *  trivfs_server is provided by libtrivfs on Hurd systems.
 *  On non-Hurd systems, we don't declare it as we can't link with libtrivfs.
 *****************************************************************************/

#if ON_HURD == 1
/* Main server function from libtrivfs */
extern int trivfs_server(void);
#endif

/*****************************************************************************
 *  STUB FUNCTIONS FOR NON-HURD
 *****************************************************************************/

#ifndef ON_HURD
/* Stub functions for non-Hurd systems */
extern int trivfs_server_loop(void);
#endif

/*****************************************************************************
 *  NEURAL NETWORK
 *****************************************************************************/

#include "neuron.h"

extern CompactNeuralNetwork global_network;

#endif /* TRIVFS_HOOKS_H */

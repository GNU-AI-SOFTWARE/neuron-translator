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
 *  This header declares the translator-specific interface.
 *  On GNU/Hurd: uses system types from <hurd/trivfs.h>.
 *  On other systems: provides opaque forward declarations.
 *  Style: Claude Delannoy - C23 standard, POSIX compliant, educational.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                        SYSTEM HEADERS                                    *
 *                                                                           *
 *****************************************************************************/

#include <errno.h>   /* For errno */
#include <sys/types.h>  /* For mode_t, off_t, size_t */

/*****************************************************************************
 *                                                                           *
 *                    HURD DETECTION                                        *
 *                                                                           *
 *  On GNU/Hurd, __GNU__ is defined by gcc. This is the standard detection. *
 *                                                                           *
 *****************************************************************************/

#if defined(__GNU__) && !defined(__GNU_LIBRARY__)
#define ON_HURD 1
#endif


/*****************************************************************************
 *                                                                           *
 *                    MACH/HURD TYPES                                       *
 *                                                                           *
 *  On Hurd: use system headers that define all necessary types.            *
 *  On other systems: provide minimal forward declarations.                 *
 *                                                                           *
 *****************************************************************************/

#if defined(ON_HURD)
/* On Hurd, include system headers - they define everything we need */
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <hurd.h>
#include <hurd/trivfs.h>
#else
/* Minimal forward declarations for non-Hurd systems (Linux, etc.) */

/* Mach types */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)

struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;

/* Hurd types - declare as opaque (pointer only) */
struct iouser;
struct node;
struct iobuf;

/* error_t is in errno.h */

#endif


/*****************************************************************************
 *                                                                           *
 *                    FUNCTION DECLARATIONS                                 *
 *                                                                           *
 *  On Hurd: trivfs_server_loop and trivfs_server are in libtrivfs.        *
 *                                                                           *
 *****************************************************************************/

#ifndef ON_HURD
/* On non-Hurd systems, provide stub declarations */
extern int trivfs_server_loop(void);
extern error_t trivfs_server(mach_msg_header_t inmsg, mach_msg_header_t outmsg);
#endif


/*****************************************************************************
 *                                                                           *
 *                    TRANSLATOR VARIABLES                                  *
 *                                                                           *
 *****************************************************************************/

/* Translator control port - defined in src/trivfs-hooks.c */
extern mach_port_t trivfs_control;

/* Help text for the translator - defined in src/trivfs-hooks.c */
extern char *fs_help;

/* Filesystem hook functions - defined in src/trivfs-hooks.c */
extern error_t fs_open(struct iouser *, int, mode_t, struct node *, struct iobuf **);
extern error_t fs_read(struct iouser *, struct iobuf *, off_t, size_t *, size_t);
extern error_t fs_write(struct iouser *, struct iobuf *, off_t, size_t, size_t);


/*****************************************************************************
 *                                                                           *
 *                    NEURAL NETWORK                                        *
 *                                                                           *
 *****************************************************************************/

#include "neuron.h"

/** Global network instance - defined in src/trivfs-hooks.c */
extern CompactNeuralNetwork global_network;


/*****************************************************************************
 *                                                                           *
 *                    HOOK FUNCTION DECLARATIONS                           *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Translator message demultiplexer - entry point for Hurd translator
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/**
 * @brief Open hook for translator node
 */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);

/**
 * @brief Read hook for translator node
 */
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);

/**
 * @brief Write hook for translator node
 */
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);


#endif /* TRIVFS_HOOKS_H */

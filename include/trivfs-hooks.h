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
 *  This header provides the interface between the neural network implementation
 *  and the GNU Hurd trivfs translator system. It follows Claude Delannoy's
 *  educational style with clear, maintainable code.
 *
 *  Note: We include minimal system headers here and declare only what's
 *  necessary. User code must include <hurd/trivfs.h> and <hurd/iohelp.h>
 *  to get the actual Hurd implementations.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                    MINIMAL SYSTEM HEADERS INCLUSION                       *
 *                                                                           *
 *  We include only what we absolutely need to declare our interface.        *
 *  POSIX types are included here; Hurd types are declared externally.     *
 *                                                                           *
 *****************************************************************************/

/* POSIX types */
#include <sys/types.h>  /* For mode_t, off_t, size_t */

/* Our neural network types */
#include "neuron.h"


/*****************************************************************************
 *                                                                           *
 *                    FORWARD DECLARATIONS FOR HURD TYPES                    *
 *                                                                           *
 *  These are forward declarations. The actual definitions come from     *
 *  Hurd headers which must be included by user code (main.c).             *
 *                                                                           *
 *****************************************************************************/

/* Hurd I/O structures */
struct iobuf;
struct node;
struct iouser;

/* Mach message type */
struct mach_msg_header_t;
typedef struct mach_msg_header_t *mach_msg_header_t;


/*****************************************************************************
 *                                                                           *
 *                    EXTERNAL DECLARATIONS                                  *
 *                                                                           *
 *  These symbols are declared in Hurd headers. We declare them here       *
 *  explicitly to ensure they are visible. User code must include           *
 *  <hurd/trivfs.h>, <hurd/iohelp.h>, <mach/mach.h> for the actual        *
 *  implementations.                                                      *
 *                                                                           *
 *****************************************************************************/

/* From <hurd/trivfs.h> */
extern mach_port_t trivfs_control;
extern char *fs_help;
extern error_t (*fs_open)(struct iouser *, int, mode_t, struct node *, struct iobuf **);
extern error_t (*fs_read)(struct iouser *, struct iobuf *, off_t, size_t *, size_t);
extern error_t (*fs_write)(struct iouser *, struct iobuf *, off_t, size_t, size_t);

/* From <mach/mach.h> */
#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL ((mach_port_t) 0)
#endif

/* From <hurd/trivfs.h> */
extern error_t trivfs_server(mach_msg_header_t, mach_msg_header_t);
extern int trivfs_server_loop(void);


/*****************************************************************************
 *                                                                           *
 *                         GLOBAL NETWORK INSTANCE                          *
 *                                                                           *
 *****************************************************************************/

/** Global network instance - defined in src/trivfs-hooks.c */
extern CompactNeuralNetwork global_network;


/*****************************************************************************
 *                                                                           *
 *                      HOOK FUNCTION DECLARATIONS                         *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Translator message demultiplexer
 *
 * @param inmsg  Incoming Mach message header
 * @param outmsg Outgoing Mach message header
 * @return       Error code (0 on success)
 */
int trivfs_demuxer(mach_msg_header_t inmsg, mach_msg_header_t outmsg);

/**
 * @brief Open hook for translator node
 *
 * @param cred  User credentials
 * @param flags File open flags
 * @param mode  File creation mode
 * @param node  Filesystem node
 * @param iobuf Output: I/O buffer
 * @return      Error code (0 on success)
 */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);

/**
 * @brief Read hook for translator node
 *
 * @param cred   User credentials
 * @param iobuf  I/O buffer
 * @param offset Read offset
 * @param len    Output: number of bytes read
 * @param count  Maximum bytes to read
 * @return       Error code (0 on success)
 */
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);

/**
 * @brief Write hook for translator node
 *
 * @param cred   User credentials
 * @param iobuf  I/O buffer
 * @param offset Write offset
 * @param len    Number of bytes to write
 * @param count  Reserved (unused)
 * @return       Error code (0 on success)
 */
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);


#endif /* TRIVFS_HOOKS_H */

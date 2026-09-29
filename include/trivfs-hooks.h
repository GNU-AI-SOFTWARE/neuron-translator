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
 *  Note: This header should be included AFTER Hurd and Mach headers
 *  (<hurd/trivfs.h>, <hurd/iohelp.h>, <mach/mach.h>, <mach/message.h>)
 *  which define all the necessary types.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                    SYSTEM HEADERS (must be included by user)             *
 *                                                                           *
 *  User code MUST include these before including this header:              *
 *  - <hurd/trivfs.h>     (for fs_open, fs_read, fs_write, etc.)            *
 *  - <hurd/iohelp.h>     (for struct iobuf, struct node, struct iouser)    *
 *  - <mach/mach.h>       (for mach_port_t, MACH_PORT_NULL)                *
 *  - <mach/message.h>    (for mach_msg_header_t)                         *
 *                                                                           *
 *****************************************************************************/

/* POSIX types */
#include <sys/types.h>  /* For mode_t, off_t, size_t */

/* Our neural network types */
#include "neuron.h"


/*****************************************************************************
 *                                                                           *
 *                    FORWARD DECLARATIONS (if needed)                      *
 *                                                                           *
 *  Only declare types that are not provided by the Hurd/Mach headers.      *
 *  Most types should come from the system headers included by user code.  *
 *                                                                           *
 *****************************************************************************/

/* Only forward declare if the Hurd headers are not available */
#ifndef __MACH_MESSAGE_H__
/* Mach message type - only forward declare if not already defined */
struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;
#endif

/* Hurd I/O structures - only forward declare if not already defined */
#ifndef __HURD_IOHELP_H__
struct iobuf;
struct node;
struct iouser;
#endif


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
 *  These match the signatures from <hurd/trivfs.h>. User code must        *
 *  include the Hurd headers to get the actual type definitions.           *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Translator message demultiplexer
 *
 * This function handles all Mach IPC messages for the translator.
 * Signature must match the declaration in <hurd/trivfs.h>.
 *
 * @param inmsg  Incoming Mach message header (pointer)
 * @param outmsg Outgoing Mach message header (pointer)
 * @return       Error code (0 on success)
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/**
 * @brief Open hook for translator node
 *
 * Called when the translator node is opened.
 * Signature must match the fs_open hook from <hurd/trivfs.h>.
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
 * Called when the translator node is read from.
 * Signature must match the fs_read hook from <hurd/trivfs.h>.
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
 * Called when data is written to the translator node.
 * Signature must match the fs_write hook from <hurd/trivfs.h>.
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

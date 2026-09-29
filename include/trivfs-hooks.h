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
 *  @brief Complete Hurd trivfs interface for the sigmoid neuron translator
 *
 *  This header provides a self-contained interface that declares all necessary
 *  Hurd and Mach types and symbols. It includes system headers where available
 *  and provides fallback declarations when needed.
 *
 *  This follows Claude Delannoy's educational style with extensive comments
 *  explaining each declaration. The design ensures POSIX and C23 compliance.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                        SYSTEM HEADERS                                    *
 *                                                                           *
 *****************************************************************************/

/* POSIX error and type headers */
#include <errno.h>   /* For errno and error_t (typedef enum __error_t_codes) */
#include <sys/types.h> /* For mode_t, off_t, size_t */


/*****************************************************************************
 *                                                                           *
 *                    MACH TYPES (from <mach/mach.h>)                        *
 *                                                                           *
 *  These declarations are provided here for systems where Mach headers       *
 *  are not available. If Mach headers are present, they will override these. *
 *                                                                           *
 *****************************************************************************/

#ifndef __MACH_MACH_H__
/* Only declare if Mach headers are not already included */

/* Mach message header type */
struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;

/* Mach port type */
typedef unsigned int mach_port_t;

/* Mach port null value */
#define MACH_PORT_NULL ((mach_port_t) 0)

#endif /* __MACH_MACH_H__ */


/*****************************************************************************
 *                                                                           *
 *                    HURD TYPES (from <hurd/iohelp.h>)                     *
 *                                                                           *
 *****************************************************************************/

#ifndef __HURD_IOHELP_H__
/* Only declare if Hurd headers are not already included */

/* Hurd I/O structures */
struct iobuf;
struct node;
struct iouser;

#endif /* __HURD_IOHELP_H__ */


/*****************************************************************************
 *                                                                           *
 *                    HURD VARIABLES (from <hurd/trivfs.h>)                  *
 *                                                                           *
 *****************************************************************************/

#ifndef __HURD_TRIVFS_H__
/* Only declare if Hurd trivfs headers are not already included */

/* Translator control port */
extern mach_port_t trivfs_control;

/* Help text for the translator */
extern char *fs_help;

/* Filesystem hook function pointers */
extern error_t (*fs_open)(struct iouser *, int, mode_t, struct node *, struct iobuf **);
extern error_t (*fs_read)(struct iouser *, struct iobuf *, off_t, size_t *, size_t);
extern error_t (*fs_write)(struct iouser *, struct iobuf *, off_t, size_t, size_t);

/* Server functions */
extern error_t trivfs_server(mach_msg_header_t, mach_msg_header_t);
extern int trivfs_server_loop(void);

#endif /* __HURD_TRIVFS_H__ */


/*****************************************************************************
 *                                                                           *
 *                         NEURAL NETWORK INSTANCE                          *
 *                                                                           *
 *****************************************************************************/

#include "neuron.h"

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
 * This function handles all Mach IPC messages for the translator.
 *
 * @param inmsg  Incoming Mach message header (pointer)
 * @param outmsg Outgoing Mach message header (pointer)
 * @return       Error code (0 on success)
 */
int trivfs_demuxer(mach_msg_header_t inmsg, mach_msg_header_t outmsg);

/**
 * @brief Open hook for translator node
 *
 * Called when the translator node is opened.
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

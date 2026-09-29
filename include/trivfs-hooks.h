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
 *  This header provides a self-contained interface for the sigmoid neuron
 *  translator. It declares all necessary Hurd and Mach types explicitly
 *  to ensure compilation works even when Hurd development headers are not
 *  available. It follows Claude Delannoy's educational style.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                        SYSTEM HEADERS                                    *
 *                                                                           *
 *****************************************************************************/

#include <errno.h>   /* For error_t */
#include <sys/types.h>  /* For mode_t, off_t, size_t */


/*****************************************************************************
 *                                                                           *
 *                    MACH TYPES                                             *
 *                                                                           *
 *****************************************************************************/

/* Mach message header type */
struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;

/* Mach port type */
typedef unsigned int mach_port_t;

/* Mach port null value */
#define MACH_PORT_NULL ((mach_port_t) 0)


/*****************************************************************************
 *                                                                           *
 *                    HURD TYPES                                             *
 *                                                                           *
 *****************************************************************************/

/* Hurd I/O structures */
struct iobuf;
struct node;
struct iouser;


/*****************************************************************************
 *                                                                           *
 *                    HURD VARIABLES                                        *
 *                                                                           *
 *****************************************************************************/

/* Translator control port */
extern mach_port_t trivfs_control;

/* Help text for the translator */
extern char *fs_help;

/* Filesystem hook function pointers */
extern error_t (*fs_open)(struct iouser *, int, mode_t, struct node *, struct iobuf **);
extern error_t (*fs_read)(struct iouser *, struct iobuf *, off_t, size_t *, size_t);
extern error_t (*fs_write)(struct iouser *, struct iobuf *, off_t, size_t, size_t);


/*****************************************************************************
 *                                                                           *
 *                    HURD FUNCTIONS                                        *
 *                                                                           *
 *****************************************************************************/

/* Server function - takes pointers to message headers */
extern error_t trivfs_server(mach_msg_header_t, mach_msg_header_t);
extern int trivfs_server_loop(void);


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
 * @param inmsg  Incoming Mach message header (pointer to pointer)
 * @param outmsg Outgoing Mach message header (pointer to pointer)
 * @return       Error code (0 on success)
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

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

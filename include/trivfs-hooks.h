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
 *  This header declares the trivfs hooks and external variables for the
 *  sigmoid neuron translator. It follows GNU Hurd translator conventions.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <hurd.h>
#include <hurd/trivfs.h>
#include <hurd/iohelp.h>

#include <mach/mach.h>
#include <mach/port.h>
#include <mach/message.h>


/*****************************************************************************
 *                                                                           *
 *                         EXTERNAL VARIABLES                                *
 *                                                                           *
 *  Variables declared by the Hurd trivfs library                            *
 *                                                                           *
 *****************************************************************************/

/** Mach port for translator control */
extern mach_port_t trivfs_control;

/** Help text for the translator */
extern char *fs_help;

/** Filesystem open hook */
extern error_t (*fs_open) (struct iouser *, int, mode_t, struct node *,
                           struct iobuf **);

/** Filesystem read hook */
extern error_t (*fs_read) (struct iouser *, struct iobuf *, off_t, size_t *,
                           size_t);

/** Filesystem write hook */
extern error_t (*fs_write) (struct iouser *, struct iobuf *, off_t, size_t,
                            size_t);


/*****************************************************************************
 *                                                                           *
 *                      FUNCTION DECLARATIONS                              *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Hurd trivfs server function
 * 
 * @param inmsg  Incoming Mach message header
 * @param outmsg Outgoing Mach message header
 * @return       Error code
 */
extern error_t trivfs_server(mach_msg_header_t *, mach_msg_header_t *);

/**
 * @brief Hurd trivfs server loop
 * 
 * @return Exit code
 */
extern int trivfs_server_loop(void);

/**
 * @brief Translator message demultiplexer
 * 
 * This function handles all Mach IPC messages for the translator.
 * 
 * @param inmsg  Incoming Mach message header
 * @param outmsg Outgoing Mach message header
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

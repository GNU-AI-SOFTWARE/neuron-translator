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
 *  This header declares ONLY our own symbols for the sigmoid neuron translator.
 *  All Hurd and Mach symbols (fs_open, fs_read, fs_write, trivfs_control,
 *  fs_help, trivfs_server_loop, mach_msg_header_t, etc.) must be obtained
 *  by including the appropriate Hurd/Mach headers BEFORE including this file.
 *
 *  Required headers to include before this one:
 *    - <hurd/trivfs.h>     (for fs_open, fs_read, fs_write, trivfs_control, fs_help)
 *    - <hurd/iohelp.h>     (for struct iobuf, struct node, struct iouser)
 *    - <mach/mach.h>       (for mach_port_t, MACH_PORT_NULL)
 *    - <mach/message.h>    (for mach_msg_header_t)
 *
 *  This follows Claude Delannoy's style: clear, minimal, no redefinitions.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                        POSIX TYPES AND ERROR HANDLING                    *
 *                                                                           *
 *****************************************************************************/

#include <sys/types.h>  /* For mode_t, off_t, size_t */


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
 *  Our hook implementations. Signatures must match those expected by    *
 *  <hurd/trivfs.h>. User code must include Hurd headers for the types.   *
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

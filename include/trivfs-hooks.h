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
 *  This header provides all necessary declarations for the sigmoid neuron
 *  translator to interface with GNU Hurd's trivfs system. It follows
 *  Claude Delannoy's educational style with clear, maintainable code.
 *
 *  Note: We include all necessary system headers FIRST, then declare
 *  our own symbols. This ensures all types are properly defined.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                        SYSTEM HEADERS INCLUSION                           *
 *                                                                           *
 *  Include ALL system headers first to ensure all types are defined        *
 *  before we use them in our declarations.                                   *
 *                                                                           *
 *****************************************************************************/

/* POSIX types */
#include <sys/types.h>  /* For mode_t, off_t, size_t */

/* Mach IPC types */
#include <mach/mach.h>    /* For mach_port_t, MACH_PORT_NULL */
#include <mach/port.h>
#include <mach/message.h>

/* Hurd trivfs */
#include <hurd.h>
#include <hurd/trivfs.h>
#include <hurd/iohelp.h>

/* Our neural network types */
#include "neuron.h"


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
 *  Declare our hook implementations. All Hurd types (error_t, struct iobuf,
 *  struct node, struct iouser, mach_port_t, etc.) are already defined by
 *  the headers included above.
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Translator message demultiplexer
 *
 * @param inmsg  Incoming Mach message header
 * @param outmsg Outgoing Mach message header
 * @return       Error code (0 on success)
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

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

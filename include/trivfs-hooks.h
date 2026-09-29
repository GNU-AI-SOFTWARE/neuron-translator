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
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H


/*****************************************************************************
 *                                                                           *
 *                    FORWARD DECLARATIONS FOR HURD TYPES                    *
 *                                                                           *
 *  These forward declarations ensure that Hurd types are visible before use.
 *  The actual definitions are provided by the Hurd headers included below.
 *                                                                           *
 *****************************************************************************/

/* Forward declare Hurd I/O structures */
struct iobuf;
struct node;
struct iouser;

/* Hurd error type */
typedef int error_t;


/*****************************************************************************
 *                                                                           *
 *                        HURD HEADERS INCLUSION                            *
 *                                                                           *
 *****************************************************************************/

#include <hurd/trivfs.h>
#include <hurd/iohelp.h>

#include <mach/mach.h>
#include <mach/port.h>
#include <mach/message.h>

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

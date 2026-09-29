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
 *  This header provides access to Hurd trivfs declarations and the global
 *  neural network instance. All Hurd-specific types and functions are
 *  obtained from the official Hurd headers.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

/*
 * Include Hurd headers first - they define all necessary types and variables.
 * <hurd/trivfs.h> provides:
 *   - fs_open, fs_read, fs_write (function pointer variables)
 *   - trivfs_control, fs_help (variables)
 *   - trivfs_server, trivfs_server_loop (functions)
 * <hurd/iohelp.h> provides:
 *   - struct iobuf, struct node, struct iouser
 */
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
 *  Declare our hook implementations with signatures matching the Hurd
 *  trivfs expectations. We use the same parameter names as in trivfs.h
 *  for maximum compatibility.
 *                                                                           *
 *****************************************************************************/

/* Message demultiplexer */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/* Filesystem hooks - signatures must exactly match the trivfs function pointers */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);


#endif /* TRIVFS_HOOKS_H */

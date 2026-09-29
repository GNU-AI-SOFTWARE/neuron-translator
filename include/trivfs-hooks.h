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
 *
 *  Note: All Hurd variables and functions (fs_open, fs_read, fs_write,
 *  trivfs_control, fs_help, trivfs_server, trivfs_server_loop) are
 *  declared in <hurd/trivfs.h> and should not be redeclared here.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <sys/types.h>  /* For mode_t, off_t, size_t */
#include <hurd.h>
#include <hurd/trivfs.h>      /* Provides: fs_open, fs_read, fs_write, trivfs_control, fs_help */
#include <hurd/iohelp.h>      /* Provides: struct iobuf, struct node, struct iouser */

#include <mach/mach.h>
#include <mach/port.h>
#include <mach/message.h>

#include "neuron.h"  /* For CompactNeuralNetwork type */


/*****************************************************************************
 *                                                                           *
 *                         GLOBAL NETWORK INSTANCE                          *
 *                                                                           *
 *****************************************************************************/

/* Global network instance - defined in src/trivfs-hooks.c */
extern CompactNeuralNetwork global_network;


/*****************************************************************************
 *                                                                           *
 *                      FUNCTION DECLARATIONS                              *
 *                                                                           *
 *  Note: Function signatures must exactly match those in <hurd/trivfs.h>  *
 *        to ensure type compatibility when assigning to fs_open, fs_read, etc.*
 *                                                                           *
 *****************************************************************************/

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

/** Open hook - signature must match fs_open from <hurd/trivfs.h> */
error_t fs_open_hook(struct iouser *, int, mode_t, struct node *, struct iobuf **);

/** Read hook - signature must match fs_read from <hurd/trivfs.h> */
error_t fs_read_hook(struct iouser *, struct iobuf *, off_t, size_t *, size_t);

/** Write hook - signature must match fs_write from <hurd/trivfs.h> */
error_t fs_write_hook(struct iouser *, struct iobuf *, off_t, size_t, size_t);


#endif /* TRIVFS_HOOKS_H */

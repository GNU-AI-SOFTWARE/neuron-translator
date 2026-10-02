/*
 * trivfs-hooks.h - Hurd Translator Interface for the Neuron Translator
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire Ivanenka <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file trivfs-hooks.h
 *  @brief Shared declarations for the trivfs translator hooks
 *
 *  On GNU/Hurd, the server routines (trivfs_S_io_read, trivfs_S_io_write,
 *  ...) are declared by <hurd/trivfs.h>; this header only pulls in the
 *  basic Mach types on Hurd, or provides minimal fallback types so the
 *  non-Hurd test harness in main.c still compiles.
 */

#ifndef TRIVFS_HOOKS_H
#define TRIVFS_HOOKS_H

#include <sys/types.h>

/*****************************************************************************
 *  HURD DETECTION
 *****************************************************************************/

/* ON_HURD should be defined by the Makefile via -DON_HURD flag
 * If compiling manually on Hurd, use: -DON_HURD
 */
#ifndef ON_HURD
#define ON_HURD 0
#endif

/*****************************************************************************
 *  BASIC MACH TYPES
 *  On Hurd: Use system headers
 *  On non-Hurd: Provide our own minimal definitions (test harness only)
 *****************************************************************************/

#if ON_HURD == 1
#if __has_include(<mach.h>)
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#include <hurd.h>
#else
#error "ON_HURD is defined but <mach.h> is not available"
#endif
#else
/* Non-Hurd systems: minimal types for the main.c test harness */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)
struct mach_msg_header;
typedef struct mach_msg_header mach_msg_header_t;
typedef int error_t;
#endif

/*****************************************************************************
 *  GLOBAL VARIABLES
 *****************************************************************************/

/* The neural network served by the translator (defined in trivfs-hooks.c) */
#include "neuron.h"

extern CompactNeuralNetwork global_network;

/* Help text (defined in trivfs-hooks.c) */
extern char *fs_help;

/* Trivfs control port placeholder for the non-Hurd test harness */
extern mach_port_t trivfs_control;

/*****************************************************************************
 *  NON-HURD TEST HARNESS
 *****************************************************************************/

#if ON_HURD != 1
/* Stub server loop: returns -1 immediately on non-Hurd systems */
extern int trivfs_server_loop(void);
#endif

#endif /* TRIVFS_HOOKS_H */

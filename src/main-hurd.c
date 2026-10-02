/*
 * main-hurd.c - Entry point for the GNU/Hurd translator
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

/** @file main-hurd.c
 *  @brief Entry point for the sigmoid neuron translator on GNU/Hurd
 *
 *  This follows the canonical trivfs translator structure (see
 *  trans/null.c in the Hurd sources).  A translator MUST:
 *
 *    1. Get the bootstrap port that settrans passed us.
 *    2. Call trivfs_startup() to reply to settrans and obtain the
 *       control port (fsys).
 *    3. Enter a server loop with ports_manage_port_operations_one_thread.
 *
 *  libtrivfs provides the demuxer (trivfs_demuxer) which dispatches
 *  incoming RPCs to our trivfs_S_* functions defined in trivfs-hooks.c.
 *  A translator that returns from main() immediately dies, and settrans
 *  reports "Translator died".
 */

#include <hurd.h>
#include <hurd/ports.h>
#include <hurd/trivfs.h>
#include <hurd/fsys.h>

#include <error.h>
#include <stdio.h>

int
main (int argc, char *argv[])
{
    error_t err;
    mach_port_t bootstrap;
    struct trivfs_control *fsys;

    /* We currently accept no command line options. */
    (void) argc;
    (void) argv;

    /* settrans gives us a bootstrap port to reply on.  Without one we
     * have not been started as a translator. */
    task_get_bootstrap_port (mach_task_self (), &bootstrap);
    if (bootstrap == MACH_PORT_NULL)
        error (1, 0, "Must be started as a translator");

    /* Reply to our parent (settrans).  Passing zeros lets libtrivfs
     * create the port classes and buckets it needs. */
    err = trivfs_startup (bootstrap, 0, 0, 0, 0, 0, &fsys);
    mach_port_deallocate (mach_task_self (), bootstrap);
    if (err)
        error (3, err, "Contacting parent");

    /* Serve RPCs until we are killed or asked to go away.  Timeout 0
     * means the loop never returns.  All messages are demultiplexed by
     * libtrivfs's trivfs_demuxer, which calls the trivfs_S_*
     * functions in trivfs-hooks.c.  A single thread serializes access
     * to the global neural network state. */
    ports_manage_port_operations_one_thread (fsys->pi.bucket,
                                              trivfs_demuxer, 0);

    return 0;
}

/*
 * main-hurd.c - Entry point for Hurd passive translators
 *
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file main-hurd.c
 *  @brief Entry point for GNU/Hurd passive translators
 *
 *  For passive translators on GNU/Hurd, the standard approach is to call
 *  trivfs_server() from main(). libtrivfs will then take over and call
 *  our fs_* functions automatically.
 *
 *  Note: trivfs_server() is provided by libtrivfs when linked with -ltrivfs.
 */

/* On Hurd, we need to provide a trivfs_demuxer function
 * which libtrivfs will call for message handling.
 * The standard approach is to call trivfs_server() from libtrivfs.
 */
extern int trivfs_server(void) __attribute__((weak));

int
main(void)
{
    /* On Hurd, call trivfs_server() from libtrivfs.
     * This starts the translator message loop.
     * If trivfs_server is not available (weak attribute), return 0.
     */
    if (trivfs_server)
        return trivfs_server();
    return 0;
}

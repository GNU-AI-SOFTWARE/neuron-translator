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

/* We need to declare trivfs_server from libtrivfs */
extern int trivfs_server(void);

int
main(void)
{
    /* On Hurd, call trivfs_server() from libtrivfs.
     * This starts the translator and handles all message dispatching.
     * It will call our fs_open, fs_read, fs_write functions automatically.
     */
    return trivfs_server();
}

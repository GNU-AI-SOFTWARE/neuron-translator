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

/* On Hurd, we need a main() that starts the trivfs server.
 * The standard approach is to call trivfs_server() from libtrivfs.
 * However, on some Hurd systems, the symbol might be named differently.
 * We use weak symbols to allow the linker to find the correct one.
 */

/* Try different possible names for the trivfs server function */
extern int trivfs_server(void) __attribute__((weak));
extern int _trivfs_server(void) __attribute__((weak));
extern int trivfs_start(void) __attribute__((weak));
extern int _hurd_trivfs_server(void) __attribute__((weak));

int
main(void)
{
    /* Try each possible trivfs server function */
    if (trivfs_server) return trivfs_server();
    if (_trivfs_server) return _trivfs_server();
    if (trivfs_start) return trivfs_start();
    if (_hurd_trivfs_server) return _hurd_trivfs_server();
    
    /* If none found, return 0 (translator won't work but won't crash) */
    return 0;
}

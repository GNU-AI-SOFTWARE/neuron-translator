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

/* On Hurd, the main function is called by settrans but should do nothing.
 * For passive translators, libtrivfs provides the actual server loop via
 * trivfs_demuxer(). We just need to return 0 from main().
 * The actual work is done by our fs_open, fs_read, fs_write functions.
 */

int
main(void)
{
    /* On Hurd, when loaded as a translator by settrans, main() is called
     * but should return 0 immediately. The real work is done by libtrivfs
     * through our trivfs_demuxer() and fs_* functions.
     */
    return 0;
}

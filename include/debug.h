/*
 * debug.h - Debug Logging Helper
 *
 * Copyright (C) 2026 GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
 */

#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

/**
 * log_debug_message - Write debug message to file or stderr
 * Tries multiple file locations, falls back to stderr
 *
 * @param message The debug message to log
 */
static inline void log_debug_message(const char *message)
{
    FILE *logfile = NULL;
    const char *paths[] = {
        "/home/claire/translator_debug.log",
        "/tmp/translator_debug.log",
        "/var/log/translator_debug.log",
        NULL
    };
    for (int i = 0; paths[i] != NULL; i++) {
        logfile = fopen(paths[i], "a");
        if (logfile) break;
    }
    if (!logfile) logfile = stderr;
    
    fprintf(logfile, "%s\n", message);
    if (logfile != stderr) {
        fflush(logfile);
        fclose(logfile);
    }
    fflush(stderr);
}

#endif /* DEBUG_H */

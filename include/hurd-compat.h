/*
 * hurd-compat.h - Compatibility definitions for Hurd types
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

/** @file hurd-compat.h
 *  @brief Compatibility definitions for Hurd-specific types
 *  This header provides Hurd type definitions for systems that don't have
 *  the actual Hurd headers installed. On GNU/Hurd systems with proper headers,
 *  the system headers take precedence.
 */

#ifndef HURD_COMPAT_H
#define HURD_COMPAT_H

#include <sys/types.h>

/*****************************************************************************
 *  HURD TYPE COMPATIBILITY
 *****************************************************************************/

/* On GNU/Hurd, use system headers if available */
#if __has_include(<hurd/iohelp.h>)
#include <hurd/iohelp.h>
#include <hurd/trivfs.h>
#elif __has_include(<mach.h>)
#include <mach.h>
#include <mach/port.h>
#include <mach/message.h>
#else
/* Non-Hurd systems or Hurd headers not available */

/*****************************************************************************
 *  BASIC MACH TYPES
 *****************************************************************************/

typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)

struct mach_msg_header;
typedef struct mach_msg_header mach_msg_header_t;

/*****************************************************************************
 *  HURD FILESYSTEM TYPES
 *****************************************************************************/

struct iouser;
struct node;
struct iobuf;

/* For non-Hurd systems, we need complete definitions to access iobuf->buf */
struct iouser {
    int uid;
    int gid;
    int *uids;
    int *gids;
    int nuids;
    int ngids;
};

struct node {
    void *data;
};

struct iobuf {
    char *buf;
    size_t size;
    off_t offset;
};

/*****************************************************************************
 *  ERROR TYPE
 *****************************************************************************/

typedef int error_t;

#endif /* __has_include(<hurd/iohelp.h>) */

#endif /* HURD_COMPAT_H */

/*
 * main-hurd.c - Entry point for the GNU/Hurd translator
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

#include <argp.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>

/* Trivfs gereksinimleri: Varsayılan port sınıfları ve kontrol yapıları */
struct trivfs_control *fsys;

const char *argp_program_version = "sigmoid-neuron-0.1";
const char *argp_program_bug_address = "<claire@gnu-ai.org>";
static char doc[] = "GNU/Hurd sigmoid neuron translator.";

/* Komut satırı opsiyonları */
static struct argp_option options[] = {
  {"bias", 'b', "FLOAT", 0, "Initial bias value for the neuron", 0},
  { 0 }
};

/* Nöron başlangıç parametreleri */
float global_bias = 0.0f;

static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  (void) state;                 /* Unused: no ARGP_KEY_ handling needs it */

  switch (key)
    {
    case 'b':
      global_bias = atof (arg);
      break;
    case ARGP_KEY_SUCCESS:
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp argp = { options, parse_opt, 0, doc, 0, 0, 0 };

int
main (int argc, char *argv[])
{
  error_t err;
  mach_port_t bootstrap;

  /* Argümanları ayrıştır */
  argp_parse (&argp, argc, argv, 0, 0, 0);

  /* Parent (settrans) tarafından sağlanan bootstrap portunu al */
  task_get_bootstrap_port (mach_task_self (), &bootstrap);
  if (bootstrap == MACH_PORT_NULL)
    error (1, 0, "Must be started as a translator");

  /* Trivfs sunucusunu başlat */
  err = trivfs_startup (bootstrap, 0, 0, 0, 0, 0, &fsys);
  mach_port_deallocate (mach_task_self (), bootstrap);
  if (err)
    error (3, err, "Contacting parent failed");

  /* RPC isteklerini dinleme döngüsüne gir */
  ports_manage_port_operations_one_thread (fsys->pi.bucket,
                                            trivfs_demuxer, 0);

  return 0;
}

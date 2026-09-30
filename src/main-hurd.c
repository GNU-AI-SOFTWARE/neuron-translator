/*
 * main-hurd.c - Minimal main stub for GNU/Hurd passive translators
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
 *  @brief Minimal main() for GNU/Hurd passive translators
 *
 *  For passive translators on GNU/Hurd, libtrivfs provides the real entry point.
 *  This minimal main() satisfies the linker's requirement for a main symbol.
 *  On Hurd, when loaded by settrans, this function will NOT be called.
 *  libtrivfs will provide its own entry point that calls our fs_* functions.
 */

#include <stdio.h>

int main(void)
{
    /* This should never be reached when loaded as a translator by settrans.
     * If it is called, it means the translator is being run directly,
     * which is not the intended use for Hurd translators.
     */
    return 0;
}

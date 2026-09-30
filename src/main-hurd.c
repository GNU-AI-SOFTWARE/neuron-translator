/*
 * main-hurd.c - Stub main() for Hurd passive translators
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
 *  @brief Stub main() function for GNU/Hurd passive translators
 *
 *  For passive translators on GNU/Hurd, libtrivfs provides the real entry point.
 *  However, the linker still requires a main() symbol. This stub satisfies that
 *  requirement while allowing libtrivfs to handle the actual translator execution.
 */

#include <stdio.h>

/**
 * @brief Stub main function for Hurd passive translators
 *
 * On GNU/Hurd, this function should never actually be called when the
 * translator is loaded by settrans. libtrivfs provides its own entry point.
 * This stub exists only to satisfy the linker's requirement for a main symbol.
 *
 * @return Always returns 0
 */
int main(void)
{
    /* This should never be reached when loaded as a translator by settrans.
     * If it is called, it means the translator is being run directly,
     * which is not the intended use.
     */
    return 0;
}

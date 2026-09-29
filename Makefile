# Makefile for LLM Sigmoid Neuron Translator - GNU Hurd
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>
#
# This Makefile builds the modular version of the sigmoid neuron translator
# for GNU Hurd using the following structure:
#   include/ - Header files
#   src/     - Source files


/*****************************************************************************
 *                                                                           *
 *                         COMPILER CONFIGURATION                             *
 *                                                                           *
 *****************************************************************************/

# Compiler
CC = gcc

# C23 Standard, POSIX compliant, with all warnings and optimizations
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L

# Linker flags
LDFLAGS =

# Target binary
TARGET = sigmoid-neuron-translator

# Source files (modular structure)
SRCS = \
	src/main.c \
	src/neuron.c \
	src/trivfs-hooks.c

# Object files
OBJS = $(SRCS:.c=.o)

# Include directory
INCLUDES = -Iinclude

# Hurd and Mach libraries
LIBS = -lm -lpthread -ltrivfs -lhurdfs -lports -lshouldbeinlibc

# Installation directory
INSTALL_DIR = /hurd


/*****************************************************************************
 *                                                                           *
 *                           BUILD RULES                                   *
 *                                                                           *
 *****************************************************************************/

# Default target: build the translator
all: $(TARGET)

# Link all object files into final binary
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile each source file with includes
src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


/*****************************************************************************
 *                                                                           *
 *                          INSTALLATION RULES                              *
 *                                                                           *
 *****************************************************************************/

# Install to /hurd/
install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use: sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"

# Uninstall
uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"


/*****************************************************************************
 *                                                                           *
 *                          UTILITY RULES                                  *
 *                                                                           *
 *****************************************************************************/

# Clean all build artifacts
clean:
	rm -f $(TARGET) $(OBJS) *~ *.o

# Show build information
help:
	@echo "LLM Sigmoid Neuron Translator for GNU Hurd - Modular Version"
	@echo ""
	@echo "Build targets:"
	@echo "  make          - Build the translator"
	@echo "  make install  - Install to /hurd/"
	@echo "  make uninstall - Remove from /hurd/"
	@echo "  make clean    - Clean build artifacts"
	@echo ""

.PHONY: all install uninstall clean help

# Makefile for LLM Sigmoid Neuron Translator - GNU Hurd Only
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>

# Project structure:
#   include/ - Header files
#   src/     - Source files
#
# This modular structure simplifies future evolution and maintenance.


/*****************************************************************************
 *                                                                           *
 *                         COMPILER CONFIGURATION                             *
 *                                                                           *
 *****************************************************************************/

# Compiler
CC = gcc

# Compiler flags: C23 standard, POSIX compliant, with all warnings enabled
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L

# Linker flags
LDFLAGS =

# Target binary name
TARGET = sigmoid-neuron-translator

# Source files
SRCS = \
	src/main.c \
	src/neuron.c \
	src/trivfs-hooks.c

# Object files
OBJS = $(SRCS:.c=.o)

# Include directories
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

# Default target
all: $(TARGET)

# Link all object files
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile each source file
src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


/*****************************************************************************
 *                                                                           *
 *                          INSTALLATION RULES                              *
 *                                                                           *
 *****************************************************************************/

# Install the translator
install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use: sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"

# Uninstall the translator
uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"


/*****************************************************************************
 *                                                                           *
 *                          UTILITY RULES                                  *
 *                                                                           *
 *****************************************************************************/

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJS) *~ *.o

# Clean everything including old files
clean-all:
	rm -f $(TARGET) $(OBJS) *~ *.o
	rm -rf include src

# Show help
default: help

help:
	@echo "LLM Sigmoid Neuron Translator for GNU Hurd"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the translator"
	@echo "  make install  - Install to /hurd/"
	@echo "  make uninstall - Remove from /hurd/"
	@echo "  make clean    - Clean build artifacts"
	@echo ""

.PHONY: all install uninstall clean clean-all default help

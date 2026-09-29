# Makefile for LLM Sigmoid Neuron Translator for GNU Hurd
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>
#
# This Makefile provides targets for building, installing, and cleaning
# the sigmoid neuron translator for GNU Hurd.


/*****************************************************************************
 *                                                                           *
 *                         COMPILER CONFIGURATION                           *
 *                                                                           *
 *****************************************************************************/

# Compiler and flags
CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS = 

# Target executable name
TARGET = sigmoid-neuron-translator

# Source file
SRC = sigmoid-neuron-translator.c

# Installation directory for Hurd translators
# Translators are typically installed in /hurd/ directory on Hurd systems
INSTALL_DIR = /hurd


/*****************************************************************************
 *                                                                           *
 *                         PLATFORM DETECTION                               *
 *                                                                           *
 *****************************************************************************/

# Detect if we're on Hurd or Linux
# On Hurd: uname -s returns "GNU"
# On Linux: uname -s returns "Linux"
UNAME := $(shell uname -s)

# On Hurd, link with Hurd-specific libraries. On Linux, only link with math and pthread.
ifeq ($(UNAME),GNU)
HURD_LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc
else
HURD_LIBS =
endif

# Standard libraries (math and pthread are needed on both platforms)
STD_LIBS = -lm -lpthread

# All libraries
LIBS = $(STD_LIBS) $(HURD_LIBS)


/*****************************************************************************
 *                                                                           *
 *                              TARGETS                                    *
 *                                                                           *
 *****************************************************************************/

# Default target - build the translator
all: $(TARGET)

# Build target - compile the translator
$(TARGET): $(SRC)
	@echo "Building sigmoid neuron translator..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# Install the translator to /hurd directory (Hurd only)
install: $(TARGET)
ifeq ($(UNAME),GNU)
	@echo "Installing translator to $(INSTALL_DIR)/..."
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo ""
	@echo "To use the translator:"
	@echo "  sudo mkdir -p /llm"
	@echo "  sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"
	@echo ""
	@echo "To test the translator:"
	@echo "  cat /llm"
	@echo "  echo '10,20,5' > /llm"
	@echo "  echo '0.5,0.3,0.8,0.1,0.9,0.2,0.4,0.6,0.0,0.7' > /llm"
else
	@echo "This target is for GNU/Hurd only."
	@echo "On Linux, just run: ./$(TARGET)"
endif

# Uninstall the translator
uninstall:
ifeq ($(UNAME),GNU)
	@echo "Removing translator from $(INSTALL_DIR)/..."
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"
else
	@echo "This target is for GNU/Hurd only."
endif

# Clean build artifacts
clean:
	rm -f $(TARGET) *~ *.o
	@echo "Clean complete"

# Show usage
help:
	@echo "LLM Sigmoid Neuron Translator for GNU Hurd"
	@echo "==========================================="
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Build the translator"
	@echo "  make install   - Install to $(INSTALL_DIR)/ (Hurd only)"
	@echo "  make uninstall - Remove from $(INSTALL_DIR)/ (Hurd only)"
	@echo "  make clean     - Clean build artifacts"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "On GNU/Hurd:"
	@echo "  make && sudo make install"
	@echo "  sudo settrans -c /llm /hurd/sigmoid-neuron-translator"
	@echo ""
	@echo "On GNU/Linux (test mode):"
	@echo "  make && ./sigmoid-neuron-translator"

# Show version
version:
	@echo "LLM Sigmoid Neuron Translator"
	@echo "Version: 1.0.0"
	@echo "Author: Claire <claire@gnu-ai.org>"
	@echo "License: GNU GPLv3"


/*****************************************************************************
 *                                                                           *
 *                              PHONY TARGETS                               *
 *                                                                           *
 *****************************************************************************/

.PHONY: all install uninstall clean help version

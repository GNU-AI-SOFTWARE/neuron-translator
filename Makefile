# Makefile for LLM Sigmoid Neuron Translator for GNU Hurd
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>

# Compiler and flags
CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS = 

# Target executable name
TARGET = sigmoid-neuron-translator

# Source file
SRC = sigmoid-neuron-translator.c

# Installation directory for Hurd translators
INSTALL_DIR = /hurd

# Detect if we're on Hurd or Linux
UNAME := $(shell uname -s)

ifeq ($(UNAME),GNU)
HURD_LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc
else
HURD_LIBS =
endif

STD_LIBS = -lm -lpthread
LIBS = $(STD_LIBS) $(HURD_LIBS)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	@echo "Building sigmoid neuron translator..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo "Build complete: $@"

# Install the translator
install: $(TARGET)
ifeq ($(UNAME),GNU)
	@echo "Installing translator to $(INSTALL_DIR)/..."
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo ""
	@echo "To use the translator:"
	@echo "  sudo mkdir -p /llm"
	@echo "  sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"
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

# Show version
version:
	@echo "LLM Sigmoid Neuron Translator"
	@echo "Version: 1.0.0"
	@echo "Author: Claire <claire@gnu-ai.org>"
	@echo "License: GNU GPLv3"

.PHONY: all install uninstall clean help version

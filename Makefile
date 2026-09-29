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
# On Hurd: uname -s returns "GNU"
# On Linux: uname -s returns "Linux"
UNAME := $(shell uname -s)

# On Hurd, link with Hurd libraries. On Linux, only link with math and pthread.
ifeq ($(UNAME),GNU)
HURD_LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc
else
HURD_LIBS =
endif

# Libraries
LIBS = -lm -lpthread $(HURD_LIBS)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# Install the translator to /hurd directory (Hurd only)
install: $(TARGET)
ifeq ($(UNAME),GNU)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use the translator:"
	@echo "  sudo mkdir -p /llm"
	@echo "  sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"
	@echo "  cat /llm"
else
	@echo "This target is for GNU/Hurd only."
	@echo "On Linux, just run: ./$(TARGET)"
endif

# Uninstall the translator
uninstall:
ifeq ($(UNAME),GNU)
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"
else
	@echo "This target is for GNU/Hurd only."
endif

# Clean build artifacts
clean:
	rm -f $(TARGET) *~ *.o

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

.PHONY: all install uninstall clean help

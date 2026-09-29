# Makefile for LLM Sigmoid Neuron Translator for GNU Hurd
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>

# Compiler and flags
CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O3 -march=native \
	-D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
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
# We are on GNU/Hurd
HURD_LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc
else
# We are on GNU/Linux or other - use stub implementations
HURD_LIBS =
endif

# Libraries (common + Hurd-specific)
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
	@echo ""
	@echo "To use the translator:"
	@echo "  1. Create mount point: sudo mkdir -p /llm"
	@echo "  2. Set translator: sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"
	@echo "  3. Test it: cat /llm"
	@echo "  4. Configure: echo '5,10,5' > /llm"
	@echo "  5. Provide input: echo '0.5,0.3,0.8,0.1,0.9' > /llm"
else
	@echo "Install target is for GNU/Hurd only. On Linux, just run: ./$(TARGET)"
endif

# Uninstall the translator
uninstall:
ifeq ($(UNAME),GNU)
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"
else
	@echo "Uninstall target is for GNU/Hurd only"
endif

# Clean build artifacts
clean:
	rm -f $(TARGET) *~ *.o

# Build and install (Hurd only)
build-install: clean all install

# Show usage
help:
	@echo "LLM Sigmoid Neuron Translator for GNU Hurd"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the translator"
	@echo "  make install  - Install to $(INSTALL_DIR)/ (Hurd only)"
	@echo "  make uninstall - Remove from $(INSTALL_DIR)/ (Hurd only)"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "On GNU/Hurd:"
	@echo "  After installing, set translator with:"
	@echo "  sudo settrans -c /llm /hurd/sigmoid-neuron-translator"
	@echo ""
	@echo "On GNU/Linux (test mode):"
	@echo "  Just run: ./sigmoid-neuron-translator"

.PHONY: all install uninstall clean help build-install

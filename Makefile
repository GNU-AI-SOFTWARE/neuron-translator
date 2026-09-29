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

# Libraries required for Hurd translators:
#   - ltrivfs: Trivial filesystem library
#   - lhurdfs: Hurd filesystem support
#   - lports: Mach port management
#   - lshouldbeinlibc: Functions that should be in libc
#   - lm: Math library (for expf, etc.)
#   - lpthread: POSIX threads (required by Hurd)
LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc -lm -lpthread

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# Install the translator to /hurd directory
install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo ""
	@echo "To use the translator:"
	@echo "  1. Create mount point: sudo mkdir -p /llm"
	@echo "  2. Set translator: sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"
	@echo "  3. Test it: cat /llm"
	@echo "  4. Configure: echo '5,10,5' > /llm"
	@echo "  5. Provide input: echo '0.5,0.3,0.8,0.1,0.9' > /llm"

# Uninstall the translator
uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"

# Clean build artifacts
clean:
	rm -f $(TARGET) *~ *.o

# Build and install
build-install: clean all install

# Show usage
help:
	@echo "LLM Sigmoid Neuron Translator for GNU Hurd"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the translator"
	@echo "  make install  - Install to $(INSTALL_DIR)/"
	@echo "  make uninstall - Remove from $(INSTALL_DIR)/"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "After installing, set translator with:"
	@echo "  sudo settrans -c /llm /hurd/sigmoid-neuron-translator"

.PHONY: all install uninstall clean help build-install

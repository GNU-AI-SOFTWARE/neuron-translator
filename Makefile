# Makefile for LLM Sigmoid Neuron Translator - GNU Hurd Only
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>

# Compiler and flags
CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS = 

# Target
TARGET = sigmoid-neuron-translator
SRC = sigmoid-neuron-translator.c

# Hurd libraries
LIBS = -lm -lpthread -ltrivfs -lhurdfs -lports -lshouldbeinlibc

# Installation directory
INSTALL_DIR = /hurd

# Build
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# Install
install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use: sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"

# Uninstall
uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"

# Clean
clean:
	rm -f $(TARGET) *~ *.o

.PHONY: all install uninstall clean

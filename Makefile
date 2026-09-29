# Makefile for LLM Sigmoid Neuron Translator - GNU Hurd
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>

CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS =
TARGET = sigmoid-neuron-translator
SRCS = src/main.c src/neuron.c src/trivfs-hooks.c
OBJS = $(SRCS:.c=.o)
INCLUDES = -Iinclude

# System detection: Detect if we can link Hurd libraries
# On GNU/Hurd: uname -s returns "GNU"
# On Linux: uname -s returns "Linux"
# We always compile; we only link on Hurd where libraries exist
UNAME := $(shell uname -s | head -n1 | tr -d '\n')

# Hurd libraries - only available on GNU/Hurd
LIBS = -lm -lpthread -ltrivfs -lfshelp -lports -lshouldbeinlibc

# Primary target: always compile
all: compile

compile: $(OBJS)
	@echo "Compilation successful. All .o files generated."

# Secondary target: link only on Hurd
$(TARGET): $(OBJS)
	@echo "Linking with Hurd libraries..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

# On Hurd systems, also make the executable when running 'make'
ifeq ($(UNAME),GNU)
all: $(TARGET)
endif

INSTALL_DIR = /hurd

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use: sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"

uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"

clean:
	rm -f $(TARGET) $(OBJS) *~ *.o

help:
	@echo "LLM Sigmoid Neuron Translator for GNU Hurd - Modular Version"
	@echo ""
	@echo "Build targets:"
	@echo "  make          - Build the translator (requires Hurd libraries)"
	@echo "  make compile  - Compile only (no linking, for Debian/Linux)"
	@echo "  make install  - Install to /hurd/"
	@echo "  make uninstall - Remove from /hurd/"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "Note: Full build with linking requires GNU/Hurd system or"
	@echo "      Hurd development libraries. On Debian/Linux, use 'make compile'."

.PHONY: all install uninstall clean help compile

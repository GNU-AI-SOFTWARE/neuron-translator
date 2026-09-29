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

# This Makefile compiles the sigmoid neuron translator for GNU/Hurd
# On any system: 'make' compiles the object files
# On GNU/Hurd: 'make link' or 'make install' performs full build with linking
# Note: Hurd libraries are only available on GNU/Hurd systems

# Compiler and flags
CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS =
TARGET = sigmoid-neuron-translator
SRCS = src/main.c src/neuron.c src/trivfs-hooks.c
OBJS = $(SRCS:.c=.o)
INCLUDES = -Iinclude

# Default target: compile only (works on all systems)
all: compile

compile: $(OBJS)
	@echo "Compilation successful. All .o files generated."
	@echo "On GNU/Hurd, use 'make link' to build the executable."

# Linking target (requires Hurd libraries)
link: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking with Hurd libraries..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lpthread -ltrivfs -lfshelp -lports -lshouldbeinlibc

INSTALL_DIR = /hurd

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

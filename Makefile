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

# System detection for library selection
# On GNU/Hurd, these libraries are available and required
# On other systems (Debian/Linux), they may not be available
UNAME := $(strip $(shell uname -s))

# Hurd libraries - only link on GNU/Hurd systems
# Standard Hurd libraries: libtrivfs, libfshelp, libports, libshouldbeinlibc
# Note: libhurdfs does NOT exist in standard GNU/Hurd - it was likely a confusion
#       with the Hurd File System. We use the correct standard libraries instead.
# On Debian/Linux, use 'make compile' to only compile without linking.
ifeq ($(UNAME),Linux)
    # On Linux, we can only compile the object files without linking
    # The full build requires Hurd libraries which are not available
    LIBS =
else
    # On GNU/Hurd or other systems, use Hurd libraries
    # Standard Hurd libraries: trivfs, fshelp, ports, shouldbeinlibc
    LIBS = -lm -lpthread -ltrivfs -lfshelp -lports -lshouldbeinlibc
endif

INSTALL_DIR = /hurd

# On Linux, 'all' just compiles without linking (no Hurd libraries)
# On Hurd, 'all' does full build with linking
ifeq ($(UNAME),Linux)
all: compile
else
all: $(TARGET)
$(TARGET): $(OBJS)
	@echo "Linking with Hurd libraries..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)
endif

# Compile-only target for systems without Hurd libraries (e.g., Debian)
compile: $(OBJS)
	@echo "Compilation successful. All .o files generated."
	@echo "Note: On Debian/Linux, use 'make compile' or 'make' to compile without linking."
	@echo "      On GNU/Hurd, use 'make' for full build with linking."

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

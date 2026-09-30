# Makefile for LLM Sigmoid Neuron Translator - GNU Hurd
# Copyright (C) 2026 GNU AI Project
# Author: Claire <claire@gnu-ai.org>
#
# This Makefile compiles the translator source files.
# - On any system: 'make' or 'make compile' compiles object files (C23/POSIX compliant)
# - On GNU/Hurd: 'make executable' links with Hurd libraries to create the translator
#
# Note: Hurd libraries (-ltrivfs, -lfshelp, -lports, -lshouldbeinlibc) are only
#       available on GNU/Hurd systems. On Debian/Linux, only compilation works.
#
# Debug symbols are enabled by default via -DDEBUG flag

CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -DDEBUG
LDFLAGS =
TARGET = sigmoid-neuron-translator

# Check if this is a Hurd system (uname -s returns "GNU" on Hurd)
UNAME_S := $(shell uname -s 2>/dev/null)
HURD_SYSTEM := $(shell echo "$(UNAME_S)" | grep -q GNU && echo yes || echo no)

# Debug: Show which system we detected
$(info Detected system: $(UNAME_S) -> Hurd=$(HURD_SYSTEM))

# Define ON_HURD macro and library flags for Hurd systems
ifeq ($(HURD_SYSTEM),yes)
CFLAGS += -DON_HURD
LDFLAGS += -ltrivfs -lfshelp -lports -lshouldbeinlibc
$(info Building for GNU/Hurd - ON_HURD defined)
else
$(info Building for $(UNAME_S) - ON_HURD not defined)
endif

# For Hurd translators, we use a passive translator approach:
# - On Hurd: No main() needed (libtrivfs provides it, -nostartfiles used)
# - On non-Hurd: Use main.c for testing/compilation
# The actual work is done by our fs_open, fs_read, fs_write functions

# Common source files
COMMON_SRCS = src/neuron.c src/trivfs-hooks.c

# On Hurd, no main.c needed; on other systems, include main.c
ifeq ($(HURD_SYSTEM),yes)
SRCS = $(COMMON_SRCS)
$(info Building for GNU/Hurd - Passive translator (no main, using -nostartfiles))
else
SRCS = src/main.c $(COMMON_SRCS)
$(info Building for $(UNAME_S) - Including main.c for non-Hurd)
endif

OBJS = $(SRCS:.c=.o)
INCLUDES = -Iinclude

# Default target: compile only (works on all systems)
all: compile

compile: $(OBJS)
	@echo "Compilation successful. All .o files generated."
	@echo "On GNU/Hurd, use 'make executable' to build the translator."

# Executable target - only works on Hurd systems with libraries
executable: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking..."
	@echo "Note: If you get 'cannot find -lhurdsig' or similar, you need to install Hurd development libraries"
	ifeq ($(HURD_SYSTEM),yes)
		$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lpthread -Wl,-nostartfiles
	else
		$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ -lm -lpthread
	endif

INSTALL_DIR = /hurd

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

install: $(TARGET)
	@echo "WARNING: This translator is designed for GNU/Hurd, not Debian/Linux!"
	@echo "On Debian/Linux, you can only compile with 'make compile'"
	@echo "On GNU/Hurd, proceed with installation..."
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use: sudo settrans -a /llm $(INSTALL_DIR)/$(TARGET)"
	@echo "NOTE: This will only work on a running GNU/Hurd system!"
	@echo "       Use -a flag for automatic startup, -c for manual startup"

uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"

clean:
	rm -f $(TARGET) $(OBJS) *~ *.o

# Minimal test translator - for debugging
minimal: src/translator-minimal.o

src/translator-minimal.o: src/translator-minimal.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

minimal-translator: src/translator-minimal.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

install-minimal: minimal-translator
	install -m 755 minimal-translator /hurd/
	@echo "Minimal translator installed to /hurd/minimal-translator"

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

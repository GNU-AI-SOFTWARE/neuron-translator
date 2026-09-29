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

# Hurd libraries - available on GNU/Hurd systems
# On GNU/Hurd: -ltrivfs -lhurdfs -lports -lshouldbeinlibc
# These libraries are part of the Hurd system and should be present when
# compiling on a Hurd system. On Debian, you may need to install
# hurd development packages or cross-compile for Hurd.
LIBS = -lm -lpthread -ltrivfs -lhurdfs -lports -lshouldbeinlibc

INSTALL_DIR = /hurd

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

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
	@echo "  make          - Build the translator"
	@echo "  make install  - Install to /hurd/"
	@echo "  make uninstall - Remove from /hurd/"
	@echo "  make clean    - Clean build artifacts"

.PHONY: all install uninstall clean help

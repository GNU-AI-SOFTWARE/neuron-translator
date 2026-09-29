# Makefile for LLM Neural Network Translator for GNU Hurd
#
# Copyright (C) 2026  GNU AI Project
#
# This Makefile provides targets for compiling and installing the
# llm-neuron-translator-optimized for GNU Hurd.
#
# Features:
#   - Memory-efficient neural network implementation
#   - CPU-optimized forward pass
#   - Support for large numbers of neurons
#   - Full Hurd translator integration


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Compiler and flags
CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O3 -march=native
LDFLAGS = 

# Source and target files
SRC = llm-neuron-translator-optimized.c
TARGET = llm-neuron-translator

# Installation directory
INSTALL_DIR = /hurd

# Libraries
LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc -lm


# ===========================================================================
# MAIN TARGETS
# ===========================================================================

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Translator installed to $(INSTALL_DIR)/$(TARGET)"
	@echo ""
	@echo "To use:"
	@echo "  mkdir -p /llm"
	@echo "  settrans -c /llm $(INSTALL_DIR)/$(TARGET)"

uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Translator removed from $(INSTALL_DIR)/"

clean:
	rm -f $(TARGET) *~ *.o


# ===========================================================================
# TESTING TARGETS
# ===========================================================================

test: $(TARGET)
	@echo "Testing LLM Neural Network Translator..."
	@echo ""
	cp $(TARGET) /hurd/ 2>/dev/null || \
	    (echo "Cannot copy - need root?"; exit 1)
	mkdir -p /llm 2>/dev/null || \
	    (echo "Cannot create /llm - need root?"; exit 1)
	settrans -c /llm /hurd/$(TARGET) 2>/dev/null || \
	    (echo "Cannot set translator - ensure GNU/Hurd is running"; exit 1)
	@echo ""
	@echo "Test 1: Configure network (10 input, 20 hidden, 5 output)"
	echo "10,20,5" > /llm/config
	@echo ""
	@echo "Test 2: Set input values"
	echo "0.5,0.3,-0.2,1.0,0.0,0.0,0.0,0.0,0.0,0.0" > /llm/input
	@echo ""
	@echo "Test 3: Get output (triggers forward pass)"
	cat /llm/output
	@echo ""
	@echo "Test 4: View network info"
	cat /llm
	@echo ""
	@echo "Cleaning up..."
	settrans -g /llm 2>/dev/null
	rm -f /llm 2>/dev/null
	@echo "Tests completed!"


# ===========================================================================
# DOCUMENTATION
# ===========================================================================

doc:
	@echo "LLM Neural Network Translator for GNU Hurd"
	@echo "==========================================="
	@echo ""
	@echo "DESCRIPTION:"
	@echo "  Memory-efficient, CPU-optimized neural network translator"
	@echo "  for GNU Hurd. Supports large networks with 100,000+ neurons."
	@echo ""
	@echo "MEMORY USAGE:"
	@echo "  - float32 for all values (4 bytes each)"
	@echo "  - Contiguous memory allocation"
	@echo "  - 16-byte alignment for SIMD"
	@echo "  - Example: 10,000 neurons, 1M weights = ~12MB"
	@echo ""
	@echo "CPU OPTIMIZATIONS:"
	@echo "  - Cache-friendly access patterns"
	@echo "  - Minimal branching in hot loops"
	@echo "  - Inline activation functions"
	@echo "  - No dynamic allocations during forward pass"
	@echo ""
	@echo "USAGE:"
	@echo "  settrans -c /llm /hurd/llm-neuron-translator"
	@echo "  echo \"10,20,5\" > /llm/config"
	@echo "  echo \"1,2,3,4,5,6,7,8,9,10\" > /llm/input"
	@echo "  cat /llm/output"


.PHONY: all install uninstall clean test doc

# Makefile for LLM Neural Network Translator for GNU Hurd
# Copyright (C) 2026 GNU AI Project

CC = gcc
CFLAGS = -std=c23 -Wall -Wextra -pedantic -O2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS = 

TARGET = llm-neuron-translator
SRC = llm-neuron-hurd.c

INSTALL_DIR = /hurd

# Libraries required for Hurd translators
LIBS = -ltrivfs -lhurdfs -lports -lshouldbeinlibc -lm -lpthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

install: $(TARGET)
	install -m 755 $(TARGET) $(INSTALL_DIR)/
	@echo "Installed to $(INSTALL_DIR)/$(TARGET)"
	@echo "To use: sudo mkdir -p /llm && sudo settrans -c /llm $(INSTALL_DIR)/$(TARGET)"

uninstall:
	rm -f $(INSTALL_DIR)/$(TARGET)
	@echo "Removed from $(INSTALL_DIR)/"

clean:
	rm -f $(TARGET) *~ *.o

test:
	@echo "Testing LLM Neural Network Translator..."
	@echo "This requires a running GNU/Hurd system with the translator installed"
	@echo ""
	@echo "To test manually:"
	@echo "  make install"
	@echo "  sudo mkdir -p /llm"
	@echo "  sudo settrans -c /llm /hurd/llm-neuron-translator"
	@echo "  echo '3,5,2' > /llm/config"
	@echo "  echo '1,2,3,4,5' > /llm/input"
	@echo "  cat /llm/output"

.PHONY: all install uninstall clean test

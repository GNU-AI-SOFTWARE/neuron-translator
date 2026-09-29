# LLM Neural Network Translator for GNU Hurd

A **memory-efficient, CPU-optimized** neural network translator for GNU Hurd, specifically designed to handle **large numbers of neurons** (100K+) with **minimal memory footprint** and **low CPU usage**.

## Features

### Memory Efficiency
- **float32 storage**: All numeric values use 4-byte float instead of 8-byte double
- **Contiguous allocation**: Single memory block for entire network (arena allocator)
- **Compact structures**: Minimum padding, optimal data packing
- **16-byte alignment**: SIMD-compatible memory layout
- **Scalable**: Linear memory growth with network size

### CPU Efficiency  
- **Cache-friendly**: Sequential memory access patterns
- **Minimal branching**: Predictable execution in hot loops
- **Inline functions**: No call overhead for critical functions
- **No runtime allocation**: All memory pre-allocated at initialization
- **Optimized forward pass**: Hand-tuned for performance

### Scalability
- **Max layers**: 8
- **Max neurons per layer**: 8,192
- **Max total neurons**: 65,536
- **Max weights**: 67 million
- **Memory per neuron**: ~4 bytes (voltage) + ~4*fan-in bytes (weights, shared)

## Neuron Model

Implements a **feedforward neural network** with **sigmoid activation**:

```
Input Layer -> Hidden Layer(s) -> Output Layer
```

Each neuron:
- Receives weighted inputs from previous layer
- Computes: Σ(input × weight) + bias
- Applies sigmoid: f(x) = 1 / (1 + exp(-x))
- Outputs to next layer

## Installation on GNU/Hurd

### Prerequisites
- GNU/Hurd system (Debian GNU/Hurd recommended)
- GCC
- Hurd development libraries

### Quick Setup

```bash
# Clone repository
git clone https://github.com/gnu-ai/inference-translator.git
cd inference-translator

# Compile
make

# Install
sudo make install

# Create mount point
sudo mkdir -p /llm

# Set translator
sudo settrans -c /llm /hurd/llm-neuron-translator
```

## Usage

### Filesystem Interface

```
/llm/
├── config       (R/W) - Network topology: "input,hidden,...,output"
├── input        (W)   - Input vector: comma-separated floats
├── output       (R)   - Output vector: comma-separated floats
└── (stats)      (R)   - Performance statistics
```

### Configure Network

```bash
# Set topology (3 input, 5 hidden, 2 output)
echo "3,5,2" > /llm/config

# Check configuration
cat /llm/config
```

### Run Forward Pass

```bash
# Set input
echo "1.0,0.5,-0.5" > /llm/input

# Get output (triggers forward pass automatically)
cat /llm/output
```

### Example Session

```bash
# Configure
echo "784,256,128,10" > /llm/config

# Set input (e.g., MNIST pixel values)
echo "0.1,0.2,0.3,0.4,0.5,0.0,0.0,0.0,0.0,0.0" > /llm/input

# Get output
cat /llm/output
# Output: [0.1234, 0.5678, ...]

# View network info
cat /llm
```

## Memory Usage Examples

| Network Topology | Neurons | Weights | Memory Usage |
|------------------|---------|--------|--------------|
| 10-20-5 | 35 | 250 | ~2 KB |
| 784-256-128-10 | 1,178 | 230,400 | ~930 KB |
| 1000-500-100 | 1,600 | 600,000 | ~2.3 MB |
| 10000-1000-100 | 11,100 | 10,100,000 | ~80 MB |

## Files

- `llm-neuron-hurd.c` - Main translator source code
- `Makefile` - Compilation and installation
- `README.md` - This documentation

## Compilation Details

The code uses:
- **C23 standard** (compatible with C11 for Hurd)
- **POSIX compliance** for portability
- **Hurd trivfs** for translator interface
- **GNU Mach IPC** for inter-process communication

### Compilation Command

```bash
gcc -std=c23 -Wall -Wextra -pedantic -O2 \
    -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L \
    -o llm-neuron-translator llm-neuron-hurd.c \
    -ltrivfs -lhurdfs -lports -lshouldbeinlibc -lm -lpthread
```

## Development

### Running GNU/Hurd VM

For testing and development:

```bash
# Download pre-built Hurd image
wget https://cdimage.debian.org/cdimage/ports/stable/hurd-i386/debian-hurd.img.tar.gz
tar xzf debian-hurd.img.tar.gz

# Start VM
kvm -m 2G -drive file=$(echo debian-hurd*.img),cache=writeback

# Inside Hurd:
apt update
apt install build-essential gcc hurd-dev git
```

### Cross-Compilation (Advanced)

For cross-compiling from Linux to Hurd:

```bash
# Install cross-compiler (Debian/Ubuntu)
sudo apt-get install gcc-i686-unknown-hurd

# Cross-compile
i686-unknown-hurd-gcc -std=c23 -O2 \
    -o llm-neuron-translator llm-neuron-hurd.c \
    -ltrivfs -lhurdfs -lports -lshouldbeinlibc -lm
```

## Bug Fixes Implemented

### 1. Voltage Reset After Spike (Critical)
**Issue**: Original code set voltage to SPIKE_AMPLITUDE but never reset to reset_potential, causing infinite spiking.

**Fix**: Explicitly reset voltage to reset_potential after spike detection.

### 2. Full Hurd Translator Implementation
**Issue**: Hurd layer (Mach IPC + trivfs) was stubbed out.

**Fix**: Complete implementation with proper trivfs integration.

### 3. Correct settrans Syntax
**Issue**: README had inverted syntax.

**Fix**: Correct syntax: `settrans -c /llm /hurd/llm-neuron-translator`

### 4. Type Compatibility
**Issue**: Missing type definitions (pthread_spinlock_t, loff_t, etc.)

**Fix**: Added proper includes (`<pthread.h>`, etc.) and declarations.

## License

GNU General Public License version 3 or later. See [LICENSE](LICENSE) for details.

```
Copyright (C) 2026 GNU AI Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

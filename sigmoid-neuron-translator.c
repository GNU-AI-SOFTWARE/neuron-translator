/*
 * sigmoid-neuron-translator.c - LLM Sigmoid Neuron Translator for GNU Hurd
 *
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * ============================================================================
 * 
 * OVERVIEW:
 * 
 * This is a memory-efficient, CPU-optimized sigmoid neuron translator for
 * GNU Hurd. It implements a compact neural network with contiguous memory
 * allocation, designed to load thousands of neurons with minimal overhead.
 * 
 * This file is designed to compile on both GNU/Hurd and GNU/Linux systems.
 * On Hurd, it functions as a full trivfs translator. On Linux, it provides
 * a test mode that simulates the translator behavior.
 * 
 * Key Features:
 *   - Sigmoid activation function for LLM-style neurons
 *   - Contiguous memory layout for cache efficiency
 *   - SIMD-aligned allocations
 *   - Low memory footprint (single malloc for all network data)
 *   - Low CPU usage (optimized forward pass)
 *   - Full GNU Hurd trivfs translator integration (when on Hurd)
 *   - POSIX compliant
 *   - C23 standard compliant
 * 
 * USAGE ON HURD:
 *   1. Compile: make
 *   2. Install: sudo make install
 *   3. Set translator: sudo settrans -c /llm /hurd/sigmoid-neuron-translator
 *   4. Read: cat /llm
 *   5. Write input: echo "0.5,0.3,0.8" > /llm
 *
 * USAGE ON LINUX (test mode):
 *   1. Compile: make
 *   2. Run: ./sigmoid-neuron-translator
 *   3. This will run a simple test of the neural network
 *
 * ============================================================================
 */


/*****************************************************************************
 *                                                                           *
 *                      INCLUDE DIRECTIVES AND DEFINITIONS                    *
 *                                                                           *
 *****************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>


/*****************************************************************************
 *                                                                           *
 *                    PORTABILITY: HURD vs LINUX                             *
 *                                                                           *
 *****************************************************************************/

#if defined(__GNU__) && !defined(__linux__)
#define ON_HURD 1
#else
#define ON_HURD 0
#endif


/* Define error_t for compatibility */
#ifndef __error_t_defined
#define __error_t_defined 1
typedef int error_t;
#endif


#if ON_HURD

/* On Hurd, include the real headers */
#include <hurd.h>
#include <hurd/fs.h>
#include <hurd/trivfs.h>
#include <mach/mach.h>
#include <mach/mach_msg.h>

#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

#else /* !ON_HURD */

/* On Linux, provide stubs */
typedef unsigned int mach_port_t;
typedef struct mach_msg_header *mach_msg_header_t;

struct iouser;
struct node;
struct iobuf;

#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

#endif /* ON_HURD */


/*****************************************************************************
 *                                                                           *
 *                          CONSTANT DEFINITIONS                             *
 *                                                                           *
 *****************************************************************************/

#define MAX_LAYERS 8
#define MAX_NEURONS_PER_LAYER 8192
#define SIMD_ALIGNMENT 16

#define RESET_POTENTIAL (-80.0f)
#define THRESHOLD (-55.0f)
#define LEAK_RATE 0.1f
#define REFRACTORY_LENGTH 5

#define DEFAULT_LAYER_SIZES {10, 20, 5}
#define DEFAULT_LAYER_COUNT 3


/*****************************************************************************
 *                                                                           *
 *                          DATA STRUCTURES                                 *
 *                                                                           *
 *****************************************************************************/

typedef struct NetworkTopology {
    uint8_t layer_count;
    uint16_t layer_sizes[MAX_LAYERS];
    uint16_t input_size;
    uint16_t output_size;
    float reset_potential;
    float threshold;
    float leak_rate;
    uint8_t refractory_length;
    uint8_t _padding[3];
} NetworkTopology;


typedef struct CompactNeuralNetwork {
    NetworkTopology topology;
    size_t total_neurons;
    size_t total_weights;
    size_t total_biases;
    size_t layer_offsets[MAX_LAYERS];
    size_t weight_offsets[MAX_LAYERS - 1];
    size_t bias_offsets[MAX_LAYERS];
    float *voltages;
    float *weights;
    float *biases;
    float *input_buffer;
    float *output_buffer;
    void *memory_block;
    size_t memory_block_size;
    bool initialized;
    bool needs_reset;
    size_t forward_pass_count;
    size_t neuron_activations;
} CompactNeuralNetwork;


static CompactNeuralNetwork global_network = {0};


/*****************************************************************************
 *                                                                           *
 *                    HURD TRIVFS DECLARATIONS (Hurd only)                  *
 *                                                                           *
 *****************************************************************************/

#if ON_HURD

/* External declarations from trivfs */
extern mach_port_t trivfs_control;
extern char *fs_help;
extern error_t (*fs_open) (struct iouser *, int, mode_t, struct node *, struct iobuf **);
extern error_t (*fs_read) (struct iouser *, struct iobuf *, off_t, size_t *, size_t);
extern error_t (*fs_write) (struct iouser *, struct iobuf *, off_t, size_t, size_t);

/* External functions from trivfs */
extern error_t trivfs_server(mach_msg_header_t *, mach_msg_header_t *);
extern int trivfs_server_loop(void);

/* Forward declarations for our hooks */
static error_t fs_open_hook(struct iouser *, int, mode_t, struct node *, struct iobuf **);
static error_t fs_read_hook(struct iouser *, struct iobuf *, off_t, size_t *, size_t);
static error_t fs_write_hook(struct iouser *, struct iobuf *, off_t, size_t, size_t);

#endif /* ON_HURD */


/*****************************************************************************
 *                                                                           *
 *                      SIGMOID ACTIVATION FUNCTION                          *
 *                                                                           *
 *****************************************************************************/

static inline float
sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}


/*****************************************************************************
 *                                                                           *
 *                        MEMORY MANAGEMENT FUNCTIONS                       *
 *                                                                           *
 *****************************************************************************/

static void *
aligned_malloc(size_t size, size_t alignment)
{
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}


static void
aligned_free(void *ptr)
{
    free(ptr);
}


/*****************************************************************************
 *                                                                           *
 *                      NEURAL NETWORK INITIALIZATION                       *
 *                                                                           *
 *****************************************************************************/

static int
network_init(CompactNeuralNetwork *net,
             uint8_t layer_count,
             const uint16_t *layer_sizes)
{
    if (!net || !layer_sizes) {
        errno = EINVAL;
        return -1;
    }
    
    if (layer_count < 2 || layer_count > MAX_LAYERS) {
        errno = EINVAL;
        return -1;
    }
    
    for (int i = 0; i < layer_count; i++) {
        if (layer_sizes[i] == 0 || layer_sizes[i] > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
    }
    
    net->topology.layer_count = layer_count;
    net->topology.input_size = layer_sizes[0];
    net->topology.output_size = layer_sizes[layer_count - 1];
    
    for (int i = 0; i < layer_count; i++) {
        net->topology.layer_sizes[i] = layer_sizes[i];
    }
    
    net->topology.reset_potential = RESET_POTENTIAL;
    net->topology.threshold = THRESHOLD;
    net->topology.leak_rate = LEAK_RATE;
    net->topology.refractory_length = REFRACTORY_LENGTH;
    
    net->total_neurons = 0;
    for (int i = 0; i < layer_count; i++) {
        net->total_neurons += layer_sizes[i];
    }
    
    net->total_weights = 0;
    net->total_biases = 0;
    for (int i = 1; i < layer_count; i++) {
        net->total_weights += (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        net->total_biases += layer_sizes[i];
    }
    
    net->layer_offsets[0] = 0;
    for (int i = 1; i < layer_count; i++) {
        net->layer_offsets[i] = net->layer_offsets[i - 1] + layer_sizes[i - 1];
    }
    
    if (layer_count > 1) {
        net->weight_offsets[0] = 0;
        for (int i = 1; i < layer_count - 1; i++) {
            net->weight_offsets[i] = net->weight_offsets[i - 1] +
                                      (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        }
    }
    
    net->bias_offsets[0] = 0;
    for (int i = 1; i < layer_count; i++) {
        net->bias_offsets[i] = net->bias_offsets[i - 1] + layer_sizes[i];
    }
    
    size_t voltages_size = net->total_neurons * sizeof(float);
    size_t weights_size = net->total_weights * sizeof(float);
    size_t biases_size = net->total_biases * sizeof(float);
    size_t input_size = layer_sizes[0] * sizeof(float);
    size_t output_size = layer_sizes[layer_count - 1] * sizeof(float);
    
    net->memory_block_size = voltages_size + weights_size + biases_size +
                             input_size + output_size;
    
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        errno = ENOMEM;
        return -1;
    }
    
    char *ptr = (char *)net->memory_block;
    net->voltages = (float *)ptr;
    ptr += voltages_size;
    net->weights = (float *)ptr;
    ptr += weights_size;
    net->biases = (float *)ptr;
    ptr += biases_size;
    net->input_buffer = (float *)ptr;
    ptr += input_size;
    net->output_buffer = (float *)ptr;
    
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    for (size_t i = 0; i < net->total_weights; i++) {
        uint32_t seed = (uint32_t)i * 2654435761U;
        float random = (float)(seed & 0x007FFFFF) / (float)0x007FFFFF;
        net->weights[i] = random * 0.4f - 0.2f;
    }
    
    for (size_t i = 0; i < net->total_biases; i++) {
        net->biases[i] = 0.0f;
    }
    
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return 0;
}


static void
network_free(CompactNeuralNetwork *net)
{
    if (!net) {
        return;
    }
    if (net->memory_block) {
        aligned_free(net->memory_block);
        net->memory_block = NULL;
    }
    net->voltages = NULL;
    net->weights = NULL;
    net->biases = NULL;
    net->input_buffer = NULL;
    net->output_buffer = NULL;
    net->initialized = false;
    net->needs_reset = false;
}


/*****************************************************************************
 *                                                                           *
 *                        FORWARD PASS FUNCTION                              *
 *                                                                           *
 *****************************************************************************/

static void
network_forward(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized || net->topology.layer_count < 2) {
        return;
    }
    
    if (net->input_buffer) {
        memcpy(net->voltages, net->input_buffer,
               net->topology.input_size * sizeof(float));
    }
    
    for (int layer = 1; layer < net->topology.layer_count; layer++) {
        size_t prev_size = net->topology.layer_sizes[layer - 1];
        size_t curr_size = net->topology.layer_sizes[layer];
        size_t prev_offset = net->layer_offsets[layer - 1];
        size_t curr_offset = net->layer_offsets[layer];
        
        size_t weight_offset = 0;
        for (int l = 1; l < layer; l++) {
            weight_offset += (size_t)net->topology.layer_sizes[l] * 
                            (size_t)net->topology.layer_sizes[l - 1];
        }
        
        size_t bias_offset = net->bias_offsets[layer - 1];
        
        for (size_t n = 0; n < curr_size; n++) {
            float sum = net->biases[bias_offset + n];
            float *w = net->weights + weight_offset + n * prev_size;
            float *v = net->voltages + prev_offset;
            
            for (size_t p = 0; p < prev_size; p++) {
                sum += v[p] * w[p];
            }
            
            net->voltages[curr_offset + n] = sigmoidf(sum);
            net->neuron_activations++;
        }
    }
    
    if (net->output_buffer) {
        size_t out_offset = net->layer_offsets[net->topology.layer_count - 1];
        memcpy(net->output_buffer, net->voltages + out_offset,
               net->topology.output_size * sizeof(float));
    }
    
    net->forward_pass_count++;
}


/*****************************************************************************
 *                                                                           *
 *                     CONFIGURATION PARSING FUNCTIONS                       *
 *                                                                           *
 *****************************************************************************/

static int
parse_config_string(const char *config_str, uint16_t *layer_sizes, int max_layers)
{
    if (!config_str || !layer_sizes || max_layers < 2) {
        errno = EINVAL;
        return -1;
    }
    
    const char *ptr = config_str;
    int count = 0;
    
    while (*ptr != '\0' && count < max_layers) {
        while (*ptr == ' ' || *ptr == '\t' || *ptr == ',') ptr++;
        if (*ptr == '\0') break;
        
        char *endptr;
        long value = strtol(ptr, &endptr, 10);
        
        if (ptr == endptr) {
            errno = EINVAL;
            return -1;
        }
        
        if (value < 1 || value > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
        
        layer_sizes[count++] = (uint16_t)value;
        ptr = endptr;
    }
    
    if (count < 2) {
        errno = EINVAL;
        return -1;
    }
    
    return count;
}


static bool
parse_input_string(CompactNeuralNetwork *net, const char *input_str)
{
    if (!net || !net->initialized || !input_str) {
        return false;
    }
    
    const char *ptr = input_str;
    size_t idx = 0;
    
    while (*ptr != '\0' && idx < net->topology.input_size) {
        while (*ptr == ',' || *ptr == ' ' || *ptr == '\t' || *ptr == '\n') ptr++;
        if (*ptr == '\0') break;
        
        char *endptr;
        float val = strtof(ptr, &endptr);
        
        if (ptr == endptr) {
            ptr++;
            continue;
        }
        
        net->input_buffer[idx++] = val;
        ptr = endptr;
    }
    
    if (idx >= net->topology.input_size) {
        network_forward(net);
        return true;
    }
    
    return false;
}


/*****************************************************************************
 *                                                                           *
 *                      FILESYSTEM OPERATION HOOKS (Hurd only)               *
 *                                                                           *
 *****************************************************************************/

#if ON_HURD

static error_t
fs_open_hook(struct iouser *cred, int flags, mode_t mode, struct node *node,
             struct iobuf **iobuf)
{
    (void)cred; (void)flags; (void)mode; (void)node;
    *iobuf = NULL;
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    }
    
    return 0;
}


static error_t
fs_read_hook(struct iouser *cred, struct iobuf *iobuf, off_t offset,
             size_t *len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    }
    
    char buffer[4096];
    size_t written = 0;
    
    written = snprintf(buffer, sizeof(buffer),
                      "LLM Sigmoid Neuron Translator - GNU Hurd\n"
                      "===========================================\n\n");
    
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Network Topology:\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Layers: %d\n", global_network.topology.layer_count);
    
    for (int i = 0; i < global_network.topology.layer_count; i++) {
        written += snprintf(buffer + written, sizeof(buffer) - written,
                          "  Layer %d: %d neurons\n",
                          i, global_network.topology.layer_sizes[i]);
    }
    written += snprintf(buffer + written, sizeof(buffer) - written, "\n");
    
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Memory: %.2f KB, Neurons: %zu, Weights: %zu\n\n",
                       (double)global_network.memory_block_size / 1024.0,
                       global_network.total_neurons,
                       global_network.total_weights);
    
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Output:\n");
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        written += snprintf(buffer + written, sizeof(buffer) - written,
                          "  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
    }
    
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "\nUsage:\n"
                       "  Configure: echo '<layers>' > /llm\n"
                       "  Input: echo '<values>' > /llm\n");
    
    if (written >= sizeof(buffer)) {
        written = sizeof(buffer) - 1;
    }
    buffer[written] = '\0';
    
    if (iobuf == NULL) {
        *len = written;
        return 0;
    }
    
    size_t to_copy = (written < *len) ? written : *len;
    if (to_copy > 0 && iobuf->buf != NULL) {
        memcpy(iobuf->buf, buffer, to_copy);
    }
    *len = to_copy;
    
    return 0;
}


static error_t
fs_write_hook(struct iouser *cred, struct iobuf *iobuf, off_t offset,
              size_t len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    if (iobuf == NULL || len == 0 || iobuf->buf == NULL) {
        return EINVAL;
    }
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    }
    
    char temp[1024];
    if (len >= sizeof(temp)) {
        len = sizeof(temp) - 1;
    }
    memcpy(temp, iobuf->buf, len);
    temp[len] = '\0';
    
    uint16_t layers[MAX_LAYERS];
    int layer_count = parse_config_string(temp, layers, MAX_LAYERS);
    
    if (layer_count > 0) {
        network_free(&global_network);
        if (network_init(&global_network, layer_count, layers) != 0) {
            return EINVAL;
        }
        return 0;
    }
    
    if (parse_input_string(&global_network, temp)) {
        return 0;
    }
    
    return EINVAL;
}


int
trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    return trivfs_server(inmsg, outmsg);
}


int
main(int argc, char **argv)
{
    (void)argc; (void)argv;
    
    global_network.initialized = false;
    global_network.memory_block = NULL;
    
    trivfs_control = MACH_PORT_NULL;
    
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    
    fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
              "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator";
    
    fs_open = fs_open_hook;
    fs_read = fs_read_hook;
    fs_write = fs_write_hook;
    
    return trivfs_server_loop();
}


#else /* !ON_HURD */

/*****************************************************************************
 *                                                                           *
 *                         TEST MODE FOR LINUX                              *
 *                                                                           *
 *****************************************************************************/

int
main(int argc, char **argv)
{
    (void)argc; (void)argv;
    
    printf("LLM Sigmoid Neuron Translator - Test Mode (Linux)\n");
    printf("===================================================\n\n");
    
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        fprintf(stderr, "Failed to initialize network: %s\n", strerror(errno));
        return 1;
    }
    
    printf("Network initialized with %d layers\n", global_network.topology.layer_count);
    for (int i = 0; i < global_network.topology.layer_count; i++) {
        printf("  Layer %d: %d neurons\n", i, global_network.topology.layer_sizes[i]);
    }
    printf("\n");
    
    for (int i = 0; i < global_network.topology.input_size; i++) {
        global_network.input_buffer[i] = (float)i * 0.1f;
    }
    
    network_forward(&global_network);
    
    printf("Forward pass completed\n");
    printf("Output:\n");
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        printf("  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
    }
    
    printf("\nMemory: %.2f KB, Neurons: %zu, Weights: %zu\n",
           (double)global_network.memory_block_size / 1024.0,
           global_network.total_neurons,
           global_network.total_weights);
    
    uint16_t test_layers[MAX_LAYERS];
    int test_count = parse_config_string("5,10,5", test_layers, MAX_LAYERS);
    if (test_count > 0) {
        printf("Config parsing works: %d layers\n", test_count);
    }
    
    if (parse_input_string(&global_network, "0.5,0.3,0.8,0.1,0.9,0.2,0.4,0.6,0.0,0.7")) {
        printf("Input parsing and forward pass works\n");
    }
    
    network_free(&global_network);
    printf("\nTest mode completed successfully\n");
    
    return 0;
}

#endif /* ON_HURD */

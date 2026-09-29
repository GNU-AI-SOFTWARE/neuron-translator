/* Define feature test macros BEFORE any includes */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

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
 */


/*****************************************************************************
 *                                                                           *
 *                           SYSTEM INCLUDES                                *
 *                                                                           *
 *  C23 Standard, POSIX compliant headers for GNU Hurd                        *
 *                                                                           *
 *****************************************************************************/

/* Standard C library headers */
#include <stdio.h>      /* Standard I/O functions */
#include <stdlib.h>     /* Memory allocation and exit codes */
#include <string.h>     /* String manipulation functions */
#include <math.h>       /* Mathematical functions (expf) */
#include <errno.h>      /* Error number definitions */
#include <stdint.h>     /* Fixed-width integer types */
#include <stdbool.h>    /* Boolean type */
#include <stddef.h>     /* Standard definitions (size_t, NULL) */
#include <pthread.h>    /* POSIX threads */

/* Define error_t for compatibility */
#ifndef __error_t_defined
#define __error_t_defined 1
typedef int error_t;
#endif


/*****************************************************************************
 *                                                                           *
 *                         GNU HURD HEADERS                                *
 *                                                                           *
 *  Core Hurd and Mach headers for translator implementation.              *
 *                                                                           *
 *****************************************************************************/

#include <hurd.h>             /* Hurd base definitions */
#include <hurd/trivfs.h>      /* Trivial filesystem translator */
#include <hurd/fs.h>          /* Filesystem interface */
#include <hurd/iohelp.h>      /* I/O buffer definitions (struct iobuf) */

#include <mach/mach.h>        /* Mach kernel interface */
#include <mach/port.h>        /* Mach port interface */
#include <mach/message.h>     /* Mach message interface */

/* Define MACH_PORT_NULL if not already defined */
#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

/* Declare trivfs variables if not already declared by Hurd headers */
#ifndef __TRIVFS_DECLARATIONS
#define __TRIVFS_DECLARATIONS

/* External trivfs variables */
extern mach_port_t trivfs_control;
extern char *fs_help;

/* External filesystem operation hooks */
extern error_t (*fs_open) (struct iouser *, int, mode_t, struct node *,
                           struct iobuf **);
extern error_t (*fs_read) (struct iouser *, struct iobuf *, off_t, size_t *,
                           size_t);
extern error_t (*fs_write) (struct iouser *, struct iobuf *, off_t, size_t,
                            size_t);

/* External trivfs server functions */
extern error_t trivfs_server(mach_msg_header_t *, mach_msg_header_t *);
extern int trivfs_server_loop(void);

#endif /* __TRIVFS_DECLARATIONS */

/* Forward declarations for our filesystem hooks */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf);
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t *len, size_t count);
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                       off_t offset, size_t len, size_t count);

/*****************************************************************************
 *                                                                           *
 *                         CONSTANT DEFINITIONS                             *
 *                                                                           *
 *  Neural network parameters and limits.                                  *
 *                                                                           *
 *****************************************************************************/

/* Maximum layers and neurons for memory pre-allocation */
#define MAX_LAYERS 8
#define MAX_NEURONS_PER_LAYER 8192
#define SIMD_ALIGNMENT 16

/* Neuron parameters (biologically plausible values in millivolts) */
#define RESET_POTENTIAL (-80.0f)
#define THRESHOLD (-55.0f)
#define LEAK_RATE 0.1f
#define REFRACTORY_LENGTH 5

/* Default network topology */
#define DEFAULT_LAYER_SIZES {10, 20, 5}
#define DEFAULT_LAYER_COUNT 3


/*****************************************************************************
 *                                                                           *
 *                          DATA STRUCTURES                                 *
 *                                                                           *
 *  Memory layout: [voltages][weights][biases][input_buffer][output_buffer] *
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
 *                      SIGMOID ACTIVATION FUNCTION                          *
 *                                                                           *
 *****************************************************************************/

static inline float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}


/*****************************************************************************
 *                                                                           *
 *                        MEMORY MANAGEMENT                                *
 *                                                                           *
 *****************************************************************************/

static void *aligned_malloc(size_t size, size_t alignment)
{
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static void aligned_free(void *ptr)
{
    free(ptr);
}


/*****************************************************************************
 *                                                                           *
 *                      NETWORK INITIALIZATION                              *
 *                                                                           *
 *****************************************************************************/

static int network_init(CompactNeuralNetwork *net,
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


static void network_free(CompactNeuralNetwork *net)
{
    if (!net) return;
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


static void network_reset(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized) return;
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    net->needs_reset = false;
}


/*****************************************************************************
 *                                                                           *
 *                        FORWARD PASS FUNCTION                              *
 *                                                                           *
 *****************************************************************************/

static void network_forward(CompactNeuralNetwork *net)
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

static int parse_config_string(const char *config_str,
                               uint16_t *layer_sizes,
                               int max_layers)
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


static bool parse_input_string(CompactNeuralNetwork *net,
                               const char *input_str)
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
 *                      FILE PERSISTENCE FUNCTIONS                          *
 *                                                                           *
 *****************************************************************************/

static bool network_save(const CompactNeuralNetwork *net,
                        const char *filename)
{
    if (!net || !net->initialized || !filename) {
        return false;
    }
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    
    if (fwrite(&net->topology, sizeof(NetworkTopology), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    if (fwrite(&net->total_neurons, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fwrite(&net->total_weights, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fwrite(&net->total_biases, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    if (fwrite(net->layer_offsets, sizeof(size_t),
               net->topology.layer_count, fp) != net->topology.layer_count) {
        fclose(fp); return false;
    }
    
    if (net->topology.layer_count > 1) {
        if (fwrite(net->weight_offsets, sizeof(size_t),
                   net->topology.layer_count - 1, fp) !=
            (size_t)(net->topology.layer_count - 1)) {
            fclose(fp); return false;
        }
    }
    
    if (fwrite(net->bias_offsets, sizeof(size_t),
               net->topology.layer_count, fp) != net->topology.layer_count) {
        fclose(fp); return false;
    }
    
    if (fwrite(net->voltages, sizeof(float), net->total_neurons, fp) !=
        net->total_neurons) {
        fclose(fp); return false;
    }
    
    if (fwrite(net->weights, sizeof(float), net->total_weights, fp) !=
        net->total_weights) {
        fclose(fp); return false;
    }
    
    if (fwrite(net->biases, sizeof(float), net->total_biases, fp) !=
        net->total_biases) {
        fclose(fp); return false;
    }
    
    fclose(fp);
    return true;
}


static bool network_load(CompactNeuralNetwork *net,
                        const char *filename)
{
    if (!net || !filename) return false;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    
    network_free(net);
    
    if (fread(&net->topology, sizeof(NetworkTopology), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    if (fread(&net->total_neurons, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fread(&net->total_weights, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fread(&net->total_biases, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    if (fread(net->layer_offsets, sizeof(size_t),
              net->topology.layer_count, fp) != net->topology.layer_count) {
        fclose(fp); return false;
    }
    
    if (net->topology.layer_count > 1) {
        if (fread(net->weight_offsets, sizeof(size_t),
                  net->topology.layer_count - 1, fp) !=
            (size_t)(net->topology.layer_count - 1)) {
            network_free(net);
            fclose(fp); return false;
        }
    }
    
    if (fread(net->bias_offsets, sizeof(size_t),
              net->topology.layer_count, fp) != net->topology.layer_count) {
        network_free(net);
        fclose(fp); return false;
    }
    
    net->memory_block_size = net->total_neurons * sizeof(float) +
                             net->total_weights * sizeof(float) +
                             net->total_biases * sizeof(float) +
                             net->topology.input_size * sizeof(float) +
                             net->topology.output_size * sizeof(float);
    
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        fclose(fp); return false;
    }
    
    char *ptr = (char *)net->memory_block;
    net->voltages = (float *)ptr;
    ptr += net->total_neurons * sizeof(float);
    net->weights = (float *)ptr;
    ptr += net->total_weights * sizeof(float);
    net->biases = (float *)ptr;
    ptr += net->total_biases * sizeof(float);
    net->input_buffer = (float *)ptr;
    ptr += net->topology.input_size * sizeof(float);
    net->output_buffer = (float *)ptr;
    
    if (fread(net->voltages, sizeof(float), net->total_neurons, fp) !=
        net->total_neurons) {
        network_free(net);
        fclose(fp); return false;
    }
    
    if (fread(net->weights, sizeof(float), net->total_weights, fp) !=
        net->total_weights) {
        network_free(net);
        fclose(fp); return false;
    }
    
    if (fread(net->biases, sizeof(float), net->total_biases, fp) !=
        net->total_biases) {
        network_free(net);
        fclose(fp); return false;
    }
    
    fclose(fp);
    
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return true;
}


/*****************************************************************************
 *                                                                           *
 *                      TRIVFS TRANSLATOR HOOKS                             *
 *                                                                           *
 *  Implementation of trivfs translator interface for GNU Hurd.             *
 *                                                                           *
 *****************************************************************************/

/**
 * fs_open_hook - Called when translator node is opened
 */
error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                     struct node *node, struct iobuf **iobuf)
{
    (void)cred; (void)flags; (void)mode; (void)node;
    *iobuf = NULL;
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return ENOMEM;
        }
    }
    
    return 0;
}


/**
 * fs_read_hook - Called when translator node is read
 */
error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                     off_t offset, size_t *len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return ENOMEM;
        }
    }
    
    char buffer[4096];
    size_t written = 0;
    
    written = snprintf(buffer, sizeof(buffer),
                      "LLM Sigmoid Neuron Translator - GNU Hurd\n"
                      "===========================================\n\n");
    
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Network: %d layers", global_network.topology.layer_count);
    for (int i = 0; i < global_network.topology.layer_count; i++) {
        written += snprintf(buffer + written, sizeof(buffer) - written,
                          ", %d", global_network.topology.layer_sizes[i]);
    }
    written += snprintf(buffer + written, sizeof(buffer) - written, "\n\n");
    
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
                       "  cat /llm                    - Show info\n"
                       "  echo '10,20,5' > /llm      - Set topology\n"
                       "  echo '0.5,0.3,0.8' > /llm  - Set input\n"
                       "  echo reset > /llm         - Reset network state\n"
                       "  echo 'save /tmp/net.bin' > /llm  - Save network\n"
                       "  echo 'load /tmp/net.bin' > /llm  - Load network\n");
    
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


/**
 * fs_write_hook - Called when translator node is written to
 */
error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                      off_t offset, size_t len, size_t count)
{
    (void)cred; (void)offset; (void)count;
    
    if (iobuf == NULL || len == 0 || iobuf->buf == NULL) {
        return EINVAL;
    }
    
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return ENOMEM;
        }
    }
    
    char temp[1024];
    if (len >= sizeof(temp)) {
        len = sizeof(temp) - 1;
    }
    memcpy(temp, iobuf->buf, len);
    temp[len] = '\0';
    
    /* Remove trailing newline for easier parsing */
    char *newline = strchr(temp, '\n');
    if (newline) *newline = '\0';
    newline = strchr(temp, '\r');
    if (newline) *newline = '\0';
    
    /* Handle special commands */
    if (strncmp(temp, "reset", 5) == 0) {
        network_reset(&global_network);
        return 0;
    }
    
    if (strncmp(temp, "save ", 5) == 0) {
        char *filename = temp + 5;
        if (*filename != '\0') {
            if (network_save(&global_network, filename)) {
                return 0;
            }
        }
        return EINVAL;
    }
    
    if (strncmp(temp, "load ", 5) == 0) {
        char *filename = temp + 5;
        if (*filename != '\0') {
            if (network_load(&global_network, filename)) {
                return 0;
            }
        }
        return EINVAL;
    }
    
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


/**
 * trivfs_demuxer - Message demultiplexer
 * 
 * This function is the entry point for all Mach messages received by the
 * translator. It demultiplexes messages and dispatches them to the
 * appropriate trivfs server function.
 * 
 * @param inmsg  Pointer to the incoming Mach message header
 * @param outmsg Pointer to the outgoing Mach message header
 * @return        Error code from message processing (0 on success)
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    /* Delegate to the trivfs server message handler */
    return trivfs_server(inmsg, outmsg);
}


/**
 * main - Translator entry point
 */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    
    global_network.initialized = false;
    global_network.memory_block = NULL;
    trivfs_control = MACH_PORT_NULL;
    
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        global_network.initialized = false;
    }
    
    fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
              "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator";
    
    fs_open = fs_open_hook;
    fs_read = fs_read_hook;
    fs_write = fs_write_hook;
    
    return trivfs_server_loop();
}


/*
 * Editor modelines
 */

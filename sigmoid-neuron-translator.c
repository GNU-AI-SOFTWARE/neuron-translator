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
 *  Standard C library headers required for both Hurd and Linux builds.     *
 *  POSIX-compliant and C23-compatible.                                    *
 *                                                                           *
 *****************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/* Standard C library headers */
#include <stdio.h>      /* Standard I/O functions */
#include <stdlib.h>     /* Memory allocation and exit codes */
#include <string.h>     /* String manipulation functions */
#include <math.h>       /* Mathematical functions (expf) */
#include <errno.h>      /* Error number definitions */
#include <stdint.h>     /* Fixed-width integer types */
#include <stdbool.h>    /* Boolean type */
#include <stddef.h>     /* Standard definitions (size_t, NULL) */
#include <pthread.h>    /* POSIX threads - required by Hurd headers */

/* Define error_t for compatibility with Hurd */
#ifndef __error_t_defined
#define __error_t_defined 1
typedef int error_t;
#endif


/*****************************************************************************
 *                                                                           *
 *                         PORTABILITY DETECTION                             *
 *                                                                           *
 *  Detect the target system:                                              *
 *    - GNU/Hurd: __GNU__ is defined, __linux__ is NOT defined              *
 *    - GNU/Linux: __linux__ is defined                                     *
 *                                                                           *
 *****************************************************************************/

#if defined(__GNU__) && !defined(__linux__)
#define ON_HURD 1
#else
#define ON_HURD 0
#endif


/*****************************************************************************
 *                                                                           *
 *                         NEURAL NETWORK CONSTANTS                          *
 *                                                                           *
 *  Configuration parameters for the neural network.                        *
 *  All values are designed for memory efficiency and CPU optimization.     *
 *                                                                           *
 *****************************************************************************/

/* Maximum layers and neurons per layer for memory pre-allocation */
#define MAX_LAYERS 8
#define MAX_NEURONS_PER_LAYER 8192

/* Memory alignment for SIMD operations (16 bytes for AVX compatibility) */
#define SIMD_ALIGNMENT 16

/* Neuron parameters - biologically plausible values in millivolts (mV) */
#define RESET_POTENTIAL (-80.0f)   /* Resting membrane potential */
#define THRESHOLD (-55.0f)          /* Voltage threshold for activation */
#define LEAK_RATE 0.1f              /* Voltage decay rate per timestep */
#define REFRACTORY_LENGTH 5        /* Post-spike silence period in timesteps */

/* Default network topology: Input(10) -> Hidden(20) -> Output(5) */
#define DEFAULT_LAYER_SIZES {10, 20, 5}
#define DEFAULT_LAYER_COUNT 3


/*****************************************************************************
 *                                                                           *
 *                          DATA STRUCTURES                                 *
 *                                                                           *
 *  Memory-efficient structures for neural network representation.          *
 *  All network data is stored in a single contiguous memory block.           *
 *                                                                           *
 *  Memory layout: [voltages][weights][biases][input_buffer][output_buffer] *
 *                                                                           *
 *****************************************************************************/

/**
 * NetworkTopology - Network configuration and parameters
 *
 * Contains all structural information about the neural network.
 * Packed to minimize memory usage with proper alignment.
 */
typedef struct NetworkTopology {
    uint8_t layer_count;               /* Number of layers in the network */
    uint16_t layer_sizes[MAX_LAYERS];  /* Neurons per layer */
    uint16_t input_size;               /* Size of input layer */
    uint16_t output_size;              /* Size of output layer */
    float reset_potential;             /* Resting potential (mV) */
    float threshold;                   /* Activation threshold (mV) */
    float leak_rate;                   /* Voltage decay rate */
    uint8_t refractory_length;         /* Refractory period in timesteps */
    uint8_t _padding[3];               /* Padding for 16-byte alignment */
} NetworkTopology;

/**
 * CompactNeuralNetwork - Complete neural network state
 *
 * Stores the entire neural network in a memory-efficient format.
 * Uses a single contiguous memory allocation for all numeric data.
 */
typedef struct CompactNeuralNetwork {
    NetworkTopology topology;        /* Network structure */
    size_t total_neurons;             /* Total neurons across all layers */
    size_t total_weights;             /* Total connection weights */
    size_t total_biases;             /* Total bias values */
    
    /* Memory layout offsets */
    size_t layer_offsets[MAX_LAYERS];
    size_t weight_offsets[MAX_LAYERS - 1];
    size_t bias_offsets[MAX_LAYERS];
    
    /* Pointers to data within the contiguous memory block */
    float *voltages;                  /* Neuron membrane potentials */
    float *weights;                  /* Connection weights */
    float *biases;                    /* Neuron biases */
    float *input_buffer;              /* Input values buffer */
    float *output_buffer;             /* Output values buffer */
    
    /* Memory management */
    void *memory_block;              /* Single contiguous memory allocation */
    size_t memory_block_size;        /* Total allocated bytes */
    
    /* Runtime statistics */
    bool initialized;                 /* Network initialization flag */
    bool needs_reset;                 /* Reset required flag */
    size_t forward_pass_count;       /* Number of forward passes executed */
    size_t neuron_activations;       /* Total neuron activations */
} CompactNeuralNetwork;


/*****************************************************************************
 *                                                                           *
 *                         GLOBAL NETWORK INSTANCE                           *
 *                                                                           *
 *****************************************************************************/

/* Single global network instance used by the translator */
static CompactNeuralNetwork global_network = {0};


/*****************************************************************************
 *                                                                           *
 *                      HURD-SPECIFIC DECLARATIONS                          *
 *                                                                           *
 *  When compiling for GNU/Hurd, include headers and declare Hurd-specific     *
 *  types and functions. When compiling for Linux, provide stubs.          *
 *                                                                           *
 *****************************************************************************/

#if ON_HURD

/* ======================================================================== */
/* GNU Hurd Headers */
/* ======================================================================== */

#include <hurd.h>             /* Hurd base definitions */
#include <hurd/fs.h>          /* Filesystem interface */
#include <hurd/trivfs.h>      /* Trivial filesystem translator interface */
#include <hurd/iohelp.h>      /* I/O buffer definitions (struct iobuf) */

/* ======================================================================== */
/* GNU Mach Headers */
/* ======================================================================== */

#include <mach/mach.h>        /* Mach kernel interface */
#include <mach/port.h>        /* Mach port interface */
#include <mach/message.h>     /* Mach message interface */

/* Ensure MACH_PORT_NULL is defined */
#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

/* Hurd filesystem types (defined in Hurd headers) */
struct iouser;              /* User credentials */
struct node;                /* Filesystem node */
struct iobuf;               /* I/O buffer */

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

/* Forward declarations for our filesystem hooks */
static error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                            struct node *node, struct iobuf **iobuf);
static error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                            off_t offset, size_t *len, size_t count);
static error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                             off_t offset, size_t len, size_t count);

#else

/* ======================================================================== */
/* Linux (Test Mode) - Stub Definitions */
/* ======================================================================== */

/* Basic Mach/Hurd types for Linux compatibility */
typedef unsigned int mach_port_t;
#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

/* Forward declarations for Hurd types (not used in Linux mode) */
struct iouser;
struct node;
struct iobuf;

#endif /* ON_HURD */


/*****************************************************************************
 *                                                                           *
 *                      SIGMOID ACTIVATION FUNCTION                          *
 *                                                                           *
 *  Core activation function for the sigmoid neuron.                       *
 *  Maps any real number to the range (0, 1).                              *
 *                                                                           *
 *  Mathematical definition: sigmoid(x) = 1 / (1 + exp(-x))                 *
 *                                                                           *
 *****************************************************************************/

/**
 * sigmoidf - Single-precision sigmoid activation function
 *
 * @param x Input value (any real number)
 * @return Sigmoid of x in range (0, 1)
 *
 * Uses expf() for single-precision floating point to match
 * the float32 storage format throughout the network.
 */
static inline float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}


/*****************************************************************************
 *                                                                           *
 *                        MEMORY MANAGEMENT FUNCTIONS                       *
 *                                                                           *
 *  Aligned memory allocation for optimal performance.                     *
 *  SIMD alignment (16 bytes) ensures compatibility with vector instructions.*
 *                                                                           *
 *****************************************************************************/

/**
 * aligned_malloc - Allocate memory with specific alignment
 *
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement (must be power of 2)
 * @return Pointer to allocated memory, or NULL on failure
 */
static void *aligned_malloc(size_t size, size_t alignment)
{
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}


/**
 * aligned_free - Free memory allocated with aligned_malloc
 *
 * @param ptr Pointer to memory to free (may be NULL)
 */
static void aligned_free(void *ptr)
{
    free(ptr);
}


/*****************************************************************************
 *                                                                           *
 *                      NETWORK INITIALIZATION FUNCTIONS                    *
 *                                                                           *
 *****************************************************************************/

/**
 * network_init - Initialize neural network with specified topology
 *
 * @param net Network structure to initialize
 * @param layer_count Number of layers
 * @param layer_sizes Array of layer sizes (neurons per layer)
 * @return 0 on success, -1 on error (with errno set)
 */
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
    
    /* Copy topology */
    net->topology.layer_count = layer_count;
    net->topology.input_size = layer_sizes[0];
    net->topology.output_size = layer_sizes[layer_count - 1];
    
    for (int i = 0; i < layer_count; i++) {
        net->topology.layer_sizes[i] = layer_sizes[i];
    }
    
    /* Set default parameters */
    net->topology.reset_potential = RESET_POTENTIAL;
    net->topology.threshold = THRESHOLD;
    net->topology.leak_rate = LEAK_RATE;
    net->topology.refractory_length = REFRACTORY_LENGTH;
    
    /* Calculate totals */
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
    
    /* Calculate offsets */
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
    
    /* Calculate memory requirements */
    size_t voltages_size = net->total_neurons * sizeof(float);
    size_t weights_size = net->total_weights * sizeof(float);
    size_t biases_size = net->total_biases * sizeof(float);
    size_t input_size = layer_sizes[0] * sizeof(float);
    size_t output_size = layer_sizes[layer_count - 1] * sizeof(float);
    
    net->memory_block_size = voltages_size + weights_size + biases_size +
                             input_size + output_size;
    
    /* Allocate contiguous memory block */
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        errno = ENOMEM;
        return -1;
    }
    
    /* Set up pointers */
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
    
    /* Initialize values */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Initialize weights with small random values */
    for (size_t i = 0; i < net->total_weights; i++) {
        uint32_t seed = (uint32_t)i * 2654435761U;
        float random = (float)(seed & 0x007FFFFF) / (float)0x007FFFFF;
        net->weights[i] = random * 0.4f - 0.2f;
    }
    
    /* Initialize biases to zero */
    for (size_t i = 0; i < net->total_biases; i++) {
        net->biases[i] = 0.0f;
    }
    
    /* Set runtime state */
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return 0;
}


/**
 * network_free - Free all memory allocated for a network
 *
 * @param net Network to free (may be NULL)
 */
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


/**
 * network_reset - Reset network state to initial values
 *
 * @param net Network to reset
 */
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

/**
 * network_forward - Perform forward pass through the network
 *
 * @param net Initialized network
 *
 * Computes the output of the network given the current input.
 * Uses pre-calculated offsets for cache-friendly memory access.
 */
static void network_forward(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized || net->topology.layer_count < 2) {
        return;
    }
    
    /* Copy input to first layer */
    if (net->input_buffer) {
        memcpy(net->voltages, net->input_buffer,
               net->topology.input_size * sizeof(float));
    }
    
    /* Process each layer */
    for (int layer = 1; layer < net->topology.layer_count; layer++) {
        size_t prev_size = net->topology.layer_sizes[layer - 1];
        size_t curr_size = net->topology.layer_sizes[layer];
        size_t prev_offset = net->layer_offsets[layer - 1];
        size_t curr_offset = net->layer_offsets[layer];
        
        /* Calculate weight offset */
        size_t weight_offset = 0;
        for (int l = 1; l < layer; l++) {
            weight_offset += (size_t)net->topology.layer_sizes[l] *
                            (size_t)net->topology.layer_sizes[l - 1];
        }
        
        /* Calculate bias offset */
        size_t bias_offset = net->bias_offsets[layer - 1];
        
        /* Process each neuron in current layer */
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
    
    /* Copy output to buffer */
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

/**
 * parse_config_string - Parse layer sizes from configuration string
 *
 * @param config_str Configuration string (comma or space separated)
 * @param layer_sizes Output array for layer sizes
 * @param max_layers Maximum number of layers to parse
 * @return Number of layers parsed (>= 2), or -1 on error
 */
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


/**
 * parse_input_string - Parse input values from string
 *
 * @param net Initialized network
 * @param input_str Input string (comma or space separated)
 * @return true if parsing and forward pass successful, false otherwise
 */
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

/**
 * network_save - Save network state to file
 *
 * @param net Network to save
 * @param filename Output filename
 * @return true on success, false on failure
 */
static bool network_save(const CompactNeuralNetwork *net, 
                        const char *filename)
{
    if (!net || !net->initialized || !filename) {
        return false;
    }
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) return false;
    
    /* Write topology */
    if (fwrite(&net->topology, sizeof(NetworkTopology), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    /* Write counts */
    if (fwrite(&net->total_neurons, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fwrite(&net->total_weights, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fwrite(&net->total_biases, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    /* Write offsets */
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
    
    /* Write data */
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


/**
 * network_load - Load network state from file
 *
 * @param net Network to load into
 * @param filename Input filename
 * @return true on success, false on failure
 */
static bool network_load(CompactNeuralNetwork *net, const char *filename)
{
    if (!net || !filename) return false;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    
    /* Free existing */
    network_free(net);
    
    /* Read topology */
    if (fread(&net->topology, sizeof(NetworkTopology), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    /* Read counts */
    if (fread(&net->total_neurons, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fread(&net->total_weights, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    if (fread(&net->total_biases, sizeof(size_t), 1, fp) != 1) {
        fclose(fp); return false;
    }
    
    /* Read offsets */
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
    
    /* Allocate memory */
    net->memory_block_size = net->total_neurons * sizeof(float) +
                             net->total_weights * sizeof(float) +
                             net->total_biases * sizeof(float) +
                             net->topology.input_size * sizeof(float) +
                             net->topology.output_size * sizeof(float);
    
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        fclose(fp); return false;
    }
    
    /* Set up pointers */
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
    
    /* Read data */
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
 *                      HURD TRANSLATOR IMPLEMENTATION                       *
 *                                                                           *
 *  These functions are only compiled when ON_HURD=1 (GNU/Hurd).             *
 *  They implement the trivfs translator interface.                          *
 *                                                                           *
 *****************************************************************************/

#if ON_HURD

/**
 * fs_open_hook - Called when translator file is opened
 *
 * @param cred User credentials (unused)
 * @param flags Open flags (unused)
 * @param mode Mode bits (unused)
 * @param node Filesystem node (unused)
 * @param iobuf Output: I/O buffer pointer
 * @return 0 on success, error code on failure
 */
static error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
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
 * fs_read_hook - Called when translator file is read
 *
 * @param cred User credentials (unused)
 * @param iobuf I/O buffer for output
 * @param offset Read offset (unused)
 * @param len Input/Output: number of bytes to read/written
 * @param count Maximum bytes to read (unused)
 * @return 0 on success, error code on failure
 */
static error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
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


/**
 * fs_write_hook - Called when translator file is written to
 *
 * @param cred User credentials (unused)
 * @param iobuf I/O buffer containing input data
 * @param offset Write offset (unused)
 * @param len Number of bytes written
 * @param count Maximum bytes to write (unused)
 * @return 0 on success, error code on failure
 */
static error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
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
 * trivfs_demuxer - Message demultiplexer for trivfs
 *
 * @param inmsg Incoming Mach message header
 * @param outmsg Outgoing Mach message header
 * @return Error code
 */
int trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    return trivfs_server(inmsg, outmsg);
}


/**
 * main - Entry point for Hurd translator
 *
 * @param argc Argument count (unused)
 * @param argv Argument vector (unused)
 * @return Exit code (never reached)
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


#else /* !ON_HURD */

/*****************************************************************************
 *                                                                           *
 *                         TEST MODE FOR LINUX                              *
 *                                                                           *
 *  Test mode that runs on GNU/Linux for development and debugging.          *
 *                                                                           *
 *****************************************************************************/

/**
 * main - Test mode entry point
 *
 * @param argc Argument count (unused)
 * @param argv Argument vector (unused)
 * @return 0 on success, 1 on failure
 */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    
    printf("LLM Sigmoid Neuron Translator - Test Mode (Linux)\n");
    printf("===================================================\n\n");
    
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        fprintf(stderr, "Failed to initialize network: %s\n", strerror(errno));
        return 1;
    }
    
    printf("Network: %d layers", global_network.topology.layer_count);
    for (int i = 0; i < global_network.topology.layer_count; i++) {
        printf(", %d", global_network.topology.layer_sizes[i]);
    }
    printf("\n\n");
    
    for (int i = 0; i < global_network.topology.input_size; i++) {
        global_network.input_buffer[i] = (float)i * 0.1f;
    }
    
    network_forward(&global_network);
    
    printf("Output:\n");
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        printf("  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
    }
    
    printf("\nMemory: %.2f KB\n",
           (double)global_network.memory_block_size / 1024.0);
    
    printf("Test passed\n");
    network_free(&global_network);
    
    return 0;
}

#endif /* ON_HURD */


/*
 * Editor modelines
 */

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
 *****************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

/* Standard C library headers - POSIX compliant */
#include <stdio.h>      /* Standard I/O functions */
#include <stdlib.h>     /* Memory allocation and exit codes */
#include <string.h>     /* String manipulation functions */
#include <math.h>       /* Mathematical functions (expf, etc.) */
#include <errno.h>      /* Error number definitions */
#include <stdint.h>     /* Fixed-width integer types */
#include <stdbool.h>    /* Boolean type */
#include <stddef.h>     /* Standard definitions (size_t, NULL, etc.) */
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
 *****************************************************************************/

/* Detect if we are compiling for GNU/Hurd or GNU/Linux */
/* On GNU/Hurd: __GNU__ is defined but __linux__ is not */
/* On GNU/Linux: __linux__ is defined */
#if defined(__GNU__) && !defined(__linux__)
#define ON_HURD 1
#else
#define ON_HURD 0
#endif


/*****************************************************************************
 *                                                                           *
 *                         HURD-SPECIFIC HEADERS                             *
 *                                                                           *
 *  When compiling for GNU/Hurd, we include the Hurd and Mach headers.      *
 *  When these headers are not available (e.g., cross-compiling), we     *
 *  provide minimal stub type definitions to allow compilation.         *
 *                                                                           *
 *****************************************************************************/

#if ON_HURD

/* GNU Hurd headers */
#include <hurd.h>           /* Hurd base definitions */
#include <hurd/fs.h>        /* Filesystem interface */
#include <hurd/trivfs.h>    /* Trivial filesystem translator interface */

/* GNU Mach headers */
#include <mach/mach.h>      /* Mach kernel interface */
#include <mach/port.h>      /* Mach port interface */
#include <mach/message.h>   /* Mach message interface */

/* Ensure MACH_PORT_NULL is defined */
#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

/* If the Hurd headers don't define these types, provide fallback definitions */
/* This handles cases where headers are missing or incomplete */
#ifndef _HURD_TRIVFS_H
/* trivfs.h should define struct iobuf, but if not, provide a minimal version */
struct iobuf {
    char *buf;              /* Buffer pointer */
    size_t buf_size;        /* Buffer size */
    off_t offset;           /* Current offset */
};

/* fs.h should define struct node, but if not, provide a minimal version */
struct node {
    void *data;             /* Node-specific data */
};

/* fs.h should define struct iouser, but if not, provide a minimal version */
struct iouser {
    int uid;                /* User ID */
    int gid;                /* Group ID */
};
#endif /* _HURD_TRIVFS_H */

#else
/* When not on Hurd, provide stub type definitions for compatibility */
/* These allow the code to compile on Linux for testing purposes */
#ifndef __mach_port_t_defined
#define __mach_port_t_defined 1
typedef unsigned int mach_port_t;
#endif

#ifndef __mach_msg_header_t_defined
#define __mach_msg_header_t_defined 1
typedef struct mach_msg_header *mach_msg_header_t;
#endif

#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif

/* Stub definitions for Hurd filesystem types */
struct iouser;      /* User credentials structure */
struct node;        /* Filesystem node structure */
struct iobuf;       /* I/O buffer structure */

#endif /* ON_HURD */


/* External trivfs variables declared in <hurd/trivfs.h> */
extern mach_port_t trivfs_control;

/* External filesystem operations table from <hurd/trivfs.h> */
extern char *fs_help;
extern error_t (*fs_open) (struct iouser *, int, mode_t, struct node *,
                           struct iobuf **);
extern error_t (*fs_read) (struct iouser *, struct iobuf *, off_t, size_t *,
                           size_t);
extern error_t (*fs_write) (struct iouser *, struct iobuf *, off_t, size_t,
                            size_t);

/* External trivfs server functions from <hurd/trivfs.h> */
extern error_t trivfs_server(mach_msg_header_t *, mach_msg_header_t *);
extern int trivfs_server_loop(void);

/* Forward declarations for our filesystem hook functions */
static error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                            struct node *node, struct iobuf **iobuf);
static error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                            off_t offset, size_t *len, size_t count);
static error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                             off_t offset, size_t len, size_t count);

#else
/* On Linux, provide stubs for Hurd-specific types */
typedef unsigned int mach_port_t;
typedef struct mach_msg_header *mach_msg_header_t;
#ifndef MACH_PORT_NULL
#define MACH_PORT_NULL 0
#endif
#endif /* ON_HURD */


/*****************************************************************************
 *                                                                           *
 *                          CONSTANT DEFINITIONS                             *
 *                                                                           *
 *  These constants define the limits and default parameters for the        *
 *  neural network implementation.                                          *
 *                                                                           *
 *****************************************************************************/

/* Maximum layers and neurons per layer for memory pre-allocation */
#define MAX_LAYERS 8
#define MAX_NEURONS_PER_LAYER 8192

/* Memory alignment for SIMD operations (16 bytes) */
#define SIMD_ALIGNMENT 16

/* Neuron parameters - biologically plausible values in millivolts */
#define RESET_POTENTIAL (-80.0f)   /* Resting membrane potential */
#define THRESHOLD (-55.0f)          /* Voltage threshold for spiking */
#define LEAK_RATE 0.1f             /* Voltage decay rate */
#define REFRACTORY_LENGTH 5       /* Post-spike silence period in timesteps */

/* Default network topology for initialization */
#define DEFAULT_LAYER_SIZES {10, 20, 5}
#define DEFAULT_LAYER_COUNT 3


/*****************************************************************************
 *                                                                           *
 *                          DATA STRUCTURES                                 *
 *                                                                           *
 *  These structures define the memory layout and organization of the       *
 *  neural network. All data is stored in contiguous memory blocks for      *
 *  efficiency and cache-friendliness.                                       *
 *                                                                           *
 *****************************************************************************/

/**
 * NetworkTopology - Defines the structure and parameters of the network
 * 
 * This structure contains all the configuration parameters for the neural
 * network, including layer sizes and neuron parameters. It is designed to
 * be compact (cache-line friendly) with minimal padding.
 */
typedef struct NetworkTopology {
    uint8_t layer_count;               /* Number of layers in the network */
    uint16_t layer_sizes[MAX_LAYERS];  /* Number of neurons in each layer */
    uint16_t input_size;               /* Size of input layer (same as layer_sizes[0]) */
    uint16_t output_size;              /* Size of output layer (same as last layer_size) */
    float reset_potential;             /* Reset voltage for neurons (mV) */
    float threshold;                   /* Spiking threshold voltage (mV) */
    float leak_rate;                   /* Voltage decay rate per timestep */
    uint8_t refractory_length;         /* Post-spike silence in timesteps */
    uint8_t _padding[3];               /* Padding to align to 16-byte boundary */
} NetworkTopology;

/**
 * CompactNeuralNetwork - Main neural network structure
 * 
 * This structure contains all the data and metadata for a complete neural
 * network. The actual neuron data (voltages, weights, biases) is stored in
 * a single contiguous memory block pointed to by 'memory_block'.
 * 
 * Memory layout: [voltages][weights][biases][input_buffer][output_buffer]
 * 
 * This design provides:
 *   - Single allocation for entire network
 *   - Better cache locality
 *   - Easier memory management
 *   - SIMD alignment for performance
 */
typedef struct CompactNeuralNetwork {
    NetworkTopology topology;        /* Network structure and parameters */
    size_t total_neurons;             /* Total number of neurons in all layers */
    size_t total_weights;             /* Total number of weight connections */
    size_t total_biases;             /* Total number of bias values */
    
    /* Offsets for each layer in the voltage array */
    size_t layer_offsets[MAX_LAYERS];
    
    /* Offsets for weights between layers */
    size_t weight_offsets[MAX_LAYERS - 1];
    
    /* Offsets for biases for each layer */
    size_t bias_offsets[MAX_LAYERS];
    
    /* Pointers to memory regions within the memory_block */
    float *voltages;                  /* Neuron voltages (size: total_neurons) */
    float *weights;                  /* Connection weights (size: total_weights) */
    float *biases;                    /* Bias values (size: total_biases) */
    float *input_buffer;              /* Input values buffer (size: input_size) */
    float *output_buffer;             /* Output values buffer (size: output_size) */
    
    /* Memory management */
    void *memory_block;              /* Pointer to single contiguous memory block */
    size_t memory_block_size;        /* Total size of memory block in bytes */
    
    /* Runtime state */
    bool initialized;                 /* Flag: is network initialized? */
    bool needs_reset;                 /* Flag: does network need reset? */
    size_t forward_pass_count;       /* Counter: number of forward passes */
    size_t neuron_activations;       /* Counter: total neuron activations */
} CompactNeuralNetwork;


/* Global network instance - used by the translator */
static CompactNeuralNetwork global_network = {0};


/*****************************************************************************
 *                                                                           *
 *                      SIGMOID ACTIVATION FUNCTION                          *
 *                                                                           *
 *  The sigmoid function is the core activation function for this neuron   *
 *  implementation. It maps any real number to the range (0, 1).            *
 *                                                                           *
 *  Mathematical definition: sigmoid(x) = 1 / (1 + exp(-x))                 *
 *                                                                           *
 *  This implementation uses expf() for single-precision floating point    *
 *  to maintain consistency with the float32 storage format.              *
 *                                                                           *
 *****************************************************************************/

/**
 * sigmoidf - Single-precision sigmoid activation function
 * 
 * @param x Input value (any real number)
 * @return Sigmoid of x in range (0, 1)
 * 
 * Computes: 1 / (1 + exp(-x))
 * 
 * This is declared inline to eliminate function call overhead in hot
 * paths (forward pass computation).
 */
static inline float
sigmoidf(float x)
{
    /* The sigmoid function: 1 / (1 + exp(-x)) */
    /* Using expf() for single-precision floating point */
    return 1.0f / (1.0f + expf(-x));
}


/*****************************************************************************
 *                                                                           *
 *                        MEMORY MANAGEMENT FUNCTIONS                       *
 *                                                                           *
 *  These functions handle aligned memory allocation and deallocation.     *
 *  Alignment to SIMD boundaries (16 bytes) is critical for performance.   *
 *                                                                           *
 *****************************************************************************/

/**
 * aligned_malloc - Allocate memory with specific alignment
 * 
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement (must be power of 2)
 * @return Pointer to allocated memory, or NULL on failure
 * 
 * Uses posix_memalign() which is POSIX-compliant and available on both
 * GNU/Linux and GNU/Hurd. The alignment must be a power of 2 and at
 * least sizeof(void*).
 */
static void *
aligned_malloc(size_t size, size_t alignment)
{
    void *ptr;
    
    /* posix_memalign allocates memory aligned to specified boundary */
    /* Returns 0 on success, EINVAL or ENOMEM on failure */
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    
    return ptr;
}


/**
 * aligned_free - Free memory allocated with aligned_malloc
 * 
 * @param ptr Pointer to memory to free (may be NULL)
 * 
 * Note: This is simply a wrapper around free() for consistency.
 * posix_memalign() allocates using malloc(), so free() is correct.
 */
static void
aligned_free(void *ptr)
{
    free(ptr);
}


/*****************************************************************************
 *                                                                           *
 *                      NEURAL NETWORK INITIALIZATION                       *
 *                                                                           *
 *  These functions initialize and clean up the neural network.            *
 *                                                                           *
 *****************************************************************************/

/**
 * network_init - Initialize a neural network with specified topology
 * 
 * @param net Pointer to network structure to initialize
 * @param layer_count Number of layers in the network
 * @param layer_sizes Array of layer sizes (number of neurons per layer)
 * @return 0 on success, -1 on error (with errno set)
 * 
 * This function:
 *   1. Validates input parameters
 *   2. Copies layer configuration
 *   3. Calculates total neurons, weights, biases
 *   4. Pre-calculates memory offsets for each layer
 *   5. Allocates single contiguous memory block
 *   6. Initializes all values (voltages, weights, biases)
 * 
 * Memory layout in memory_block:
 *   [voltages (total_neurons)][weights (total_weights)]
 *   [biases (total_biases)][input_buffer (input_size)][output_buffer (output_size)]
 */
static int
network_init(CompactNeuralNetwork *net,
             uint8_t layer_count,
             const uint16_t *layer_sizes)
{
    /* Input validation - check for NULL pointers */
    if (!net || !layer_sizes) {
        errno = EINVAL;
        return -1;
    }
    
    /* Validate layer count - must have at least input and output layers */
    if (layer_count < 2 || layer_count > MAX_LAYERS) {
        errno = EINVAL;
        return -1;
    }
    
    /* Validate each layer size - must be within bounds */
    for (int i = 0; i < layer_count; i++) {
        if (layer_sizes[i] == 0 || layer_sizes[i] > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
    }
    
    /* Copy topology configuration */
    net->topology.layer_count = layer_count;
    net->topology.input_size = layer_sizes[0];
    net->topology.output_size = layer_sizes[layer_count - 1];
    
    for (int i = 0; i < layer_count; i++) {
        net->topology.layer_sizes[i] = layer_sizes[i];
    }
    
    /* Set default neuron parameters */
    net->topology.reset_potential = RESET_POTENTIAL;
    net->topology.threshold = THRESHOLD;
    net->topology.leak_rate = LEAK_RATE;
    net->topology.refractory_length = REFRACTORY_LENGTH;
    
    /* Calculate total neurons across all layers */
    net->total_neurons = 0;
    for (int i = 0; i < layer_count; i++) {
        net->total_neurons += layer_sizes[i];
    }
    
    /* Calculate total weights and biases */
    net->total_weights = 0;
    net->total_biases = 0;
    for (int i = 1; i < layer_count; i++) {
        net->total_weights += (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        net->total_biases += layer_sizes[i];
    }
    
    /* Pre-calculate layer offsets for voltage array */
    net->layer_offsets[0] = 0;
    for (int i = 1; i < layer_count; i++) {
        net->layer_offsets[i] = net->layer_offsets[i - 1] + layer_sizes[i - 1];
    }
    
    /* Pre-calculate weight offsets between layers */
    if (layer_count > 1) {
        net->weight_offsets[0] = 0;
        for (int i = 1; i < layer_count - 1; i++) {
            net->weight_offsets[i] = net->weight_offsets[i - 1] +
                                      (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        }
    }
    
    /* Pre-calculate bias offsets for each layer */
    net->bias_offsets[0] = 0;
    for (int i = 1; i < layer_count; i++) {
        net->bias_offsets[i] = net->bias_offsets[i - 1] + layer_sizes[i];
    }
    
    /* Calculate sizes for each memory region */
    size_t voltages_size = net->total_neurons * sizeof(float);
    size_t weights_size = net->total_weights * sizeof(float);
    size_t biases_size = net->total_biases * sizeof(float);
    size_t input_size = layer_sizes[0] * sizeof(float);
    size_t output_size = layer_sizes[layer_count - 1] * sizeof(float);
    
    /* Total memory block size */
    net->memory_block_size = voltages_size + weights_size + biases_size +
                             input_size + output_size;
    
    /* Allocate single contiguous memory block with SIMD alignment */
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        errno = ENOMEM;
        return -1;
    }
    
    /* Set up pointers to each region within the memory block */
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
    
    /* Initialize all neuron voltages to reset potential */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Initialize weights with small random values using a simple PRNG */
    /* Using a simple linear congruential generator for reproducibility */
    for (size_t i = 0; i < net->total_weights; i++) {
        /* Simple pseudo-random number generator */
        /* Multiplier is a large prime number for good distribution */
        uint32_t seed = (uint32_t)i * 2654435761U;
        /* Extract 23 bits for mantissa (0x007FFFFF = 2^23 - 1) */
        float random = (float)(seed & 0x007FFFFF) / (float)0x007FFFFF;
        /* Scale to range [-0.2, 0.2] for small initial weights */
        net->weights[i] = random * 0.4f - 0.2f;
    }
    
    /* Initialize all biases to zero */
    for (size_t i = 0; i < net->total_biases; i++) {
        net->biases[i] = 0.0f;
    }
    
    /* Set runtime state flags */
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return 0;
}


/**
 * network_free - Free all memory allocated for a neural network
 * 
 * @param net Pointer to network to free (may be NULL)
 * 
 * This function safely deallocates all memory used by the network.
 * It handles NULL pointers gracefully and clears all pointers after freeing.
 */
static void
network_free(CompactNeuralNetwork *net)
{
    if (!net) {
        return;
    }
    
    /* Free the single memory block if allocated */
    if (net->memory_block) {
        aligned_free(net->memory_block);
        net->memory_block = NULL;
    }
    
    /* Clear all pointers */
    net->voltages = NULL;
    net->weights = NULL;
    net->biases = NULL;
    net->input_buffer = NULL;
    net->output_buffer = NULL;
    
    /* Reset state flags */
    net->initialized = false;
    net->needs_reset = false;
}


/**
 * network_reset - Reset network state to initial values
 * 
 * @param net Pointer to network to reset
 * 
 * Resets all neuron voltages to the reset potential without deallocating
 * memory. This is useful for re-initializing between runs.
 */
static void
network_reset(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized) {
        return;
    }
    
    /* Reset all neuron voltages to reset potential */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Reset runtime counters */
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    net->needs_reset = false;
}


/*****************************************************************************
 *                                                                           *
 *                        FORWARD PASS FUNCTION                              *
 *                                                                           *
 *  This is the core computation function that propagates input through   *
 *  the network from input layer to output layer.                          *
 *                                                                           *
 *****************************************************************************/

/**
 * network_forward - Perform forward pass through the network
 * 
 * @param net Pointer to initialized network
 * 
 * This function:
 *   1. Copies input buffer to input layer voltages
 *   2. For each layer (starting from first hidden layer):
 *      a. For each neuron in the layer:
 *         i. Compute weighted sum of inputs from previous layer
 *         ii. Add bias
 *         iii. Apply sigmoid activation function
 *   3. Copy output layer voltages to output buffer
 *   4. Update runtime counters
 * 
 * The algorithm is optimized for:
 *   - Sequential memory access (cache-friendly)
 *   - Minimal branching in hot loops
 *   - Pre-calculated memory offsets (no runtime calculations)
 */
static void
network_forward(CompactNeuralNetwork *net)
{
    /* Validate network state */
    if (!net || !net->initialized || net->topology.layer_count < 2) {
        return;
    }
    
    /* Copy input buffer to input layer voltages */
    if (net->input_buffer) {
        memcpy(net->voltages, net->input_buffer,
               net->topology.input_size * sizeof(float));
    }
    
    /* Process each layer starting from the first hidden layer */
    for (int layer = 1; layer < net->topology.layer_count; layer++) {
        size_t prev_size = net->topology.layer_sizes[layer - 1];
        size_t curr_size = net->topology.layer_sizes[layer];
        size_t prev_offset = net->layer_offsets[layer - 1];
        size_t curr_offset = net->layer_offsets[layer];
        
        /* Calculate weight offset for this layer */
        size_t weight_offset = 0;
        for (int l = 1; l < layer; l++) {
            weight_offset += (size_t)net->topology.layer_sizes[l] * 
                            (size_t)net->topology.layer_sizes[l - 1];
        }
        
        /* Calculate bias offset for this layer */
        size_t bias_offset = net->bias_offsets[layer - 1];
        
        /* Process each neuron in current layer */
        for (size_t n = 0; n < curr_size; n++) {
            /* Start with bias value for this neuron */
            float sum = net->biases[bias_offset + n];
            
            /* Pointer to weights for this neuron */
            float *w = net->weights + weight_offset + n * prev_size;
            
            /* Pointer to previous layer voltages */
            float *v = net->voltages + prev_offset;
            
            /* Compute weighted sum of inputs from previous layer */
            for (size_t p = 0; p < prev_size; p++) {
                sum += v[p] * w[p];
            }
            
            /* Apply sigmoid activation function */
            net->voltages[curr_offset + n] = sigmoidf(sum);
            
            /* Increment activation counter */
            net->neuron_activations++;
        }
    }
    
    /* Copy output layer voltages to output buffer */
    if (net->output_buffer) {
        size_t out_offset = net->layer_offsets[net->topology.layer_count - 1];
        memcpy(net->output_buffer, net->voltages + out_offset,
               net->topology.output_size * sizeof(float));
    }
    
    /* Increment forward pass counter */
    net->forward_pass_count++;
}


/*****************************************************************************
 *                                                                           *
 *                     CONFIGURATION PARSING FUNCTIONS                       *
 *                                                                           *
 *  These functions parse configuration and input strings.                *
 *                                                                           *
 *****************************************************************************/

/**
 * parse_config_string - Parse layer sizes from a configuration string
 * 
 * @param config_str String containing layer sizes (comma or space separated)
 * @param layer_sizes Array to store parsed layer sizes
 * @param max_layers Maximum number of layers to parse
 * @return Number of layers parsed (>= 2), or -1 on error
 * 
 * This function parses a string like "10,20,5" or "10 20 5" into an array
 * of layer sizes. It handles various delimiters and whitespace.
 */
static int
parse_config_string(const char *config_str, uint16_t *layer_sizes, int max_layers)
{
    /* Validate inputs */
    if (!config_str || !layer_sizes || max_layers < 2) {
        errno = EINVAL;
        return -1;
    }
    
    const char *ptr = config_str;
    int count = 0;
    
    /* Parse until end of string or maximum layers reached */
    while (*ptr != '\0' && count < max_layers) {
        /* Skip whitespace and commas */
        while (*ptr == ' ' || *ptr == '\t' || *ptr == ',') ptr++;
        if (*ptr == '\0') break;
        
        /* Convert numeric value */
        char *endptr;
        long value = strtol(ptr, &endptr, 10);
        
        /* Check for conversion errors */
        if (ptr == endptr) {
            errno = EINVAL;
            return -1;
        }
        
        /* Validate layer size */
        if (value < 1 || value > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
        
        /* Store layer size */
        layer_sizes[count++] = (uint16_t)value;
        ptr = endptr;
    }
    
    /* Must have at least 2 layers (input and output) */
    if (count < 2) {
        errno = EINVAL;
        return -1;
    }
    
    return count;
}


/**
 * parse_input_string - Parse input values from a string
 * 
 * @param net Pointer to initialized network
 * @param input_str String containing input values (comma or space separated)
 * @return true if parsing and forward pass successful, false otherwise
 * 
 * This function parses input values from a string and triggers a forward
 * pass through the network. It expects the number of values to match the
 * input layer size.
 */
static bool
parse_input_string(CompactNeuralNetwork *net, const char *input_str)
{
    /* Validate inputs */
    if (!net || !net->initialized || !input_str) {
        return false;
    }
    
    const char *ptr = input_str;
    size_t idx = 0;
    
    /* Parse until end of string or input buffer filled */
    while (*ptr != '\0' && idx < net->topology.input_size) {
        /* Skip delimiters */
        while (*ptr == ',' || *ptr == ' ' || *ptr == '\t' || *ptr == '\n') ptr++;
        if (*ptr == '\0') break;
        
        /* Convert floating-point value */
        char *endptr;
        float val = strtof(ptr, &endptr);
        
        /* Check for conversion errors */
        if (ptr == endptr) {
            ptr++;
            continue;
        }
        
        /* Store input value */
        net->input_buffer[idx++] = val;
        ptr = endptr;
    }
    
    /* Only trigger forward pass if we have enough inputs */
    if (idx >= net->topology.input_size) {
        network_forward(net);
        return true;
    }
    
    return false;
}


/**
 * network_save - Save network state to a file
 * 
 * @param net Pointer to network to save
 * @param filename Path to save file
 * @return true on success, false on failure
 * 
 * Saves the complete network state including topology, weights, and biases.
 * This allows networks to be trained offline and loaded into the translator.
 */
static bool
network_save(const CompactNeuralNetwork *net, const char *filename)
{
    if (!net || !net->initialized || !filename) {
        return false;
    }
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return false;
    }
    
    /* Write topology */
    if (fwrite(&net->topology, sizeof(NetworkTopology), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    /* Write total counts */
    if (fwrite(&net->total_neurons, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    if (fwrite(&net->total_weights, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    if (fwrite(&net->total_biases, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    /* Write layer offsets */
    if (fwrite(net->layer_offsets, sizeof(size_t), net->topology.layer_count, fp) != 
        net->topology.layer_count) {
        fclose(fp);
        return false;
    }
    
    /* Write weight offsets */
    if (net->topology.layer_count > 1) {
        if (fwrite(net->weight_offsets, sizeof(size_t), net->topology.layer_count - 1, fp) != 
            net->topology.layer_count - 1) {
            fclose(fp);
            return false;
        }
    }
    
    /* Write bias offsets */
    if (fwrite(net->bias_offsets, sizeof(size_t), net->topology.layer_count, fp) != 
        net->topology.layer_count) {
        fclose(fp);
        return false;
    }
    
    /* Write voltages */
    if (fwrite(net->voltages, sizeof(float), net->total_neurons, fp) != 
        net->total_neurons) {
        fclose(fp);
        return false;
    }
    
    /* Write weights */
    if (fwrite(net->weights, sizeof(float), net->total_weights, fp) != 
        net->total_weights) {
        fclose(fp);
        return false;
    }
    
    /* Write biases */
    if (fwrite(net->biases, sizeof(float), net->total_biases, fp) != 
        net->total_biases) {
        fclose(fp);
        return false;
    }
    
    fclose(fp);
    return true;
}


/**
 * network_load - Load network state from a file
 * 
 * @param net Pointer to network to load into
 * @param filename Path to load file
 * @return true on success, false on failure
 * 
 * Loads a complete network state from a file saved with network_save().
 */
static bool
network_load(CompactNeuralNetwork *net, const char *filename)
{
    if (!net || !filename) {
        return false;
    }
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return false;
    }
    
    /* Free existing network memory */
    network_free(net);
    
    /* Read topology */
    if (fread(&net->topology, sizeof(NetworkTopology), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    /* Read total counts */
    if (fread(&net->total_neurons, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    if (fread(&net->total_weights, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    if (fread(&net->total_biases, sizeof(size_t), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    /* Read layer offsets */
    if (fread(net->layer_offsets, sizeof(size_t), net->topology.layer_count, fp) != 
        net->topology.layer_count) {
        fclose(fp);
        return false;
    }
    
    /* Read weight offsets */
    if (net->topology.layer_count > 1) {
        if (fread(net->weight_offsets, sizeof(size_t), net->topology.layer_count - 1, fp) != 
            net->topology.layer_count - 1) {
            fclose(fp);
            return false;
        }
    }
    
    /* Read bias offsets */
    if (fread(net->bias_offsets, sizeof(size_t), net->topology.layer_count, fp) != 
        net->topology.layer_count) {
        fclose(fp);
        return false;
    }
    
    /* Allocate memory block */
    net->memory_block_size = net->total_neurons * sizeof(float) +
                             net->total_weights * sizeof(float) +
                             net->total_biases * sizeof(float) +
                             net->topology.input_size * sizeof(float) +
                             net->topology.output_size * sizeof(float);
    
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        fclose(fp);
        return false;
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
    
    /* Read voltages */
    if (fread(net->voltages, sizeof(float), net->total_neurons, fp) != 
        net->total_neurons) {
        network_free(net);
        fclose(fp);
        return false;
    }
    
    /* Read weights */
    if (fread(net->weights, sizeof(float), net->total_weights, fp) != 
        net->total_weights) {
        network_free(net);
        fclose(fp);
        return false;
    }
    
    /* Read biases */
    if (fread(net->biases, sizeof(float), net->total_biases, fp) != 
        net->total_biases) {
        network_free(net);
        fclose(fp);
        return false;
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
 *                      FILESYSTEM OPERATION HOOKS (Hurd only)               *
 *                                                                           *
 *  These functions implement the trivfs translator interface for Hurd.     *
 *  They are only compiled when building for GNU/Hurd.                      *
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
 * 
 * This function initializes the network if not already initialized.
 * It is called when a user opens the translator (e.g., cat /llm).
 */
static error_t
fs_open_hook(struct iouser *cred, int flags, mode_t mode,
             struct node *node, struct iobuf **iobuf)
{
    /* Suppress unused parameter warnings */
    (void)cred; (void)flags; (void)mode; (void)node;
    
    /* Initialize output parameter */
    *iobuf = NULL;
    
    /* Lazy initialization: only initialize if not already initialized */
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
 * @param count Maximum number of bytes to read (unused)
 * @return 0 on success, error code on failure
 * 
 * This function generates the output shown when the user reads the
 * translator (e.g., cat /llm). It displays network information, memory
 * usage, and current output values.
 */
static error_t
fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
             off_t offset, size_t *len, size_t count)
{
    /* Suppress unused parameter warnings */
    (void)cred; (void)offset; (void)count;
    
    /* Lazy initialization */
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return ENOMEM;
        }
    }
    
    /* Buffer for constructing response */
    char buffer[4096];
    size_t written = 0;
    
    /* Build header */
    written = snprintf(buffer, sizeof(buffer),
                      "LLM Sigmoid Neuron Translator - GNU Hurd\n"
                      "===========================================\n\n");
    
    /* Add network topology information */
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
    
    /* Add memory and statistics */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Memory: %.2f KB, Neurons: %zu, Weights: %zu\n\n",
                       (double)global_network.memory_block_size / 1024.0,
                       global_network.total_neurons,
                       global_network.total_weights);
    
    /* Add current output values */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Output:\n");
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        written += snprintf(buffer + written, sizeof(buffer) - written,
                          "  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
    }
    
    /* Add usage instructions */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "\nUsage:\n"
                       "  Configure: echo '<layers>' > /llm\n"
                       "  Input: echo '<values>' > /llm\n"
                       "  Save: echo 'save <filename>' > /llm\n"
                       "  Load: echo 'load <filename>' > /llm\n");
    
    /* Ensure buffer is null-terminated and within bounds */
    if (written >= sizeof(buffer)) {
        written = sizeof(buffer) - 1;
    }
    buffer[written] = '\0';
    
    /* If iobuf is NULL, just set the length and return */
    if (iobuf == NULL) {
        *len = written;
        return 0;
    }
    
    /* Copy data to output buffer */
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
 * @param count Maximum number of bytes to write (unused)
 * @return 0 on success, error code on failure
 * 
 * This function handles user input to the translator. It parses commands
 * for configuration, input, save, and load operations.
 */
static error_t
fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
              off_t offset, size_t len, size_t count)
{
    /* Suppress unused parameter warnings */
    (void)cred; (void)offset; (void)count;
    
    /* Validate input */
    if (iobuf == NULL || len == 0 || iobuf->buf == NULL) {
        return EINVAL;
    }
    
    /* Lazy initialization */
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
            return ENOMEM;
        }
    }
    
    /* Copy input data to temporary buffer */
    char temp[1024];
    if (len >= sizeof(temp)) {
        len = sizeof(temp) - 1;
    }
    memcpy(temp, iobuf->buf, len);
    temp[len] = '\0';
    
    /* Check for save command */
    if (strncmp(temp, "save ", 5) == 0) {
        const char *filename = temp + 5;
        if (network_save(&global_network, filename)) {
            return 0;
        }
        return EINVAL;
    }
    
    /* Check for load command */
    if (strncmp(temp, "load ", 5) == 0) {
        const char *filename = temp + 5;
        if (network_load(&global_network, filename)) {
            return 0;
        }
        return EINVAL;
    }
    
    /* Try to parse as configuration string (layer sizes) */
    uint16_t layers[MAX_LAYERS];
    int layer_count = parse_config_string(temp, layers, MAX_LAYERS);
    
    if (layer_count > 0) {
        /* Reconfigure network with new topology */
        network_free(&global_network);
        if (network_init(&global_network, layer_count, layers) != 0) {
            return EINVAL;
        }
        return 0;
    }
    
    /* Try to parse as input string */
    if (parse_input_string(&global_network, temp)) {
        return 0;
    }
    
    /* Unrecognized input */
    return EINVAL;
}


/**
 * trivfs_demuxer - Message demultiplexer for trivfs translator
 * 
 * @param inmsg Incoming Mach message header
 * @param outmsg Outgoing Mach message header
 * @return error code
 * 
 * This function delegates message handling to the standard trivfs server.
 * It is required by the trivfs interface.
 */
int
trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    return trivfs_server(inmsg, outmsg);
}


/**
 * main - Entry point for the translator
 * 
 * @param argc Argument count (unused)
 * @param argv Argument vector (unused)
 * @return Exit code (never reached on Hurd)
 * 
 * This function:
 *   1. Initializes the global network
 *   2. Sets up the trivfs interface (help string, hooks)
 *   3. Starts the trivfs server loop
 * 
 * On Hurd, this function runs as a translator and never exits.
 */
int
main(int argc, char **argv)
{
    /* Suppress unused parameter warnings */
    (void)argc; (void)argv;
    
    /* Initialize global network state */
    global_network.initialized = false;
    global_network.memory_block = NULL;
    
    /* Initialize trivfs control port */
    trivfs_control = MACH_PORT_NULL;
    
    /* Initialize network with default topology */
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        /* If initialization fails, still continue with empty network */
        global_network.initialized = false;
    }
    
    /* Set help string for translator */
    fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
              "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator\n"
              "  Commands:\n"
              "    echo '<layers>' > /node  - Configure network topology\n"
              "    echo '<values>' > /node  - Set input and compute output\n"
              "    echo 'save <file>' > /node - Save network to file\n"
              "    echo 'load <file>' > /node - Load network from file\n"
              "    cat /node               - View network info and output";
    
    /* Register our filesystem hook functions */
    fs_open = fs_open_hook;
    fs_read = fs_read_hook;
    fs_write = fs_write_hook;
    
    /* Start the trivfs server loop - this never returns */
    return trivfs_server_loop();
}


#else /* !ON_HURD */

/*****************************************************************************
 *                                                                           *
 *                         TEST MODE FOR LINUX                              *
 *                                                                           *
 *  This section provides a test mode that runs on GNU/Linux systems      *
 *  without Hurd. It demonstrates the neural network functionality.        *
 *                                                                           *
 *****************************************************************************/

/**
 * main - Test mode entry point for Linux
 * 
 * @param argc Argument count (unused)
 * @param argv Argument vector (unused)
 * @return 0 on success, 1 on failure
 * 
 * This test mode:
 *   1. Initializes a network with default topology
 *   2. Sets sample input values
 *   3. Performs a forward pass
 *   4. Displays the output
 *   5. Tests configuration parsing
 *   6. Tests input parsing and forward pass
 *   7. Cleans up and exits
 */
int
main(int argc, char **argv)
{
    /* Suppress unused parameter warnings */
    (void)argc; (void)argv;
    
    /* Display header */
    printf("LLM Sigmoid Neuron Translator - Test Mode (Linux)\n");
    printf("===================================================\n\n");
    
    /* Initialize network with default topology */
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    if (network_init(&global_network, DEFAULT_LAYER_COUNT, layers) != 0) {
        fprintf(stderr, "Failed to initialize network: %s\n", strerror(errno));
        return 1;
    }
    
    /* Display network information */
    printf("Network initialized with %d layers\n", global_network.topology.layer_count);
    for (int i = 0; i < global_network.topology.layer_count; i++) {
        printf("  Layer %d: %d neurons\n", i, global_network.topology.layer_sizes[i]);
    }
    printf("\n");
    
    /* Set sample input values */
    printf("Setting sample input values...\n");
    for (int i = 0; i < global_network.topology.input_size; i++) {
        global_network.input_buffer[i] = (float)i * 0.1f;
        printf("  Input[%d] = %.2f\n", i, global_network.input_buffer[i]);
    }
    printf("\n");
    
    /* Perform forward pass */
    network_forward(&global_network);
    
    /* Display results */
    printf("Forward pass completed\n");
    printf("Output:\n");
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        printf("  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
    }
    
    printf("\nMemory: %.2f KB, Neurons: %zu, Weights: %zu\n",
           (double)global_network.memory_block_size / 1024.0,
           global_network.total_neurons,
           global_network.total_weights);
    
    printf("Forward passes: %zu, Neuron activations: %zu\n\n",
           global_network.forward_pass_count,
           global_network.neuron_activations);
    
    /* Test configuration parsing */
    printf("Testing configuration parsing...\n");
    uint16_t test_layers[MAX_LAYERS];
    int test_count = parse_config_string("5,10,5", test_layers, MAX_LAYERS);
    if (test_count > 0) {
        printf("  Config parsing works: %d layers [", test_count);
        for (int i = 0; i < test_count; i++) {
            printf("%d%s", test_layers[i], (i < test_count - 1) ? ", " : "");
        }
        printf("]\n\n");
    } else {
        printf("  Config parsing failed\n\n");
    }
    
    /* Test input parsing and forward pass */
    printf("Testing input parsing and forward pass...\n");
    if (parse_input_string(&global_network, "0.5,0.3,0.8,0.1,0.9,0.2,0.4,0.6,0.0,0.7")) {
        printf("  Input parsing and forward pass works\n");
        printf("  New output:\n");
        for (size_t i = 0; i < global_network.topology.output_size; i++) {
            printf("    [%zu]: %.6f\n", i, global_network.output_buffer[i]);
        }
        printf("\n");
    } else {
        printf("  Input parsing failed\n\n");
    }
    
    /* Test network reset */
    printf("Testing network reset...\n");
    network_reset(&global_network);
    printf("  Network reset completed\n\n");
    
    /* Test network save and load */
    printf("Testing network save and load...\n");
    if (network_save(&global_network, "/tmp/test-network.bin")) {
        printf("  Network saved successfully\n");
        
        CompactNeuralNetwork test_net = {0};
        if (network_load(&test_net, "/tmp/test-network.bin")) {
            printf("  Network loaded successfully\n");
            printf("  Loaded network has %zu neurons, %zu weights\n",
                   test_net.total_neurons, test_net.total_weights);
            network_free(&test_net);
        } else {
            printf("  Network load failed\n");
        }
        remove("/tmp/test-network.bin");
    } else {
        printf("  Network save failed\n");
    }
    printf("\n");
    
    /* Clean up */
    network_free(&global_network);
    
    printf("Test mode completed successfully\n");
    
    return 0;
}

#endif /* ON_HURD */


/*
 * Editor modelines - for automatic mode setting in text editors
 */

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
 * Key Features:
 *   - Sigmoid activation function for LLM-style neurons
 *   - Contiguous memory layout for cache efficiency
 *   - SIMD-aligned allocations
 *   - Low memory footprint (single malloc for all network data)
 *   - Low CPU usage (optimized forward pass)
 *   - Full GNU Hurd trivfs translator integration
 *   - POSIX compliant
 *   - C23 standard compliant
 * 
 * USAGE:
 *   1. Compile: make
 *   2. Install: sudo make install
 *   3. Set translator: sudo settrans -c /llm /hurd/sigmoid-neuron-translator
 *   4. Read: cat /llm
 *   5. Write input: echo "0.5,0.3,0.8" > /llm
 *
 * ============================================================================
 */


/*****************************************************************************
 *                                                                           *
 *                      INCLUDE DIRECTIVES AND DEFINITIONS                    *
 *                                                                           *
 *****************************************************************************/

/*
 * Standard C headers
 * We define _GNU_SOURCE to get POSIX extensions on GNU systems
 * and _POSIX_C_SOURCE to ensure POSIX compliance
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>      /* Standard I/O operations */
#include <stdlib.h>     /* Memory allocation, exit() */
#include <string.h>     /* memcpy(), memset(), strerror() */
#include <math.h>       /* expf(), mathematical functions */
#include <errno.h>      /* Error number definitions */
#include <error.h>      /* error() function for error reporting */
#include <stdint.h>     /* Fixed-width integer types */
#include <stdbool.h>    /* Boolean type and values */
#include <stddef.h>     /* NULL, size_t, ptrdiff_t */
#include <stdalign.h>   /* alignas, alignof for C23 alignment */
#include <pthread.h>    /* POSIX threads for Hurd compatibility */


/*
 * Hurd-specific headers
 * These provide the Mach IPC and trivfs (trivial filesystem) interfaces
 * required for creating a GNU Hurd translator
 */
#include <hurd.h>       /* GNU Hurd basic definitions */
#include <hurd/fs.h>    /* Filesystem interface */
#include <hurd/trivfs.h>/* Trivial filesystem implementation */


/*
 * Mach message header for IPC communication
 * This is required for the trivfs demuxer function
 */
#include <mach/mach.h>
#include <mach/mach_msg.h>


/*****************************************************************************
 *                                                                           *
 *                          CONSTANT DEFINITIONS                             *
 *                                                                           *
 *****************************************************************************/

/*
 * Network Topology Limits
 * 
 * MAX_LAYERS: Maximum number of layers in the neural network
 *     - Limited to 8 to keep memory usage predictable
 *     - Can be increased if needed, but affects stack usage
 * 
 * MAX_NEURONS_PER_LAYER: Maximum neurons in a single layer
 *     - 8192 provides a good balance between capacity and memory
 *     - Each neuron uses 4 bytes (float32), so 8192 = 32KB per layer
 * 
 * SIMD_ALIGNMENT: Memory alignment for SIMD operations
 *     - 16 bytes is typical for SSE/AVX
 *     - Ensures efficient vectorized operations
 */
#define MAX_LAYERS 8
#define MAX_NEURONS_PER_LAYER 8192
#define SIMD_ALIGNMENT 16


/*
 * Neuron Constants
 * 
 * These values control the behavior of individual neurons:
 * 
 * RESET_POTENTIAL: Membrane potential when neuron is at rest (mV)
 *     - Typical biological value: -70 to -80 mV
 *     - Used after a spike to reset the neuron
 * 
 * THRESHOLD: Voltage threshold for firing (mV)
 *     - Typical biological value: -55 to -40 mV
 *     - When voltage exceeds this, neuron fires
 * 
 * LEAK_RATE: Rate at which voltage decays toward reset potential
 *     - Range: 0.0 (no leak) to 1.0 (instant reset)
 *     - Typical value: 0.1 to 0.3
 *     - Controls how quickly neurons forget their state
 * 
 * REFRACTORY_LENGTH: Number of timesteps neuron cannot fire after spiking
 *     - Prevents immediate re-firing
 *     - Typical value: 5-10 timesteps
 *     - 0 means no refractory period
 */
#define RESET_POTENTIAL (-80.0f)
#define THRESHOLD (-55.0f)
#define LEAK_RATE 0.1f
#define REFRACTORY_LENGTH 5


/*
 * Default Network Topology
 * 
 * This defines the structure of the neural network if none is specified:
 *   - Input layer: 10 neurons
 *   - Hidden layer: 20 neurons
 *   - Output layer: 5 neurons
 * 
 * This can be changed at runtime by writing a configuration string
 * to the translator (e.g., "10,20,5" or "5,10,10,3")
 */
#define DEFAULT_LAYER_SIZES {10, 20, 5}
#define DEFAULT_LAYER_COUNT 3


/*****************************************************************************
 *                                                                           *
 *                          DATA STRUCTURES                                 *
 *                                                                           *
 *****************************************************************************/

/*
 * NetworkTopology - Defines the structure of the neural network
 * 
 * This structure contains all the metadata about the network's architecture:
 *   - Number of layers
 *   - Size of each layer (number of neurons)
 *   - Input and output sizes
 *   - Neuron parameters
 * 
 * The structure is padded to ensure proper alignment when used in arrays.
 * This is important for performance on modern CPUs with cache lines.
 */
typedef struct NetworkTopology {
    /* Number of layers in the network (2-8) */
    uint8_t layer_count;
    
    /* Size of each layer (number of neurons) */
    uint16_t layer_sizes[MAX_LAYERS];
    
    /* Size of input and output layers */
    uint16_t input_size;
    uint16_t output_size;
    
    /* Neuron parameters */
    float reset_potential;   /* Reset potential in mV */
    float threshold;         /* Firing threshold in mV */
    float leak_rate;         /* Leak rate (0.0 to 1.0) */
    uint8_t refractory_length; /* Refractory period length */
    
    /* Padding to align to 8 bytes */
    uint8_t _padding[3];
} NetworkTopology;


/*
 * CompactNeuralNetwork - Main neural network structure
 * 
 * This is the heart of our implementation. It stores:
 *   - Network topology (structure)
 *   - All neural data in contiguous memory blocks
 *   - Memory management information
 *   - Usage statistics
 * 
 * Key Design Decisions:
 * 
 * 1. CONTIGUOUS MEMORY: All network data (voltages, weights, biases, buffers)
 *    is stored in a single memory block allocated with aligned_malloc().
 *    This provides several benefits:
 *      - Reduces memory fragmentation
 *      - Improves cache locality (all data is close together)
 *      - Requires only one malloc/free call
 *      - Easier memory management
 * 
 * 2. PRE-CALCULATED OFFSETS: We calculate all memory offsets once during
 *    initialization, so the forward pass can access data directly without
 *    pointer arithmetic at each step.
 * 
 * 3. MINIMAL STATE: We store only what we need. Each neuron's state is just
 *    its current voltage. Weights and biases are stored once and reused.
 * 
 * 4. ALIGNED ALLOCATION: All memory is aligned to SIMD_ALIGNMENT (16 bytes)
 *    to enable efficient vector operations on modern CPUs.
 */
typedef struct CompactNeuralNetwork {
    /* Network structure */
    NetworkTopology topology;
    
    /* Sizes */
    size_t total_neurons;    /* Total number of neurons in all layers */
    size_t total_weights;    /* Total number of weights (connections) */
    size_t total_biases;     /* Total number of bias values */
    
    /* Memory layout offsets (pre-calculated for efficiency) */
    size_t layer_offsets[MAX_LAYERS];     /* Offset of each layer in voltages */
    size_t weight_offsets[MAX_LAYERS - 1]; /* Offset of each layer's weights */
    size_t bias_offsets[MAX_LAYERS];      /* Offset of each layer's biases */
    
    /* Data pointers (into memory_block) */
    float *voltages;         /* Neuron voltages (mV) */
    float *weights;          /* Connection weights */
    float *biases;           /* Bias values for each neuron */
    float *input_buffer;     /* Input buffer */
    float *output_buffer;    /* Output buffer */
    
    /* Memory management */
    void *memory_block;      /* Single contiguous memory block */
    size_t memory_block_size;/* Size of memory_block in bytes */
    
    /* State */
    bool initialized;        /* Has the network been initialized? */
    bool needs_reset;        /* Does the network need to be reset? */
    
    /* Statistics (for debugging and monitoring) */
    size_t forward_pass_count;   /* Number of forward passes performed */
    size_t neuron_activations;   /* Total number of neuron activations */
    
} CompactNeuralNetwork;


/*
 * Global Network Instance
 * 
 * We maintain a single global network instance that is shared by all
 * file operations. This is the translator's state.
 * 
 * Why a global?
 *   - Translators are typically single-threaded (Hurd's trivfs handles
 *     one request at a time via the demuxer)
 *   - Simplifies memory management
 *   - Easy to access from all callback functions
 * 
 * Note: If we wanted to support concurrent access, we would need to
 * add proper locking. But for a translator, this is usually unnecessary.
 */
static CompactNeuralNetwork global_network = {0};


/*****************************************************************************
 *                                                                           *
 *                    HURD TRIVFS DECLARATIONS                              *
 *                                                                           *
 *****************************************************************************/

/*
 * Trivfs Variables
 * 
 * These are the global variables that the trivfs library uses to dispatch
 * filesystem operations. We declare them as extern and then define them
 * in our main() function.
 * 
 * The trivfs library expects these to be defined in the translator.
 */

/* Control port for the translator */
extern mach_port_t trivfs_control;

/* Help text for the translator */
extern char *fs_help;

/* File operation function pointers */
extern error_t (*fs_open) (struct iouser *cred, int flags, mode_t mode,
                           struct node *node, struct iobuf **iobuf);
extern error_t (*fs_read) (struct iouser *cred, struct iobuf *iobuf,
                           off_t offset, size_t *len, size_t count);
extern error_t (*fs_write) (struct iouser *cred, struct iobuf *iobuf,
                            off_t offset, size_t len, size_t count);


/*
 * Trivfs Function Declarations
 * 
 * These are the functions provided by the trivfs library that we need
 * to call or implement.
 */

/*
 * trivfs_server - The main server function that handles incoming messages
 * 
 * This function is called by trivfs_server_loop() to process each incoming
 * Mach message. It dispatches to the appropriate filesystem operation
 * based on the message type.
 * 
 * Parameters:
 *   inmsg  - Pointer to the incoming Mach message header
 *   outmsg - Pointer to the outgoing Mach message header
 * 
 * Returns:
 *   error_t - Error code (0 for success)
 */
extern error_t trivfs_server(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg);

/*
 * trivfs_server_loop - The main server loop
 * 
 * This function runs the main message loop, receiving and processing
 * Mach messages indefinitely. It calls trivfs_server() for each message.
 * 
 * Returns:
 *   int - Exit code (never returns under normal operation)
 */
extern int trivfs_server_loop(void);

/*
 * trivfs_startup - Translator startup function
 * 
 * This function is called when the translator is started. It performs
 * initialization and sets up the translator's state.
 * 
 * Parameters:
 *   bootstrap - Bootstrap port for Mach IPC
 *   flags     - Startup flags
 *   port_class_list - Port class for control ports
 *   port_bucket_list - Port bucket for control ports
 *   fs_class   - Port class for filesystem ports
 *   fs_bucket   - Port bucket for filesystem ports
 *   control    - Output: pointer to store the control port
 * 
 * Returns:
 *   error_t - Error code (0 for success)
 */
extern error_t trivfs_startup(mach_port_t bootstrap, int flags,
                                struct port_class *port_class_list,
                                struct port_bucket *port_bucket_list,
                                struct port_class *fs_class,
                                struct port_bucket *fs_bucket,
                                struct trivfs_control **control);


/*
 * Forward declarations for our filesystem hooks
 * 
 * These are the functions we implement to handle filesystem operations.
 */
static error_t fs_open_hook(struct iouser *cred, int flags, mode_t mode,
                           struct node *node, struct iobuf **iobuf);
static error_t fs_read_hook(struct iouser *cred, struct iobuf *iobuf,
                           off_t offset, size_t *len, size_t count);
static error_t fs_write_hook(struct iouser *cred, struct iobuf *iobuf,
                            off_t offset, size_t len, size_t count);


/*****************************************************************************
 *                                                                           *
 *                      SIGMOID ACTIVATION FUNCTION                          *
 *                                                                           *
 *****************************************************************************/

/*
 * sigmoidf - Sigmoid activation function (float version)
 * 
 * The sigmoid function is defined as: 1 / (1 + e^(-x))
 * 
 * Properties:
 *   - Output range: (0, 1)
 *   - S-shaped curve (hence "sigmoid")
 *   - Smooth and differentiable (important for backpropagation)
 *   - Used in many neural network architectures
 * 
 * Why use sigmoid for LLM neurons?
 *   - Provides non-linear activation
 *   - Outputs can be interpreted as probabilities
 *   - Well-behaved gradients (though they can vanish for extreme inputs)
 *   - Computationally efficient (single expf call)
 * 
 * Numerical Considerations:
 *   - We use float (float32) instead of double for memory efficiency
 *   - expf() is the float version of exp()
 *   - For very large negative x, result approaches 0
 *   - For very large positive x, result approaches 1
 * 
 * Parameters:
 *   x - Input value (any real number)
 * 
 * Returns:
 *   Sigmoid of x, in range (0, 1)
 */
static inline float
sigmoidf(float x)
{
    /* 
     * Direct computation: 1 / (1 + exp(-x))
     * 
     * This is the standard formula. We could add optimizations like:
     *   - Approximation for large |x| (return 0 or 1)
     *   - Polynomial approximation
     *   - Lookup table
     * 
     * But for now, we use the direct computation for accuracy.
     * The expf() function is hardware-accelerated on most modern CPUs.
     */
    return 1.0f / (1.0f + expf(-x));
}


/*
 * sigmoidf_derivative - Derivative of sigmoid function
 * 
 * The derivative of sigmoid(x) is: sigmoid(x) * (1 - sigmoid(x))
 * 
 * This is used during backpropagation (though not in this translator,
 * which is currently forward-pass only).
 * 
 * Properties:
 *   - Maximum value is 0.25 at x = 0
 *   - Approaches 0 as |x| increases (vanishing gradient problem)
 * 
 * Parameters:
 *   x - Input value
 * 
 * Returns:
 *   Derivative of sigmoid at x
 */
static inline float
sigmoidf_derivative(float x)
{
    float s = sigmoidf(x);
    return s * (1.0f - s);
}


/*****************************************************************************
 *                                                                           *
 *                        MEMORY MANAGEMENT FUNCTIONS                       *
 *                                                                           *
 *****************************************************************************/

/*
 * aligned_malloc - Allocate aligned memory
 * 
 * Allocates a block of memory aligned to a specified boundary.
 * This is important for:
 *   - SIMD instructions (SSE, AVX) which require aligned data
 *   - Cache line alignment for better performance
 *   - DMA operations (though not used here)
 * 
 * We use posix_memalign() which is the POSIX-standard way to allocate
 * aligned memory. It's available on GNU/Linux and GNU/Hurd.
 * 
 * Parameters:
 *   size      - Number of bytes to allocate
 *   alignment - Required alignment (must be power of 2)
 * 
 * Returns:
 *   Pointer to allocated memory, or NULL on failure
 */
static void *
aligned_malloc(size_t size, size_t alignment)
{
    void *ptr;
    
    /* 
     * posix_memalign() allocates memory aligned to 'alignment' bytes.
     * The alignment must be a power of 2 and at least sizeof(void*).
     * 
     * On success, it stores the pointer in *ptr and returns 0.
     * On failure, it returns errno and *ptr is undefined.
     */
    if (posix_memalign(&ptr, alignment, size) != 0) {
        /* 
         * posix_memalign failed
         * errno will be set to EINVAL (bad alignment) or ENOMEM (out of memory)
         */
        return NULL;
    }
    
    return ptr;
}


/*
 * aligned_free - Free aligned memory
 * 
 * Frees memory allocated with aligned_malloc().
 * Since we use posix_memalign(), we can use the standard free().
 * 
 * Parameters:
 *   ptr - Pointer to memory to free (may be NULL)
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
 *****************************************************************************/

/*
 * network_init - Initialize a neural network with given topology
 * 
 * This function sets up the neural network with the specified architecture.
 * It allocates a single contiguous memory block for all network data.
 * 
 * Steps:
 *   1. Validate the topology parameters
 *   2. Copy the layer sizes
 *   3. Calculate total sizes (neurons, weights, biases)
 *   4. Calculate memory offsets for each layer
 *   5. Allocate memory block
 *   6. Set up data pointers
 *   7. Initialize voltages to reset potential
 *   8. Initialize weights with random values
 *   9. Initialize biases to zero
 * 
 * Memory Layout:
 *   [voltages][weights][biases][input_buffer][output_buffer]
 * 
 * This contiguous layout provides:
 *   - Fewer memory allocations (just one)
 *   - Better cache locality
 *   - Easier memory management
 *   - Potential for memory-mapped files in the future
 * 
 * Parameters:
 *   net         - Pointer to network structure to initialize
 *   layer_count - Number of layers (2-8)
 *   layer_sizes - Array of layer sizes (number of neurons per layer)
 * 
 * Returns:
 *   0 on success, -1 on failure (with errno set)
 */
static int
network_init(CompactNeuralNetwork *net,
             uint8_t layer_count,
             const uint16_t *layer_sizes)
{
    /* Validate inputs */
    if (!net || !layer_sizes) {
        errno = EINVAL;
        return -1;
    }
    
    if (layer_count < 2 || layer_count > MAX_LAYERS) {
        errno = EINVAL;
        return -1;
    }
    
    /* Validate each layer size */
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
    
    /* Set neuron parameters (can be customized later) */
    net->topology.reset_potential = RESET_POTENTIAL;
    net->topology.threshold = THRESHOLD;
    net->topology.leak_rate = LEAK_RATE;
    net->topology.refractory_length = REFRACTORY_LENGTH;
    
    /* Calculate total neuron count */
    net->total_neurons = 0;
    for (int i = 0; i < layer_count; i++) {
        net->total_neurons += layer_sizes[i];
    }
    
    /* Calculate total weight and bias counts */
    net->total_weights = 0;
    net->total_biases = 0;
    for (int i = 1; i < layer_count; i++) {
        /* Each neuron in layer i has connections from all neurons in layer i-1 */
        net->total_weights += (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        net->total_biases += layer_sizes[i];
    }
    
    /* Calculate layer offsets (for voltage array) */
    net->layer_offsets[0] = 0;
    for (int i = 1; i < layer_count; i++) {
        net->layer_offsets[i] = net->layer_offsets[i - 1] + layer_sizes[i - 1];
    }
    
    /* Calculate weight offsets */
    if (layer_count > 1) {
        net->weight_offsets[0] = 0;
        for (int i = 1; i < layer_count - 1; i++) {
            net->weight_offsets[i] = net->weight_offsets[i - 1] +
                                      (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        }
    }
    
    /* Calculate bias offsets */
    net->bias_offsets[0] = 0;
    for (int i = 1; i < layer_count; i++) {
        net->bias_offsets[i] = net->bias_offsets[i - 1] + layer_sizes[i];
    }
    
    /* Calculate memory sizes */
    size_t voltages_size = net->total_neurons * sizeof(float);
    size_t weights_size = net->total_weights * sizeof(float);
    size_t biases_size = net->total_biases * sizeof(float);
    size_t input_size = layer_sizes[0] * sizeof(float);
    size_t output_size = layer_sizes[layer_count - 1] * sizeof(float);
    
    /* Total memory needed */
    net->memory_block_size = voltages_size + weights_size + biases_size +
                             input_size + output_size;
    
    /* Allocate memory block with SIMD alignment */
    net->memory_block = aligned_malloc(net->memory_block_size, SIMD_ALIGNMENT);
    if (!net->memory_block) {
        errno = ENOMEM;
        return -1;
    }
    
    /* Set up pointers into the memory block */
    char *ptr = (char *)net->memory_block;
    
    /* Voltage array */
    net->voltages = (float *)ptr;
    ptr += voltages_size;
    
    /* Weight array */
    net->weights = (float *)ptr;
    ptr += weights_size;
    
    /* Bias array */
    net->biases = (float *)ptr;
    ptr += biases_size;
    
    /* Input buffer */
    net->input_buffer = (float *)ptr;
    ptr += input_size;
    
    /* Output buffer */
    net->output_buffer = (float *)ptr;
    
    /* Initialize voltages to reset potential */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Initialize weights with small random values */
    /* 
     * We use a simple deterministic pseudo-random number generator
     * based on the golden ratio. This ensures:
     *   - Reproducible results (same weights on each run)
     *   - No dependency on external RNG libraries
     *   - Fast initialization
     * 
     * The golden ratio prime (2654435761) is used as a multiplier.
     * This is a well-known constant that produces good pseudo-random sequences.
     */
    for (size_t i = 0; i < net->total_weights; i++) {
        uint32_t seed = (uint32_t)i * 2654435761U; /* Golden ratio prime */
        /* Mask to get 23 bits (maintains sign bit for float) */
        float random = (float)(seed & 0x007FFFFF) / (float)0x007FFFFF;
        /* Scale to range [-0.2, 0.2] for small initial weights */
        net->weights[i] = random * 0.4f - 0.2f;
    }
    
    /* Initialize biases to zero */
    for (size_t i = 0; i < net->total_biases; i++) {
        net->biases[i] = 0.0f;
    }
    
    /* Mark as initialized */
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return 0;
}


/*
 * network_free - Free neural network memory
 * 
 * This function releases all memory allocated for a neural network.
 * It should be called when the network is no longer needed or when
 * reinitializing with a new topology.
 * 
 * Parameters:
 *   net - Pointer to network structure to free
 */
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
    
    /* Reset all pointers */
    net->voltages = NULL;
    net->weights = NULL;
    net->biases = NULL;
    net->input_buffer = NULL;
    net->output_buffer = NULL;
    
    /* Mark as uninitialized */
    net->initialized = false;
    net->needs_reset = false;
}


/*
 * network_reset - Reset network state
 * 
 * Resets all neuron voltages to the reset potential without changing
 * the weights or topology. This is useful for starting a new computation
 * sequence.
 * 
 * Parameters:
 *   net - Pointer to network structure
 */
static void
network_reset(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized) {
        return;
    }
    
    /* Reset all voltages */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Reset statistics */
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
}


/*****************************************************************************
 *                                                                           *
 *                        FORWARD PASS FUNCTION                              *
 *                                                                           *
 *****************************************************************************/

/*
 * network_forward - Perform a forward pass through the network
 * 
 * This is the core computation function. It takes the current input,
 * propagates it through all layers using the sigmoid activation function,
 * and produces output in the output buffer.
 * 
 * Algorithm:
 *   For each layer (starting from the first hidden layer):
 *     For each neuron in the layer:
 *       sum = bias + sum of (input_neuron_value * weight)
 *       output = sigmoid(sum)
 * 
 * Implementation Notes:
 *   - We use pre-calculated offsets for efficient memory access
 *   - The input to each layer comes from the previous layer's voltages
 *   - We compute one layer at a time to minimize memory usage
 *   - No dynamic memory allocation during forward pass
 * 
 * Cache Optimization:
 *   - We access memory sequentially (good for cache prefetching)
 *   - We reuse pointers to avoid repeated pointer arithmetic
 *   - Weight access pattern is optimized for the common case
 * 
 * CPU Optimization:
 *   - Uses float32 instead of float64 (half the memory, similar precision)
 *   - No function calls in inner loop (sigmoidf is inlined)
 *   - Minimal branching in hot path
 * 
 * Parameters:
 *   net - Pointer to initialized network structure
 */
static void
network_forward(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized || net->topology.layer_count < 2) {
        return;
    }
    
    /* Copy input buffer to first layer voltages if provided */
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
        
        /* Calculate weight offset for this layer */
        size_t weight_offset = 0;
        for (int l = 1; l < layer; l++) {
            weight_offset += (size_t)net->topology.layer_sizes[l] * 
                            (size_t)net->topology.layer_sizes[l - 1];
        }
        
        size_t bias_offset = net->bias_offsets[layer - 1];
        
        /* Process each neuron in current layer */
        for (size_t n = 0; n < curr_size; n++) {
            /* Start with bias */
            float sum = net->biases[bias_offset + n];
            
            /* Pointer to weights for this neuron */
            float *w = net->weights + weight_offset + n * prev_size;
            
            /* Pointer to previous layer voltages */
            float *v = net->voltages + prev_offset;
            
            /* Sum weighted inputs */
            for (size_t p = 0; p < prev_size; p++) {
                sum += v[p] * w[p];
            }
            
            /* Apply sigmoid activation and store result */
            net->voltages[curr_offset + n] = sigmoidf(sum);
            net->neuron_activations++;
        }
    }
    
    /* Copy output layer voltages to output buffer */
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

/*
 * parse_config_string - Parse a configuration string into layer sizes
 * 
 * The configuration string format is a comma or space-separated list of
 * integers, each representing the number of neurons in a layer.
 * 
 * Examples:
 *   "10,20,5"    - 3 layers: 10, 20, 5 neurons
 *   "5 10 10 3"  - 4 layers: 5, 10, 10, 3 neurons
 *   "100"        - Invalid (need at least 2 layers)
 *   "10,0,5"     - Invalid (layer size must be > 0)
 * 
 * Parameters:
 *   config_str - Configuration string to parse
 *   layer_sizes - Output array for layer sizes
 *   max_layers - Maximum number of layers to parse
 * 
 * Returns:
 *   Number of layers parsed (2-max_layers), or -1 on error
 */
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
        /* Skip whitespace and delimiters */
        while (*ptr == ' ' || *ptr == '\t' || *ptr == ',') {
            ptr++;
        }
        
        if (*ptr == '\0') {
            break;
        }
        
        /* Parse integer */
        char *endptr;
        long value = strtol(ptr, &endptr, 10);
        
        if (ptr == endptr) {
            /* No digits found */
            errno = EINVAL;
            return -1;
        }
        
        /* Validate value */
        if (value < 1 || value > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
        
        /* Store layer size */
        layer_sizes[count++] = (uint16_t)value;
        ptr = endptr;
    }
    
    /* Need at least 2 layers */
    if (count < 2) {
        errno = EINVAL;
        return -1;
    }
    
    return count;
}


/*
 * parse_input_string - Parse an input string into the input buffer
 * 
 * The input string is a comma or space-separated list of floating-point values.
 * 
 * Examples:
 *   "0.5,0.3,0.8"     - Three input values
 *   "1.0 0.0 -1.0"    - Three input values
 *   "0.5,0.3"         - Two input values (may be incomplete)
 * 
 * Parameters:
 *   net - Pointer to initialized network
 *   input_str - Input string to parse
 * 
 * Returns:
 *   true if enough inputs were provided and forward pass was performed,
 *   false otherwise
 */
static bool
parse_input_string(CompactNeuralNetwork *net, const char *input_str)
{
    if (!net || !net->initialized || !input_str) {
        return false;
    }
    
    const char *ptr = input_str;
    size_t idx = 0;
    
    while (*ptr != '\0' && idx < net->topology.input_size) {
        /* Skip whitespace and delimiters */
        while (*ptr == ',' || *ptr == ' ' || *ptr == '\t' || *ptr == '\n') {
            ptr++;
        }
        
        if (*ptr == '\0') {
            break;
        }
        
        /* Parse floating-point value */
        char *endptr;
        float val = strtof(ptr, &endptr);
        
        if (ptr == endptr) {
            /* No number found, skip this character */
            ptr++;
            continue;
        }
        
        /* Store value */
        net->input_buffer[idx++] = val;
        ptr = endptr;
    }
    
    /* Only perform forward pass if we have enough inputs */
    if (idx >= net->topology.input_size) {
        network_forward(net);
        return true;
    }
    
    return false;
}


/*****************************************************************************
 *                                                                           *
 *                      FILESYSTEM OPERATION HOOKS                           *
 *                                                                           *
 *****************************************************************************/

/*
 * fs_open_hook - Handle file open requests
 * 
 * This function is called when a client tries to open the translator.
 * For a simple translator like this, we don't need to maintain any
 * per-client state, so we just return success.
 * 
 * Parameters:
 *   cred   - User credentials (unused)
 *   flags  - Open flags (O_READ, O_WRITE, etc.)
 *   mode   - File mode (permissions)
 *   node   - Filesystem node (unused)
 *   iobuf  - Output: pointer to iobuf structure
 * 
 * Returns:
 *   error_t - 0 for success, error code otherwise
 */
static error_t
fs_open_hook(struct iouser *cred, int flags, mode_t mode, struct node *node, struct iobuf **iobuf)
{
    (void)cred;   /* Unused parameter */
    (void)flags;  /* Unused parameter */
    (void)mode;   /* Unused parameter */
    (void)node;   /* Unused parameter */
    
    /* We don't need to allocate an iobuf for this simple translator */
    *iobuf = NULL;
    
    /* Initialize network if not already done */
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    }
    
    return 0;
}


/*
 * fs_read_hook - Handle file read requests
 * 
 * This function is called when a client reads from the translator.
 * It generates a status report containing:
 *   - Translator information
 *   - Network topology
 *   - Memory usage
 *   - Statistics
 *   - Current output values
 * 
 * The output is formatted as plain text for easy reading.
 * 
 * Parameters:
 *   cred   - User credentials (unused)
 *   iobuf  - I/O buffer to read into
 *   offset - Read offset (unused, we always read from start)
 *   len    - Input/Output: requested/actual read length
 *   count  - Maximum number of bytes to read (unused)
 * 
 * Returns:
 *   error_t - 0 for success, error code otherwise
 */
static error_t
fs_read_hook(struct iouser *cred, struct iobuf *iobuf, off_t offset, size_t *len, size_t count)
{
    (void)cred;   /* Unused parameter */
    (void)offset; /* Unused parameter */
    (void)count;  /* Unused parameter */
    
    /* Initialize network if not already done */
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    }
    
    /* Build status buffer */
    char buffer[4096];
    size_t written = 0;
    
    /* Header */
    written = snprintf(buffer, sizeof(buffer),
                      "LLM Sigmoid Neuron Translator - GNU Hurd\n"
                      "===========================================\n\n");
    
    /* Network topology */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Network Topology:\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Layers: %d\n", global_network.topology.layer_count);
    
    for (int i = 0; i < global_network.topology.layer_count; i++) {
        written += snprintf(buffer + written, sizeof(buffer) - written,
                          "  Layer %d: %d neurons\n",
                          i, global_network.topology.layer_sizes[i]);
    }
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "\n");
    
    /* Neuron parameters */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Neuron Parameters:\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Reset potential: %.2f mV\n",
                       global_network.topology.reset_potential);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Threshold: %.2f mV\n",
                       global_network.topology.threshold);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Leak rate: %.2f\n",
                       global_network.topology.leak_rate);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Refractory length: %d\n\n",
                       global_network.topology.refractory_length);
    
    /* Memory usage */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Memory Usage:\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Total memory: %.2f KB\n",
                       (double)global_network.memory_block_size / 1024.0);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Neurons: %zu\n",
                       global_network.total_neurons);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Weights: %zu\n",
                       global_network.total_weights);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Biases: %zu\n\n",
                       global_network.total_biases);
    
    /* Statistics */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Statistics:\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Forward passes: %zu\n",
                       global_network.forward_pass_count);
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Neuron activations: %zu\n\n",
                       global_network.neuron_activations);
    
    /* Current output */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "Current Output:\n");
    
    for (size_t i = 0; i < global_network.topology.output_size; i++) {
        written += snprintf(buffer + written, sizeof(buffer) - written,
                          "  [%zu]: %.6f\n", i, global_network.output_buffer[i]);
    }
    
    /* Usage instructions */
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "\nUsage:\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  To reconfigure: echo '<layers>' > /llm\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Example: echo '5,10,5' > /llm\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  To provide input: echo '<values>' > /llm\n");
    written += snprintf(buffer + written, sizeof(buffer) - written,
                       "  Example: echo '0.5,0.3,0.8,0.1,0.9' > /llm\n");
    
    /* Ensure we don't overflow the buffer */
    if (written >= sizeof(buffer)) {
        written = sizeof(buffer) - 1;
    }
    buffer[written] = '\0';
    
    /* If no iobuf provided, just set length and return */
    if (iobuf == NULL) {
        *len = written;
        return 0;
    }
    
    /* Copy data to iobuf */
    size_t to_copy = (written < *len) ? written : *len;
    if (to_copy > 0 && iobuf->buf != NULL) {
        memcpy(iobuf->buf, buffer, to_copy);
    }
    *len = to_copy;
    
    return 0;
}


/*
 * fs_write_hook - Handle file write requests
 * 
 * This function is called when a client writes to the translator.
 * It interprets the input as either:
 *   - A configuration string (e.g., "10,20,5" to set topology)
 *   - Input values (e.g., "0.5,0.3,0.8" to set inputs)
 * 
 * Parameters:
 *   cred   - User credentials (unused)
 *   iobuf  - I/O buffer containing data to write
 *   offset - Write offset (unused)
 *   len    - Number of bytes to write
 *   count  - Total bytes available (unused)
 * 
 * Returns:
 *   error_t - 0 for success, error code otherwise
 */
static error_t
fs_write_hook(struct iouser *cred, struct iobuf *iobuf, off_t offset, size_t len, size_t count)
{
    (void)cred;   /* Unused parameter */
    (void)offset; /* Unused parameter */
    (void)count;  /* Unused parameter */
    
    /* Validate inputs */
    if (iobuf == NULL || len == 0 || iobuf->buf == NULL) {
        return EINVAL;
    }
    
    /* Ensure global network is initialized */
    if (!global_network.initialized) {
        uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
        network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    }
    
    /* Copy input data to a null-terminated buffer */
    char temp[1024];
    if (len >= sizeof(temp)) {
        len = sizeof(temp) - 1;
    }
    memcpy(temp, iobuf->buf, len);
    temp[len] = '\0';
    
    /* Try to parse as configuration string first */
    uint16_t layers[MAX_LAYERS];
    int layer_count = parse_config_string(temp, layers, MAX_LAYERS);
    
    if (layer_count > 0) {
        /* It's a configuration string - reconfigure the network */
        network_free(&global_network);
        if (network_init(&global_network, layer_count, layers) != 0) {
            return EINVAL;
        }
        return 0;
    }
    
    /* Try to parse as input values */
    if (parse_input_string(&global_network, temp)) {
        return 0;
    }
    
    /* If we get here, the input didn't match any expected format */
    return EINVAL;
}


/*****************************************************************************
 *                                                                           *
 *                          TRIVFS DEMUXER                                 *
 *                                                                           *
 *****************************************************************************/

/*
 * trivfs_demuxer - Message demultiplexer for trivfs
 * 
 * This function is called by the trivfs library to handle incoming
 * Mach messages. It dispatches to the appropriate filesystem operation
 * based on the message type.
 * 
 * For a standard trivfs translator, we simply call trivfs_server() which
 * handles the demultiplexing for us.
 * 
 * Parameters:
 *   inmsg  - Pointer to incoming Mach message header
 *   outmsg - Pointer to outgoing Mach message header
 * 
 * Returns:
 *   error_t - Error code (typically 0 or KERN_SUCCESS)
 */
error_t
trivfs_demuxer(mach_msg_header_t *inmsg, mach_msg_header_t *outmsg)
{
    /* Delegate to the standard trivfs server */
    return trivfs_server(inmsg, outmsg);
}


/*****************************************************************************
 *                                                                           *
 *                            MAIN FUNCTION                                 *
 *                                                                           *
 *****************************************************************************/

/*
 * main - Translator entry point
 * 
 * This is the main entry point for the translator. It:
 *   1. Initializes the global network
 *   2. Sets up the trivfs control port
 *   3. Configures the filesystem operation hooks
 *   4. Sets the help text
 *   5. Enters the main server loop
 * 
 * The translator runs indefinitely, processing filesystem operations
 * from clients via Mach IPC.
 * 
 * Parameters:
 *   argc - Argument count (unused)
 *   argv - Argument vector (unused)
 * 
 * Returns:
 *   int - Exit code (should never return)
 */
int
main(int argc, char **argv)
{
    (void)argc;  /* Unused parameter */
    (void)argv;  /* Unused parameter */
    
    /* Initialize global network state */
    global_network.initialized = false;
    global_network.memory_block = NULL;
    
    /* Initialize the trivfs control port */
    trivfs_control = MACH_PORT_NULL;
    
    /* Initialize the network with default topology */
    uint16_t layers[MAX_LAYERS] = DEFAULT_LAYER_SIZES;
    network_init(&global_network, DEFAULT_LAYER_COUNT, layers);
    
    /* Set up filesystem operation hooks */
    fs_help = "LLM Sigmoid Neuron Translator for GNU Hurd\n"
              "Memory-efficient, CPU-optimized neural network\n"
              "Usage: settrans -c <node> /hurd/sigmoid-neuron-translator\n"
              "Example: settrans -c /llm /hurd/sigmoid-neuron-translator";
    
    /* Assign our hook functions */
    fs_open = fs_open_hook;
    fs_read = fs_read_hook;
    fs_write = fs_write_hook;
    
    /* Enter the main server loop */
    /* This will process incoming Mach messages indefinitely */
    return trivfs_server_loop();
}


/*
 * Editor modelines  https://www.wu.ac.at/usr/local/info/modeline.html
 * vim: set ts=8 sw=4 sts=4 tw=78 expandtab:
 * Emacs: -*- mode: c; tab-width: 8; c-basic-offset: 4; -*- 
 */

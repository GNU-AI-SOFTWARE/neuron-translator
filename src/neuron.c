/*
 * neuron.c - Neural Network Implementation for LLM Translator
 *
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/** @file neuron.c
 *  @brief Neural network implementation with memory-efficient design
 *
 *  This file implements a compact sigmoid neural network for the GNU Hurd
 *  translator. It follows Claude Delannoy's educational style with extensive
 *  comments explaining each algorithmic step.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "neuron.h"


/*****************************************************************************
 *                                                                           *
 *                        MEMORY MANAGEMENT                                *
 *                                                                           *
 *  Uses posix_memalign for SIMD-compatible allocations                    *
 *                                                                           *
 *****************************************************************************/

/**
 * @brief Aligned memory allocation
 * 
 * Allocates memory with specified alignment for SIMD operations.
 * 
 * @param size      Size of memory to allocate
 * @param alignment Alignment requirement (e.g., 16 for SIMD)
 * @return          Pointer to allocated memory, or NULL on failure
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
 * @brief Aligned memory deallocation
 * 
 * @param ptr Pointer to memory to free
 */
static void aligned_free(void *ptr)
{
    free(ptr);
}


/*****************************************************************************
 *                                                                           *
 *                      NETWORK INITIALIZATION                              *
 *                                                                           *
 *  Initializes network topology and allocates contiguous memory block        *
 *                                                                           *
 *****************************************************************************/

int network_init(CompactNeuralNetwork *net,
                uint8_t layer_count,
                const uint16_t *layer_sizes)
{
    /* Input validation */
    if (!net || !layer_sizes) {
        errno = EINVAL;
        return -1;
    }
    
    /* Validate layer count */
    if (layer_count < 2 || layer_count > MAX_LAYERS) {
        errno = EINVAL;
        return -1;
    }
    
    /* Validate each layer size */
    for (uint8_t i = 0; i < layer_count; i++) {
        if (layer_sizes[i] == 0 || layer_sizes[i] > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
    }
    
    /* Store topology */
    net->topology.layer_count = layer_count;
    net->topology.input_size = layer_sizes[0];
    net->topology.output_size = layer_sizes[layer_count - 1];
    
    for (uint8_t i = 0; i < layer_count; i++) {
        net->topology.layer_sizes[i] = layer_sizes[i];
    }
    
    /* Set neuron parameters */
    net->topology.reset_potential = RESET_POTENTIAL;
    net->topology.threshold = THRESHOLD;
    net->topology.leak_rate = LEAK_RATE;
    net->topology.refractory_length = REFRACTORY_LENGTH;
    
    /* Calculate total neuron count */
    net->total_neurons = 0;
    for (uint8_t i = 0; i < layer_count; i++) {
        net->total_neurons += layer_sizes[i];
    }
    
    /* Calculate total weights (sum of layer_i * layer_{i-1}) */
    net->total_weights = 0;
    net->total_biases = 0;
    for (uint8_t i = 1; i < layer_count; i++) {
        net->total_weights += (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        net->total_biases += layer_sizes[i];
    }
    
    /* Calculate layer offsets */
    net->layer_offsets[0] = 0;
    for (uint8_t i = 1; i < layer_count; i++) {
        net->layer_offsets[i] = net->layer_offsets[i - 1] + layer_sizes[i - 1];
    }
    
    /* Calculate weight offsets */
    if (layer_count > 1) {
        net->weight_offsets[0] = 0;
        for (uint8_t i = 1; i < layer_count - 1; i++) {
            net->weight_offsets[i] = net->weight_offsets[i - 1] +
                                      (size_t)layer_sizes[i] * (size_t)layer_sizes[i - 1];
        }
    }
    
    /* Calculate bias offsets */
    net->bias_offsets[0] = 0;
    for (uint8_t i = 1; i < layer_count; i++) {
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
    
    /* Set up pointers into the memory block */
    char *ptr = net->memory_block;
    net->voltages = (float *)ptr;
    ptr += voltages_size;
    net->weights = (float *)ptr;
    ptr += weights_size;
    net->biases = (float *)ptr;
    ptr += biases_size;
    net->input_buffer = (float *)ptr;
    ptr += input_size;
    net->output_buffer = (float *)ptr;
    
    /* Initialize voltages to reset potential */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Initialize weights with small random values */
    for (size_t i = 0; i < net->total_weights; i++) {
        /* Simple deterministic pseudo-random initialization */
        uint32_t seed = (uint32_t)i * 2654435761U; /* Knuth's multiplicative constant */
        float random = (float)(seed & 0x007FFFFF) / (float)0x007FFFFF;
        net->weights[i] = random * 0.4f - 0.2f; /* Range: [-0.2, 0.2] */
    }
    
    /* Initialize biases to zero */
    for (size_t i = 0; i < net->total_biases; i++) {
        net->biases[i] = 0.0f;
    }
    
    /* Set initialization flag */
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return 0;
}


/*****************************************************************************
 *                                                                           *
 *                        NETWORK OPERATIONS                               *
 *                                                                           *
 *****************************************************************************/

void network_free(CompactNeuralNetwork *net)
{
    if (!net) return;
    
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
    
    net->initialized = false;
    net->needs_reset = false;
}


void network_reset(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized) return;
    
    /* Reset all neuron voltages to resting potential */
    for (size_t i = 0; i < net->total_neurons; i++) {
        net->voltages[i] = net->topology.reset_potential;
    }
    
    /* Reset counters */
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    net->needs_reset = false;
}


void network_forward(CompactNeuralNetwork *net)
{
    if (!net || !net->initialized || net->topology.layer_count < 2) {
        return;
    }
    
    /* Copy input buffer to first layer voltages */
    if (net->input_buffer) {
        memcpy(net->voltages, net->input_buffer,
               net->topology.input_size * sizeof(float));
    }
    
    /* Process each layer (starting from first hidden layer) */
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
            
            /* Add weighted sum of previous layer */
            float *w = net->weights + weight_offset + n * prev_size;
            float *v = net->voltages + prev_offset;
            
            for (size_t p = 0; p < prev_size; p++) {
                sum += v[p] * w[p];
            }
            
            /* Apply sigmoid activation */
            net->voltages[curr_offset + n] = sigmoidf(sum);
            net->neuron_activations++;
        }
    }
    
    /* Copy output layer to output buffer */
    if (net->output_buffer) {
        size_t out_offset = net->layer_offsets[net->topology.layer_count - 1];
        memcpy(net->output_buffer, net->voltages + out_offset,
               net->topology.output_size * sizeof(float));
    }
    
    net->forward_pass_count++;
}


/*****************************************************************************
 *                                                                           *
 *                     CONFIGURATION PARSING                                *
 *                                                                           *
 *****************************************************************************/

int parse_config_string(const char *config_str,
                       uint16_t *layer_sizes,
                       int max_layers)
{
    if (!config_str || !layer_sizes || max_layers < 2) {
        errno = EINVAL;
        return -1;
    }
    
    const char *ptr = config_str;
    int count = 0;
    
    /* Parse comma or space separated values */
    while (*ptr != '\0' && count < max_layers) {
        /* Skip separators */
        while (*ptr == ' ' || *ptr == '\t' || *ptr == ',') ptr++;
        if (*ptr == '\0') break;
        
        /* Parse integer */
        errno = 0;
        char *endptr;
        long value = strtol(ptr, &endptr, 10);
        
        if (ptr == endptr || errno == ERANGE) {
            /* No digit found or out of range */
            errno = EINVAL;
            return -1;
        }
        
        /* Validate range */
        if (value < 1 || value > MAX_NEURONS_PER_LAYER) {
            errno = EINVAL;
            return -1;
        }
        
        layer_sizes[count++] = (uint16_t)value;
        ptr = endptr;
    }
    
    /* Must have at least 2 layers */
    if (count < 2) {
        errno = EINVAL;
        return -1;
    }
    
    return count;
}


bool parse_input_string(CompactNeuralNetwork *net,
                       const char *input_str)
{
    if (!net || !net->initialized || !input_str) {
        return false;
    }
    
    const char *ptr = input_str;
    size_t idx = 0;
    
    /* Parse comma or space separated values */
    while (*ptr != '\0' && idx < net->topology.input_size) {
        /* Skip separators */
        while (*ptr == ',' || *ptr == ' ' || *ptr == '\t' || *ptr == '\n') ptr++;
        if (*ptr == '\0') break;
        
        /* Parse float */
        errno = 0;
        char *endptr;
        float val = strtof(ptr, &endptr);
        
        if (ptr == endptr || errno == ERANGE) {
            /* Invalid number - fail parsing */
            return false;
        }
        
        net->input_buffer[idx++] = val;
        ptr = endptr;
    }
    
    /* Only execute forward pass if we got all required inputs */
    if (idx == net->topology.input_size) {
        network_forward(net);
        return true;
    }
    
    return false;
}


/*****************************************************************************
 *                                                                           *
 *                      FILE PERSISTENCE                                    *
 *                                                                           *
 *****************************************************************************/

bool network_save(const CompactNeuralNetwork *net,
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
    
    /* Write data arrays */
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


bool network_load(CompactNeuralNetwork *net,
                 const char *filename)
{
    if (!net || !filename) return false;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) return false;
    
    /* Free existing network */
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
        network_free(net);
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
    char *ptr = net->memory_block;
    net->voltages = (float *)ptr;
    ptr += net->total_neurons * sizeof(float);
    net->weights = (float *)ptr;
    ptr += net->total_weights * sizeof(float);
    net->biases = (float *)ptr;
    ptr += net->total_biases * sizeof(float);
    net->input_buffer = (float *)ptr;
    ptr += net->topology.input_size * sizeof(float);
    net->output_buffer = (float *)ptr;
    
    /* Read data arrays */
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
    
    /* Mark as initialized */
    net->initialized = true;
    net->needs_reset = false;
    net->forward_pass_count = 0;
    net->neuron_activations = 0;
    
    return true;
}

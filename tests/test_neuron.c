/*
 * test_neuron.c - Unit tests for the sigmoid neuron core
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026  GNU AI Project
 * Author: Claire Ivanenka <claire@gnu-ai.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This file is the unit test suite for src/neuron.c. It follows the
 * project conventions: no external test framework, plain C23/POSIX,
 * fully deterministic. The neuron core is pure POSIX code, so the
 * suite runs on any system (Linux development box or GNU/Hurd).
 *
 * Run with 'make check' from the repository root.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neuron.h"

/*****************************************************************************
 *                                                                           *
 *                         MINIMAL TEST HARNESS                             *
 *                                                                           *
 *  Two macros and two counters - deliberately no framework. Each CHECK     *
 *  records one assertion; the binary exits non-zero if any failed.          *
 *                                                                           *
 *****************************************************************************/

static int tests_run = 0;
static int tests_failed = 0;

/** Record the truth of a boolean condition as one test assertion. */
#define CHECK(cond)                                                     \
    do {                                                                \
        tests_run++;                                                    \
        if (!(cond)) {                                                  \
            tests_failed++;                                             \
            fprintf(stderr, "FAIL %s:%d: %s\n",                         \
                    __FILE__, __LINE__, #cond);                         \
        }                                                               \
    } while (0)

/** Record a floating-point comparison within an absolute tolerance. */
#define CHECK_FLT(actual, expected, tol)                                \
    do {                                                                \
        tests_run++;                                                    \
        if (!(fabsf((float)(actual) - (float)(expected)) <= (tol))) {   \
            tests_failed++;                                             \
            fprintf(stderr, "FAIL %s:%d: %s = %.9g, expected %.9g\n",   \
                    __FILE__, __LINE__, #actual,                        \
                    (double)(actual), (double)(expected));              \
        }                                                               \
    } while (0)

/** Temporary model file used by the persistence tests. */
#define TEST_FILE "test_neuron_tmp.nn"

/*****************************************************************************
 *                                                                           *
 *                             SIGMOID FUNCTION                             *
 *                                                                           *
 *****************************************************************************/

static void test_sigmoid(void)
{
    /* sigmoid(0) is exactly one half */
    CHECK_FLT(sigmoidf(0.0f), 0.5f, 1e-7f);

    /* Saturation at both ends of the range */
    CHECK_FLT(sigmoidf(50.0f), 1.0f, 1e-6f);
    CHECK_FLT(sigmoidf(-50.0f), 0.0f, 1e-6f);

    /* Symmetry: sigmoid(-x) = 1 - sigmoid(x) */
    const float xs[] = { 0.5f, 1.0f, 2.0f, 5.0f };
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        CHECK_FLT(sigmoidf(-xs[i]) + sigmoidf(xs[i]), 1.0f, 1e-6f);
    }

    /* Strictly increasing */
    CHECK(sigmoidf(0.1f) > sigmoidf(0.0f));
    CHECK(sigmoidf(1.0f) > sigmoidf(0.1f));
}

/*****************************************************************************
 *                                                                           *
 *                          NETWORK INITIALIZATION                          *
 *                                                                           *
 *****************************************************************************/

static void test_init_valid(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[3] = { 3, 4, 2 };

    CHECK(network_init(&net, 3, sizes) == 0);
    CHECK(net.initialized);
    CHECK(net.needs_reset == false);

    /* Topology fields */
    CHECK(net.topology.layer_count == 3);
    CHECK(net.topology.input_size == 3);
    CHECK(net.topology.output_size == 2);
    CHECK(net.topology.layer_sizes[0] == 3);
    CHECK(net.topology.layer_sizes[1] == 4);
    CHECK(net.topology.layer_sizes[2] == 2);

    /* Derived counts: 3+4+2 neurons, 4*3 + 2*4 weights, 4+2 biases */
    CHECK(net.total_neurons == 9);
    CHECK(net.total_weights == 20);
    CHECK(net.total_biases == 6);

    /* Offset tables */
    CHECK(net.layer_offsets[0] == 0);
    CHECK(net.layer_offsets[1] == 3);
    CHECK(net.layer_offsets[2] == 7);
    CHECK(net.weight_offsets[0] == 0);
    CHECK(net.weight_offsets[1] == 12);
    CHECK(net.bias_offsets[0] == 0);
    CHECK(net.bias_offsets[1] == 4);
    CHECK(net.bias_offsets[2] == 6);

    /* Runtime counters start at zero */
    CHECK(net.forward_pass_count == 0);
    CHECK(net.neuron_activations == 0);

    /* Every voltage starts at the resting potential */
    for (size_t i = 0; i < net.total_neurons; i++) {
        CHECK_FLT(net.voltages[i], RESET_POTENTIAL, 1e-6f);
    }

    /* Every bias starts at zero */
    for (size_t i = 0; i < net.total_biases; i++) {
        CHECK_FLT(net.biases[i], 0.0f, 1e-7f);
    }

    /* Every weight stays in the documented [-0.2, 0.2] range */
    for (size_t i = 0; i < net.total_weights; i++) {
        CHECK(net.weights[i] >= -0.2f);
        CHECK(net.weights[i] <= 0.2f);
    }

    /* Free clears the state */
    network_free(&net);
    CHECK(net.memory_block == NULL);
    CHECK(net.initialized == false);
}

static void test_init_rejects_bad_topology(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t ok[2] = { 2, 2 };

    /* NULL arguments */
    CHECK(network_init(NULL, 2, ok) == -1);
    CHECK(network_init(&net, 2, NULL) == -1);

    /* Layer count out of the documented 2..MAX_LAYERS range */
    const uint16_t two[1] = { 4 };
    (void)two; /* silence unused warnings on builds that reorder */
    CHECK(network_init(&net, 1, ok) == -1);
    CHECK(network_init(&net, MAX_LAYERS + 1, ok) == -1);

    /* Zero-sized layer and oversized layer */
    const uint16_t zeroed[2] = { 0, 2 };
    CHECK(network_init(&net, 2, zeroed) == -1);
    const uint16_t huge[2] = { MAX_NEURONS_PER_LAYER + 1, 2 };
    CHECK(network_init(&net, 2, huge) == -1);

    /* A rejected init must leave the network uninitialized */
    CHECK(net.initialized == false);
}

static void test_init_deterministic(void)
{
    /* The pseudo-random weight initialization is deterministic:
     * two networks with the same topology must be identical,
     * which is what the reproductibility requirements of the
     * orchestrator rely on. */
    CompactNeuralNetwork a = {0}, b = {0};
    const uint16_t sizes[3] = { 10, 20, 5 };

    CHECK(network_init(&a, 3, sizes) == 0);
    CHECK(network_init(&b, 3, sizes) == 0);

    CHECK(a.total_weights == b.total_weights);
    for (size_t i = 0; i < a.total_weights; i++) {
        CHECK(a.weights[i] == b.weights[i]);
    }

    network_free(&a);
    network_free(&b);
}

/*****************************************************************************
 *                                                                           *
 *                              NETWORK RESET                                *
 *                                                                           *
 *****************************************************************************/

static void test_reset(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[2] = { 2, 1 };

    CHECK(network_init(&net, 2, sizes) == 0);

    /* Tamper with the runtime state */
    net.voltages[0] = 5.0f;
    net.forward_pass_count = 7;
    net.neuron_activations = 100;

    network_reset(&net);

    /* Everything is back to the post-init state */
    for (size_t i = 0; i < net.total_neurons; i++) {
        CHECK_FLT(net.voltages[i], RESET_POTENTIAL, 1e-6f);
    }
    CHECK(net.forward_pass_count == 0);
    CHECK(net.neuron_activations == 0);
    CHECK(net.needs_reset == false);

    network_free(&net);
}

/*****************************************************************************
 *                                                                           *
 *                              FORWARD PASS                                 *
 *                                                                           *
 *  Hand-computed cases: the expected values are derived from the math,     *
 *  not from a re-implementation of the code under test.                     *
 *                                                                           *
 *****************************************************************************/

static void test_forward_two_layers(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[2] = { 2, 1 };

    CHECK(network_init(&net, 2, sizes) == 0);

    /* Known weights and bias for a single output neuron */
    net.weights[0] = 0.5f;
    net.weights[1] = -0.5f;
    net.biases[0] = 0.25f;

    /* Input (1.0, -1.0): sum = 0.25 + 0.5*1.0 + (-0.5)*(-1.0) = 1.25 */
    CHECK(parse_input_string(&net, "1.0,-1.0") == true);

    const float expected = sigmoidf(0.25f + 0.5f * 1.0f + 0.5f);
    CHECK_FLT(net.output_buffer[0], expected, 1e-6f);

    /* Counters: one pass, one output neuron activated */
    CHECK(net.forward_pass_count == 1);
    CHECK(net.neuron_activations == 1);

    /* Same input again: deterministic, counter advances */
    const float first = net.output_buffer[0];
    CHECK(parse_input_string(&net, "1.0,-1.0") == true);
    CHECK_FLT(net.output_buffer[0], first, 1e-7f);
    CHECK(net.forward_pass_count == 2);

    network_free(&net);
}

static void test_forward_three_layers(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[3] = { 1, 1, 1 };

    CHECK(network_init(&net, 3, sizes) == 0);

    /* Chain of two unit weights, zero biases:
     * hidden = sigmoid(1.0), output = sigmoid(hidden) */
    net.weights[0] = 1.0f;   /* layer 1 (hidden) */
    net.weights[1] = 1.0f;   /* layer 2 (output) */
    net.biases[0] = 0.0f;
    net.biases[1] = 0.0f;

    CHECK(parse_input_string(&net, "1.0") == true);

    const float expected = sigmoidf(sigmoidf(1.0f));
    CHECK_FLT(net.output_buffer[0], expected, 1e-6f);
    CHECK(net.neuron_activations == 2);

    network_free(&net);
}

static void test_forward_weight_offsets(void)
{
    /* A 2->2->1 network exercises the weight-matrix offset walk:
     * layer 2's matrix starts right after layer 1's 2x2 matrix. */
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[3] = { 2, 2, 1 };

    CHECK(network_init(&net, 3, sizes) == 0);

    /* Layer 1 matrix (2x2): identity */
    net.weights[0] = 1.0f; net.weights[1] = 0.0f;
    net.weights[2] = 0.0f; net.weights[3] = 1.0f;
    net.biases[0] = 0.0f;
    net.biases[1] = 0.0f;

    /* Layer 2 matrix (1x2): sum both hidden neurons */
    net.weights[4] = 1.0f;
    net.weights[5] = 1.0f;
    net.biases[2] = 0.0f;

    /* Input (1.0, 0.0): hidden = (sigmoid(1), sigmoid(0))
     * output = sigmoid(sigmoid(1) + sigmoid(0)) */
    CHECK(parse_input_string(&net, "1.0,0.0") == true);

    const float expected = sigmoidf(sigmoidf(1.0f) + sigmoidf(0.0f));
    CHECK_FLT(net.output_buffer[0], expected, 1e-6f);

    network_free(&net);
}

/*****************************************************************************
 *                                                                           *
 *                          CONFIGURATION PARSING                           *
 *                                                                           *
 *****************************************************************************/

static void test_parse_config(void)
{
    uint16_t sizes[MAX_LAYERS];

    /* Canonical form: "10,20,5" */
    memset(sizes, 0, sizeof(sizes));
    CHECK(parse_config_string("10,20,5", sizes, MAX_LAYERS) == 3);
    CHECK(sizes[0] == 10 && sizes[1] == 20 && sizes[2] == 5);

    /* Mixed separators: spaces and tabs are accepted */
    memset(sizes, 0, sizeof(sizes));
    CHECK(parse_config_string("10, 20\t5", sizes, MAX_LAYERS) == 3);
    CHECK(sizes[0] == 10 && sizes[1] == 20 && sizes[2] == 5);

    /* max_layers caps the parse */
    memset(sizes, 0, sizeof(sizes));
    CHECK(parse_config_string("1,2,3", sizes, 2) == 2);
    CHECK(sizes[0] == 1 && sizes[1] == 2);

    /* Single layer: not a network */
    CHECK(parse_config_string("5", sizes, MAX_LAYERS) == -1);
    /* Empty string */
    CHECK(parse_config_string("", sizes, MAX_LAYERS) == -1);
    /* Garbage */
    CHECK(parse_config_string("abc,def", sizes, MAX_LAYERS) == -1);
    /* Zero and oversized layers are rejected */
    CHECK(parse_config_string("0,4", sizes, MAX_LAYERS) == -1);
    CHECK(parse_config_string("4,99999", sizes, MAX_LAYERS) == -1);

    /* NULL arguments and absurd max_layers */
    CHECK(parse_config_string(NULL, sizes, MAX_LAYERS) == -1);
    CHECK(parse_config_string("1,2", NULL, MAX_LAYERS) == -1);
    CHECK(parse_config_string("1,2", sizes, 1) == -1);
}

static void test_parse_input(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[2] = { 2, 1 };

    CHECK(network_init(&net, 2, sizes) == 0);

    /* Complete input: parsed, forward pass executed */
    CHECK(parse_input_string(&net, "0.5,0.25") == true);
    CHECK_FLT(net.input_buffer[0], 0.5f, 1e-7f);
    CHECK_FLT(net.input_buffer[1], 0.25f, 1e-7f);
    CHECK(net.forward_pass_count == 1);

    /* Too few values: rejected, no forward pass */
    CHECK(parse_input_string(&net, "0.5") == false);
    CHECK(net.forward_pass_count == 1);

    /* Garbage token: rejected */
    CHECK(parse_input_string(&net, "abc,1.0") == false);

    /* Extra values beyond the input size are ignored */
    CHECK(parse_input_string(&net, "1.0,2.0,3.0") == true);
    CHECK_FLT(net.input_buffer[0], 1.0f, 1e-7f);
    CHECK_FLT(net.input_buffer[1], 2.0f, 1e-7f);

    /* NULL string */
    CHECK(parse_input_string(&net, NULL) == false);

    /* Uninitialized network: rejected */
    CompactNeuralNetwork fresh = {0};
    CHECK(parse_input_string(&fresh, "1.0,2.0") == false);
    CHECK(parse_input_string(NULL, "1.0") == false);

    network_free(&net);
}

/*****************************************************************************
 *                                                                           *
 *                           FILE PERSISTENCE                                *
 *                                                                           *
 *****************************************************************************/

static void test_save_then_load(void)
{
    CompactNeuralNetwork net = {0}, loaded = {0};
    const uint16_t sizes[3] = { 3, 4, 2 };

    CHECK(network_init(&net, 3, sizes) == 0);

    /* Run one pass so voltages hold something else than the reset value */
    CHECK(parse_input_string(&net, "0.1,0.2,0.3") == true);
    const float expected_out = net.output_buffer[0];

    remove(TEST_FILE);
    CHECK(network_save(&net, TEST_FILE) == true);
    CHECK(network_load(&loaded, TEST_FILE) == true);

    /* The loaded network describes the same topology */
    CHECK(loaded.initialized);
    CHECK(loaded.topology.layer_count == 3);
    CHECK(loaded.topology.input_size == 3);
    CHECK(loaded.topology.output_size == 2);
    CHECK(loaded.total_neurons == 9);
    CHECK(loaded.total_weights == 20);
    CHECK(loaded.total_biases == 6);

    /* The loaded network holds the same data, value for value */
    for (size_t i = 0; i < net.total_neurons; i++) {
        CHECK(loaded.voltages[i] == net.voltages[i]);
    }
    for (size_t i = 0; i < net.total_weights; i++) {
        CHECK(loaded.weights[i] == net.weights[i]);
    }
    for (size_t i = 0; i < net.total_biases; i++) {
        CHECK(loaded.biases[i] == net.biases[i]);
    }

    /* And it computes the same answer from the same input */
    CHECK(parse_input_string(&loaded, "0.1,0.2,0.3") == true);
    CHECK_FLT(loaded.output_buffer[0], expected_out, 1e-6f);

    network_free(&net);
    network_free(&loaded);
    remove(TEST_FILE);
}

static void write_garbage_at(const char *path, long offset, char byte)
{
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    fseek(fp, offset, SEEK_SET);
    fputc(byte, fp);
    fclose(fp);
}

static void test_load_rejects_corrupt_files(void)
{
    CompactNeuralNetwork net = {0};
    const uint16_t sizes[2] = { 2, 2 };

    CHECK(network_init(&net, 2, sizes) == 0);
    net.weights[0] = 0.123456f;
    const float saved_w0 = net.weights[0];

    remove(TEST_FILE);
    CHECK(network_save(&net, TEST_FILE) == true);

    /* --- Corrupted magic: rejected, net untouched --- */
    write_garbage_at(TEST_FILE, 0, '\x00');
    CHECK(network_load(&net, TEST_FILE) == false);
    CHECK(net.initialized);
    CHECK(net.weights[0] == saved_w0);

    /* Restore a valid file for the next case */
    CHECK(network_save(&net, TEST_FILE) == true);

    /* --- Trailing garbage: rejected, net untouched --- */
    FILE *fp = fopen(TEST_FILE, "ab");
    CHECK(fp != NULL);
    if (fp) {
        fputc('X', fp);
        fclose(fp);
    }
    CHECK(network_load(&net, TEST_FILE) == false);
    CHECK(net.initialized);
    CHECK(net.weights[0] == saved_w0);

    /* --- Truncated file: rejected, net untouched --- */
    CHECK(network_save(&net, TEST_FILE) == true);
    FILE *in = fopen(TEST_FILE, "rb");
    CHECK(in != NULL);
    if (in) {
        fseek(in, 0, SEEK_END);
        long len = ftell(in);
        char *buf = malloc((size_t)len - 10);
        CHECK(buf != NULL);
        if (buf) {
            fseek(in, 0, SEEK_SET);
            if (fread(buf, 1, (size_t)len - 10, in) == (size_t)len - 10) {
                FILE *out = fopen(TEST_FILE, "wb");
                if (out) {
                    fwrite(buf, 1, (size_t)len - 10, out);
                    fclose(out);
                }
            }
            free(buf);
        }
        fclose(in);
    }
    CHECK(network_load(&net, TEST_FILE) == false);
    CHECK(net.initialized);
    CHECK(net.weights[0] == saved_w0);

    /* --- Nonexistent file: rejected --- */
    CHECK(network_load(&net, "no_such_model_file.nn") == false);
    CHECK(net.initialized);
    CHECK(net.weights[0] == saved_w0);

    /* NULL arguments */
    CHECK(network_load(NULL, TEST_FILE) == false);
    CHECK(network_load(&net, NULL) == false);

    network_free(&net);
    remove(TEST_FILE);
}

/*****************************************************************************
 *                                                                           *
 *                                   MAIN                                    *
 *                                                                           *
 *****************************************************************************/

int main(void)
{
    test_sigmoid();
    test_init_valid();
    test_init_rejects_bad_topology();
    test_init_deterministic();
    test_reset();
    test_forward_two_layers();
    test_forward_three_layers();
    test_forward_weight_offsets();
    test_parse_config();
    test_parse_input();
    test_save_then_load();
    test_load_rejects_corrupt_files();

    printf("test_neuron: %d assertions, %d failed\n",
           tests_run, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}

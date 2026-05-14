#ifndef NN_H
#define NN_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#define LEARNING_RATE 0.01f
#define EPOCHS 10

#include "mnist_loader.h"

typedef struct {
    float w1[HIDDEN_SIZE][INPUT_SIZE];
    float b1[HIDDEN_SIZE];

    float w2[OUTPUT_SIZE][HIDDEN_SIZE];
    float b2[OUTPUT_SIZE];
} NeuralNetwork;

static float random_weight(void) {
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
}

static void init_network(NeuralNetwork *net) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->w1[i][j] = random_weight();
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        net->b1[i] = 0.0f;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->w2[i][j] = random_weight();
        }
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net->b2[i] = 0.0f;
    }
}

static float relu(float x) {
    if (x > 0.0f) {
        return x;
    }

    return 0.0f;
}

static float relu_derivative(float x) {
    if (x > 0.0f) {
        return 1.0f;
    }

    return 0.0f;
}

static void softmax(float input[OUTPUT_SIZE], float output[OUTPUT_SIZE]) {
    float max_val = input[0];

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = output[i] / sum;
    }
}

static void forward(
    NeuralNetwork *net,
    float input[INPUT_SIZE],
    float z1[HIDDEN_SIZE],
    float a1[HIDDEN_SIZE],
    float z2[OUTPUT_SIZE],
    float output[OUTPUT_SIZE]
) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        z1[i] = net->b1[i];

        for (int j = 0; j < INPUT_SIZE; j++) {
            z1[i] += net->w1[i][j] * input[j];
        }

        a1[i] = relu(z1[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        z2[i] = net->b2[i];

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            z2[i] += net->w2[i][j] * a1[j];
        }
    }

    softmax(z2, output);
}

static int predict_digit(float output[OUTPUT_SIZE]) {
    int best_index = 0;
    float best_value = output[0];

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > best_value) {
            best_value = output[i];
            best_index = i;
        }
    }

    return best_index;
}

static float cross_entropy_loss(float output[OUTPUT_SIZE], int label) {
    return -logf(output[label] + 0.000001f);
}

static float train_sample(
    NeuralNetwork *net,
    float input[INPUT_SIZE],
    int label,
    float learning_rate
) {
    float z1[HIDDEN_SIZE];
    float a1[HIDDEN_SIZE];
    float z2[OUTPUT_SIZE];
    float output[OUTPUT_SIZE];
    float dz2[OUTPUT_SIZE];
    float da1[HIDDEN_SIZE];
    float dz1[HIDDEN_SIZE];

    forward(net, input, z1, a1, z2, output);
    float loss = cross_entropy_loss(output, label);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        dz2[i] = output[i];
    }
    dz2[label] -= 1.0f;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        da1[j] = 0.0f;

        for (int i = 0; i < OUTPUT_SIZE; i++) {
            da1[j] += net->w2[i][j] * dz2[i];
        }

        dz1[j] = da1[j] * relu_derivative(z1[j]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->w2[i][j] -= learning_rate * dz2[i] * a1[j];
        }

        net->b2[i] -= learning_rate * dz2[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->w1[i][j] -= learning_rate * dz1[i] * input[j];
        }

        net->b1[i] -= learning_rate * dz1[i];
    }

    return loss;
}

static void train_network(
    NeuralNetwork *net,
    MnistDataset *train_data,
    int epochs,
    float learning_rate
) {
    float input[INPUT_SIZE];
    int label;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        int samples = 0;

        mnist_rewind(train_data);

        while (mnist_read_sample(train_data, input, &label)) {
            float z1[HIDDEN_SIZE];
            float a1[HIDDEN_SIZE];
            float z2[OUTPUT_SIZE];
            float output[OUTPUT_SIZE];

            forward(net, input, z1, a1, z2, output);
            if (predict_digit(output) == label) {
                correct++;
            }

            total_loss += train_sample(net, input, label, learning_rate);
            samples++;
        }

        printf("epoch %d/%d - loss %.4f - accuracy %.2f%%\n",
               epoch + 1,
               epochs,
               total_loss / (float)samples,
               100.0f * (float)correct / (float)samples);
    }
}

static float evaluate_network(NeuralNetwork *net, MnistDataset *test_data) {
    float input[INPUT_SIZE];
    int label;
    int correct = 0;
    int samples = 0;

    mnist_rewind(test_data);

    while (mnist_read_sample(test_data, input, &label)) {
        float z1[HIDDEN_SIZE];
        float a1[HIDDEN_SIZE];
        float z2[OUTPUT_SIZE];
        float output[OUTPUT_SIZE];

        forward(net, input, z1, a1, z2, output);

        if (predict_digit(output) == label) {
            correct++;
        }

        samples++;
    }

    if (samples == 0) {
        return 0.0f;
    }

    return (float)correct / (float)samples;
}

#endif

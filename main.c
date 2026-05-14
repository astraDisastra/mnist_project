#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn.h"

int main(void) {
    NeuralNetwork net;
    MnistDataset train_data;
    MnistDataset test_data;

    srand((unsigned int)time(NULL));

    if (!mnist_open(&train_data,
                    "images/train-images-idx3-ubyte",
                    "images/train-labels-idx1-ubyte")) {
        return 1;
    }

    if (!mnist_open(&test_data,
                    "images/t10k-images-idx3-ubyte",
                    "images/t10k-labels-idx1-ubyte")) {
        mnist_close(&train_data);
        return 1;
    }

    init_network(&net);
    train_network(&net, &train_data, EPOCHS, LEARNING_RATE);

    float accuracy = evaluate_network(&net, &test_data);
    printf("test accuracy %.2f%%\n", accuracy * 100.0f);

    mnist_close(&test_data);
    mnist_close(&train_data);

    return 0;
}

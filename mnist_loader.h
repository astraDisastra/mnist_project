#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>
#include <stdio.h>

#ifndef INPUT_SIZE
#define INPUT_SIZE 784
#endif

typedef struct {
    FILE *images;
    FILE *labels;
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
    uint32_t index;
} MnistDataset;

int mnist_open(MnistDataset *dataset, const char *image_path, const char *label_path);
void mnist_close(MnistDataset *dataset);
int mnist_read_sample(MnistDataset *dataset, float input[INPUT_SIZE], int *label);
void mnist_rewind(MnistDataset *dataset);

#endif

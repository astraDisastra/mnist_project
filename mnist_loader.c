#include "mnist_loader.h"

#include <stdio.h>
#include <stdlib.h>

#define MNIST_ROWS 28
#define MNIST_COLS 28
#define MNIST_IMAGE_MAGIC 2051
#define MNIST_LABEL_MAGIC 2049



// mnist stores headers big-endian
static uint32_t read_u32_be(FILE *f) {
    unsigned char b[4];

    if (fread(b, 1, 4, f) != 4) {
        fprintf(stderr, "Failed to read MNIST header\n");
        exit(1);
    }

    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8) |
           ((uint32_t)b[3]);
}


int mnist_open(MnistDataset *dataset, const char *image_path, const char *label_path) {
    uint32_t image_magic;
    uint32_t label_magic;
    uint32_t label_count;

    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
    dataset->rows = 0;
    dataset->cols = 0;
    dataset->index = 0;

    dataset->images = fopen(image_path, "rb");
    if (dataset->images == NULL) {
        fprintf(stderr, "Failed to open image file: %s\n", image_path);
        return 0;
    }

    dataset->labels = fopen(label_path, "rb");
    if (dataset->labels == NULL) {
        fprintf(stderr, "Failed to open label file: %s\n", label_path);
        fclose(dataset->images);
        dataset->images = NULL;
        return 0;
    }

    image_magic = read_u32_be(dataset->images);
    dataset->count = read_u32_be(dataset->images);
    dataset->rows = read_u32_be(dataset->images);
    dataset->cols = read_u32_be(dataset->images);

    label_magic = read_u32_be(dataset->labels);
    label_count = read_u32_be(dataset->labels);

    if (image_magic != MNIST_IMAGE_MAGIC) {
        fprintf(stderr, "Invalid MNIST image magic number: %u\n", image_magic);
        mnist_close(dataset);
        return 0;
    }

    if (label_magic != MNIST_LABEL_MAGIC) {
        fprintf(stderr, "Invalid MNIST label magic number: %u\n", label_magic);
        mnist_close(dataset);
        return 0;
    }

    if (dataset->count != label_count) {
        fprintf(stderr, "MNIST image count does not match label count\n");
        mnist_close(dataset);
        return 0;
    }

    if (dataset->rows != MNIST_ROWS || dataset->cols != MNIST_COLS) {
        fprintf(stderr, "Expected 28x28 MNIST images, got %ux%u\n",
                dataset->rows, dataset->cols);
        mnist_close(dataset);
        return 0;
    }

    dataset->index = 0;
    return 1;
}

void mnist_close(MnistDataset *dataset) {
    if (dataset->images != NULL) {
        fclose(dataset->images);
        dataset->images = NULL;
    }

    if (dataset->labels != NULL) {
        fclose(dataset->labels);
        dataset->labels = NULL;
    }

    dataset->count = 0;
    dataset->rows = 0;
    dataset->cols = 0;
    dataset->index = 0;
}

int mnist_read_sample(MnistDataset *dataset, float input[INPUT_SIZE], int *label) {
    unsigned char label_byte;
    unsigned char pixels[INPUT_SIZE];

    if (dataset->index >= dataset->count) {
        return 0;
    }

    if (fread(&label_byte, 1, 1, dataset->labels) != 1) {
        return 0;
    }

    if (fread(pixels, 1, INPUT_SIZE, dataset->images) != INPUT_SIZE) {
        return 0;
    }

    *label = label_byte;

    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = pixels[i] / 255.0f;
    }

    dataset->index++;
    return 1;
}

void mnist_rewind(MnistDataset *dataset) {
    fseek(dataset->images, 16, SEEK_SET);
    fseek(dataset->labels, 8, SEEK_SET);
    dataset->index = 0;
}

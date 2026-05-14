# MNIST Neural Network Project

This project is a simple neural network written in C that classifies handwritten digits from the MNIST dataset.

It uses a basic feedforward structure with:

- One input layer
- One hidden layer
- One output layer

The goal of this project was to learn how a neural network works by building the main parts manually instead of using a machine learning library.

---

## What It Does

The program loads the MNIST image and label files, trains the neural network on the training data, and then tests its accuracy on the test data.

Each MNIST image is 28x28 pixels, so each image is converted into 784 input values. These values are passed through the network to predict which digit the image represents.

The project includes:

- Loading MNIST data from binary files
- Normalizing image pixels
- Running forward propagation
- Using ReLU in the hidden layer
- Using softmax in the output layer
- Calculating loss
- Updating weights with backpropagation
- Printing training progress and final test accuracy

---

## Project Files

```text
main.c
```

Runs the program, loads the dataset, trains the network, and prints the results.

```text
nn.h
```

Contains the neural network structure and training logic.

```text
mnist_loader.c
```

Contains the functions for reading MNIST image and label files.

```text
mnist_loader.h
```

Contains the dataset structure and function declarations for the MNIST loader.

---

## Building and Running

Compile the project with:

```bash
gcc main.c mnist_loader.c -o mnist_lab -lm
```

Run it with:

```bash
./mnist_lab
```

The MNIST files should be placed in an `images/` directory:

```text
images/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

---

## Conclusion

This project was a hands-on way to understand the basic learning process behind neural networks.

By building the network in C, I was able to see how input data, weights, activations, loss, and backpropagation work together to let a model improve over time.

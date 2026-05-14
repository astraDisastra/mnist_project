# MNIST Neural Network Lab

This project is a learning-focused implementation of a simple neural network in C for classifying handwritten digits from the MNIST dataset.

The goal of this project is not to build a production-ready machine learning library. Instead, it is meant to function as a lab for understanding the core ideas behind neural networks by implementing each major step manually.

The network uses:

- One input layer
- One hidden layer
- One output layer

The model reads MNIST image and label files, trains on the training set, and evaluates its accuracy on the test set.

---

## Purpose

This project was built to study how a neural network learns.

Rather than using a machine learning framework, the network is written directly in C so that the important parts are visible:

- How image data becomes numeric input
- How weights and biases are initialized
- How a forward pass produces a prediction
- How activation functions affect learning
- How loss measures error
- How backpropagation updates weights
- How accuracy changes over training

This makes the project more of a lab experiment than a tool.

---

## What the Network Learns From

The project uses the MNIST dataset, which contains images of handwritten digits from `0` to `9`.

Each image is:

```text
28 pixels × 28 pixels = 784 input values
```

Each pixel is converted into a floating-point value between `0.0` and `1.0`.

```c
input[i] = pixels[i] / 255.0f;
```

This means the network does not see an image the way a person does. It sees a long array of 784 numbers.

---

## Network Architecture

The network is defined in `nn.h`.

```text
Input Layer:   784 values
Hidden Layer:  128 neurons
Output Layer:  10 values
```

The output layer has 10 values because there are 10 possible digit classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The output with the highest value is treated as the network's prediction.

---

## Learning Concepts in This Project

### 1. Input Representation

The MNIST image starts as pixels, but the neural network needs numbers.

A 28x28 image is flattened into a one-dimensional array:

```text
[784 pixel values]
```

This is the input layer.

---

### 2. Weights and Biases

The network stores weights and biases in the `NeuralNetwork` struct:

```c
float w1[HIDDEN_SIZE][INPUT_SIZE];
float b1[HIDDEN_SIZE];

float w2[OUTPUT_SIZE][HIDDEN_SIZE];
float b2[OUTPUT_SIZE];
```

The weights decide how strongly one layer connects to the next. The biases allow each neuron to shift its output.

At the beginning, weights are initialized randomly. The network starts without knowledge and improves only through training.

---

### 3. Forward Pass

The forward pass moves data through the network:

```text
input -> hidden layer -> output layer -> prediction
```

The hidden layer uses ReLU:

```c
relu(x) = max(0, x)
```

The output layer uses softmax, which converts raw output values into probability-like values.

---

### 4. Loss

The network uses cross-entropy loss:

```c
loss = -log(output[label])
```

The loss is high when the network is confident but wrong. The loss is low when the network gives a high score to the correct digit.

Training tries to make this loss smaller over time.

---

### 5. Backpropagation

Backpropagation calculates how much each weight contributed to the error.

The project manually computes:

- Error at the output layer
- Error passed backward into the hidden layer
- Weight updates for the hidden-to-output layer
- Weight updates for the input-to-hidden layer

This is the main learning step of the project.

---

### 6. Training

Training happens one sample at a time.

For each image:

1. The image is read from the MNIST file.
2. The network makes a prediction.
3. The loss is calculated.
4. Backpropagation updates the weights.
5. The next image is processed.

After each epoch, the program prints the average loss and training accuracy.

Example output:

```text
epoch 1/15 - loss 0.8421 - accuracy 74.32%
epoch 2/15 - loss 0.4218 - accuracy 87.11%
...
test accuracy 91.45%
```

The exact accuracy may be different each time because the weights are randomly initialized.

---

## Project Files

```text
main.c
```

Runs the full experiment. It loads the dataset, initializes the network, trains it, evaluates it, and prints test accuracy.

```text
nn.h
```

Contains the neural network structure and learning logic, including forward propagation, softmax, loss, prediction, training, and evaluation.

```text
mnist_loader.c
```

Reads MNIST image and label files from disk.

```text
mnist_loader.h
```

Defines the MNIST dataset structure and loader functions.

---

## Expected Dataset Layout

The program expects the MNIST files to be inside an `images/` directory:

```text
images/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

These are the raw MNIST files, not PNG or JPG images.

---

## Building the Project

Compile with:

```bash
gcc main.c mnist_loader.c -o mnist_lab -lm
```

The `-lm` flag links the math library, which is needed for functions like `expf()` and `logf()`.

---

## Running the Lab

After compiling, run:

```bash
./mnist_lab
```

The program will train the neural network and then print the final test accuracy.

---

## Things to Experiment With

This project is useful because small changes can show how neural networks behave.

Try changing these values in `nn.h`:

```c
#define HIDDEN_SIZE 128
#define LEARNING_RATE 0.01f
#define EPOCHS 15
```

### Possible Experiments

#### Change the Hidden Layer Size

Try:

```c
#define HIDDEN_SIZE 32
#define HIDDEN_SIZE 64
#define HIDDEN_SIZE 256
```

Questions to observe:

- Does a larger hidden layer learn faster?
- Does it improve test accuracy?
- Does it take longer to train?

---

#### Change the Learning Rate

Try:

```c
#define LEARNING_RATE 0.1f
#define LEARNING_RATE 0.01f
#define LEARNING_RATE 0.001f
```

Questions to observe:

- What happens if the learning rate is too high?
- What happens if it is too low?
- Does the loss decrease smoothly?

---

#### Change the Number of Epochs

Try:

```c
#define EPOCHS 1
#define EPOCHS 5
#define EPOCHS 30
```

Questions to observe:

- Does accuracy improve every epoch?
- Does it stop improving after a while?
- Can the model overfit?

---

## What This Project Does Not Do

This project keeps the neural network intentionally simple.

It does not include:

- Multiple hidden layers
- Convolutional layers
- Mini-batch training
- Model saving or loading
- External machine learning libraries
- Optimizers like Adam or RMSProp

These are all useful features, but leaving them out makes the basic learning process easier to understand.

---

## Main Takeaway

This project shows that a neural network is not magic.

At its core, it is a sequence of numeric steps:

```text
input values
-> weighted sums
-> activation functions
-> output probabilities
-> loss
-> gradient updates
```

By implementing these steps manually, this lab helps explain what a neural network is actually doing when it learns.

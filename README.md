# MNIST Neural Network Project Lab

This project is my implementation of a simple neural network in C for classifying handwritten digits from the MNIST dataset.

I built this project as a learning lab, not as a polished machine learning tool. The main goal was to understand what is happening inside a basic neural network by writing the important parts myself instead of relying on a machine learning library.

The network has:

- One input layer
- One hidden layer
- One output layer

The model trains on the MNIST training set and then tests its accuracy on the MNIST test set.

---

## Project Goal

The purpose of this project was to learn how a neural network works at a lower level.

Instead of using a framework like TensorFlow or PyTorch, I implemented the network logic directly in C. This helped me see the steps that are usually hidden inside machine learning libraries.

Through this project, I practiced:

- Loading raw MNIST data from files
- Converting image pixels into numeric input values
- Initializing weights and biases
- Running a forward pass
- Applying activation functions
- Calculating loss
- Using backpropagation to update weights
- Measuring training and testing accuracy

The project helped me understand that neural networks are built from a series of mathematical steps rather than being a black box.

---

## Dataset

This project uses the MNIST dataset, which contains images of handwritten digits from `0` to `9`.

Each image is 28 pixels wide and 28 pixels tall.

```text
28 * 28 = 784 input values
```

Each pixel is converted into a floating-point value between `0.0` and `1.0`.

```c
input[i] = pixels[i] / 255.0f;
```

This normalization step makes the data easier for the neural network to work with.

The network does not process the image visually like a person would. Instead, each image is flattened into an array of 784 numbers.

---

## Network Architecture

The neural network uses a simple feedforward structure:

```text
Input Layer -> Hidden Layer -> Output Layer
```

The architecture is:

```text
Input Layer:   784 values
Hidden Layer:  128 neurons
Output Layer:  10 values
```

The input layer represents the 784 pixels in each image.

The hidden layer is where the network begins learning patterns in the input data.

The output layer has 10 values, one for each possible digit:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The digit with the highest output value is treated as the network's prediction.

---

## What I Implemented

### MNIST Data Loading

I wrote helper functions to read the MNIST image and label files.

The dataset is stored in a binary format, so part of the project involved understanding how to:

- Open binary files
- Read header information
- Read image data
- Read label data
- Store the dataset in memory

This was important because the network cannot train until the raw dataset is converted into a usable format.

---

### Forward Propagation

I implemented the forward pass, which moves one image through the network.

The general flow is:

```text
Input values
-> Hidden layer weighted sums
-> ReLU activation
-> Output layer weighted sums
-> Softmax
-> Prediction
```

This helped me understand how the network turns raw pixel values into a prediction.

---

### ReLU Activation

The hidden layer uses ReLU as its activation function.

```c
relu(x) = max(0, x)
```

ReLU allows the network to model more complex patterns by introducing nonlinearity. Without an activation function, the network would mostly behave like a linear equation, even with multiple layers.

---

### Softmax Output

The output layer uses softmax.

Softmax converts the raw output values into probability-like values.

For example, the network might output something like:

```text
Digit 0: 0.01
Digit 1: 0.03
Digit 2: 0.02
Digit 3: 0.88
Digit 4: 0.01
Digit 5: 0.02
Digit 6: 0.00
Digit 7: 0.01
Digit 8: 0.01
Digit 9: 0.01
```

In this example, the network would predict `3` because it has the highest value.

---

### Loss Calculation

I used cross-entropy loss to measure how wrong the network was.

The loss is based on how much confidence the network gave to the correct answer.

If the correct digit is `3`, then the model should give a high output value to index `3`.

A high loss means the network made a poor prediction. A lower loss means the network is improving.

---

### Backpropagation

Backpropagation was the most important learning part of this project.

I implemented the process of calculating how much each weight contributed to the final error, then updating the weights to reduce that error.

The project updates:

- Hidden-to-output weights
- Output biases
- Input-to-hidden weights
- Hidden biases

This helped me understand that training is not just guessing. The network improves by using the error to make small adjustments to its weights.

---

### Training Loop

The training loop processes the dataset over multiple epochs.

For each training example, the program:

1. Converts the image into input values
2. Runs a forward pass
3. Calculates the loss
4. Runs backpropagation
5. Updates the weights and biases
6. Tracks accuracy

After each epoch, the program prints the loss and accuracy so I can see whether the network is learning.

Example output:

```text
epoch 1/15 - loss 0.8421 - accuracy 74.32%
epoch 2/15 - loss 0.4218 - accuracy 87.11%
...
test accuracy 91.45%
```

The exact values can change because the weights are initialized randomly.

---

## Project Files

```text
main.c
```

Contains the main program. It loads the data, initializes the network, trains the model, evaluates it, and prints the results.

```text
nn.h
```

Contains the neural network structure and the main learning logic, including forward propagation, prediction, loss, training, and evaluation.

```text
mnist_loader.c
```

Contains the functions that read the MNIST image and label files.

```text
mnist_loader.h
```

Defines the dataset structure and function declarations for loading and freeing MNIST data.

---

## Expected Dataset Layout

The program expects the MNIST files to be stored in an `images/` directory.

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

Compile the program with:

```bash
gcc main.c mnist_loader.c -o mnist_lab -lm
```

The `-lm` flag links the math library, which is needed for functions such as `expf()` and `logf()`.

---

## Running the Project

Run the compiled program with:

```bash
./mnist_lab
```

The program trains the network and then prints the final test accuracy.

---

## Experiments I Can Run

One useful part of this project is that I can change a few values and observe how the network behaves.

For example, I can experiment with:

```c
#define HIDDEN_SIZE 128
#define LEARNING_RATE 0.01f
#define EPOCHS 15
```

### Hidden Layer Size

Changing the hidden layer size lets me test how model capacity affects learning.

Examples:

```c
#define HIDDEN_SIZE 32
#define HIDDEN_SIZE 64
#define HIDDEN_SIZE 256
```

Questions I can explore:

- Does a larger hidden layer improve accuracy?
- Does it make training slower?
- Is there a point where adding more neurons does not help much?

---

### Learning Rate

Changing the learning rate lets me see how step size affects training.

Examples:

```c
#define LEARNING_RATE 0.1f
#define LEARNING_RATE 0.01f
#define LEARNING_RATE 0.001f
```

Questions I can explore:

- Does a high learning rate make training unstable?
- Does a low learning rate make learning too slow?
- Which value gives the smoothest loss decrease?

---

### Number of Epochs

Changing the number of epochs lets me see how repeated training affects accuracy.

Examples:

```c
#define EPOCHS 1
#define EPOCHS 5
#define EPOCHS 30
```

Questions I can explore:

- Does accuracy improve after every epoch?
- Does the model eventually stop improving?
- Does the training accuracy increase faster than test accuracy?

---

## What I Learned

This project helped me understand the basic structure of a neural network.

The biggest takeaway is that a neural network is a sequence of connected mathematical operations:

```text
Input data
-> Weights and biases
-> Activation functions
-> Output values
-> Loss
-> Gradients
-> Weight updates
```

By implementing these steps myself, I got a clearer understanding of how training works.

I also learned that machine learning depends heavily on small design choices, such as:

- How weights are initialized
- Which activation function is used
- How large the hidden layer is
- What learning rate is chosen
- How many epochs the model trains for

Even though this network is simple, it demonstrates the core learning process used in larger neural networks.

---

## What This Project Does Not Include

This project intentionally keeps the network simple.

It does not include:

- Multiple hidden layers
- Convolutional layers
- Mini-batch training
- Model saving or loading
- Advanced optimizers
- External machine learning libraries

Leaving these features out made it easier to focus on the fundamentals.

---

## Conclusion

This project was a hands-on lab for understanding neural networks by building one from scratch in C.

It showed me how a model can start with random weights and gradually improve through training. More importantly, it helped me understand the individual steps behind that learning process.

The final result is not just a digit classifier. It is a working demonstration of forward propagation, loss calculation, backpropagation, and gradient-based learning.

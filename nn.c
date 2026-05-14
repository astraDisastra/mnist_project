#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#define LEARNING_RATE 0.01
#define EPOCHS 10



typedef struct {
	float w1[HIDDEN_SIZE][INPUT_SIZE];
	float b1[HIDDEN_SIZE];

	float w2[OUTPUT_SIZE][HIDDEN_SIZE];
	float b2[OUTPUT_SIZE];
} NeuralNetwork;


// random value between -0.05 and 0.05
float random_weight() {
	return ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
}

void init_network(NeuralNetwork *net) {
	
	// initialize w1
	for (int i = 0; i < HIDDEN_SIZE; i++) {
		for (int j = 0; j < INPUT_SIZE; j++) {
			net->w1[i][j] = random_weight();
		}
	}

	// initialize b1 -> each hidden neuron gets 1 bias
	for (int i = 0; i < HIDDEN_SIZE; i++) {
		net->b1[i] = 0.0f;
	}

	//initialize w2
	for (int = 0; i < OUTPUT_SIZE; i++) {
		for (int j = 0; j < HIDDEN_SIZE; j++) {
			net->w2[i][j] = random_weight();
		}
	}

	// initialize b2
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		net->b2[i] = 0.0f;
	}
}


// RelU implementation
float relu(float x) {
	if (x > 0.0f) {
		return x;
	}

	return 0.0f;

}


// softmax normalizes raw scores into probabailities
void softmax(float input[OUTPUT_SIZE], float output[OUTPUT_SIZE]) {
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



// forwarding gets outputs
void forward(
	NeuralNetwork *net, float input[INPUT_SIZE],
	float z1[HIDDEN_SIZE],
	float a1[HIDDEN_SIZE],
	float z2[OUTPUT_SIZE],
	float output[OUTPUT_SIZE]
){

	//input layer to hidden layer
	for (int i = 0; i < HIDDEN_SIZE; i++) {
		z1[i] = net->b1[i];

		for (int j = 0; j < INPUT_SIZE; j++) {
			z1[i] += net->w1[i][j] * input[j];
		}

		a1[i] = relu(z1[i]);
	}
	
	// hidden layer to output layer
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		z2[i] = net->b2[i];

		for (int j = 0; j < HIDDEN_SIZE; j++) {
			z2[i] += net->w2[i][j] * a1[j];
		}
	}

	softmax(z2, output);
}


int predict_digit(float output[OUTPUT_SIZE]) {
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

float cross_entropy_loss(float output[OUTPUT_SIZE], int label) {
	return -logf(output[label] + 0.000001); // added to avoid log(0)
}


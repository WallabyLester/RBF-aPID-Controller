#include "rbf_model.h"

/**
 * @brief Constructor to initialize the RBF model.
 */
RBFModel::RBFModel(int n_centers, int input_dim, double sigma, bool random_centers) 
    : n_centers(n_centers), input_dim(input_dim), sigma(sigma) {
    centers = new double*[n_centers]; // Allocate memory for centers
    weights = new double[n_centers]; // Allocate memory for weights

    // Check if memory allocation was successful
    if (!centers || !weights) {
        if (centers) delete[] centers; // Clean up if centers allocation was successful
        return; // Indicate failure (Use whatever exception handling you have)
    }

    // Initialize centers and weights
    for (int i = 0; i < n_centers; ++i) {
        centers[i] = new double[input_dim]; // Allocate memory for each center
        if (random_centers) {
            for (int j = 0; j < input_dim; ++j) {
                centers[i][j] = static_cast<double>(rand()) / RAND_MAX; // Random centers
            }
        } else {
            for (int j = 0; j < input_dim; ++j) {
                centers[i][j] = static_cast<double>(i); // Fixed centers
            }
        }
        weights[i] = 0.0; // Initialize weights to zero
    }
}

/**
 * @brief Destructor to free allocated memory.
 */
RBFModel::~RBFModel() {
    for (int i = 0; i < n_centers; ++i) {
        delete[] centers[i]; // Free memory for each center inside centers
    }
    delete[] centers; // Free memory for centers
    delete[] weights; // Free memory for weights
}

/**
 * @brief Gaussian function used in RBF prediction.
 */
double RBFModel::gaussian(const double* input, const double* center) {
    double norm = 0.0;
    for (int i = 0; i < input_dim; ++i) {
        norm += pow(input[i] - center[i], 2);
    }
    return exp(-0.5 * norm / (sigma * sigma));
}

/**
 * @brief Predict the RBF output for a given input.
 */
double RBFModel::predict(const double* input) {
    double output = 0.0;
    for (int i = 0; i < n_centers; ++i) {
        output += weights[i] * gaussian(input, centers[i]);
    }
    return output;
}

/**
 * @brief Adapt weights based on the error and learning rate.
 */
void RBFModel::adapt(double error, double learning_rate, const double* input) {
    for (int i = 0; i < n_centers; ++i) {
        double influence = gaussian(input, centers[i]); // Calculate influence based on input
        weights[i] += learning_rate * error * influence; // Update weight based on error and influence
    }
}

/**
 * @brief Train the RBF model using recorded data.
 */
void RBFModel::train(const double* inputs, const double* targets, int n_samples, int epochs, double learning_rate) {
    for (int iter = 0; iter < epochs; ++iter) {
        for (int sample = 0; sample < n_samples; ++sample) {
            double output = predict(&inputs[sample * input_dim]); 
            double error = targets[sample] - output;

            adapt(error, learning_rate, &inputs[sample * input_dim]); // Adapt weights based on the error
        }
    }
}

/**
 * @brief Get the weight at a specific index.
 */
double RBFModel::get_weight(int index) const {
    if (index < 0 || index >= n_centers) return 0.0;
    return weights[index];
}

/**
 * @brief Set the weight at a specific index.
 */
void RBFModel::set_weight(int index, double value) {
    if (index < 0 || index >= n_centers) return;
    weights[index] = value;
}

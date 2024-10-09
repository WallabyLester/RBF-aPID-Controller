#include "rbf_model.h"

/**
 * @brief Constructor to initialize the RBF model.
 */
RBFModel::RBFModel(int numCenters, int inputDim, double sigma, bool randomCenters) 
    : numCenters(numCenters), inputDim(inputDim), sigma(sigma) {
    centers = new double*[numCenters]; // Allocate memory for centers
    weights = new double[numCenters]; // Allocate memory for weights

    // Check if memory allocation was successful
    if (!centers || !weights) {
        if (centers) delete[] centers; // Clean up if centers allocation was successful
        return; // Indicate failure (Use whatever exception handling you have)
    }

    // Initialize centers and weights
    for (int i = 0; i < numCenters; ++i) {
        centers[i] = new double[inputDim]; // Allocate memory for each center
        if (randomCenters) {
            for (int j = 0; j < inputDim; ++j) {
                centers[i][j] = static_cast<double>(rand()) / RAND_MAX; // Random centers
            }
        } else {
            for (int j = 0; j < inputDim; ++j) {
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
    for (int i = 0; i < numCenters; ++i) {
        delete[] centers[i]; // Free memory for each center inside centers
    }
    delete[] centers; // Free memory for centers
    delete[] weights; // Free memory for weights
}

/**
 * @brief Gaussian function used in RBF evaluation.
 */
double RBFModel::gaussian(const double* input, const double* center) {
    double norm = 0.0;
    for (int i = 0; i < inputDim; ++i) {
        norm += pow(input[i] - center[i], 2);
    }
    return exp(-0.5 * norm / (sigma * sigma));
}

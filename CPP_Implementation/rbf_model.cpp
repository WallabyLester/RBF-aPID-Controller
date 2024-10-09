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

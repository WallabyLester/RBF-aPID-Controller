#ifndef RBF_MODEL_H
#define RBF_MODEL_H

#include <cmath>
#include <cstdlib>

/**
 * @class RBFModel
 * @brief Radial Basis Function (RBF) Model for function approximation.
 * 
 * This class implements an RBF model that evaluates outputs based
 * on a set of radial basis functions centered at specified locations.
 */
class RBFModel {
public:

    /**
     * @brief Constructor to initialize the RBF model.
     * 
     * Initializes the RBF model with a specified number
     * of centers, input dimensions, and the spread (sigma) of the RBFs.
     * Random initialization of the centers can be turned off.
     * 
     * @param numCenters The number of radial basis function centers.
     * @param inputDim The dimensionality of the input data.
     * @param sigma The spread of the RBFs (default is 1.0).
     * @param randomCenters Boolean to initialize centers randomly (default is true).
     */
    RBFModel(int numCenters, int inputDim, double sigma = 1.0, bool randomCenters = true);
    
    /**
     * @brief Destructor to free allocated memory.
     */
    ~RBFModel();
    
    /**
     * @brief Predict the RBF output for a given input.
     * 
     * @param input A pointer to an array of input values.
     * @return The computed output of the RBF model.
     */
    double predict(const double* input);
    
    /**
     * @brief Adapt weights based on the error and learning rate.
     * 
     * @param error The difference between the desired output and the actual output.
     * @param learningRate The rate at which the weights are adjusted.
     * @param input A pointer to an array of input values used for adaptation.
     */
    void adapt(double error, double learningRate, const double* input);
    
    /**
     * @brief Get the weight at a specific index.
     * 
     * @param index The index of the weight to retrieve.
     * @return The weight at the specified index.
     */
    double get_weight(int index) const;
    
    /**
     * @brief Set the weight at a specific index.
     * 
     * @param index The index of the weight to set.
     * @param value The new value for the weight.
     */
    void set_weight(int index, double value);

private:
    double** centers; // 2D array for centers
    double* weights;  // Array of weights
    int numCenters;   // Number of RBF centers
    int inputDim;     // Dimension of the input
    double sigma;     // Spread of the RBF

    /**
     * @brief Gaussian function used in RBF evaluation.
     * 
     * @param input A pointer to an array of input values.
     * @param center A pointer to an array representing the center of the Gaussian function.
     * @return The output of the Gaussian function.
     */
    double gaussian(const double* input, const double* center);
};

#endif // RBF_MODEL_H

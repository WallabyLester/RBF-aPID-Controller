#include <gtest/gtest.h>
#include "rbf_model.h"

// Test fixture for RBFModel
class RBFModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up an RBF model with 5 centers, 3D input, and random centers
        n_centers = 5;
        input_dim = 3;
        rbf = new RBFModel(n_centers, input_dim);
    }

    void TearDown() override {
        delete rbf;
    }

    RBFModel* rbf;
    int n_centers;
    int input_dim;
};

// Test constructor and initial weight values
TEST_F(RBFModelTest, Constructor_And_Initial_Weights) {
    for (int i = 0; i < n_centers; ++i) {
        EXPECT_EQ(rbf->get_weight(i), 0.0);
    }
}

// Test predict method
TEST_F(RBFModelTest, Predict_Output) {
    // Set some weights and define centers
    for (int i = 0; i < n_centers; ++i) {
        rbf->set_weight(i, 1.0);
    }

    double input[] = {0.5, 0.5, 0.5};
    double output = rbf->predict(input);
    EXPECT_GT(output, 0.0); 
}

// Test adaptation of weights
TEST_F(RBFModelTest, Adapt_Weights) {
    double input[] = {1.0, 2.0, 3.0};
    rbf->set_weight(0, 1.0); 
    double error = 0.5; 
    double learning_rate = 0.1;

    double initial_weight = rbf->get_weight(0);
    
    rbf->adapt(error, learning_rate, input);
    
    double updated_weight = rbf->get_weight(0);

    EXPECT_NE(updated_weight, initial_weight);
}

// Test training function
TEST_F(RBFModelTest, Train_Function) {
    const int numSamples = 5;
    double inputs[] = {
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0,
        4.0, 4.0, 4.0
    };
    double targets[] = {0.0, 1.0, 0.0, 1.0, 0.0};
    int numIterations = 200;
    double learningRate = 0.01;

    rbf->train(inputs, targets, numSamples, numIterations, learningRate);

    for (int i = 0; i < n_centers; ++i) {
        EXPECT_NE(rbf->get_weight(i), 0.0);
    }
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

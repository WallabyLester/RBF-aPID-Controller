#include <gtest/gtest.h>
#include "rbf_model.h"

// Test fixture for RBFModel
class RBFModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up an RBF model with 3 centers, 3D input, and random centers
        n_centers = 3;
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

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

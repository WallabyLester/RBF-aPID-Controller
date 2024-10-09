import unittest
import numpy as np

from RBF_numpy import RBFNetwork

class TestRBFNetwork(unittest.TestCase):
    def setUp(self):
        """Set up an RBFNetwork instance for testing."""
        self.input_dim = 3
        self.n_centers = 5
        self.x = np.array([6.0, 0.5, 0.2])
        self.rbf_network = RBFNetwork(self.input_dim, self.n_centers)

    def test_gaussian(self):
        """Test the Gaussian function."""
        center = np.random.rand(1, self.input_dim)
        expected_output = np.exp(-np.linalg.norm(self.x - center) ** 2 / (2 * self.rbf_network.sigma ** 2))
        output = self.rbf_network.gaussian(self.x, center)
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_predict(self):
        """Test the predict function."""
        output_before = self.rbf_network.predict(self.x)
        self.assertIsInstance(output_before, float)
        [self.assertNotAlmostEqual(self.rbf_network.weights[i], self.rbf_network.weights[i+1]) 
         for i in range(len(self.rbf_network.weights)-1)]

        target = 1.0
        self.rbf_network.train(self.x, target)
        output_after = self.rbf_network.predict(self.x)
        self.assertIsInstance(output_after, float) 
        self.assertNotEqual(output_before, output_after)

    def test_train(self):
        """Test the training function."""
        target = 1.0
        initial_weights = self.rbf_network.weights.copy()
        output_before = np.dot(np.array([self.rbf_network.gaussian(self.x, center) 
                                         for center in self.rbf_network.centers]), initial_weights)
    
        self.rbf_network.train(self.x, target)
        
        self.assertFalse(np.array_equal(initial_weights, self.rbf_network.weights))
        
        output_after = self.rbf_network.predict(self.x)
        
        self.assertNotEqual(output_before, output_after)
        if not abs(target - output_after) < abs(target - output_before):
            print("Output did not move closer to the target after training.")

if __name__ == "__main__":
    unittest.main()

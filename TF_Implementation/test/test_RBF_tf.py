import unittest
import numpy as np
import tensorflow as tf
from RBF_tf import RBFLayer, RBFAdaptiveModel, train_rbf_adaptive

class TestRBFLayer(unittest.TestCase):
    def setUp(self):
        """ Set up a RBFLayer class instance."""
        self.n_centers = 5
        self.input_dim = 3
        self.rbf_layer = RBFLayer(self.n_centers, self.input_dim)

    def test_initialization(self):
        """ Test the initialization of the layer."""
        self.assertEqual(self.rbf_layer.n_centers, self.n_centers)
        self.assertEqual(self.rbf_layer.centers.shape, (self.n_centers, self.input_dim))
        self.assertEqual(self.rbf_layer.sigmas.shape, (self.n_centers,))


if __name__ == '__main__':
    unittest.main()

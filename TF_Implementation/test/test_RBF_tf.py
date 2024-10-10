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

    def test_call(self):
        """ Test the call method."""
        inputs = tf.random.normal((3, self.input_dim))
        output = self.rbf_layer(inputs)

        self.assertEqual(output.shape, (3, self.n_centers))
        self.assertTrue(tf.reduce_all(output >= 0)) 

class TestRBFAdaptiveModel(unittest.TestCase):
    def setUp(self):
        """ Set up a RBFAdaptiveModel class instance."""
        self.n_centers = 5
        self.input_dim = 3
        self.model = RBFAdaptiveModel(self.n_centers, self.input_dim)

    def test_initialization(self):
        """ Test the initialization of the model."""
        self.assertIsInstance(self.model.rbf_layer, RBFLayer)
        self.assertEqual(self.model.output_layer.units, 1)

        
if __name__ == '__main__':
    unittest.main()

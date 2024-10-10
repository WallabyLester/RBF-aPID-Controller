import unittest
import numpy as np
import tensorflow as tf
from aPID_tf import AdaptivePIDTf
from RBF_tf import RBFAdaptiveModel

class TestAdaptivePIDTf(unittest.TestCase):
    def setUp(self):
        """ Set up RBFAdaptiveModel and AdaptivePIDTf class instances."""
        self.Kp = 7.0
        self.Ki = 0.5
        self.Kd = 0.01
        self.n_centers = 5
        self.input_dim = 3
        self.rbf_model = RBFAdaptiveModel(self.n_centers, self.input_dim)
        self.pid_controller = AdaptivePIDTf(self.Kp, self.Ki, self.Kd, self.rbf_model)


if __name__ == '__main__':
    unittest.main()

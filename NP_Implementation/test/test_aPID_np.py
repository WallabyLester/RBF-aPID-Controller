import unittest
import numpy as np

from aPID_numpy import AdaptivePIDNP
from RBF_numpy import RBFNetwork

class TestAdaptivePIDNP(unittest.TestCase):
    def setUp(self):
        """Set up an AdaptivePIDNP instance for testing."""
        Kp = 4.0
        Ki = 0.1
        Kd = 0.01
        self.input_dim = 3 
        self.n_centers = 5
        self.rbf = RBFNetwork(self.input_dim, self.n_centers)
        self.apid = AdaptivePIDNP(Kp, Ki, Kd, self.rbf)

if __name__ == '__main__':
    unittest.main()

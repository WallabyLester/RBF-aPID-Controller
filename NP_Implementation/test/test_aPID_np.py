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

    def test_update(self):
        """Test the update method."""
        target = 10.0
        measured_value = 8.0
        dt = 0.1
        
        control_signal = self.apid.update(target, measured_value, dt)
        
        self.assertIsInstance(control_signal, float)        
        self.assertGreater(control_signal, 0)
        
        measured_value = 9.0
        control_signal_2 = self.apid.update(target, measured_value, dt)
        self.assertNotEqual(control_signal, control_signal_2)

if __name__ == '__main__':
    unittest.main()

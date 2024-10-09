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
        self.target = 10.0
        self.measured_value = 8.0
        self.dt = 0.1

    def test_proportional_action(self):
        """Test the proportional action."""
        self.apid.update(self.target, self.measured_value, self.dt)

        self.assertAlmostEqual(self.apid.error, self.target-self.measured_value)

    def test_update(self):
        """Test the update method."""
        control_signal = self.apid.update(self.target, self.measured_value, self.dt)
        
        self.assertIsInstance(control_signal, float)        
        self.assertGreater(control_signal, 0)
        
        self.measured_value = 9.0
        control_signal_2 = self.apid.update(self.target, self.measured_value, self.dt)
        self.assertNotEqual(control_signal, control_signal_2)

    def test_integral_action(self):
        """Test the integral action."""
        self.apid.update(self.target, self.measured_value, self.dt)
        prev_integral = self.apid.integral

        measured_value = 9.0
        self.apid.update(self.target, measured_value, self.dt)

        self.assertGreater(self.apid.integral, prev_integral)

    def test_derivative_action(self):
        """Test the derivative action."""
        self.apid.update(self.target, self.measured_value, self.dt)
        prev_derivative = self.apid.derivative

        self.measured_value = 9.0
        self.apid.update(self.target, self.measured_value, self.dt)

        self.assertLess(self.apid.derivative, prev_derivative)

if __name__ == '__main__':
    unittest.main()

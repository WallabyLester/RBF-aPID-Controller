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
        self.apid = AdaptivePIDTf(self.Kp, self.Ki, self.Kd, self.rbf_model)
        self.target = 10.0
        self.measured_value = 8.0
        self.dt = 0.1

    def test_initialization(self):
        """ Test initialization of the controller."""
        self.assertEqual(self.apid.Kp, self.Kp)
        self.assertEqual(self.apid.Ki, self.Ki)
        self.assertEqual(self.apid.Kd, self.Kd)
        self.assertIsInstance(self.apid.rbf_model, RBFAdaptiveModel)
        self.assertEqual(self.apid.prev_err, 0)
        self.assertEqual(self.apid.error, 0)
        self.assertEqual(self.apid.integral, 0)
        self.assertEqual(self.apid.derivative, 0)

    def test_update(self):
        """ Test the update method."""
        control_signal = self.apid.update(self.target, self.measured_value, self.dt)

        self.assertIsInstance(control_signal, float)
        self.assertEqual(self.apid.prev_err, self.apid.error)

        integral = self.apid.integral
        self.assertAlmostEqual(integral, (self.target - self.measured_value) * self.dt)

        derivative = self.apid.derivative
        self.assertAlmostEqual(derivative, (self.target - self.measured_value)/self.dt)


if __name__ == '__main__':
    unittest.main()

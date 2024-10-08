import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from RBF_numpy import RBFNetwork
from aPID_numpy import AdaptivePIDNP
from RBF_tf import RBFAdaptiveModel
from aPID_tf import AdaptivePIDTf

def simulate_system(controller, target, dt, T):
    """ Simulate control model as first order system.

    Parameters
    ----------
    controller : AdaptivePID
        Any AdaptivePID class instance.
    target : float64
        Target setpoint.
    dt : float64
        Timestep.
    T : float64
        Total time range to simulate.
    
    Returns
    -------
    Timesteps and measured_value at each.
    
    """
    time = np.arange(0, T, dt)
    measured_value = 0
    measurements = []

    for t in time:
        control_signal = controller.update(target, measured_value, dt)
        measured_value += (control_signal - measured_value) * dt  # Simple first-order system
        measurements.append(measured_value)
        print(f"Control Signal: {control_signal:.2f}, Measurement: {measured_value:.2f}")

    return time, measurements

if __name__ == "__main__":
    # numpy implementation
    np.random.seed(20)

    rbf_np = RBFNetwork(input_dim=3, n_centers=5)
    apid_np = AdaptivePIDNP(Kp=4.0, Ki=0.1, Kd=0.01, rbf_network=rbf_np)

    target = 1.0
    dt = 0.1
    T = 10.0
    time, measurements = simulate_system(apid_np, target, dt, T)

    plt.plot(time, measurements, label="Measured Value")
    plt.axhline(y=target, color="r", linestyle="--", label="Target")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title("Adaptive RBF Neural PID Controller Numpy")
    plt.legend()
    plt.grid()
    plt.show()

    # tensorflow implementation
    rbf_tf = RBFAdaptiveModel(n_centers=5, input_dim=3)
    rbf_tf.compile(optimizer="adam", loss="mean_squared_error")

    apid_tf = AdaptivePIDTf(Kp=7.0, Ki=0.5, Kd=0.01, rbf_model=rbf_tf)

    target = 1.0
    dt = 0.1
    T = 10.0
    time, measurements = simulate_system(apid_tf, target, dt, T)

    plt.plot(time, measurements, label="Measured Value")
    plt.axhline(y=target, color="r", linestyle="--", label="Target")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.title("Adaptive RBF Neural PID Controller TF")
    plt.legend()
    plt.grid()
    plt.show()

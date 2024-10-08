import numpy as np

class AdaptivePIDNP:
    """ PID class implemented for numpy integration. 

    ...

    Attributes
    ----------
    Kp : float64
        Proportional gain.
    Ki : float64
        Integral gain.
    Kd : float64
        Derivative gain.
    rbf_network : RBFNetwork object
        RBF network class instance.

    Methods
    -------
    update(target, measured_value, dt):
        Updates the control signal.    
    """
    def __init__(self, Kp, Ki, Kd, rbf_network):
        """ Constructs PID gains and RBF network.

        Parameters
        ----------
            Kp : float64
                Proportional gain.
            Ki : float64
                Integral gain.
            Kd : float64
                Derivative gain.
            rbf_network : RBFNetwork object
                RBF network class instance.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.rbf_network = rbf_network
        self.prev_err = 0
        self.error = 0
        self.integral = 0
        self.derivative = 0

    def update(self, target, measured_value, dt):
        """ Update the control signal according to error and adapt with RBF
        network predictions. 

        Parameters
        ----------
            target : float64
                Target setpoint.
            measured_value : float64
                Actual value.
            dt : float64
                Timestep.

        Returns
        -------
        Control signal.
        """
        self.error = target - measured_value
        self.integral += self.error * dt
        self.derivative = (self.error - self.prev_err) / dt

        u = (self.Kp * self.error) + (self.Ki * self.integral) + (self.Kd*self.derivative)

        gain_adapt = self.rbf_network.predict(np.array([self.error, self.integral, self.derivative]))
        u += gain_adapt

        self.prev_err = self.error
        return u
        
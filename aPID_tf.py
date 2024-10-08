import tensorflow as tf

class AdaptivePIDTf:
    """ PID class implemented for TensorFlow integration. 

    ...

    Attributes
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain.
    rbf_model : RBFAdaptiveModel object
        RBF adaptive model class instance.

    Methods
    -------
    update(target, measured_value, dt):
        Updates the control signal.    
    """
    def __init__(self, Kp, Ki, Kd, rbf_model):
        """ Constructs PID gains, RBF model, and initial PID components.

        Parameters
        ----------
            Kp : float
                Proportional gain.
            Ki : float
                Integral gain.
            Kd : float
                Derivative gain.
            rbf_model : RBFAdaptiveModel object
                RBF adaptive model class instance.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.rbf_model = rbf_model
        self.prev_err = 0
        self.error = 0
        self.integral = 0
        self.derivative = 0

    def update(self, target, measured_value, dt):
        """ Update the control signal according to error and adapt with RBF
        model predictions. 

        Parameters
        ----------
            target : float
                Target setpoint.
            measured_value : float
                Actual value.
            dt : float
                Timestep.

        Returns
        -------
        Control signal.
        """
        self.error = target - measured_value
        self.integral += self.error * dt
        self.derivative = (self.error - self.prev_err) / dt

        u = (self.Kp * self.error) + (self.Ki * self.integral) + (self.Kd*self.derivative)

        control_signal_adapt = self.rbf_model(tf.constant([[self.error, self.integral, self.derivative]])).numpy().flatten()[0]
        u += control_signal_adapt

        self.prev_err = self.error
        return u
        
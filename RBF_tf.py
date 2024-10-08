import tensorflow as tf

class RBFLayer(tf.keras.layers.Layer):
    """ RBF Layer using TF Subclassing API.

    ...

    Attributes
    ----------
    n_centers : int
        The number of RBF centers.
    input_dim : int
        The dimensions of the RBF centers.

    Methods
    -------
    call(inputs):
        TF call method to implement forward pass.
    """
    def __init__(self, n_centers, input_dim=3):
        """ Constructs RBF centers and standard deviations as trainable weights.

        Parameters
        ----------
            n_centers : int
                The number of RBF centers.
            input_dim : int
                The dimensions of the RBF centers. Default of 3 for Kp, Ki, Kd   
        """
        super().__init__()
        self.n_centers = n_centers
        self.centers = self.add_weight(shape=(n_centers, input_dim), 
                                       initializer="random_normal",
                                       trainable=True,)
        self.sigmas = self.add_weight(shape=(n_centers,),
                                      initializer="ones",
                                      trainable=True,)

    def call(self, inputs):
        """ Find likelihood of inputs under Gaussian distribution given center and 
        standard deviation. 

        Parameters
        ----------
            inputs : tensor
                The points in space to evaluate the Gaussian.

        Returns
        -------
        Height of Gaussian curve at inputs.          
        """
        distances = tf.norm(tf.expand_dims(inputs, axis=1) - self.centers, axis=2) 
        rbf_output = tf.exp(-tf.square(distances) / (2 * tf.square(self.sigmas))) 
        return rbf_output


class RBFAdaptiveModel(tf.keras.Model):
    """ RBF Adaptive Model using TF Subclassing API.

    Outputs one control signal adaptation. Determined by output layer
    number of neurons.

    ...

    Attributes
    ----------
    n_centers : int
        The number of RBF centers.

    Methods
    -------
    call(inputs):
        TF call method to implement forward pass of model.
    """
    def __init__(self, n_centers):
        """ Constructs RBF and output layers.

        Parameters
        ----------
            n_centers : int
                The number of RBF centers.
        """
        super().__init__()
        self.rbf_layer = RBFLayer(n_centers)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """ Implement forward pass.  

        Parameters
        ----------
            inputs : tensor
                The points in space to adapt with.

        Returns
        -------
        Adapted control signal.          
        """
        rbf_output = self.rbf_layer(inputs)
        control_signal = self.output_layer(rbf_output)
        return control_signal

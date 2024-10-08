import numpy as np

class RBFNetwork:
    """ Basic radial basis function (RBF) neural network class, numpy implementation. 

    Contains input layer, one hidden layer with RBF activation, and linear output layer. 
    RBF activation is a Gaussian.

    ...

    Attributes
    ----------
    input_dim : int
        The dimension of the RBF centers. 
    n_centers : int
        The number of RBF centers.

    Methods
    -------
    gaussian(x, center):
        Forms the gaussian given the center and input.
    predict(x, center):
        Predicts from the model using the gaussian and saved weights.
    train(x, target):
        Train the RBF model on stored data. 
    """
    def __init__(self, input_dim, n_centers):
        """ Constructs distribution parameters and initializes weights.

        Parameters
        ----------
            input_dim : int
                The dimension of the RBF centers.
            n_centers : int
                The number of RBF centers.
        """
        self.input_dim = input_dim
        self.n_centers = n_centers
        self.centers = np.random.rand(n_centers, input_dim)     # expected value
        self.sigma = 1.0                                        # variance
        self.weights = np.random.rand(n_centers)

    def gaussian(self, x, center):
        """ Find likelihood of x under Gaussian distribution centered at center with 
        standard deviation sigma. 

        Parameters
        ----------
            x : ndarray[Any, dtype[float64]]
                The point in space to evaluate the Gaussian.
            center : ndarray[Any, dtype[float64]]
                Mean/center of Gaussian distribution.

        Returns
        -------
        Height of Gaussian curve at x. 
        """
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * self.sigma ** 2))

    def predict(self, x):
        """ Prediction function of form dot(Activations, Weights).

        Parameters
        ----------
            x : ndarray[Any, dtype[float64]]
                The point in space to evaluate the Gaussian.

        Returns
        -------
        Approximation of the target function. 
        """
        activations = np.array([self.gaussian(x, center) for center in self.centers])
        return np.dot(activations, self.weights)

    def train(self, x, target):
        """ Training function to adapt weights to known datapoints.

        Parameters
        ----------
            x : ndarray[Any, dtype[float64]]
                The point in space to evaluate the Gaussian.
            target : float64
                Target data point.
        """
        activations = np.array([self.gaussian(x, center) for center in self.centers])
        self.weights += 0.01 * (target - np.dot(activations, self.weights)) * activations
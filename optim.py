import numpy as np


class SGD:
    """
    Implements vanilla SGD update
    """

    def update(self, w, d_w, learning_rate):
        """
        Performs SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        return w - d_w * learning_rate


class MomentumSGD:
    """
    Implements Momentum SGD update
    """

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = 0

    def update(self, w, d_w, learning_rate):
        """
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        self.velocity = self.momentum * self.velocity - learning_rate * d_w
        w += self.velocity
        return w


class AdamSGD:
    """
    Implements Adam SGD update
    """

    def __init__(self, alpha=0.5, beta=0.6):
        self.alpha = alpha
        self.beta = beta
        self.velocity = 0
        self.accumulated = 0

    def update(self, w, d_w, learning_rate):
        """
        Performs Adam SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        self.velocity = self.alpha * self.velocity + (1 - self.alpha) * d_w
        self.accumulated = self.beta * self.accumulated + (1 - self.beta) * d_w**2
        adaptive_learning_rate = learning_rate/np.sqrt(self.accumulated)
        w -= adaptive_learning_rate * self.velocity
        return w

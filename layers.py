import numpy as np
from sklearn.metrics import log_loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength*np.sum(np.square(W))
    grad = 2*reg_strength*W
    return loss, grad


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    norm_predictions = predictions - np.amax(predictions, axis=1)[:, None]
    exp_array = np.exp(norm_predictions)
    return exp_array/np.sum(exp_array, axis=1)[:, None]


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
      batch_size is a number of batches in probs array
    Returns:
      loss: single value
    """
    mask_array = np.zeros(probs.shape, dtype=int)
    for i in range(probs.shape[0]):
        mask_array[i, target_index[i]] = 1
    ce_loss = log_loss(mask_array, probs)
    return ce_loss


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    target_array = np.zeros(probs.shape, dtype=int)
    for i in range(target_array.shape[0]):
        target_array[i, target_index[i]] = 1
    dprediction = (probs - target_array)/probs.shape[0]
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(self.X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = np.multiply(d_out, np.int64(self.X > 0))
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        W = self.params()['W'].value
        B = self.params()['B'].value
        predictions = np.dot(X, W) + B
        return predictions

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        X = self.X
        W = self.params()['W'].value
        dresult = np.dot(d_out, W.T)
        dW = np.dot(X.T, d_out)
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out)
        self.params()['W'].grad += dW
        self.params()['B'].grad += dB
        return dresult

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        # padding of X with zeroes
        if self.padding > 0:
            pad_X = np.zeros((batch_size, height+2*self.padding, width+2*self.padding, channels))
            pad_X[:, self.padding:X.shape[0]+self.padding, self.padding:X.shape[1]+self.padding, :] = X
            self.X = pad_X
        # calculate output height ant width
        out_height = (height-self.filter_size+2*self.padding)+1
        out_width = (width-self.filter_size+2*self.padding)+1
        prediction = (np.zeros((batch_size, out_height, out_width, self.out_channels)))
        # get weight and bias parameters, reshape weight parameter to 2D array
        W = self.params()['W'].value.reshape(-1, self.out_channels)
        B = self.params()['B'].value
        # calculate forward path
        for y in range(out_height):
            for x in range(out_width):
                inp = self.X[:, y:(y+self.filter_size), x:(x+self.filter_size), :].reshape(batch_size, 1, -1)
                prediction[:, y, x, :] = (np.tensordot(inp, W, axes=([2], [0])) + B)[:, 0, :]
        return prediction

    def backward(self, d_out):
        X = self.X
        # get weight parameter, reshape weight parameter to 2D array
        W = self.params()['W'].value.reshape(-1, self.out_channels)
        filter_size = self.filter_size
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        # prepare variables for gradients
        dresult = np.zeros((batch_size, height, width, channels))
        dW = np.zeros((filter_size, filter_size, channels, out_channels))
        dB = np.zeros(out_channels)
        # calculate back path
        for y in range(out_height):
            for x in range(out_width):
                inp = X[:, y:(y+self.filter_size), x:(x+self.filter_size), :].reshape(batch_size, 1, -1)
                d_X = np.dot(d_out[:, y, x, :], W.T).reshape((batch_size, filter_size, filter_size, channels))
                dresult[:, y:(y+self.filter_size), x:(x+self.filter_size), :] += d_X
                d_W = np.tensordot(inp.T, d_out[:, y, x, :], axes=([2], [0])).reshape((filter_size, filter_size,
                                                                                       channels, out_channels))
                dW += d_W
                d_B = np.dot(np.ones((1, d_out[:, y, x, :].shape[0])), d_out[:, y, x, :]).reshape(out_channels)
                dB += d_B
        # update parameters
        self.params()['W'].grad += dW
        self.params()['B'].grad += dB
        dresult = dresult[:, self.padding:dresult.shape[1]-self.padding,
                          self.padding:dresult.shape[2]-self.padding, :]
        return dresult

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = np.zeros_like(X)
        out_height = int((height - self.pool_size)/self.stride + 1)
        out_width = int((width - self.pool_size)/self.stride + 1)
        result = np.zeros((batch_size, out_height, out_width, channels))
        for b in range(batch_size):
            for c in range(channels):
                for y in range(0, height, self.stride):
                    for x in range(0, width, self.stride):
                        inp = X[b, y:(y+self.pool_size), x:(x+self.pool_size), c]
                        max_ind = np.unravel_index(np.argmax(inp), inp.shape)
                        self.X[b, y+max_ind[0], x+max_ind[1], c] = 1
                        result[b, int(y/self.stride), int(x/self.stride), c] = inp[max_ind]
        return result

    def backward(self, d_out):
        dresult = np.zeros_like(self.X)
        batch_size, height, width, channels = self.X.shape
        for b in range(batch_size):
            for c in range(channels):
                for y in range(0, height, self.stride):
                    for x in range(0, width, self.stride):
                        out = self.X[b, y:(y+self.pool_size), x:(x+self.pool_size), c]
                        max_ind = np.unravel_index(np.argmax(out), out.shape)
                        dresult[b, y+max_ind[0], x+max_ind[1], c] = d_out[b, int(y/self.stride), int(x/self.stride), c]
        return dresult

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}

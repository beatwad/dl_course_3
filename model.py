import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.height = input_shape[0]
        self.width = input_shape[1]
        self.input_channels = input_shape[2]
        self.n_output_classes = n_output_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels

        self.conv1_layer = ConvolutionalLayer(self.input_channels, self.conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(4, 4)

        self.conv2_layer = ConvolutionalLayer(self.conv1_channels, self.conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(4, 4)

        self.flattener = Flattener()
        self.fc_layer = FullyConnectedLayer(self.conv2_channels, self.n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # nullify layers gradients
        # Conv1 Layer
        self.params()['W1'].grad = np.zeros((3, 3, self.input_channels, self.conv1_channels))
        self.params()['B1'].grad = np.zeros(self.conv1_channels)
        # Conv2 Layer
        self.params()['W2'].grad = np.zeros((3, 3, self.conv1_channels, self.conv2_channels))
        self.params()['B2'].grad = np.zeros(self.conv2_channels)
        # FC Layer
        self.params()['W3'].grad = np.zeros((self.conv2_channels, self.n_output_classes))
        self.params()['B3'].grad = np.zeros(self.n_output_classes)

        # forward conv layer 1
        conv_forward1 = self.conv1_layer.forward(X)
        # forward relu activation funtcion 1
        relu_forward1 = self.relu1.forward(conv_forward1)
        # forward maxpool layer 1
        maxpool_forward1 = self.maxpool1.forward(relu_forward1)

        # forward conv layer 2
        conv_forward2 = self.conv2_layer.forward(maxpool_forward1)
        # forward relu activation funtcion 2
        relu_forward2 = self.relu2.forward(conv_forward2)
        # forward maxpool layer 2
        maxpool_forward2 = self.maxpool2.forward(relu_forward2)

        # forward flattener layer
        flattener_forward = self.flattener.forward(maxpool_forward2)
        # calculate flattener output data shape and create FC layer
        batch_size, height, width, channels = self.flattener.X_shape
        self.fc_layer = FullyConnectedLayer(height*width*channels, self.n_output_classes)
        self.params()['W3'].grad = np.zeros((height*width*channels, self.n_output_classes))
        self.params()['B3'].grad = np.zeros((1, self.n_output_classes))
        # forward FC layer
        fc_forward = self.fc_layer.forward(flattener_forward)

        # calculate loss and grad
        loss, grad = softmax_with_cross_entropy(fc_forward, y)

        # backward FC layer
        fc_backward = self.fc_layer.backward(grad)
        # backward flattener layer
        flattener_backward = self.flattener.backward(fc_backward)

        # backward maxpool layer 2
        maxpool_backward2 = self.maxpool2.backward(flattener_backward)
        # backward relu activation funtcion 2
        relu_backward2 = self.relu2.backward(maxpool_backward2)
        # forward conv layer 2
        conv_backward2 = self.conv2_layer.backward(relu_backward2)

        # backward maxpool layer 1
        maxpool_backward1 = self.maxpool1.backward(conv_backward2)
        # backward relu activation funtcion 1
        relu_backward1 = self.relu1.backward(maxpool_backward1)
        # forward conv layer 1
        conv_backward1 = self.conv1_layer.backward(relu_backward1)
        return loss

    def predict(self, X):
        # forward conv layer 1
        conv_forward1 = self.conv1_layer.forward(X)
        # forward relu activation funtcion 1
        relu_forward1 = self.relu1.forward(conv_forward1)
        # forward maxpool layer 1
        maxpool_forward1 = self.maxpool1.forward(relu_forward1)

        # forward conv layer 2
        conv_forward2 = self.conv2_layer.forward(maxpool_forward1)
        # forward relu activation funtcion 2
        relu_forward2 = self.relu2.forward(conv_forward2)
        # forward maxpool layer 2
        maxpool_forward2 = self.maxpool2.forward(relu_forward2)

        # forward flattener layer
        flattener_forward = self.flattener.forward(maxpool_forward2)
        # calculate flattener output data shape and create FC layer
        batch_size, height, width, channels = self.flattener.X_shape
        self.fc_layer = FullyConnectedLayer(height * width * channels, self.n_output_classes)
        self.params()['W3'].grad = np.zeros((height * width * channels, self.n_output_classes))
        self.params()['B3'].grad = np.zeros((1, self.n_output_classes))
        # forward FC layer
        fc_forward = self.fc_layer.forward(flattener_forward)
        # make prediction
        prediciton = fc_forward.argmax(axis=1)
        return prediciton

    def params(self):
        result = {'W1': self.conv1_layer.params()['W'], 'B1': self.conv1_layer.params()['B'],
                  'W2': self.conv2_layer.params()['W'], 'B2': self.conv2_layer.params()['B'],
                  'W3': self.fc_layer.params()['W'], 'B3': self.fc_layer.params()['B']}
        return result

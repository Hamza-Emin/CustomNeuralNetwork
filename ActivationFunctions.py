import numpy as np

class ActivationFunction:
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def activate(self, x):
        if self.activation_type == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_type == 'tanh':
            return self.tanh(x)
        elif self.activation_type == 'relu':
            return self.relu(x)
        elif self.activation_type == 'softmax':
            return self.softmax(x)

    def derivative(self, x):
        if self.activation_type == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_type == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation_type == 'relu':
            return self.relu_derivative(x)
        elif self.activation_type == 'softmax':
            return self.softmax_derivative(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanh_derivative(self, x):
        return 1 - pow((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 2)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def softmax_derivative(self, x):

        pass

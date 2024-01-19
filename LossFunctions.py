import numpy as np

class LossFunctions:
    def __init__(self, loss_type):
        self.loss_type = loss_type

    def calculate_loss(self, y_true, y_pred):
        if self.loss_type == 'mean_squared_error':
            return self.mean_squared_error(y_true, y_pred)
        elif self.loss_type == 'binary_crossentropy':
            return self.binary_crossentropy(y_true, y_pred)
        elif self.loss_type == 'categorical_crossentropy':
            return self.categorical_crossentropy(y_true, y_pred)
        elif self.loss_type == 'hinge_loss':
            return self.hinge_loss(y_true, y_pred)
        elif self.loss_type == 'huber_loss':
            return self.huber_loss(y_true, y_pred)



    def mean_squared_error(self, y_true, y_pred):
        n = len(y_true)
        return np.sum((y_true - y_pred) ** 2) / (2 * n)



    def binary_crossentropy(self, y_true, y_pred):
        n = len(y_true)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n



    def categorical_crossentropy(self, y_true, y_pred):
        n = len(y_true)
        return -np.sum(y_true * np.log(y_pred)) / n



    def hinge_loss(self, y_true, y_pred):
        n = len(y_true)
        return np.sum(np.maximum(0, 1 - y_true * y_pred)) / n


    def huber_loss(self, y_true, y_pred, delta=1.0):
        n = len(y_true)
        diff = np.abs(y_true - y_pred)
        quadratic_part = np.minimum(diff, delta)
        linear_part = diff - quadratic_part
        return np.sum(0.5 * quadratic_part ** 2 + delta * linear_part) / n




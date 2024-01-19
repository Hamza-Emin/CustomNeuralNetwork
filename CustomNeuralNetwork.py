import numpy as np
from ActivationFunctions import ActivationFunction
from LossFunctions import LossFunctions

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation, loss):

        self.weights_of_input_hidden = np.random.rand(input_size, hidden_size) * 0.0001
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_of_hidden_output = np.random.rand(hidden_size, output_size) * 0.0001
        self.bias_output = np.zeros((1, output_size))


        self.activation = activation
        self.loss = loss

    def forward(self, inputs):

        hidden_input = np.dot(inputs, self.weights_of_input_hidden) + self.bias_hidden
        self.hidden_output = self.activation.activate(hidden_input)


        output_input = np.dot(self.hidden_output, self.weights_of_hidden_output) + self.bias_output
        self.output = self.activation.activate(output_input)

        return self.output

    def backward(self, inputs, targets, learning_rate):
        # Calculate the error and delta for the output layer
        error_output = targets - self.output
        delta_output = error_output * self.activation.derivative(self.output)


        error_hidden = delta_output.dot(self.weights_of_hidden_output.T)
        delta_hidden = error_hidden * self.activation.derivative(self.hidden_output)


        self.weights_of_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_of_input_hidden += inputs.T.dot(delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_sample = inputs[i:i + 1]
                target_sample = targets[i:i + 1]


                output = self.forward(input_sample)


                self.backward(input_sample, target_sample, learning_rate)


                if epoch % 1000 == 0:
                    error = self.loss.calculate_loss(target_sample, output)
                    print(f"Epoch {epoch}, Error: {error}")

# Example usage:
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [1]])


activation_function = ActivationFunction(activation_type='sigmoid')
loss_function = LossFunctions(loss_type='binary_crossentropy')


nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, activation=activation_function, loss=loss_function)


nn.train(inputs, targets, epochs=10000, learning_rate=0.1)


predictions = nn.forward(inputs)
print("\nFinal Predictions:")
for i in range(len(predictions)):
    print(f"Input: {inputs[i]}, Predicted Output: {predictions[i][0]:.4f}")

import numpy as np
import pandas as pd
np.random.seed(42)  # for reproducible results

class NeuronalNet():
    """
    A simple feedforward neural network with multiple hidden layers.

    Attributes:
        input_size (int): Number of input features.
        hidden_sizes (list[int]): List containing the number of neurons in each hidden layer.
        output_size (int): Number of output neurons.
        learning_rate (float): Learning rate for weight updates.
        layer_sizes (list[int]): List containing sizes of all layers (input + hidden + output).
        weight_matrix (list[np.ndarray]): List of weight matrices for each layer.
    """

    def __init__(self, input_size: int, hidden_sizes: int | list[int], output_size: int, learning_rate: float):
        """
        Initialize the neural network with random weights.

        Args:
            input_size (int): Number of input neurons.
            hidden_sizes (int | list[int]): Number of neurons in each hidden layer.
            output_size (int): Number of output neurons.
            learning_rate (float): Learning rate for weight updates.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layer_sizes = [self.input_size] + (self.hidden_sizes if isinstance(self.hidden_sizes, list) else [self.hidden_sizes]) + [self.output_size]

        # Initialize weight matrices for all layers with random values in range [-0.5, 0.5]
        self.weight_matrix = [
            np.random.rand(self.layer_sizes[i+1], self.layer_sizes[i]) - 0.5 
            for i in range(len(self.layer_sizes) - 1)
        ]

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of the sigmoid function.
        
        Note: x here is the activation output, not the linear input.
        """
        return x * (1 - x)
    
    @staticmethod
    def one_hot_encode(y, num_classes):
        """
        Convert labels to one-hot encoding.
        
        Args:
            y (array-like): Labels as integers.
            num_classes (int): Number of classes.
        
        Returns:
            ndarray: One-hot encoded labels.
        """
        y_encoded = np.zeros((len(y), num_classes)) + 0.01
        for i, label in enumerate(y):
            y_encoded[i, int(label)] = 0.99
        return y_encoded

    def feedforward(self, inputs: np.ndarray):
        """
        Perform a forward pass through the network.

        Args:
            inputs (list): Input values for the network.

        Returns:
            np.ndarray: Output activations of the network.
        """
        inputs_v = np.array(inputs, ndmin=2).T
        activations = inputs_v

        for matrix in self.weight_matrix:
            activations = self.sigmoid(np.dot(matrix, activations)) 
        return activations

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Train the network with a single input-target pair using backpropagation.
        """
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # Forward pass
        activations = [inputs]
        actv = inputs
        for matrix in self.weight_matrix:
            actv = self.sigmoid(np.dot(matrix, actv))
            activations.append(actv)

        # Calculate output error
        errors = [targets - activations[-1]]
        deltas = [errors[0] * self.sigmoid_derivative(activations[-1])]

        # Backpropagate errors
        for i in range(len(self.weight_matrix)-1, 0, -1):
            delta = np.dot(self.weight_matrix[i].T, deltas[0]) * self.sigmoid_derivative(activations[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weight_matrix)):
            self.weight_matrix[i] += self.learning_rate * np.dot(deltas[i], activations[i].T)

    def fit(self, X_train: pd.DataFrame|pd.Series|np.ndarray, y_train: pd.DataFrame|pd.Series|np.ndarray, epochs: int) -> None:
        """
        Train the network over multiple epochs using training data.
        """
        # ---- Prepare inputs ----
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            # If each cell contains an array → stack them
            if isinstance(X_train.iloc[0], (np.ndarray, list)):
                X_train = np.stack(X_train.apply(lambda x: np.ravel(x)))
            else:
                X_train = X_train.values
        else:
            X_train = np.array(X_train)

        # ---- Prepare targets ----
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            if isinstance(y_train.iloc[0], (np.ndarray, list)):
                # Already one-hot encoded
                y_train = np.stack(y_train.apply(lambda x: np.ravel(x)))
            else:
                y_train = y_train.values
        else:
            y_train = np.array(y_train)

        # If labels are 1D → one-hot encode
        if y_train.ndim == 1:
            y_train = self.one_hot_encode(y_train, self.output_size)

        for _ in range(epochs):
            for i in range(len(X_train)):
                self.train(X_train[i], y_train[i])

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict output labels for given input data.
        """
        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            # If each cell contains an array → stack them
            if isinstance(X_test.iloc[0], (np.ndarray, list)):
                X_test = np.stack(X_test.apply(lambda x: np.ravel(x)))
            else:
                X_test = X_test.values
        else:
            X_test = np.array(X_test)
        return np.array([np.argmax(self.feedforward(X_test[i])) for i in range(len(X_test))])
    
    def accuracy(self, X_test: pd.DataFrame, y_test: pd.DataFrame)-> float:
        """
        Calculate the accuracy of the network on test data.
        """
        # ---- Prepare inputs ----
        if isinstance(X_test, (pd.DataFrame, pd.Series)):
            # If each cell contains an array → stack them
            if isinstance(X_test.iloc[0], (np.ndarray, list)):
                X_test = np.stack(X_test.apply(lambda x: np.ravel(x)))
            else:
                X_test = X_test.values
        else:
            X_test = np.array(X_test)

        # ---- Prepare targets ----
        if isinstance(y_test, (pd.DataFrame, pd.Series)):
            if isinstance(y_test.iloc[0], (np.ndarray, list)):
                # Already one-hot encoded
                y_test = np.stack(y_test.apply(lambda x: np.ravel(x)))
            else:
                y_test = y_test.values
        else:
            y_test = np.array(y_test)
        
        # Predictions
        predictions = self.predict(X_test)
        
        # Decode one-hot labels if necessary
        if y_test.ndim > 1:
            true_labels = np.array([np.argmax(y) for y in y_test])
        else:
            true_labels = y_test

        return np.mean(predictions == true_labels)

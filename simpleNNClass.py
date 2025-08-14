#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 2025
@author: apostolosmav
"""


import numpy as np
import pandas as pd
np.random.seed(42) # results reproduct

class NeuronalNet():
    """
    A simple feedforward neural network with one hidden layer.
    
    Attributes:
        input_nodes (int): Number of input nodes.
        hidden_nodes (int): Number of hidden nodes.
        output_nodes (int): Number of output nodes.
        learning_rate (float): Learning rate for weight updates.
        weights_input_hidden (ndarray): Weights matrix between input and hidden layer.
        weights_hidden_output (ndarray): Weights matrix between hidden and output layer.
    """

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        """
        Initialize network parameters and weight matrices.
        Weights are initialized randomly in range [-0.5, 0.5].
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize weights with small random numbers
        self.weights_input_hidden = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weights_hidden_output = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        
        Args:
            x (ndarray): Input array.
        
        Returns:
            ndarray: Sigmoid of input.
        """
        return 1 / (1 + np.exp(-x))
    
    def feedforward(self, inputs: np.ndarray):
        """
        Perform a forward pass through the network.
        
        Args:
            inputs (ndarray): Input features as 1D array.
        
        Returns:
            ndarray: Output layer activations.
        """
        inputs_v = np.array(inputs, ndmin=2).T  # Convert to column vector
        hidden_layer = self.sigmoid(np.dot(self.weights_input_hidden, inputs_v))
        output_layer = self.sigmoid(np.dot(self.weights_hidden_output, hidden_layer))
        return output_layer
        
    def train(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Train the network on a single example using backpropagation.
        
        Args:
            inputs (ndarray): Input features as 1D array.
            targets (ndarray): Target outputs as 1D array (one-hot if classification).
        """
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # Forward pass
        hidden_layer = self.sigmoid(np.dot(self.weights_input_hidden, inputs))
        output_layer = self.sigmoid(np.dot(self.weights_hidden_output, hidden_layer))

        # Compute errors
        output_error = targets - output_layer
        hidden_error = np.dot(self.weights_hidden_output.T, output_error)

        # Update weights
        self.weights_hidden_output += np.dot((output_error * output_layer * (1 - output_layer)), hidden_layer.T) * self.learning_rate
        self.weights_input_hidden += np.dot((hidden_error * hidden_layer * (1 - hidden_layer)), inputs.T) * self.learning_rate

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

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, epochs: int):
        """
        Train the network on the whole dataset for a number of epochs.
        
        Args:
            X_train (DataFrame): Training features.
            y_train (DataFrame): Training labels.
            epochs (int): Number of times to iterate over the training data.
        """
        X_train = X_train.values
        y_train = y_train.values

        # Convert labels to one-hot encoding if necessary
        if y_train.ndim == 1:
            y_train = self.one_hot_encode(y_train, self.output_nodes)
        
        # Train for multiple epochs
        for _ in range(epochs):
            for i in range(len(X_train)):
                self.train(X_train[i], y_train[i])

    def predict(self, X_test: pd.DataFrame):
        """
        Predict the class labels for given inputs.
        
        Args:
            X_test (DataFrame): Test features.
        
        Returns:
            ndarray: Predicted class labels.
        """
        X_test = X_test.values
        return np.array([np.argmax(self.feedforward(X_test[i])) for i in range(len(X_test))])
    
    def accuracy(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Compute the accuracy of the network on test data.
        
        Args:
            X_test (DataFrame): Test features.
            y_test (DataFrame): True labels.
        
        Returns:
            float: Accuracy as a proportion of correct predictions.
        """
        X_test = X_test.values
        y_test = y_test.values

        predictions = self.predict(X_test)

        # Decode one-hot labels if necessary
        if y_test.ndim > 1:
            true_labels = np.array([np.argmax(y) for y in y_test])
        else:
            true_labels = y_test

        return np.mean(predictions == true_labels)


        
    
        





  

        
        





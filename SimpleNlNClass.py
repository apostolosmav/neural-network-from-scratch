import numpy as np
import pandas as pd
np.random.seed(42) # results reproduct

class NeuronalNetz():

    def __init__(self,input_nodes:int,hidden_nodes:int,output_dotes:int,learning_rate:float):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_dotes = output_dotes
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.rand(self.hidden_nodes,self.input_nodes) - 0.5 
        self.weights_hidden_output = np.random.rand(self.output_dotes,self.hidden_nodes) - 0.5

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def feedforward(self,inputs:list):
        inputs_v = np.array(inputs,ndmin=2).T
        
        hidden_layer= self.sigmoid(np.dot(self.weights_input_hidden,inputs_v))
        output_layer = self.sigmoid(np.dot(self.weights_hidden_output,hidden_layer))

        return output_layer
        

    
    def train(self,inputs:list,targets:np.array):
        # inputs,targets in arrays
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T


        # forward propagation
        hidden_layer= self.sigmoid(np.dot(self.weights_input_hidden,inputs))
        output_layer = self.sigmoid(np.dot(self.weights_hidden_output,hidden_layer))

        # calculate error 
        output_error = targets - output_layer
        hidden_error = np.dot(self.weights_hidden_output.T,output_error)

        # adjust weights
        self.weights_hidden_output += np.dot((output_error * output_layer * (1 - output_layer)),hidden_layer.T) * self.learning_rate
        self.weights_input_hidden += np.dot((hidden_error * hidden_layer * (1 - hidden_layer)),inputs.T) * self.learning_rate

    def fit(self,X_train:pd.DataFrame,y_train:pd.DataFrame,epochs:int):
        X_train = X_train.values.tolist()
        y_train = y_train.values.tolist()

        for epoch in range(epochs):
            for i in range(len(X_train)):
                targets_v = np.zeros(self.output_dotes) + 0.01
                targets_v[int(y_train[i])] = 0.99 
                self.train(X_train[i],targets_v)

    def predict(self,X_test:pd.DataFrame):
        X_test = X_test.values.tolist()
        return [np.argmax(self.feedforward(X_test[i])) for i in range(len(X_test))]

        
        



